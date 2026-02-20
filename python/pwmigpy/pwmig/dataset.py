#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 07:58:26 2026

@author: pavlis
"""
from mspasspy.util.db_utils import MongoDBWorker
from mspasspy.ccore.utility import AntelopePf
from pwmigpy.pwmig.pwmig import migrate_event
from pwmigpy.pwmig.pwstack import pwstack
from pwmigpy.db.database import GCLdbsave
from mspasspy.util.seismic import print_metadata

def pwmig_dataset(mspass_client,
                 dbname,
                 pfname="pwmig.pf",
                 source_collection="telecluster",
                 pwstack_data_tag="pseudostation_stacks",
                 base_fieldname="pwmigdata",
                 outdir=None,
                 restart=False,
                 minimum_data=10000,
                 verbose=False,
                 initialize_workers=True,
                 ):
    """
    Driver to process all output from pwstack to produce a suite of 
    pwmig output image volumes stored in the database.  This function 
    can be rerun multiple times if the processing is aborted prematurely. 
    That can happen too easily on HPC systems when you guess badly
    on the time required for to run the function. 
    
    The primary use of this function to drive a workflow at the stage when the 
    data have all been run through pwstack.  i.e. this function 
    expects to read output created by pwstack and managed by 
    MongoDB in the database defined by "dbname".   It is more-or-less 
    a loop over all event ids (normally defined by "telecluster_id" values).  
    If the restart options is set True the function will first check 
    for completed id values and not reprocess those.  That is how 
    checkpointing is handled for this commonly long-running process.  
    It works because data from each event is not dependent upon others. 
    The final image is produced by summing all the componets form the 
    entire data set created by running this function. 
    
    Note if you create a mega job to run this function in the same 
    HPC job as pwstack set initialize_workers to False. 
    
    :param mspass_client:  As the name suggests the instance of 
      `mspasspy.client.Client` controlling this run. 
    :param dbname:  name (not a Database instance the name only) of the 
      MongoDB database being used to manage this dataset.
    :param pfname:  name of the "pf-file" containing control parameters for pwmig.
    :param source_collection:  Name of the collection that defines source id 
      values used as a unique idenifier for each event.  Default "telecluster"
      is unlikely to ever need to be changed but is possible to allow 
      alternative workflows to get to this stage.  
    :param pwstack_data_tag:   data tag to identify the instance of outputs 
      from pwstack to use. The default matches default tag created by 
      pwstack so this need to be change if and only if you change the 
      output tag set by pwstack.  If pwstack is run on the same data 
      with different options or inputs the data tag should change.   
      Use this parameter to select the unique run of pwstack you want to 
      use as the input.  The default matches the default output for pwstack. 
    :param base_fieldname:   output images have a "name" tag used to 
      create unique tags in the MongodB database.   The tags created by 
      migrate_event, which this function drives, are this name + the str 
      representation of the source id (normally str(telecluster_id)).  
      Change this name only if you plan to rerun pwmig multiple times on 
      the same data with variations in inputs to either upstream processing 
      and/or the pf control parameters. 
    :param outdir:  The image data is written to file in a binary format with 
      file names derived from "base_fieldname".  Those files will be written 
      in this directory.  If it does not exist it will be created.  
      Default is "pwmigdata" which means "./pwmigdata".   
    :param restart: boolean that when set True causes the function to 
      scan the database and edit the list of source ids to process.  As noted 
      in the description paragraphs this is how checkpointing is implemented 
      for this very compute and memory intensive process.   The default is 
      False which cause the prefiltering to be bypassed.  If you rerun without 
      setting this True you will get duplicate copies of previously processed 
      data. 
    :param minimum_data:   source regions with small numbers of events 
      stacked into composite events by pseudosource_stacker often have 
      only fractional coverage.  If they are used they do more harm than 
      good to a final image as you are summing data with large voids 
      inside a 3d volume.   Use this parameter to not process data from 
      an event when the total number of pwstack data for that event is 
      less than this threshold.  Note is is very important to realize 
      the total count is the number of pseudostation cells time the
      number of plane components.   Hence, if, for example, pwstack was
      used 121 plane wave components (an 11x11 slowness grid) the the 
      threshold for number of pseudostation with data would be 
      minimum_data/121.   The default is 10000 which means drop data if 
      coverage falls below 10000/121 surface grid points.  
    :param intialize_workers: boolean controlling initialization of workers 
      to setup database clients.  Default is True which is appropriate in 
      the (normally recommended) run where the only thing the job does is 
      run pwmig.  Since pwmig is a resource pig that usually requires 
      special jobs specification that would be normal.  
    :param verbose:  when True (default is False) prints more information 
      to stdout.  
    """
    # lelt this throw an exception if pffile is not found or messed up
    pf = AntelopePf(pfname)
    db = mspass_client.get_database(dbname)
    daskclient=mspass_client.get_scheduler()
    if initialize_workers:
        print("Creating worker plugin")
        dbplugin = MongoDBWorker(dbname)
        print("Running register_plugin")
        daskclient.register_plugin(dbplugin)
    else:
        print("WARNING:  running parallel but initialize_workers was set False")
        print("This job may abort if workers were not previously initalized with the MongoDBWorker plugin")
        

    # this complicated incantation is needed to extract ids only 
    # from those matching the data tag from pwstack.   The "distinct" 
    # method only works on an entire collection.  Here I use 
    # a set container to accumulalte a list of unique ids
    # works because a python set is not a "multiset" so the only unique 
    # values are kept
    idkey=source_collection + "_id"
    idset=set()
    base_query={"data_tag" : pwstack_data_tag}
    cursor=db.wf_Seismogram.find(base_query)
    for doc in cursor:
        if idkey in doc:
            idset.add(doc[idkey])
    idlist = list(idset) # list accepts a set in its constructor
    if restart:
        print("Running in restart mode:  checking for completed events")
        completed = db.GCLfielddata.distinct(idkey)
        print("Found ",len(completed)," documents in GCLfielddata that appear to be pwmig output")
        newlist=list()
        for tcid in idlist:
            if tcid not in completed:
                newlist.append(tcid)
        print("Size of original list of all event ids=",len(idlist))
        print("Size of edited list=",len(newlist))
        idlist = newlist
    if verbose:
        print("Starting processing of ",len(idlist)," events")
    for sid in idlist:
        fieldname = base_fieldname + "_" + str(sid)
        imagedata = migrate_event(mspass_client,db.name,sid,pf,fieldname,
          minimum_data=minimum_data,
          verbose=verbose,
          base_query=base_query,
          )
        if imagedata is not None:
            auxdata={"source_collection":source_collection,idkey:sid}
            md = GCLdbsave(db,imagedata,dir=outdir,dfile=fieldname,auxdata=auxdata)
            if verbose:
                print("Document created for file=",fieldname)
                print_metadata(md)
    

def pwstack_dataset(mspass_client,
                    dbname,
                    pfname="pwstack.pf",
                    wf_query=None,
                    data_tag="pseudosource_stacks",
                    pseudosource_stacker_algorithm="weighted_stack",
                    source_collection="telecluster",
                    parallel=True,
                    initialize_workers=True,
                    output_data_tag="pwstack_data",
                    outdir="pwstack_output",
                    verbose=False,
                    restart=False,
                    ):
    """
    Run the pwstack part of the pwmig package on an entire data set. 
    
    This function is mostly a wrapper to provide a consistent, simplifer
    API to run pwstack on an entire assembled dataset.   The input 
    of this function is expected to be the output of pseudosource_stacker
    saved in a MsPASS database with a unique data dag defined the 
    data_tag valued set by a run of pseudosource_stacker.  In addition, 
    pseudosource_stacker is usually run with multiple stacking algorithms 
    distinguished by a secondary key-value pair with the key 
    "stacking_algorithm".   i.e. the data set is defined by a match of 
    the two keys "data_tag" and "stack_algorithm" with documents in 
    wf_Seismogram.   The default value of "data_tag" is the default 
    value for outputs from pseudosource_stacker.   If you run that 
    algorithm more than once with different data tags use the 
    data_tag argument of this function to select an alternative run. 
    A more common need is to change which output with  "stacking_algorithm"
    is to be used.  The default is "weighted_stack".  Alternatives, which 
    are constants in pseudosource_stacker, are: "average", "median", and 
    "robust_dbxcor".   If you need a more elaborate method of selecting 
    the data set to be processed that can be specified with the wf_query 
    argument.  If defined the default prefiltering of inputs with data_tag 
    and stacking_algorithm will be ignored ans wf_query will be passed 
    directly to pwstack.  Use that feature only if you completely understand 
    how it will be uses in pwstack.  
    
    This function has a restart option.   Use that option if a first try 
    at running this function was aborted from a data problem or 
    (more commonly) a mistake in estimating the run time that caused the 
    first run to aborted by a batch scheduler like SLURM for exceeding 
    the run time request.  Note it is HIGHLY RECOMMENDED if you have a 
    job abort you run the function pwmigpy.pwmig.pwstack.checkpoint_report 
    and look carefully at the output.   Partially processed events may need 
    to be manually removed from the database before a restart if the 
    report shows an issue.  
    
    :param mspass_client: instance of MsPASS client used to initialize your 
      job.  A stock incantation at the top of almost all MsPASS jobs.
    :param dbname:  name of the database containing the output of
      pseudosource_stacker.  
    :param pfname:  name of the "pf-file" of control parameters for pwstack. 
      The function loads the contents of this file as an AntelopePf object 
      passed to pwstack.  It defines all the parameters that control details 
      of how pwstack is to be run.  Default is "pwstack.pf" which means the 
      function expects to find a file "pwstack.pf" in the current directory 
      or in a search path defined by PFPATH (if defined).   It can also 
      define a fully qualified file pathname as long as the leaf file is 
      what pwstack expects.  The recommended use is the default which is to 
      put the file pwstack.pf in the run directory.  
      
    :param wf_query:
    :param data_tag:
    :param pseudosource_stacker_algorithm: These three kwargs are tightly 
      interconnected as described above.   Defaults are the norm as they 
      match defaults of outputs from pseudousource_stacker.  

    :param source_collection:   used to define keys that define sources. 
       Default is telecluster which means the key telecluster_id is used 
       to define the source geometry for each ensemble sent for processing. 
       The default should never be changed unless the input has been 
       created in a completely different way than through pseudosource_stacker.. 
    :param parallel:  boolean that when True (default) run pwstack on a cluster
       accessed via mspass_client.  
    :param initialize_workers:  boolean that when True (default) forces an 
      initialziation of dask workers to allow parallel database reads and 
      writes.  Ignored when parallel is set False.  A warning is issued if 
      parallel is True and this is set False.  False is appropriate only 
      for a workflow that does other parallel steps earlier an the workers 
      were alreeady initialized.   
    :param output_data_tag:   data tag to use for outputs of pwststack. 
       Default is "pwstack_data".  If you rerun pwstack multiple times 
       (normal if you use more than one of the output options of pseudosource_stacker)
       you should can this argument for each run.  Otherwise pwmig may 
       run with overlapping inputs.  
    :param outdir:  directory to hold output files of pwstack data.
    :param verbose:  be more verbose.  Default is mostly silent.
    :param restart:  Set True if (default is False) if you are rerunning 
      this function after it was aborted before finishing processsing of 
      one event.   Think of this as a checkpointing functionality.  
      Npte when this option is enabled the checkpoint_report function is 
      always run and printed in the output.  Events not completely processed
      will cause programs if passed downstream to pwmig.  
    
    """
    valid_algorithms=["average","median","weighted_stack","robust_dbxcor"]        
    # handle wf query complexity feature
    if wf_query is None:
        wf_query = {"data_tag" : data_tag}
        if pseudosource_stacker_algorithm in valid_algorithms:
            wf_query["stacking_algorithm"]=pseudosource_stacker_algorithm
        else:
            message = "pwstack_dataset:  Illegal value for pseudosource_stacker_algorith={}".format(pseudosource_stacker_algorithm)
            raise ValueError(message)
    pf = AntelopePf(pfname)
    db = mspass_client.get_database(dbname)
    daskclient=mspass_client.get_scheduler()
    if parallel:
        if initialize_workers:
            print("Creating worker plugin")
            dbplugin = MongoDBWorker(dbname)
            print("Running register_plugin")
            daskclient.register_plugin(dbplugin)
        else:
            print("WARNING:  running parallel but initialize_workers was set False")
            print("This job may abort if workers were not previously initalized with the MongoDBWorker plugin")
    pwstack(db,pf,wf_query=wf_query,source_collection=source_collection,
            verbose=verbose,storage_mode='file',outdir=outdir,
            run_serial=(not parallel),output_data_tag=output_data_tag,
            restart=restart)  

        
        