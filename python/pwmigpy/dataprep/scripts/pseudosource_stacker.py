#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file defines a CLI tool that can be used for stacking data 
grouped by source region with the closely related CLI tool telecluster.
It can be thought of as a python translation of a C++ program in the 
original pwmig package called RFeventstacker.   The resemblance, however, 
is only in what it does.  This implementation is drastically differentn 
adding additional functionality and using MongoDB instead of 
the Antelope RDBMS system used by the C++ implementation.  

TODO:   document CLI usage line

Created on Sun Sep 28 07:23:12 2025

@author: pavlis
"""

import sys
import argparse
import copy
import time
from bson.objectid import ObjectId
import dask.distributed as ddist
from mspasspy.client import Client
from mspasspy.db.normalize import ObjectIdMatcher,normalize
from mspasspy.ccore.utility import AntelopePf,Metadata,ErrorSeverity
from mspasspy.ccore.seismic import SeismogramEnsemble
from mspasspy.algorithms.basic import rotate_to_standard
from mspasspy.util.Janitor import Janitor
from mspasspy.util.db_utils import fetch_dbhandle,MongoDBWorker
from mspasspy.util.seismic import number_live
import pwmigpy.dataprep.binned_stacking as bsm
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees

class bsscontrol():
    """
    This class is a conveneice structure to simplify use of the control 
    data loaded for this application from a pf file.   The complexity is 
    needed to allow multiple stacking methods to be applied to the 
    same data.  Although never actually tested other experience makes it 
    clear that computing multiple stacks while the data is in memory 
    is order of magnitude faster than reading it multiple times and running 
    a tool with a different stack option.   The reason is the compuations 
    here are guaraneed to be io bound for any converted wave imaging 
    data set I can conceive at this time on any machine suitable for 
    running pwmig.  
    
    The class is driven by an AntelopePf object read from a file 
    assed to the construtor.   That constructor translates the pf to 
    this class with convenience methods used to drive ths CLI tool.
    """
    def __init__(self,pf):
        """
        Constructor.  Input is an AntelopePf object read from a "pf-file".
        """
        # enabled has boolean values defining if an algorithm is to be run
        # for each True algorithm we load argdoc.  Note there will be no 
        # no data in argdoc when the corresonding enabled entry is False
        self.alg_enabled = dict()
        self.argdoc = dict()
        
        # the structure of the pf allows this simple loop to 
        # parse the pf file.
        self.algorithm_list = ["average","weighted_average","median","robust_dbxcor"]
        # the data w need for this object are on this branch 
        pfalg = pf.get_branch("stacking_parameters")
        for key in self.algorithm_list:
            pfb = pfalg.get_branch(key)
            useme = pfb["enable"]
            self.alg_enabled[key] = useme
            self.argdoc[key] = pfb.todict()
        
        #TODO:  may want some sanity checks on values here
    def enabled(self,algorithm)->bool:
        """
        Returns a True if algorithm is marked to be run and False otherwise. 
        If algorithm does match the objects list of known algorithms it 
        will raise a ValueError exception
        """
        if algorithm in self.algorithm_list:
            return self.alg_enabled[algorithm]
        else:
            message="bscontrol.enabled:  do not grok algorithm={}\n".format(algorithm)
            message += "Must be one of: {}".format(str(self.algorithm_list))
            raise ValueError(message)
    def getargs(self,algorithm)->dict:
        """`
        Returns a dictionary of the parameters for an algorithm.
        Caller must handle the keys and options.  This is just a wrapper 
        to fetch teh dictionary linked tot he algorithm.   
        It can throw a value error is the algorithm value is illegal.
        """
        if algorithm in self.algorithm_list:
            return self.argdoc[algorithm]
        else:
            message = "bscontrol.getarg:  do not grok algorithm={}\n".format(algorithm)
            message += "Must be one of: {}".format(str(self.algorithm_list))
            raise ValueError(message)
        
def parse_telecluster_source_ids(doc)->list:
    """
    The documents telecluster creates have a list of source ids converted to 
    strings.  Here we basically convert that to a list of source_id ObjectId 
    values.   Those are then used by the load_and_sort reader. 
    """
    id_string_list = doc["events"]
    idlist = list()
    for idstr in id_string_list:
        sid = ObjectId(idstr)
        idlist.append(sid)
    return idlist

def load_special_attributes(ensemble,algorithm,argdoc,clusterdoc)->SeismogramEnsemble:
    """
    Small function to post Metadata attributes used to drive the stacks
    in ensemble.   It saves all the pf branch attribues for the 
    algorithm used and most of the content of the document driving the
    processing created by telecluser.  The list of source_ids is dropped
    to reduce he size what is saved as that data can be linked to the 
    ensemble data when saved via the gridcell data. 
    
    :param ensemble:   ensemble to which attributes are to be posted 
      (returned wih only ensmble Metadata changed)
    :param algorithm:  name of algorithm run to produce stacks in ensemble
    :param algdoc:  pf branch data extraced for algorithm from full pf object
    :param clusterdoc:  document created by telecluster that was used to 
      create this ensemble.
    """
    ensemble["stacking_algorithm"] = algorithm
    ensemble["stacking_parameters"] = argdoc
    cdkeys=["gridname","hypocentroid","gridcell"]
    for key in cdkeys:
        ensemble[key] = clusterdoc[key]
    ensemble["telecluster_id"] = clusterdoc["_id"]
    return ensemble
        

def make_dfile_name(clusterdoc)->str:
    """
    Creates dfile name to save this ensemble.  File name is created from 
    telecluster document using a base name and grid index values.  
    A typical example would be:  stackdata_bin_2_5
    """
    subdoc = clusterdoc['gridcell']
    azindex=subdoc["azimuth_index"]
    delta_index = subdoc["distance_index"]
    dfile = "stackdata_bin_{}_{}".format(azindex,delta_index)
    return dfile

def create_stack_md(keyed_ensembles,stack_mdlist)->Metadata:
    """
    The robust_stack function in mspass requires a Metadata container 
    with content that is copied to the stack it computes.   It requires 
    that in the use here where there is no initial stack from which it 
    can clone required Metadata.   
    
    For robustness this algorithm will scan the entire dataset, if 
    necessary to fetch a value for keys in the stack_mdlist argument.   
    That is a bit overkill as in most cases the first live datum 
    should have the required data if the keys are defined correctly.  
    A corollary is if a key is not unique what will be set is the first 
    one found in keyed_ensembles
    
    :param ensemble:  dictionary of ensemble values defining the data to 
      be stacked.  Output of load_and_sort or equivalent.  That is, it is 
      assumed to be a dicationary with SeismogramEnsemles as the values 
      associated with each key.  The algorithms over each esnemble and 
      inside the members of each ensemble until it finds a value for each 
      key in stack_mdlist.  Caution that this will be expensive for a large 
      dataset if a key in stack_mdlist is not defined for any datum in 
      the entire data set.  
    :param stack_mdlist:  list of strings defining keys to be extracted 
      and copied to output.
    :return:  Metadata contaienr with copys of attributes defined b 
      stack_mdlist.   Note if a key is not found in any member it will 
      be silently absent from the output.  If that is a problem caller 
      should compare len(stack_mdlist) and len(result.keys) for 
      equality and use an error handler if that is an issue.
    
    """
    keys_still_to_copy=copy.deepcopy(stack_mdlist)
    md = Metadata()
    for ekey in keyed_ensembles:
        ensemble = keyed_ensembles[ekey]
        for i in range(len(ensemble.member)):
            if ensemble.member[i].live:
                # this is necessary because we pop keys when they are found
                current_list = copy.deepcopy(keys_still_to_copy)
                for key in current_list:
                    if key in ensemble.member[i]:
                        md[key] = ensemble.member[i][key]
                        keys_still_to_copy.pop(key)
                if len(keys_still_to_copy) == 0:
                    return md
    return md
            
def pf2snrwt_control(pf)->dict:
    """
    Create control structure to drive signal-to-noise ratio conversions 
    to weights for weighted stacks.
    
    The structure of the pf is expected to resolve to nested dictionaries.
    Easier explained with an example:
        
        ```
        snr_weighting &Arr{
            snr_RF.snr_H &Arr{
                weight_key RF_snrwt
                snr_floor 10.0
                weight_floor 0.01
                full_weight_snr 500.0
            }
            Parrival.snr_filtered_peak &Arr{
                weight_key snr_filtered_peak_wt
                snr_floor 10.0
                weight_floor 0.01
                full_weight_snr 500.0
            }
            Parrival.median_snr &Arr{
                weight_key median_snr_wt
                snr_floor 10.0
                weight_floor 0.01
                full_weight_snr 500.0
            }
        ]

        ```
    where the top level key is "snr_weight".  The enclosed &Arr 
    are returned as dictionaries with key defined by the associated
    &Arr tag.  e.g. result["snr_RF.snr_H"] would retrieve a dictionary
    with the data in the "snr_RF.snr_H" block.
    
    An more generic perspective of this function is it converts a 
    set of nested Arr blocks to a dictionary of dictionaries.  
    
    :param pf: AntelopePf object created from a pf file.  Must contain an 
      Arr section keyed by "snr_weighting" with content as described above.
    :type pf:  mspasspy.ccore.utility.AntelopePf object
    """
    pfsnr = pf.get_branch("snr_weighting")
    wt_control = dict()
    for key in pfsnr.arr_keys():
        md = pfsnr.get_branch(key)
        doc = dict(md)
        wt_control[key] = doc
    return wt_control

def set_weights(dataset,
                wt_control,
                summary_weight_output_key="weight",
                magnitude_weight_key=None)->dict:
    """
    Sets weights in dataset using control information passed via the 
    wt_control dictionary created by the function above called pf2snrwt_control.
    Note this could be parallelized but for now left serial as it isn't 
    a challenging calculation.
    """
    summary_weight_key_list=list()
    for wkey in wt_control:
        this_wt_control = wt_control[wkey]
        summary_weight_key_list.append(this_wt_control["weight_key"])
    if magnitude_weight_key:
        summary_weight_key_list.append(magnitude_weight_key)
    
    for dkey in dataset.keys():
        ens = dataset[dkey]
        for wkey in wt_control:
            wtctrl = wt_control[wkey]
            ens = bsm.set_snr_weights(ens,
                                  wkey,
                                  wtctrl["weight_key"],
                                  snr_floor=wtctrl["snr_floor"],
                                  weight_floor=wtctrl["weight_floor"],
                                  full_weight_snr=wtctrl["full_weight_snr"],
                            )
        ens = bsm.set_ensemble_summary_weights(ens,
                                               summary_weight_key_list,
                                               summary_weight_output_key)
        dataset[dkey] = ens
                                  
    return dataset
def get_base_query_from_pf(pf)->dict:
    """
    Small functin to convert pf data in the branch wf_Seismogram_base_query 
    to a dictionary defining a query for MongoDB.  
    """
    md = pf.get_branch("wf_Seismogram_base_query")    
    return md.todict()
def srcidlist2querylist(base_query,srcidlist)->list:
    """
    Creates a list of dictionaries that are MongoDB queries combining 
    the base query and a query for a single source_id in each list element.
    """
    querylist=list()
    for srcid in srcidlist:
        query = copy.deepcopy(base_query)
        query["source_id"] = srcid
        querylist.append(query)
    return querylist

def set_fake_starttimes(stacked_data,clusterdoc,refmodel):
    """
    Set t0 value of all live data in stacked_data using P times computed from 
    hypocentroid coordinates in clusterdoc and site coordinates for each 
    station.  Assumes stacked_data have been normalized.   Because of 
    internal use in this tool there is no checking that required metadata
    exist.   
    """
    # in this program source data was posted to the ensemble Metadata 
    # container so we pull it out here
    srclat = bsm.get_subdoc_value(stacked_data, "hypocentroid.lat")
    srclon = bsm.get_subdoc_value(stacked_data, "hypocentroid.lon")
    srcdepth = bsm.get_subdoc_value(stacked_data, "hypocentroid.depth")
    otime = bsm.get_subdoc_value(stacked_data, "hypocentroid.time")
    for d in stacked_data.member:
        if d.live:
            stalat = d['site_lat']
            stalon = d['site_lon']
            georesult = gps2dist_azimuth(srclat,srclon,stalat,stalon)
            dist=kilometers2degrees(georesult[0]/1000.0)
            arrivals=refmodel.get_travel_times(source_depth_in_km=srcdepth,
                                                distance_in_degree=dist,
                                                phase_list=['P'])
            d['epicentral_distance']=dist
            esaz = georesult[1]
            seaz = georesult[2]
            d['esaz']=esaz
            d['seaz']=seaz
            if len(arrivals)>=1:
                Ptime = otime + arrivals[0].time
                newt0 = Ptime + d.t0
                d['Ptime'] = Ptime
                d.force_t0_shift(newt0)
            else:
                message = "Failed to compute P wave arrival time to set timing for this stack"
                d.kill()
                d.elog.log_error("set_fake_starttimes",message,ErrorSeverity.Invalid)
    return stacked_data

def special_index_state(db,index_name="pseudosource_stacker_index",keys=["data_tag", "source_id"])->int:
    """
    This function is used in pseudsource_stacker to test if teh special index required by the application 
    exists.   The index must match in two ways:  (1) the "name" field and (2) the keys defined for the 
    index IN ORDER (indexes are hierarchic).   

    Note the **kwargs paameters for this function should not be changed.  They there to make this 
    slightly more generic.   The index name, in particular, could change.  

    :param db:  Database object referencing the dataset of interest.
    :return:  
      0  - index with name defined by index_name does not exist and can be created
      1  - index exists and matches name and keys
      -1  - index exists with name of index_name but the keys do not match 
    """
    cursor=db.wf_Seismogram.list_indexes()
    indexdoc=None
    for doc in cursor:
        if doc['name'] == index_name:
            indexdoc=doc
            break
    if indexdoc is None:
        return 0
    # indexdoc is not a dictionary or a list but has some concepts from both
    # This algorithm is used because len is defined but "in" does not work
    indexkeys = indexdoc['key']
    if len(indexkeys)!=len(keys):
        return -1
    i=0
    for k in keys:
        if keys[i]!=k:
            return -1
        i+=1
    return 1

def verify_wfindex(db,verbose=False):
    """
    Called from main to assure the special index with the fixed name 
    "pseudosource_stacker_index" is defined and of the required form.  
    That form is the two keys in order:  "data_tag" and "source_id".  
    If the index exists and matches reuirements the function does nothing. 
    If an index with the special name does not exist it is created. 
    If the index exists but is not of the required form the function 
    throws a RuntimeError exception to abort the application.  
    """
    index_state = special_index_state(db)
    match index_state:
        case -1:
            message = "verify_wfindex:  special index for wf_Seismogram called pseudosource_stacker_index exists but does not match requirements\n"
            message += "remove the existing index and run rerun this application - it will create the correct index automatically"
            raise RuntimeError(message)
        case 1:
            if verbose:
                print("index with name pseudosource_stacker_index exists in correct form - no action needed")
        case 0:
            if verbose:
                print("pseudosource_stacker_index does not exist - creating it now")
                db.wf_Seismogram.create_index([("data_tag" , 1),('source_id' , 1)],name="pseudosource_stacker_index")
        case _:
            message = "verify_wfindex:  special_index_state returned an unexpected falue={}\n".format(index_state)
            message += "This should not happen and is a bug that needs to be fixed"
            raise RuntimeError(message)

def process_group(clusterdoc,
                  dbname,
                  base_query,
                  magwt_control,
                  site_matcher,
                  control,
                  janitor,
                  snrwt_control,
                  refmodel,
                  output_directory,
                  output_data_tag,
                  verbose=False,
                  ):
    db = fetch_dbhandle(dbname)
    if verbose:
        print("Begin processing for data with telecluster_id=",clusterdoc[['_id']])
    source_id_list = parse_telecluster_source_ids(clusterdoc)
    query_list = srcidlist2querylist(base_query,source_id_list)

    if magwt_control["enable"]:
        dataset = bsm.load_and_sort(db, 
                                    query_list,
                                    magnitude_key=magwt_control["magnitude_key"],
                                    full_weight_magnitude=magwt_control["full_weight_magnitude"],
                                    floor_magnitude=magwt_control["floor_magnitude"],
                                    minimum_weight=magwt_control["minimum_weight"],
                                    default_magnitude=magwt_control["default_magnitude"],
                                    magnitude_weight_key=magwt_control["magnitude_weight_key"],
                                    )
    else:
        dataset = bsm.load_and_sort(db)
    input_size = 0
    input_live_size = 0
    for key in dataset.keys():
        ens = dataset[key]
        input_size += number_live(ens)
        # this loads metadata from site collection
        # important to note that default used here loads the data into 
        # the ensemble Metadata container.  That is corect as the 
        # load_and_sort function returns data sorted by site_id.  
        ens = normalize(ens,site_matcher)
        # TODO:  this is a temporary workaround for a bug in decorators
        # use the ensemble version when it is resolved
        dataset[key] = rotate_to_standard(ens)
        #for d in ens.member:
        #    if d.live:
        #        d.rotate_to_standard()
        dataset[key]=ens
        input_live_size += number_live(ens)
    output_size = 0
    output_number_live = 0
    for algorithm in control.algorithm_list:
        if control.enabled(algorithm):
            argdoc = control.getargs(algorithm)
            if verbose:
                print("Computing stacks with algorithm=",algorithm)
            match algorithm:
                case "average":
                    # average currently has no options so argdoc is ignored
                    stacked_data = bsm.stack_groups(dataset,
                                                method=algorithm,
                                                janitor=janitor)
                case "weighted_average":
                    dataset = set_weights(dataset,
                                          snrwt_control,
                                          summary_weight_output_key=argdoc["weight_key"],
                                    )
                    stacked_data = bsm.stack_groups(dataset,
                                    method=algorithm,
                                    weight_key=argdoc["weight_key"],
                                    undefined_weight=argdoc["undefined_weight"],
                                    )
                case "median":
                    stacked_data = bsm.stack_groups(dataset,
                                    method=algorithm,
                                    janitor=janitor,
                                    timespan_method=argdoc["timespan_method"],
                                    pad_fraction_cutoff=argdoc["pad_fraction_cutoff"],
                                    )
                case "robust_dbxcor":
                    stacked_data = bsm.stack_groups(dataset,
                                    method=algorithm,
                                    janitor=janitor,
                                    timespan_method=argdoc["timespan_method"],
                                    pad_fraction_cutoff=argdoc["pad_fraction_cutoff"],
                                    residual_norm_floor=argdoc["residual_norm_floor"],
                                    )
            if stacked_data.live:
                output_size += number_live(stacked_data)
                stacked_data=load_special_attributes(stacked_data,
                                                     algorithm,
                                                     argdoc,
                                                     clusterdoc,
                                                         )
                # this operation could be done in stacked_data but 
                # more maintanable if done here for a minor cost
                # problem is site metadata is not retained for stacked 
                # data but site_id values are.  Hence we have to renormalzie
                # the stacked data.
                # note the override of default to force normalization by members
                stacked_data = normalize(stacked_data,site_matcher,handles_ensembles=False)
                stacked_data = set_fake_starttimes(stacked_data,clusterdoc,refmodel)
                # this is loaded from the telecluster collection but 
                # we change the key name here.  Cleare this way than 
                # if it had been pushed to the load_special_attributes
                # function.
                stacked_data["telecluster_events"] = source_id_list
                dfile=make_dfile_name(clusterdoc)
                output_number_live += number_live(stacked_data)
                db.save_data(stacked_data,
                             collection="wf_Seismogram",
                             storage_mode="file",
                             dir=output_directory,
                             dfile=dfile,
                             data_tag=output_data_tag,
                             )
    # cleanup before exiting and return the id from clusterdoc
    del stacked_data
    del dataset
    return [clusterdoc,input_size,input_live_size,output_size,output_number_live]
def main(args=None):
    """
    Command line tool replacement for original C++ program in original 
    pwmig that was called RFeventstacker.   
    
    This tool stacks data binned previouisly by telecluster the 
    cluster components stored in documents in a collection called 
    "telecluster".   It is primarily intended to deconvolved data but 
    with proper use it should work for an alternative to implement the 
    ancient idea from Scott Neal's dissertation with decon of 
    pseeudostation stacked data.  That is a research problem though.  
    
    Experience has shown this algorithm is almost immediately io bound 
    even with only a handful of workers.   The problem seems worse 
    on HPC lustre file systems where reading the data requires a fair 
    number of file open-closes.   To be concrete on the IU cluster 
    I found running this reading lustre files any more than 8 workers 
    was pointless.   
    """
    t0=time.time()
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog = "pseudosource_stacker",
        usage="%(prog)s dbname [-pf pffile -outdir output_directory -tag dtag -model 1dmodel -w sliding_window_size -p -v]",
        description="Stack groups of common source gathers using groupings created by telecluster",
    )
    parser.add_argument(
        "dbname",
        metavar="dbname",
        type=str,
        help="MongoDB database name containg source collection to be processed",
    )
    parser.add_argument(
        "-pf",
        "--pffile",
        action="store",
        type=str,
        default="pseudosource_stacker.pf",
        help="Define parameter file that defines tool options (default pseudosouorce_stacker.pf)",
    )
    parser.add_argument(
        "-outdir",
        "--output_directory",
        action="store",
        type=str,
        default="binned_stacks",
        help="Set directory for stacked output files (default ./binned_stacks)",
        )
    parser.add_argument(
        "-tag",
        "--data_tag",
        action="store",
        type=str,
        default="pseudosource_stacks",
        help="data_tag value passed to Dataase.save_data for all outputs (default pseudosource_stacks)",
        )
    parser.add_argument(
        "-m",
        "--model",
        action="store",
        type=str,
        default="iasp91",
        help="Reference 1d earth model for travel time calculations (default iasp91)",
        )
    parser.add_argument(
            "-p",
            "--parallel",
            action="store_true",
            help="Run parallel (default is serial)",
        )
    parser.add_argument(
            "-w",
            "--swsize",
            action="store",
            type=int,
            default="20",
            help="Sliding window of futures size (default 20)",
        )
    parser.add_argument(
           "-v",
           "--verbose",
           action="store_true",
           help="Print some progress data instead of start and end info (off by default)"
        )
    args = parser.parse_args(args)
    if args.verbose:
        verbose = True
        print("pseudosource_stacker:  started and running in verbose mode")
    else:
        verbose = False
    if args.parallel:
        parallel=True
        print("Parallel mode - assuming dask cluster is active")
    else:
        parallel=False
    output_directory = args.output_directory
    dtag = args.data_tag
    refmodname = args.model
    refmodel = TauPyModel(refmodname)
    mspass_client=Client()
    dbname = args.dbname
    db = mspass_client.get_database(dbname)
    if parallel:
        dask_client = mspass_client.get_scheduler()
        print("Creating worker plugin")
        dbplugin = MongoDBWorker(dbname)
        print("Running register_plugin")
        dask_client.register_plugin(dbplugin)
    # we always require site data to calculate travel times 
    # Could do this more generically but for require this be site with 
    # ObjectId matching
    site_matcher = ObjectIdMatcher(db,collection="site",attributes_to_load=["_id","lat","lon","elev"])
    pffile = args.pffile
    pf = AntelopePf(pffile)
    control = bsscontrol(pf)
    md = pf.get_branch("magnitude_weighting")
    magwt_control = dict(md)
    if control.enabled("weighted_average"):
        snrwt_control = pf2snrwt_control(pf) 
    janitor=Janitor()
    auxmdkeys = pf.get_tbl("stack_add2keepers_list")
    for key in auxmdkeys:
        janitor.add2keepers(key)
    base_query = get_base_query_from_pf(pf)
    # for efficiency it is important to create this index if it doesn't exist
    # this function defined above handles that 
    # if it returns the index exists with the right form.
    # it can throw an exception but when it does the program will and should abort
    verify_wfindex(db,verbose=verbose)
    N = db.telecluster.count_documents({})
    if N==0:
        message="pseudosource_stacker:   telecluster collection is empty - no data to process"
        raise RuntimeError(message)
    print("pseudosource_stacker:  working on {} source clusters defined in telecluster collection".format(N))
    if verbose:
        print("telecluster_id number_in number_in_live number_out number_live_out")
    if parallel:
        swsize = args.swsize
        if verbose:
            print(f"Using sliding window size={swsize}")
        f_site_matcher = dask_client.scatter(site_matcher,broadcast=True)
        f_refmodel = dask_client.scatter(refmodel,broadcast=True)
        # cache the entire list of documents or we will get cursor timeouts
        doclist=list()
        cursor = db.telecluster.find({})
        for doc in cursor:
            doclist.append(doc)
        cursor.close()
        futureslist=list()
        i_p = 0

        for doc in doclist:
            # assumes process_group default for verbose is false
            f=dask_client.submit(process_group,
                                 doc,
                                 dbname,
                                 base_query,
                                 magwt_control,
                                 f_site_matcher,
                                 control,
                                 janitor,
                                 snrwt_control,
                                 f_refmodel,
                                 output_directory,
                                 dtag,
               )
            futureslist.append(f)
            i_p += 1
            if i_p>=swsize:
                break
        
        # Use the sliding window algorithm
        # similar to pwmig use but there is not accumulation here
        seq = ddist.as_completed(futureslist)
        for f in seq:
            f_out = f.result()
            dask_client.cancel(f)
            del f
            if verbose:
                idout = f_out[0]['_id']
                print(f"{idout} {f_out[1]} {f_out[2]}  {f_out[3]} {f_out[4]}")
            if i_p < N:
                f=dask_client.submit(process_group,
                                     doclist[i_p],
                                     dbname,
                                     base_query,
                                     magwt_control,
                                     f_site_matcher,
                                     control,
                                     janitor,
                                     snrwt_control,
                                     f_refmodel,
                                     output_directory,
                                     dtag,
                   )
                seq.add(f)
                i_p += 1
    else:
        # outer loop over groupings defined by telecluster
        # parallel version driven by list of docs
        cursor = db.telecluster.find({})   
        for clusterdoc in cursor:
            f_out = process_group(clusterdoc,
                              dbname,
                              base_query,
                              magwt_control,
                              site_matcher,
                              control,
                              janitor,
                              snrwt_control,
                              refmodel,
                              output_directory,
                              dtag,
                              verbose=verbose,
            )
            if verbose:
                idout = f_out[0]['_id']
                print(f"{idout} {f_out[1]} {f_out[2]}  {f_out[3]} {f_out[4]}")
    t=time.time()
    print("Elapsed time=",t-t0)

if __name__ == "__main__":
    main()

