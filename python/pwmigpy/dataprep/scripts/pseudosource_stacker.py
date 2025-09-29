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
from bson.objectid import ObjectId
from mspasspy.db.client import DBClient
from mspasspy.db.database import Database
from mspasspy.ccore.utility import AntelopePf,Metadata
from mspasspy.ccore.seismic import SeismogramEnsemble
import pwmigpy.dataprep.binned_stacking as bsm

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
        self.enabled = dict()
        self.argdoc = dict()
        
        # the structure of the pf allows this simple loop to 
        # parse the pf file.
        self.algorithm_list = ["average","weighted_average","median","robust_dbxcor"]
        # the data w need for this object are on this branch 
        pfalg = pf.get_branch("stacking_parameters")
        for key in self.algorithm_list:
            pfb = pfalg.get_branch(key)
            useme = pfb["enable"]
            self.enabled[key] = useme
            self.argdoc = pfb.todict()
        
        #TODO:  may want some sanity checks on values here
    def enabled(self,algorithm)->bool:
        """
        Returns a True if algorithm is marked to be run and False otherwise. 
        If algorithm does match the objects list of known algorithms it 
        will raise a ValueError exception
        """
        if algorithm in self.algorithm_list:
            return self.enabled[algorithm]
        else:
            message = "bscontrol.enabled:  do not grok algorithm={}\n",format(algorithm)
            message += "Must be one of: {}".format(self.algorithm_list)
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
            message = "bscontrol.getarg:  do not grok algorithm={}\n",format(algorithm)
            message += "Must be one of: {}".format(self.algorithm_list)
            raise ValueError(message)
        
def parse_telecluster_source_ids(doc)->list:
    """
    The documents telecluster creates have a list of source ids conerted to 
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
    return ensemble
        

def make_dfile_name(clusterdoc)->str:
    """
    Creates dfile name to save this ensemble.  File name is created from 
    telecluster document using a base name and grid index values.  
    A typical example would be:  stackdata_bin_2_5
    """
    azindex=clusterdoc["azimuth_index"]
    delta_index = clusterdoc["distance_index"]
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
            
    

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog = "pseudosource_staacker",
        usage="%(prog)s dbname [-pf pffile -outdir output_directory -tag dtag]",
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
        "pffile",
        action="store",
        type=str,
        default="pseudostation_stacker.pf",
        help="Define parameter file that defines tool options",
    )
    parser.add_argument(
        "-outdir",
        "output_directory",
        action="store",
        type=str,
        default="binned_stacks",
        help="Defined directory for stacked outputs - files of stack for each pseudostation point",
        )
    parser.add_argument(
        "-tag",
        "data_tag",
        action="store",
        type=str,
        default="pseudostation_stacks",
        help="data_tag value passed to Dataase.save_data for all outputs",
        )
    args = parser.parse_args(args)
    output_directory = args.outdir
    dtag = args.tag
    dbname = args.dbname
    dbclient=DBClient()
    db=Database(dbclient,dbname)
    pffile = args.pffile
    pf = AntelopePf(pffile)
    control = bsscontrol(pf)
    stack_mdlist = pf.gettbl("stack_mdlist")
    # outer loop over groupings defined by telecluster
    # currently read all - TODO:  add optional query of telecluster collction
    # note this could be parallelized over this outer loop
    cursor = db.telecluster.find({})   
    for clusterdoc in cursor:
        source_id_list = parse_telecluster_source_ids(clusterdoc)
        dataset = bsm.load_and_sort(db)
        for algorithm in control.algorithm_list:
            if control.enabled(algorithm):
                argdoc = control.getargs(algorithm)
                match algorithm:
                    case "average":
                        # average currently has no options so argdoc is ignored
                        stacked_data = bsm.stack_groups(dataset,
                                                    method=algorithm)
                    case "weighted_stack":
                        stacked_data = bsm.stack_groups(dataset,
                                        method=algorithm,
                                        weight_key=argdoc["weight_key"],
                                        undefined_weight=argdoc["undefined_weight"],
                                        )
                    case "median":
                        md_to_clone = create_stack_md(dataset, stack_mdlist)
                        stacked_data = bsm.stack_groups(dataset,
                                        method=algorithm,
                                        stack_md=md_to_clone,
                                        timespan_method=argdoc["timespan_method"],
                                        pad_fracton_cutoff=argdoc["pad_fraction_cutoff"],
                                        )
                    case "robust_dbxcor":
                        # note for now we always default stack0 to None which 
                        # caused a median stack to be used as the initial 
                        # estimator
                        md_to_clone = create_stack_md(dataset, stack_mdlist)
                        stacked_data = bsm.stack_groups(dataset,
                                        method=algorithm,
                                        stack_md=md_to_clone,
                                        timespan_method=argdoc["timespan_method"],
                                        pad_fracton_cutoff=argdoc["pad_fraction_cutoff"],
                                        residual_norm_floor=argdoc["residual_norm_floor"],
                                        )
                if stacked_data.live:
                    stacked_data=load_special_attributes(stacked_data,
                                                         algorithm,
                                                         argdoc,
                                                         clusterdoc,
                                                             )
                    dfile=make_dfile_name(clusterdoc)
                    db.save_data(stacked_data,
                                 collection="wf_Seismogram",
                                 storage_mode="file",
                                 dir=output_directory,
                                 dfile=dfile,
                                 data_tag=dtag,
                                 )
    

if __name__ == "__main__":
    main()

