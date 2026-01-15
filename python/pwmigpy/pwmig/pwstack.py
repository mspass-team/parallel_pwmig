#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the python implementation of the pwstack
algorithm.  pwstack was originally a c++ executable.  This set of functions
allow pwstack to be implemented in a mspass workflow.  The file first 
defines a set of functions and a python required by the primary driver 
function called pwstack.  The pwstack function is at the end of the file.   
To run the pwstack function requires these preliminary steps:
    1.  Instiate an instance of a mspass database client
    2.  Loading a parameter file (".pf") to create and instance of an
        AntelopePf object.   The pf defines the main control inputs 
        for pwstack.
    3.  Parallel workflows need to create and install the MongoDBWorker 
        plugin.   The details of that will evolve as something similar 
        to MongodBWorker inside this module is in the process of being 
        implemented in MsPASS.  

Author:  Gary L. Pavlis with contributions from Chenbo Yin and Ian Wang.
"""
import dask
import math
import copy
# for parallell debugging - remove for production tests
import dask.distributed as ddist


from mspasspy.ccore.seismic import SeismogramEnsemble
# We have a python wrapper for the C++ implementation of top mutes (_TopMute)
# but here we need the direct C++ call because we are calling the
# pwstack_ensemble function that wants the C++ function not the wrapper
#from mspasspy.algorithms.window import TopMute
from mspasspy.ccore.algorithms.basic import _TopMute
from mspasspy.ccore.utility import MsPASSError,ErrorSeverity
from mspasspy.util.seismic import number_live
from mspasspy.db.normalize import ObjectIdMatcher,normalize
from mspasspy.db.database import Database

from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth,kilometers2degrees

from pwmigpy.ccore.pwmigcore import (RectangularSlownessGrid,
                                   DepthDependentAperture,
                                   pwstack_ensemble)
from pwmigpy.db.database import GCLdbread


def TopMuteFromPf(pf,tag):
    """
    The C++ class TopMute does not have an AntelopePf driven constructor.
    This function is a front end to work around that in python.
    Could have been done in C++ but a trivial constuct either way.
    The mutetype is required for mspass implementation.  We default it
    here to cosine for backward compability with old pwstack pf files.

    :param pf:  AntelopePf that is assumed to have a branch with label tag
    :param tag: string defining branch name to extract

    :return: TopMute constructed from pf data

    """
    pfbranch=pf.get_branch(tag)
    if pfbranch.is_defined('mute_type'):
        mutetype=pfbranch.get_string('mute_type')
    else:
        mutetype = 'cosine'
    # the old pwstack names for these parameters were very confusing
    # documentation should say to use t0 and t1 but accept the old
    # names for backward compatibility with old pf files.
    if pfbranch.is_defined('t0'):
        t0=pfbranch.get_double('t0')
    else:
        t0=pfbranch.get_double('zero_end_time')
    if pfbranch.is_defined('t1'):
        t1=pfbranch.get_double('t1')
    else:
        t1=pfbranch.get_double('end_time')
    return _TopMute(t0,t1,mutetype)

class pwstack_control:
    """
    This class is used to contain all the control parameters that
    define processing with the pwstack algorithm.   The constructor
    is pretty much like the initialization section of the old
    pwstack main.  Most content is loaded from the AntelopePf object 
    passed as the required arg pf.   The kwarg keys with "tag" in the name 
    are all optional name tags for pf components where optional names 
    could be helpful.   
    """
    def __init__(self,db,pf,slowness_grid_tag='RectangularSlownessGrid',
            data_mute_tag='data_top_mute',
                 stack_mute_tag='stack_top_mute',
                     save_history=False,instance='undefined'):
        # These are control parameters passed to pwstack_ensemble
        self.tstart = pf.get_double('data_time_window_start')
        self.tend = pf.get_double('data_time_window_end')
        self.SlowGrid=RectangularSlownessGrid(pf,slowness_grid_tag)
        self.data_mute=TopMuteFromPf(pf,data_mute_tag)
        self.stack_mute=TopMuteFromPf(pf,stack_mute_tag)
        use_fresnel=pf.get_bool('use_fresnel_aperture')
        if use_fresnel:
            fvs=pf.get_double("fresnel_vs")
            fvp=pf.get_double("fresnel_vp")
            fcm=pf.get_double("fresnel_cutoff_multiplier")
            fperiod=pf.get_double("fresnel_period")
            fdt=pf.get_double("fresnel_lag_time_sampling_interval")
            fnt=pf.get_long("fresnel_number_lag_samples")
            self.aperture=DepthDependentAperture(fvs,fvp,fperiod,fdt,fnt,fcm,True)
        else:
            self.aperture=DepthDependentAperture(pf,"depth_dependent_aperture")
        self.aperture_taper_length=pf.get_double("aperture_taper_length")
        self.centroid_cutoff=pf.get_double("centroid_cutoff")
        self.stack_count_cutoff = pf.get_long("stack_count_cutoff")
        # These were added for mspass converion - old pfs will need to
        # add these
        # None is used for a null for data_tag.   We default that here
        if pf.is_defined('data_tag'):
            self.data_tag=pf.get_string('data_tag')
        else:
            self.data_tag = None
        self.save_history=save_history
        self.algid=instance
        # default this one to iasp91
        if pf.is_defined('global_earth_model_name'):
            modelname = pf.get_string('global_earth_model_name')
        else:
            modelname = 'iasp91'
        self.model = TauPyModel(model=modelname)
        # These parameters define thing things outside pwstack_ensemble
        # but are important control
        # keep only dbname - caching the Database object creates serialization issues
        self.dbname=db.name
        gridname=pf.get_string("pseudostation_grid_name")
        self.pseudostation_gridname=gridname
        # We query by name and object type - this may require additional
        # tags to find the unique grid requested but for how keep it simple
        query={'name' : gridname,'object_type' : 'pwmig::gclgrid::GCLgrid'}
        nfound=db.GCLfielddata.count_documents(query)
        if nfound<1:
            raise MsPASSError("GCLgrid with name="+gridname+" was not found in database",
                              "Fatal")
        elif nfound>1:
            raise MsPASSError("GCLgrid with name="+gridname+" is ambiguous - multiple documents have that name tag",
                              "Fatal")
        doc=db.GCLfielddata.find_one(query)
        self.stagrid=GCLdbread(db,doc)

def site_query(db,lat,lon,ix1,ix2,cutoff,units='km')->dict:
    """
    This function creates a dictionary that is a query for MongoDB 
    to find site documents inside a circle with radius `cutoff` 
    centered at defined geograpic coordinates (lat,lon)

    :param db:  Database class (top level MongoDB handle)
    :param lat:  latitude (in degrees) of center point
    :param lon: longitude (in degrees) of center point
    :param cutoff:  distance in degrees of circlular region centered on
       lat,lon to select.
    :param units:  string defining units of cutoff argument.   Note
      lon and lat are always assumed to be degrees.  Acceptable values
      are 'km', 'degrees', or 'radians'.  Default is km

    :return:  python list of ObjectIds of site documents linked to
      stations within the search area (note this includes all times so
      a station location can appear more than once. Assume when matched
      against wf collection this will become one-to-one.  Returns and
      empty list if the query returns no site documents.

    """
    if units=='radians':
        rcutoff = math.degrees(cutoff)
        rcutoff = rcutoff*111120.0
    elif units=='km':
        rcutoff = cutoff*1000.0
    elif units=='degrees':
        rcutoff = cutoff*111120.0

    query = { "location" : {
      "$nearSphere" : {
        "$geometry" : {"type": "Point", "coordinates" : [lon,lat]},
        "$maxDistance" : rcutoff,
        }
      }
    }

    idlist=list()
    cursor=db.site.find(query)
    for doc in cursor:
        thisid=doc['_id']
        idlist.append(thisid)
    cursor.close()
    result=dict()
    result['idlist']=idlist
    result['lat']=lat
    result['lon']=lon
    result['ix1']=ix1
    result['ix2']=ix2
    return result


def build_wfquery(sid,rids,source_collection="telecluster",base_query=None):
    """
    This function is more or less a reformatter.  It expands each
    source_id by a set of queries for a pseudostation grid
    defined by the ridlist which a nexted data structre (details below).
    The result is a list of dict data structures that are passed down
    this workflow chain to build ensembles with lat, lon and the
    each pseudostation and seismogram objects linked to stations
    inside the search radius.  Those ultimately are passed to the
    function that actually runs the pwstack algorithm.

    :param sid:  expected to contain a single ObjectID the resolves
      with the source collection and defines the seismic source
      data to be processed (pwmig is prestack in that sense)
    :param rids: is a nested data structure.  It is expected to
      contain a list of dict structures.  As a minimum each dict for
      this function is expected to contain data with the key 'idlist'.
      The value associated with idlist is assumed to be a list of
      ObjectIDs in the site collection that define the set of seismic
      station data that can be stacked to define each pseudostation grid
      position.  The function passes along the attributes that define
      the pseudostation point - lat, lon, and two integer indicec
      (ix1, and ix2)   The return is mostly the contents of this plus
      the source data
    :param source_collection:  name of MongoDB collection to fetch 
      source documents.   
    :param base_query:   additional add clause for query to be generated.  
      Assumed to be a standard MongoDB query that does not reference
      "site_id" or the source collection id derived from source_collection 
      (currently that means either "source_id" or "telecluster_id").  
      If None (default) 
    :return: a list of dict data expanded from sid with one entry
      per pseudostation defined in ridlist.  The contents of each dict
      are copies of lat, lon, ix1, and ix2 plus  'query' attribute that
      should be used to query the wf_Seismogram collection to fetch
      data to be processed.  Note an empty list will be returned immediately
      if the ridlist received is empty.   That is common with a sparse
      and/or irregular station geometry.
    """
    allqdata = rids.copy()
    if base_query is None:
        q=dict()
    else:
        q = base_query.copy()
    idname = source_collection + "_id"
    q[idname] = { '$eq' : sid }
    idlist = rids['idlist']
    q['site_id'] = { '$in' : rids['idlist']}
    allqdata['query']=q
    # We use this field for efficiency.   It is known from experience
    # with pwmig that a stack cutoff is required because low fold
    # pseudostation bins produce bad results.  Hence, we post the number of
    # stations in each ensemble and drop anything when this count is too
    # small
    allqdata['fold'] = len(idlist)
    return allqdata
def get_source_metadata(ensemble)->dict:
    """
    Helper for below.  Returns source metadata from the first live
    member of ensemble as a dict.  That is safer than just using the first member
    as there are many ways a dead datum could lack the required metadata.
    Less likely here but a minor cost for robustness.   Returns an empty
    dict if there are no live members.   Caller must handle that condition.
    
    There is some complexity in this function to allow source data to 
    come from either a source or telecluster collection.   The later is 
    detected by the presence of the special key "hypocentroid". When found
    the function assumes it should fetch source data as the hypocentroid 
    data stored as a subducument with that key - the output of telecluster.  
    In all cases duplicate copies of the source metadata are posted with 
    two different strings prependend.  The generic data are posted with the 
    string "pwmig_source_".  e.g. source latitude is then posted with the key
    "pwmig_source_lat".   The key for the copy depends on the input collection. 
    If using hypocentroid data the prefix "telecluster_" is used.  Hence, for 
    the lat example the key "telecluster_lat" would contain the same data as
    "pwmig_source_lat".   If "source" is the parent then the comparable copy 
    would be found with the key "source_lat".   "lon", "depth", and "time" 
    are all handled similarly. Only the id field is not duplicated.  
    i.e. the output will contain an id with either the key "telecluster_id"
    or "source_id".
    """
    result=dict()
    if ensemble.dead() or number_live(ensemble)==0:
        return result
    found = False
    for d in ensemble.member:
        if d.live: 
            if "hypocentroid" in d:
                subdoc = d["hypocentroid"]
                result['pwmig_source_lat'] = subdoc["lat"]
                result['pwmig_source_lon'] = subdoc["lon"]
                result['pwmig_source_depth'] = subdoc["depth"]
                result['pwmig_source_time'] = subdoc["time"]
                result['pwmig_source_id'] = d['telecluster_id']
                result['telecluster_id'] = d['telecluster_id']
                # duplicate as noted above
                result["telecluster_lat"] = result["pwmig_source_lat"]
                result["telecluster_lon"] = result["pwmig_source_lon"]
                result["telecluster_depth"] = result["pwmig_source_depth"]
                result["telecluster_time"] = result["pwmig_source_time"]
            else:
                result['pwmig_source_lat'] = d.get_double('source_lat')
                result['pwmig_source_lon'] = d.get_double('source_lon')
                result['pwmig_source_depth'] = d.get_double('source_depth')
                result['pwmig_source_time'] = d.get_double('source_time')
                result['pwmig_source_id'] = d['source_id']
                # duplicate as noted above
                result["source_lat"] = result["pwmig_source_lat"]
                result["source_lon"] = result["pwmig_source_lon"]
                result["source_depth"] = result["pwmig_source_depth"]
                result["source_time"] = result["pwmig_source_time"]
            found = True
            break
    if not found:
        message = "get_source_metadata:   found no source location data\n"
        message += "Data need either source collection data loaded by normalization or a hypocentroid subdoc\n"
        message += "Neither were found any any member of this ensemble"
        raise MsPASSError("get_source_metadata",message,ErrorSeverity.Fatal)
    return result

def handle_relative_time(ensemble,arrival_key):
    """
    Implements algorithm described in docstring of read_ensemble to 
    allow mixing relative and absolute time.  d is the input ensemble 
    and arrival_key is a metadata key used to fetch arrival times 
    for any live datum with time set as UTC.  I intentionally let this 
    throw an exception of the arrival key is missing as it is a data 
    error that should not be permitted. 
    """
    for d in ensemble.member:
        # this will just skip any dead data
        if d.live:
            if d.time_is_UTC():
                atime = d[arrival_key]
                d.ator(atime)
    return ensemble
        
def read_ensemble(querydata,
                   dbname_or_handle,
                       control,
                           source_matcher,
                               site_matcher,
                                   arrival_key="Ptime"):
    """
    Constructs a query from dict created by build_wfquery, runs it 
    on the wf_Seismogram collection of db, and then calls read_ensemble_data 
    on the cursor returned by MongoDB.  It the sets the ensemble 
    metadata for lat, lon, ix1, and ix2 before returning the ensembles.
    Most variable parameters come through control but there is a data 
    arg (arrival_key) handling an issue that is data dependent.  Input 
    data can be in either UTC or relative time.  If the data read are 
    found to be in relative time this program assumes it has already been 
    shifted to what I call the "arrival time reference" that normally means 
    t0 is the P arrival time which is the maximum of the spike in t                                    normalize=['source','site'],he 
    data at the arrival time.   If the data have a UTC time standard 
    the function will attempt to extract a time from each member's 
    metadata container with that key and then call the ator method of 
    Seismogram to make the data into the expected relative time standard. 
    The entire run will be aborted with an exception if any live datum 
    is missing the arrival_key field.  (Not relevant, of course, for 
    all data input with relative time already set.)
    
    The function always performs normalization internally for site and 
    source data.   These are applied through the required `source_matcher` 
    and `site_matcher` agruments.   Most applications will use source 
    info that comes from the "telecluster" application instead of the 
    MsPASS "source" collection because real data always requires source 
    side stacking to producde reasonable results.  The only exception is 
    simulation data where full fold with controlled noise properties is
    possible,  

    :param querydata:  python dictionary created by build_wfquery (see that function)
    :param dbname_or_handle:   For normal parallel processing this is expected 
      to be the name of the database to access.  For serial processing it 
      must be an actual instance of a MsPASS Database object.  The function 
      will raise a ValueError exception if is is anything else. 
    :param control:  special class with control parameters created from pf
    :param source_matcher:   Concrete implementation of a 
      `mspasspy.db.normalize.BasicMatcher` used to load source metadata.  
    :param site_matcher:  Concrete implemation of a 
      'mspasspy.db.normalize.BasicMatcher' to load receiver side metadata. 
    :param arrival_key:  key for fetching arrival time using algorithm noted 
      above.  Default is "Ptime"
    """
    if isinstance(dbname_or_handle,str):
        worker = ddist.get_worker()
        dbclient = worker.data["dbclient"]
        db = dbclient.get_database(dbname_or_handle)
    elif isinstance(dbname_or_handle,Database):
        db = dbname_or_handle
    else:
        message = "read_ensembles:   illegal type for arg1={}\n".format(type(dbname_or_handle))
        message += "Must be str defining a db name or a Database object"
        raise ValueError(message)
    worker = ddist.get_worker()
    fold=querydata['fold']
    if fold<=control.stack_count_cutoff:
        d=SeismogramEnsemble()
    else:
        query=querydata['query']
        n=db.wf_Seismogram.count_documents(query)
        if n==0:
            # This shouldn't ever really be executed unless stack_count_cutoff 
            # is 0 or the query is botched
            d=SeismogramEnsemble()
        else:
            cursor=db.wf_Seismogram.find(query)
            # Note control.data_tag can be a None type here - see 
            # control object constructor
            d=db.read_data(cursor,collection='wf_Seismogram',
                                    data_tag=control.data_tag)
            cursor.close()
            # this is subject to change as this is a workaround for a bug
            # in mspass.  Eventually ensemble elog should always be empty 
            # and any read errors get posted member components
            #if d.dead():
            #    ddist.print("Read failed for ensemble with query: ",query)
            #    ddist.print("Error logs from members")
            #    elog = d.elog.get_error_log()
            #    for e in elog:
            #        ddist.print(e.message)
            #else:
            #ddist.print("Running normalize")
            d = normalize(d,source_matcher,handles_ensembles=False)
            d = normalize(d,site_matcher,handles_ensembles=False)
                            
        if len(d.member) > 0:
            d = handle_relative_time(d,arrival_key)
            # When the ensemble is not empty we have to compute the 
            # slowness vector of the incident wavefield using source 
            # coordinates and the pseudostation location.  This section 
            # depends on a feature of the reader that it the normalize 
            # parameter causes all members to have source coordinates loaded
            # this small function copies source metadata from the first 
            # live member
            srcdata = get_source_metadata(d)
            if len(srcdata)==0:
                d.kill()
                d.elog.log_error("pwstack",
                        "Ensembled read from database has no live members",
                        ErrorSeverity.Invalid)
            else:
                # post the source metadata to the ensemble metadata
                for k in srcdata:
                    d[k] = srcdata[k]
                # a bit of a weird way to fetch the pseudostation 
                # coordinates but a fast an efficnet way to do it
                # Note also a unit mismatch - gps2dist_azimuth requires
                # coordinates in degrees but the C++ code here uses 
                # radians internally
                pslat=querydata['lat']
                pslon=querydata['lon']
                georesult=gps2dist_azimuth(srcdata['pwmig_source_lat'],
                                        srcdata['pwmig_source_lon'],pslat,pslon)
                # obspy's function we just called returns distance in m in element 0 of a tuple
                # their travel time calculator it is degrees so we need this conversion
                Rearth=6378.164
                dist=kilometers2degrees(georesult[0]/1000.0)
                # We pass the model object through control because I think 
                # there is a nontrivial overhead in creating it
                arrivals=control.model.get_travel_times(
                    source_depth_in_km=srcdata['pwmig_source_depth'],
                    distance_in_degree=dist,phase_list=['P']
                    )
                if len(arrivals)>0:
                    ray_param=arrivals[0].ray_param
                    umag=ray_param/Rearth    # need slowness in s/km but ray_param is s/radian
                    baz=georesult[2]   # The obspy function seems to return back azimuth
                    az=baz+180.0
                    # az can be > 360 here but all known trig function algs handl this automatically
                    ux=umag*math.sin(math.radians(az))
                    uy=umag*math.cos(math.radians(az))
                    d.put('ux0', ux)
                    d.put('uy0',uy)
                    d.put('pseudostation_lat',pslat)
                    d.put('pseudostation_lon',pslon) 
                    # this more obscure name is needed as an alias by pwstack_ensemble
                    # we save the longer tag for less obscure database tags
                    d.put('lat0',pslat)
                    d.put('lon0',pslon) 
                    d.put('ix1',querydata['ix1']) 
                    d.put('ix2',querydata['ix2'])
                    d.put('gridname',control.pseudostation_gridname)
                else:
                    d.kill()
                    message = "Travel time calculator failed to compute P wave arrival time\n"
                    message += "Epicentral distance(deg)={}".format(dist)
                    d.elog.log_error("pwstack",message,ErrorSeverity.Invalid)
    return d
def save_ensemble(ens, dbname_or_handle, data_tag, storage_mode="gridfs", outdir=None):
    """
    Writer function for ensembles created by pwstack.   It is largely a 
    wrapper around db.save_data to handle:
        1.  Handling details of how sample data are organized in files 
            or gridfs.
        2.  Work on this algorithm led to a discovery of a probolem with 
            earlier implementations of how Database object were handled 
            in MsPASS.   (db clients were being serialized that caused a 
            rsrouce leak of open db connections.)   This code uses a 
            prototype approach that may be replaced by a more generic 
            form under development for MsPASS.
    
    :param: ens:  data to be saved
    :param dbname_or_handle:   For serial proclessing this arg must 
      contain an instance of a Database object.  For parallel processing 
      it MUST contain a string defining the database name to use.   
      Note MonogDBWorker must have been used previously to instantiate a 
      client on each worker when running this algorithm in parallel or it 
      will fail on the first call to the function.
    :param data_tag:  required unique data tag for output waveform documents. 
    :param storage_mode:   Must be either "gridfs" (default) or "file".  
      "file" is recommended but if so you should always define outdir 
      as the default writes files to the current directory.   When 
      set to "file" the file names are dogmatically derived from 
      the id of the source document.  That means the file name is usually 
      the str value of the internal key "pwmig_source_id".   That key is 
      a required tag and the function will abort with a MsPASSError if it 
      is not defined.  
    :param outdir:  optional string defining a valid directory in which to 
      save sample data files.   Ignored when storage_mode is set to gridfs. 
      Default is None which causes files to be written to the currenct directory. 
      Otherwise the value is passed to Database.save_data which currently, 
      at least, will create the directory if it does not yet exist. 
    """
    if ens.dead():
        return None
    if isinstance(dbname_or_handle,str):
        worker = ddist.get_worker()
        dbclient = worker.data["dbclient"]
        db = dbclient.get_database(dbname_or_handle)
    elif isinstance(dbname_or_handle,Database):
        db = dbname_or_handle
    else:
        message = "read_ensembles:   illegal type for arg1={}\n".format(type(dbname_or_handle))
        message += "Must be str defining a db name or a Database object"
    if storage_mode=="file":
        if outdir:
            odir=outdir
        else:
            odir="."
        # since it shouldn't happen let this abort with a Metadata 
        # fetching error if this key is not defined.  Not user 
        # friendly but appropriate since it is a bug if that happens
        dfile = str(ens["pwmig_source_id"]) + ".dat"
    sdret = db.save_data(ens,
                         collection="wf_Seismogram",
                             storage_mode=storage_mode,
                                 dir=odir,
                                     dfile=dfile,
                                         data_tag=data_tag)
    return sdret
    

def pwstack_ensemble_python(*arg):
    """
    Temporary workaround for a problem with return in dask. Patch we can
    hopefully remove when we understand this problem better.
    """ 
    return pwstack_ensemble(*arg)

def pwstack(db,pf,source_query=None,
    wf_query=None,
     minimum_input_data=None,
      source_collection="telecluster",
        slowness_grid_tag='RectangularSlownessGrid',
            data_mute_tag='Data_Top_Mute',
                 stack_mute_tag='Stack_Top_Mute',
                     save_history=False,instance='undefined',
                        storage_mode='gridfs',
                             outdir=None,
                                 output_data_tag='test_pwstack_output',
                                     run_serial=True,
                                         dask_client=None,
                                             verbose=False):
    """
    Driver function for the pwstack algorithm.
    
    Runs the pwstack algorithn on all data in wf_Seismogram in the 
    Database defined by db (an instance of the MsPASS Database classs).
    The algorithm is an outer loop over source ids with an internal 
    loop over a 2d grid of points created in a control structure 
    instantiated from the input parameter pf.   The inputs are complex 
    and interrelated.  See the user manual page under construction for 
    details on how to run this function.   A CLI tool may also be 
    eventually produced to run this function.  
    """
    # must be dogmatic about this
    if not run_serial and dask_client is None:
        message = "pwstack:   illegal argument combination\n"
        message += "When run_serial is False you must define dask_client as the result of mspass_client.get_scheduler()"
        raise ValueError(message)
    # the control structure pretty much encapsulates the args for
    # this driver function
    if verbose:
        print("Start pwstack processing of this dataset")
        print("Using collection=",source_collection," for source data")
        if source_query is not None:
            print("Source collection will be limited by the following query:")
            print(source_query)
        print("Attempting to build control structure from pf data input")
    control=pwstack_control(db,pf,slowness_grid_tag,data_mute_tag,
                    stack_mute_tag,save_history,instance)
    if source_query==None:
        base_query={}
    else:
        base_query=source_query
    dbcol = db[source_collection]
    base_cursor=dbcol.find(base_query)
    # We make an assumption here that the array of source ids created
    # here is tiny and not a memory problem.  Unambiguously true when
    # source side stack was previously completed.
    source_id_list=list()
    for doc in base_cursor:
        id=doc['_id']
        source_id_list.append(id)
    base_cursor.close()
    if verbose:
        print("Number of sources ids used to drive this run=",len(source_id_list))
    if source_collection=="source":
        srcmatcher=ObjectIdMatcher(db,collection="source",
                                   attributes_to_load=['_id','lat','lon','depth','time'],
                            )
    elif source_collection=="telecluster":
        srcmatcher=ObjectIdMatcher(db,collection="telecluster",
                                   attributes_to_load=['_id','hypocentroid'],
                            )
    else:
        message = "pwstack:   Illegal value for source_collection={}".format(source_collection)
        message += "Must be either source or telecluster"
        raise ValueError(message)
    sitematcher=ObjectIdMatcher(db,
                                    collection="site",
                                        attributes_to_load=["_id","lat","lon","elev"],
                                )
    if not run_serial:
        srcm_f = dask_client.scatter(srcmatcher, broadcast=True)
        sitem_f = dask_client.scatter(sitematcher, broadcast=True)
    cutoff=control.aperture.maximum_cutoff()
    staids=list()
    for i in range(control.stagrid.n1):
        for j in range(control.stagrid.n2):

            # We use this instead of the lat and lon methods for
            # a minor efficiency difference.  lat and lon have to call
            # both call this method and return only one of 3 elements
            gc=control.stagrid.geo_coordinates(i,j)
            lat=math.degrees(gc.lat)
            lon=math.degrees(gc.lon)
            # I could not make this work.   I don't think there is a
            # huge overhead in creating this geometry object defining
            # how to find which stations are related to each pseudostation
            # point.
            #ids=dask.delayed(site_query)(db,lat,lon,i,j,cutoff)
            ids=site_query(db,lat,lon,i,j,cutoff)
            staids.append(ids)
    
    
    for sid in source_id_list:
        # skip this sid if there size filter is enabled and the size is small
        if (minimum_input_data is not None):
            query = copy.deepcopy(base_query)
            idkey = source_collection + "_id"
            query[idkey] = sid
            n_this_sid = db.wf_Seismogram.count_documents(query)
            if n_this_sid<minimum_input_data:
                print("Number of waveforms for sid=",sid,
                      " is ",n_this_sid)
                print("That number is below minimum_input_data threshold set as ",minimum_input_data)
                print("Skipping data for this sid")
                continue
        allqueries=list()
        for rids in staids:
            # build_wfquery returns a dict with lat, lon,
            # i, j, and a query string  That is a simple
            # construct so don't think it will be a bottleneck.
            # May have been better done with a dataframe
            #q=dask.delayed(build_wfquery)(sid,rids)
            q=build_wfquery(sid,
                        rids,
                            source_collection=source_collection,
                                base_query=wf_query)
            # debug
            #print(q['ix1'],q['ix2'],q['fold'])
            allqueries.append(q)
        if verbose:
            print("Number of ensembles to processed=",len(allqueries))
            
        if run_serial:
            if verbose:
                print("Starting main processing using serial algorithm")
            for q in allqueries:
                ens = read_ensemble(q, db, control,srcmatcher,sitematcher)
                ens = pwstack_ensemble(ens,
                    control.SlowGrid,
                      control.data_mute,
                        control.stack_mute,
                          control.stack_count_cutoff,
                            control.tstart,
                              control.tend,
                                control.aperture,
                                  control.aperture_taper_length,
                                    control.centroid_cutoff,
                                        False,'')
    
                # This repeats code in save_ensemble_parallel and maybe 
                # should be a function called both places
                sdret = save_ensemble(ens,
                                        db,
                                          output_data_tag,
                                            storage_mode=storage_mode,
                                              outdir=outdir)
                if verbose:
                    print(sdret)
        else:
            
            if verbose:
                print("Starting main processing using parallel algorithm for source with id=",sid)
            # these common data are large and are best pushed to all 
            # the nodes like this to reduce the size of the dag
            
            mybag=dask.bag.from_sequence(allqueries)
            # parallel reader - result is a bag of ensembles created from
            # queries held in query
            mybag = mybag.map(read_ensemble,db.name,control,srcm_f,sitem_f)
            # Now run pwstack_ensemble - it has a long arg list
            mybag = mybag.map(pwstack_ensemble_python,
                    control.SlowGrid,
                      control.data_mute,
                        control.stack_mute,
                          control.stack_count_cutoff,
                            control.tstart,
                              control.tend,
                                control.aperture,
                                  control.aperture_taper_length,
                                    control.centroid_cutoff,
                                        save_history,'')
            mybag = mybag.map(save_ensemble,
                              db.name,
                                  output_data_tag,
                                      storage_mode=storage_mode,
                                          outdir=outdir,
                                )
            sdret = mybag.compute()
        if verbose:
            print("Finished processing data for source with id=",sid)
            print("Number of ensembled processed=",len(sdret))
