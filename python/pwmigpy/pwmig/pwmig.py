import time  # for testing only - remove when done

import math
import dask
import dask.distributed as ddist
from mspasspy.ccore.utility import (Metadata,
                                    MsPASSError,
                                    ErrorSeverity)
from mspasspy.ccore.seismic import (SlownessVector)
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees

from pwmigpy.ccore.gclgrid import (GCLvectorfield3d)
from pwmigpy.ccore.seispp import RayPathSphere
from pwmigpy.ccore.pwmigcore import (SlownessVectorMatrix,
                                     ComputeIncidentWaveRaygrid,
                                     migrate_one_seismogram,
                                     migrate_component)
from pwmigpy.db.database import (vmod1d_dbread_by_name,
                                 GCLdbread_by_name)
from pwmigpy.utility.earthmodels import Velocity3DToSlowness

class worker_memory_estimator:
    """
    Utility class used to estimate memory footprint for a run of pwmig 
    from database content and parameters defined in pf that will 
    be used in running pwmig.   Constructor is initialization here. 
    Standard use is to instantiate and instance of this class.  
    If successful then run the report method to print results in a 
    clean report forma. 
    
    :param db:  database handle for db to be used to drive pwmig
    :param sid:  source is to use as pattern for checking data ensemble 
      sizes.  subsets by source_collection_id matching this value.  
    :param pf:   AntelopePf object created from pf file defining 
      this run of pwmig.   Normally created from "pwmig.pf".   

    """
    def __init__(self,db,sid,pf,source_collection="telecluster"):
        self.ensemble_size = self.input_data_size(db,sid,source_collection)
        # this 2d grid has size data that impacts most of the memory 
        # use.  We read it here as does pwmig
        parent_grid_name = pf.get_string("Parent_GCLgrid_Name");
        parent = GCLdbread_by_name(db, parent_grid_name, object_type="pwmig::gclgrid::GCLgrid")
        # inlined as it is pretty trivial
        self.parentsize = 8*3*parent.n1*parent.n2
        self.wrgsize = self.worker_raygrid_size(db,pf,parent)
        self.vmod3dsize = self.vel3dfield_size(db,pf)
        self.imggridsize = self.imagegrid_size(db,pf)
        self.incidentgridsize = self.incident_ptimegrid_size(db,parent,pf)
        self.ugridsize = self.slowness_grid_size(parent)
    @staticmethod
    def input_data_size(db,sid,source_collection):
        """
        Set an estimate of the memory required to store an ensembled 
        for sid.   Internal to this class but could be used in isolation.,
        """
        idname = source_collection + "_id"
        query = {idname : sid,"gridid" : 0}
        doc = db.wf_Seismogram.find_one(query)
        # pwstack produces seismograms of the same size for all outputs 
        # so we only need the first one to make an estimate
        # firther the ensemble size is fixed for all components so 
        # we can compute it thsi way
        #query["gridid"] = doc["gridid"]
        n_members = db.wf_Seismogram.count_documents(query)
        # reading one member and using sizeof will give a more accurate 
        # memory estimate than just using the sample size
        d = db.read_data(doc,collection="wf_Seismogram")
        dsize=d.__sizeof__()
        input_data_size_bytes = n_members*dsize
        return input_data_size_bytes
    @staticmethod
    def worker_raygrid_size(db,pf,parent):
        """
        Computes an estimate of the size of the raygrid computed 
        internally in each worker.
        
        Internal to this class but could be used in isolation.,
        """
        # this computation is modeled on C++ function Build_GCLraygrid
        n1 = parent.n1
        n2 = parent.n2
        # this is a nominal slowness  needed to build a nominal raypath 
        # of typcal size.  This is an approximation but reasonable for 
        # the purpose here
        u_nominal = 1.0/8.0
        zmax = pf.get_double("maximum_depth")
        tmax = pf.get_double("maximum_time_lag")
        # dux=pf.get_double("slowness_grid_deltau")
        dt = pf.get_double("data_sample_interval")
        Smodel1d_name = pf.get_string("S_velocity_model1d_name");
        Vs1d = vmod1d_dbread_by_name(db, Smodel1d_name)
        ray = RayPathSphere(Vs1d,u_nominal,zmax,tmax,dt,"t")
        n3=ray.npts
        # these grids use a 5 component vector.  DAta + domega and grt weight
        nv=5
        npts_grid = n1*n2*n3*nv
        coords_size = 3*n1*n2*n3
        # return size in bytes
        return 8*(npts_grid+coords_size)
    @staticmethod
    def vel3dfield_size(db,pf,collection="GCLfielddata"):
        """
        Computes an estimate of the size of the grid used to store the 
        3d model for P and S wave travel times.  
        
        Internal to this class but could be used in isolation.,
        
        Note this one needs to return 0 if the model name is not defined
        """
        Pvel3dname = pf.get_string('P_velocity_model3d_name') 
        query={ "name" : Pvel3dname,
               "object_type" : "pwmig::gclgrid::GCLscalarfield3d"}
        doc = db[collection].find_one(query)
        if doc:
            n1 = doc["n1"]
            n2 = doc["n2"]
            n3 = doc["n3"]
            data_size = n1*n2*n3
            coords_size = 3*n1*n2*n3
            return 8*(data_size+coords_size)
        else:
            message = "worker_memory_estimator.vel3dfield_size:  "
            message += "no documents found with name="+Pvel3dname
            print(message)
            print("Check pf file if you intend to use a 3d earth model")
            return 0
    
    @staticmethod
    def incident_ptimegrid_size(db,parent,pf):
        borderpad = pf.get_long("border_padding")
        n1=parent.n1 + 2*borderpad
        n2=parent.n2 + 2*borderpad
        # n3 size has to handle padding and decimation
        zmax = pf.get_double("maximum_depth")
        tmax = pf.get_double("maximum_time_lag")
        dt = pf.get_double("data_sample_interval")
        zdecfac = pf.get_long("Incident_TTgrid_zdecfac")
        zpad = pf.get_double("depth_padding_multiplier")
        Pmodel1d_name = pf.get_string("P_velocity_model1d_name");
        Vp1d = vmod1d_dbread_by_name(db, Pmodel1d_name)
        u_nominal = 1.0/12.0
        ray = RayPathSphere(Vp1d,u_nominal,zmax*zpad,tmax,dt,"t")
        n3 = round(float(ray.npts)/zdecfac)
        coords_size = 8*3*n1*n2*n3
        data_size = 8*n1*n2*n3
        return coords_size + data_size

    @staticmethod
    def imagegrid_size(db,pf,collection="GCLfielddata"):
        """
        Compute and store an estimate of the size of the image grid 
        used to store the final image in frontend script.  
        
        Internal to this class but could be used in isolation.,
        """
        imgname = pf.get_string("stack_grid_name")
        query = {"name" : imgname,
                 "object_type" : "pwmig::gclgrid::GCLgrid3d"}
        doc = db[collection].find_one(query)
        if doc:
            n1 = doc["n1"]
            n2 = doc["n2"]
            n3 = doc["n3"]
            # image has 5 components at each node
            data_size = n1*n2*n3*5
            coords_size = 3*n1*n2*n3
            return 8*(data_size+coords_size)
        else:
            message = "worker_memory_estimator.imagegrid_size:  "
            message += "no documents found with name="+imgname
            raise RuntimeError(message)
            
            
    @staticmethod
    def slowness_grid_size(parent):
        """
        Compute and store an estimate of the size of the 2d slowness grid 
        used in pwmig.  This object is usually sall compared to any f the 
        3d fields but is posted by the report method.
        """
        n1=parent.n1
        n2=parent.n2
        # each slowness vector has 2 components
        data_size = 2*n1*n2
        # slowness grid doesn is made conguent with parent so has no
        # coordinate arrays
        return 8*data_size
    
    def report(self,print_report=True):
        """
        Generate a nicely formatted report from the memory 
        size data computed on construction.   This is the primary 
        purpose of this class.  The method internally generates a 
        string holding the report.   When the print_report is 
        True (default) that string is pushed to the python 
        print function and returns a None.  When False the 
        report string is returned.  The False options provides more 
        flexibility in how the output is handled.   
        """
        report = "////////////// pwmig run time data memory use estimates////////////////\n"
        report += "////////////// Major pwmig data object sizes (Mbytes) ///////////////////\n"
        report += "Approximate average ensemble size={}\n".format(self.ensemble_size/1e6)
        report += "pseudostation 2d grid size (parent symbol in code)={}\n".format(self.parentsize/1e6)
        report += "3d velocity model field={}\n".format(self.vmod3dsize/1e6)
        report += "Incident wave travel time field size={}\n".format(self.incidentgridsize/1e6)
        report += "2d slowness array grid size={}\n".format(self.ugridsize/1e6)
        report += "raygrid created for each plane wave component size={}\n".format(self.wrgsize/1e6)
        report += "Image grid size={}\n".format(self.imggridsize/1e6)
        report += "////////////// estimaed memory use per worker  ////////////////////////\n"
        workersize = self.ensemble_size 
        workersize += self.parentsize
        workersize += self.vmod3dsize 
        workersize += self.incidentgridsize
        workersize += self.ugridsize
        workersize += self.wrgsize
        report += "Minimum memory use={}\n".format(workersize/1e6)
        report += "////////////// estimated memory use of frontend process////////////////\n"
        fesize = self.parentsize
        fesize += self.vmod3dsize 
        fesize += self.incidentgridsize
        fesize += self.ugridsize
        fesize += self.imggridsize
        report += "Base size without worker data in transit={}\n".format(fesize/1e6)
        report += "Size of raygrid data returned by each worker={}\n".format(self.wrgsize/1e6)
        report += "Note:  actual use may buffer multiple copies of raygrid data being returned by workers\n"
        report += "///////////////////////////////////////////////////////////////////////\n"
        if print_report:
            print(report)
            return None
        else:
            return report

# from pwmigpy.ccore.seispp import VelocityModel_1d

def _print_default_used_message(key, defval):
    print("Parameter warning:  AntelopePf file does not have key=", key,
          " defined.\nUsing default value=", defval, " which is of type ", type(defval))


def _build_control_metadata(control):
    """
    Parses AntelopPf container (required arg0) for parameters required by
    pwmig's inner function (migrate_one_seismogram).  Returns the subset of
    the input that are required by that function in a Metadata container
    (note AntelopePf is a child of Metadata).  This function is bombproof
    as all parameters are defaulted.  Because we expect it to be used
    outside parallel constructs any time the default is used a warning
    print message is posted.

    """
    result = Metadata()
    if control.is_defined("use_3d_velocity_model"):
        use_3d_vmodel = control.get_bool("use_3d_velocity_model")
    else:
        use_3d_vmodel = True
        _print_default_used_message("use_3d_velocity_model", use_3d_vmodel)
    if control.is_defined("use_grt_weights"):
        use_grt_weights = control.get_bool("use_grt_weights")
    else:
        use_grt_weights = True
        _print_default_used_message("use_grt_weights", use_grt_weights)
    if control.is_defined("stack_only"):
        stack_only = control.get_bool("stack_only")
    else:
        stack_only = True
        _print_default_used_message("stack_only", stack_only)
    if control.is_defined("border_padding"):
        border_pad = control.get_long("border_padding")
    else:
        border_pad = 20
        _print_default_used_message("border_padding", border_pad)
    if control.is_defined("depth_padding_multiplier"):
        zpad = control.get_double("depth_padding_multiplier")
    else:
        zpad = 1.2
        _print_default_used_message("depth_padding_multiplier", zpad)
    if control.is_defined("taper_length_turning_rays"):
        taper_length = control.get_double("taper_length_turning_rays")
    else:
        taper_length = 2.0
        _print_default_used_message("taper_length_turning_rays", taper_length)
    if control.is_defined("recompute_weight_functions"):
        rcomp_wt = control.get_bool("recompute_weight_functions")
    else:
        rcomp_wt = True
        _print_default_used_message("recompute_weight_functions", rcomp_wt)
    if control.is_defined("weighting_function_smoother_length"):
        nwtsmooth = control.get_long("weighting_function_smoother_length")
    else:
        nwtsmooth = 10
        _print_default_used_message("weighting_function_smoother_length", nwtsmooth)
    if control.is_defined("slowness_grid_deltau"):
        dux = control.get_double("slowness_grid_deltau")
    else:
        dux = 0.01
        _print_default_used_message("slowness_grid_deltau", dux)
    if control.is_defined("ray_trace_depth_increment"):
        dz = control.get_double("ray_trace_depth_increment")
    else:
        dz = 1.0
        _print_default_used_message("ray_trace_depth_increment", dz)

    result.put("use_3d_velocity_model", use_3d_vmodel)
    result.put("use_grt_weights", use_grt_weights)
    result.put("stack_only", stack_only)
    result.put("border_padding", border_pad)
    result.put("depth_padding_multiplier", zpad)
    result.put("taper_length_turning_rays", taper_length)
    result.put("recompute_weight_functions", rcomp_wt)
    result.put("weighting_function_smoother_length", nwtsmooth)
    result.put("slowness_grid_deltau", dux)
    result.put("ray_trace_depth_increment", dz)
    # these have no defaults
    result.put("maximum_depth", control["maximum_depth"])
    result.put("maximum_time_lag", control["maximum_time_lag"])
    result.put("data_sample_interval", control["data_sample_interval"])
    return result;


def BuildSlownessGrid(g, source_lat, source_lon, source_depth, model='iasp91', phase='P'):
    model = TauPyModel(model=model)
    Rearth = 6378.164
    svm = SlownessVectorMatrix(g.n1, g.n2)
    for i in range(g.n1):
        for j in range(g.n2):
            stalat = g.lat(i, j)
            stalon = g.lon(i, j)
            georesult = gps2dist_azimuth(source_lat, source_lon,
                                         math.degrees(stalat), math.degrees(stalon))
            # obspy's function we just called returns distance in m in element 0 of a tuple
            # their travel time calculator it is degrees so we need this conversion
            dist = kilometers2degrees(georesult[0] / 1000.0)
            arrivals = model.get_travel_times(source_depth_in_km=source_depth,
                                              distance_in_degree=dist, phase_list=phase)
            ray_param = arrivals[0].ray_param
            umag = ray_param / Rearth  # need slowness in s/km but ray_param is s/radian
            baz = georesult[2]  # The obspy function seems to return back azimuth
            az = baz + 180.0
            # az can be > 360 here but all known trig function algs handl this automatically
            ux = umag * math.sin(math.radians(az))
            uy = umag * math.cos(math.radians(az))
            u = SlownessVector(ux, uy, 0.0)
            svm.set_slowness(u, i, j)
    return svm


def query_by_id(gridid, db, source_id, collection='wf_Seismogram'):
    """
    Small function used in map below to parallelize mongodb query.
    Querys and returns a cursor of all wf_Seismogram entries matching
    source_id and gridid (keys are those names).  Cursors are the output
    and used in a map to parallelize the query.

    :param gridid:  integer gridid for plane wave component to be selected
    :param db:  Database handle
    :param source_id:  ObjectID of the source to set in query

    """
    query = {'source_id': source_id, "gridid": gridid}
    collection = db[collection]
    return collection.find(query)


@dask.delayed
def _set_incident_slowness_metadata(d, svm):
    """
    Internal function used in map to set metadata fields in
    mspass Seismogram, d, from incident wave slowness data stored in
    the SlownessVectorMatrix object svm.

    """
    i = d['ix1']
    j = d['ix2']
    # not sure this is in the bindings but idea is to fetch a SlownessVector for i,j
    slowness = svm.get_slowness(i, j)
    # Warning:  these keys must be consistent with C++ function migrate_one_seismogram
    # corresponding get
    d['ux0'] = slowness.ux
    d['uy0'] = slowness.uy
    return d


def _add_fieldata(f1, f2):
    """
    Applies += operator and returns f1+f2.  If geometries of f1 and f2
    differ the returned field will have the geometry of f1
    """
    f1 += f2
    return f1


def migrate_one_seismogram_python(*args):
    return migrate_one_seismogram(*args)


@dask.delayed
def accumulate_python(grid, migseis):
    grid.accumulate(migseis)
    return grid


def _migrate_component(query,dbname, parent, TPfield, VPsvm, Us3d, Vp1d, Vs1d, control):
    """
    This small function is largely a wrapper for the C++ function
    with the same name sans the _ (i.e. migrate_component).

    """
    worker = ddist.get_worker()
    dbclient = worker.data["dbclient"]
    db = dbclient.get_database(dbname)
    t0 = time.time()
    cursor = db.wf_Seismogram.find(query)
    pwensemble = db.read_data(cursor, collection="wf_Seismogram")
    cursor.close()
    t1 = time.time()
    pwdgrid = migrate_component(pwensemble, parent, TPfield, VPsvm, Us3d,
                                Vp1d, Vs1d, control)
    t2 = time.time()
    ddist.print("Time to run read_ensemble=", t1 - t0, " Time to run migrate_component=", t2 - t1)
    return pwdgrid
    # just return a nessage for debugging
    #del pwdgrid
    #return "finished ensemble with {} members".format(len(pwensemble.member))


def pwmig_verify(db, pffile="pwmig.pf", GCLcollection='GCLfielddata',
                 check_waveform_data=False):
    """
    Run this function to verify input to pwmig is complete as can be
    established before actually running the algorithm on a complete data set.
    The function writes a standard report to stdout that should be examined
    carefully before commiting to a long running job.


    """
    print('Warning:  this function is not yet implemented.   Checks only if pf is readable')
    # First we test the parameter file for completeness.
    # we use a generic pf testing function in mspass.
    # algorithm="pwmig"
    # pf=AntelopePf(pffile)
    # TODO  this does not exists and is subject to design changes from github discussion
    # pfverify(algorithm,pf)
    # this also doesn't exists.  Idea is to verify all model files are
    # present and valid
    # pwmig_verify_files(pf)
    # now verify the database collection and print a report
    # TODO;  need to implement this fully - this is rough
    # 1  verify all gclfeield and vmodel data
    # print a report for waveform inputs per event

def compute_3dfieldsize(f)->int:
    """`
    Returns the size in bytes of a GCLfield3D object.  Works for 
    vector or scalar fileds.   Assumes all array elements are 8 bytes
    which is true of the C++ function that defines these objects.
    Input is the field.  Output is an integer with a size estimate.
    
    Note the size returned neglects object scalar attributes that 
    are always tiny compared to the field data. 
    """
    ngridpoints=f.n1*f.n2*f.n3
    if hasattr(f,"nv"):
        nv=f.nv
    else:
        nv=1
    fielddatasize=nv*ngridpoints
    total_arraysize = 3*ngridpoints + fielddatasize
    return 8*total_arraysize

def migrate_event(mspass_client, dbname, sid, pf, 
                    source_collection="telecluster",
                    parallel=True,
                    verbose=False,
                    dryrun=False):
    """
    Top level map function for pwmig.   pwmig is a "prestack" method
    meaning we migrate data from individual source regions (always best
    done as the output of stacked data produced by running telecluster
    followed by RFeventStacker.)  This function creates a 3d image
    volume from migration of one ensemble of data assumed linked to
    a single source region.
    
    The linkto source region is through a database id.  The function 
    allows the source data to be defined in either the "telecluster"
    collection (default) or the "source" collection - the normal 
    MsPASS collection for source data.   Most data will use telecluster
    unless a nonstandard method is used to assemble the data for processing.
    
    :param mspass_client:  as he name implies the instance of 
      `mspasspy.client.Client` driving this workflow.   The mspass 
      client contains a hook for both the database and dask cluster clients.
      Both are required for this algorithm.
    :param dbname:   string defining the name of the database containing 
      data to be processed.
    :param sid:  ObjectId of the document containing source data used by 
      pwstack to generate plane wave estimates used as input for set of 
      data to be migrated by this function.  This id is linked to the 
      argument "source_collection" as the actual key used is the 
      standard composite key for a normalizing collection id 
      (i.e collection_id).  T
    :param pf:  AntelopePf object containing the control parameters for
      the program (run the pwmig_pfverify function to assure the parameters
      in this file are complete before starting a complete processing run.)
    :param source_collection:   Collection containing source data documents 
      that define the event to be processed by this function.
    :param parallel:   boolean that when true runs the algorithm in 
      paallel using dask.   When false processing is serial.
    :param verbose:  boolean that when true will generate more volumious 
      output.   Default is mostly silent. 
      
    Warning:   the source data extracted from the database is used to 
    compute 1d and 3d model travel times.   The algorithm depends upon the 
    id passed as "sid" matching the source data posted in the Metadata of 
    all data saved by pwstack used as input.   The function does not currently 
    check for consistency.  That is assured if the input is from stacks 
    produced by pseudosource_stacker but custom data assembly processing 
    must understand how all that works.  

    """
    if parallel:
        db = mspass_client.get_database(dbname)
        dask_client = mspass_client.get_scheduler()
    else:
        db = mspass_client.get_database(dbname)
        dask_client = None
        if verbose:
            print("Wanring:  running serial mode which may run for a long time")
    # force verbose if dryrun is enabled
    if dryrun:
        wmem = worker_memory_estimator(db, sid, pf)
        wmem.report()
        exit(1)
    # Freeze use of source collection for source_id consistent with MsPASS
    # default schema.
    doc = db[source_collection].find_one({'_id': sid})
    if doc == None:
        # This is fatal because expected use is parallel processing of multiple event
        # if any are not defined the job should abort.  Also a preprocess
        # checker is planned to check for such problems
        raise MsPASSError("migrate_event:  source_id=" + str(sid) + " not found in database", ErrorSeverity.Fatal)
    # This is needed to handle use either source or telecluster for 
    # source data - they store the data differently
    # note the algorithm uses relative time and does not need origin time
    if source_collection=="telecluster":
        subdoc=doc["hypocentroid"]
        source_lat=subdoc["lat"]
        source_lon=subdoc["lon"]
        source_depth=subdoc["depth"]
    elif source_collection=="source":
        source_lat = doc['source_lat']
        source_lon = doc['source_lon']
        source_depth = doc['source_depth']
    else:
        message = "migrate_event:   illegal value received with source_collection={}\n".format(source_collection)
        message += "Must be either telecluster (default) or source"
        raise ValueError(message)
    if verbose:
        print("Working on source with latitude=",source_lat,
              " lon=",source_lon,
              ", and depth=",source_depth)
    # source_time=doc['time']

    # This function is the one that extracts parameters required in
    # what were once inner loops of this program.  As noted there are
    # so many parameters it makes the code more readable to pass just
    # this control metadata container around.  The dark side is if any
    # new parameters are added changes are required in this function,
    control = _build_control_metadata(pf)
    if verbose:
        print("Successfully built internal control structure for pf input")

    # This builds the image volume used to accumulate plane wave
    # components.   We assume it was constructed earlier and saved
    # to the database.  This parameter is outside control because it
    # is only used in this function
    imgname = pf.get_string("stack_grid_name")
    imggrid = GCLdbread_by_name(db, imgname, object_type="pwmig::gclgrid::GCLgrid3d")
    migrated_image = GCLvectorfield3d(imggrid, 5)
    del imggrid
    migrated_image.zero()
    migrated_image.name="pwmigimage"
    if verbose:
        print("Creeated image grid")
        imagevolsize=compute_3dfieldsize(migrated_image)
        print("Size (bytes) of image grid used for this run=",imagevolsize)

    # This function extracts parameters passed around through a Metadata
    # container (what it returns).   These are a subset of those extracted
    # in this function.  This should, perhaps, be passed into this function
    # but the cost of extracting it from pf in this function is assumed tiny
    base_message = "migrate_evnet:  :  "
    border_pad = pf.get_long("border_padding")
    # This isn't used in this function, but retain this for now for this test
    # This test will probably be depricated when we the pfmig_verify functions is
    # completed as it is planned to have a constraints on the data a parameter allows
    zpad = pf.get_double("depth_padding_multiplier")
    if zpad > 1.5 or zpad <= 1.0:
        message = 'Illegal value for depth_padding_multiplier={zpad}\nMust be between 1 and 1.5'.format(zpad=zpad)
        raise MsPASSError(base_message + message, ErrorSeverity.Invalid)
    # these were used in file based io - revision uses mongodb but some of this
    # may need to be put into the control Metadata container
    # fielddir=pf.get_string("output_field_directory");
    # if os.path.exists(fielddir):
    #    if not os.path.isdir():
    #        message='fielddir parameter defined in parameter file as {}\n'.format(          fielddir)
    #        message+='File exists but is not a directory as required'
    #        raise MsPASSError(base_message+message,ErrorSeverity.Invalid)
    # else:
    #     os.mkdir(fielddir)
    # dfilebase=pf.get_string("output_filename_base");
    # Following C++ pwmig the output fieldnames, which become file names
    # defined with a dir/dfile combo i a MongoDB collection, are dfilebase+'_'+source_id
    # This is now always true - TODO:  verify that and when sure delete this next line
    # use_depth_variable_transformation=pf.get_bool("use_depth_variable_transformation")
    zmax = pf.get_double("maximum_depth")
    tmax = pf.get_double("maximum_time_lag")
    # dux=pf.get_double("slowness_grid_deltau")
    dt = pf.get_double("data_sample_interval")
    zdecfac = pf.get_long("Incident_TTgrid_zdecfac")
    # duy=dux
    # dumax=pf.get_double("delta_slowness_cutoff")
    # dz=pf.get_double("ray_trace_depth_increment")
    # rcomp_wt=pf.get_bool("recompute_weight_functions")
    # nwtsmooth=pf.get_long("weighting_function_smoother_length")
    # if nwtsmooth<=0:
    #    smooth_wt=False
    # else:
    #    smooth_wt=True
    # taper_length=pf.get_double("taper_length_turning_rays")
    # Parameters for handling elevation statics.
    # These are depricated because we now assume statics are applied with
    # an separate mspass function earlier in the workflow
    # ApplyElevationStatics=pf.get_bool("apply_elevation_statics")
    # static_velocity=pf.get_double("elevation_static_correction_velocity")
    # use_grt_weights=pf.get_bool("use_grt_weights")
    # use_3d_vmodel=pf.get_bool("use_3d_velocity_model")
    # The C++ version of pwmig loaded velocity model data and constructed
    # slowness grids from that.  Here we always use a 3D slowness model
    # loaded from the database using the new gclgrid library that
    # is deriven by mongodb.  Conversion from velocity to slowness and
    # building on from a 1d model is considered a preprocessing step
    # to assure the following loads will succeed.
    # WARNING - these names are new and not in any old pf files driving C++ version
    
    # load the slowness field directly if it is define.  If not try to 
    # derive alternate tag of a velocity fieldl name
    if "P_slowness_model_name" in pf:
        up3dname = pf.get_string('P_slowness_model_name')
        us3dname = pf.get_string('S_slowness_model_name')
        Up3d = GCLdbread_by_name(db, up3dname, object_type="pwmig::gclgrid::GCLscalarfield3d")
        Us3d = GCLdbread_by_name(db, us3dname, object_type="pwmig::gclgrid::GCLscalarfield3d")
    elif "P_velocity_model3d_name" in pf:
        up3dname = pf.get_string("P_velocity_model3d_name")
        Up3d = GCLdbread_by_name(db, up3dname, object_type="pwmig::gclgrid::GCLscalarfield3d")
        # convert to slowness
        Up3d = Velocity3DToSlowness(Up3d)
    else:
        message="missing required value for 3d P model name\n"
        message += "defined slowness model with key=P_slowness_model_name\n"
        message += "or velocity model with key=P_velocity_model3d_name"
        raise MsPASSError(message,ErrorSeverity.Fatal)
    if "S_slowness_model_name" in pf:
        us3dname = pf.get_string('S_slowness_model_name')
        Us3d = GCLdbread_by_name(db, us3dname, object_type="pwmig::gclgrid::GCLscalarfield3d")
    elif "S_velocity_model3d_name" in pf:
        us3dname = pf.get_string("S_velocity_model3d_name")
        Us3d = GCLdbread_by_name(db, us3dname, object_type="pwmig::gclgrid::GCLscalarfield3d")
        # convert to slowness
        Us3d = Velocity3DToSlowness(Us3d)
    else:
        message="missing required value for 3d S odel name\n"
        message += "defined slowness model with key=S_slowness_model_name\n"
        message += "or velocity model with key=S_velocity_model3d_name"
        raise MsPASSError(message,ErrorSeverity.Fatal)
    # Similar for 1d models.   The velocity name key is the pf here is the
    # same though since we don't convert to slowness in a 1d model
    # note the old program used files.  Here we store these in mongodb
    Pmodel1d_name = pf.get_string("P_velocity_model1d_name");
    Smodel1d_name = pf.get_string("S_velocity_model1d_name");
    Vp1d = vmod1d_dbread_by_name(db, Pmodel1d_name)
    Vs1d = vmod1d_dbread_by_name(db, Smodel1d_name)
    # Now bring in the grid geometry.  First the 2d surface of pseudostation points
    parent_grid_name = pf.get_string("Parent_GCLgrid_Name");
    parent = GCLdbread_by_name(db, parent_grid_name, object_type="pwmig::gclgrid::GCLgrid")
    # This functions is implemented in python because we currently know of
    # no stable and usable, open-source, travel time calculator in a lower
    # level language.  The performance hit doesn't seem horrible anyway
    # since we only compute this once per event
    if verbose:
        t0=time.time()
        print("Starting to compute incident P wave raygrid volume")
    svm0 = BuildSlownessGrid(parent, source_lat, source_lon, source_depth)
    TPfield = ComputeIncidentWaveRaygrid(parent, border_pad,
                                         Up3d, Vp1d, svm0, 
                                           zmax * zpad, tmax, dt, 
                                             zdecfac, True)
    del Up3d
    if verbose:
        print("Finished computatio in ",time.time()-t0," seconds")
        ttgsize=compute_3dfieldsize(TPfield)
        print("Size in bytes of 3d scalar volue=",ttgsize)

    # The loop over plane wave components is driven by a list of gridids
    # retried this way.   We also need, however, to subset by source id.  
    # some complexity here to handle source or telecluster as collection 
    # used for defining source data
    key = source_collection + "_id"
    query = {key: sid}
    gridid_list = db.wf_Seismogram.find(query).distinct('gridid')

    if parallel:
        f_parent = dask_client.scatter(parent,broadcast=True)
        f_TPfield = dask_client.scatter(TPfield,broadcast=True)
        f_svm0 = dask_client.scatter(svm0,broadcast=True)
        f_Us3d = dask_client.scatter(Us3d,broadcast=True)
        f_Vp1d = dask_client.scatter(Vp1d,broadcast=True)
        f_Vs1d = dask_client.scatter(Vs1d,broadcast=True)
        # this one isn't that large but probably better pushed this way
        f_control = dask_client.scatter(control,broadcast=True)

    if parallel:
        futures_list = []
        sidkey = source_collection + "_id"
        # these are used for block submits
        # if this works this variable will become an arg
        N_submit_buffer=8
        N_q = len(gridid_list)
        i_q = 0
        for gridid in gridid_list:
            query = {sidkey: sid, "gridid": gridid}
            print("Submitting data to cluster defined by this database query: ",query)
            f = dask_client.submit(_migrate_component, query, db.name, f_parent, f_TPfield,
                                   f_svm0, f_Us3d, f_Vp1d, f_Vs1d, f_control)
            #f = dask_client.submit(_migrate_component, query, db.name, parent, TPfield,
            #                       svm0, Us3d, Vp1d, Vs1d, control)
            futures_list.append(f)
            i_q += 1
            if i_q >= N_submit_buffer:
                break
        """
        # used for testing - deleted when fully resolved
        for f in as_completed(futures_list):
            x=f.result()
            del x
            print("Finished one")
        """
        """
        # Binary tree reduction for parallel accumulation with timely garbage collection
        def add_images(a, b):
            # Function to add two migrated image components.
            ddist.print("Summing raygrid into image volume")
            ddist.print("a.name=",a.name," a.n3=",a.n3)
            ddist.print("b.name=",b.name," b.n3=",b.n3)
            if a.name=="pwmigimage":
                a += b
                return a
            else:
                b += a
                return b
            #return a + b
        """
        """
        from dask.distributed import as_completed
        for f in as_completed(futures_list):
            t0sum=time.time()
            pwdgrid = f.result()
            ddist.print("Summing raygrid data into final image field")
            migrated_image += pwdgrid
            del pwdgrid
            ddist.print("Time to accumulate these data in master=",time.time()-t0sum)
        """
        """
        Gemini gives this algorithm a terminology called a "moving window pattern"
        The AI gives a variant of this algorithm with a useful description 
        when asked "with dask distributed how can you use as_completed to create 
        a fixed length buffer of futures"  An important variation Gemini 
        gives we might use if this works as expected is to replace the loop 
        above with a map consruct:
            
            futures_list = client.aap(migrate_component,first_batch,...)
            
        where first_batch is the first buffer size in the gridid list.
        
        My reduction algorithm below is actually cleaner than what Gemini 
        suggests.   I also add some diagnostic print statements. 
        
        Obviously this block commentshould be deleted if this works as hoped.
"
        """
        from dask.distributed import as_completed
        seq=as_completed(futures_list)
        for f in seq:
            t0sum=time.time()
            pwdgrid = f.result()
            ddist.print("Summing raygrid data into final image field")
            migrated_image += pwdgrid
            del pwdgrid
            # this seems necessar to force dask to release worker memory 
            # used by f
            del f
            ddist.print("Time to accumulate these data in master=",time.time()-t0sum)
            if i_q<N_q:
                print("submitting data for gridid=",gridid_list[i_q]," to cluster for processing")
                query = {sidkey: sid, "gridid": gridid_list[i_q]}
                new_f = dask_client.submit(_migrate_component, query, db.name, f_parent, f_TPfield,
                                       f_svm0, f_Us3d, f_Vp1d, f_Vs1d, f_control)
                seq.add(new_f)
                i_q += 1
            
            
        """
        while len(futures_list) > 1:
            new_futures = []
            for i in range(0, len(futures_list), 2):
                if i + 1 < len(futures_list):
                    # Submit a task to add two futures concurrently.
                    sum_future = dask_client.submit(add_images, futures_list[i], futures_list[i+1])
                    new_futures.append(sum_future)
                else:
                    # Carry forward the odd future.
                    new_futures.append(futures_list[i])
            futures_list = new_futures

        migrated_image = futures_list[0].result()
        """
    else:
        idkey = source_collection + "_id"
        query = {idkey: sid}
        i=0
        for gridid in gridid_list:
            print("Working on gridid=",gridid)
            t0 = time.time()
            query["gridid"] = gridid
            cursor = db.wf_Seismogram.find(query)
            pwensemble = db.read_data(cursor, collection="wf_Seismogram")
            cursor.close()
            t1 = time.time()
            pwdgrid = migrate_component(pwensemble, parent, TPfield, svm0, Us3d,
                                        Vp1d, Vs1d, control)
            t2=time.time()
            if i==0:
                migrated_image = pwdgrid
            else:
                migrated_image += pwdgrid
            t3 = time.time()
            print("Time to read data=",t1=t0) 
            print("Time to run migrate_component=",t2-t1)
            print("Time to sum grids=",t3-t2)
            i += 1
        
    return migrated_image