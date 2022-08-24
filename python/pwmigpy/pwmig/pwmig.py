
import math
import dask
from mspasspy.ccore.utility import (AntelopePf,
                    Metadata,
                    MsPASSError,
                    ErrorSeverity)
from mspasspy.ccore.seismic import (SlownessVector)
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth,kilometers2degrees

from pwmigpy.ccore.gclgrid import (GCLvectorfield3d,
                    PWMIGfielddata)

from pwmigpy.ccore.pwmigcore import (SlownessVectorMatrix,
                    Build_GCLraygrid,
                    ComputeIncidentWaveRaygrid,
                    migrate_one_seismogram)
from pwmigpy.db.database import (GCLdbread,
                                 vmod1d_dbread_by_name,
                                 GCLdbread_by_name)
#from pwmigpy.ccore.seispp import VelocityModel_1d

def _print_default_used_message(key,defval):
    print("Parameter warning:  AntelopePf file does not have key=",key,
          " defined.\nUsing default value=",defval," which is of type ",type(defval))
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
    result=Metadata()
    if control.is_defined("use_3d_velocity_model"):
        use_3d_vmodel=control.get_bool("use_3d_velocity_model")
    else:
        use_3d_vmodel=True
        _print_default_used_message("use_3d_velocity_model",use_3d_vmodel)
    if control.is_defined("use_grt_weights"):
        use_grt_weights=control.get_bool("use_grt_weights")
    else:
        use_grt_weights=True
        _print_default_used_message("use_grt_weights",use_grt_weights)
    if control.is_defined("stack_only"):
        stack_only=control.get_bool("stack_only")
    else:
        stack_only=True
        _print_default_used_message("stack_only",stack_only)
    if control.is_defined("border_padding"):
        border_pad = control.get_long("border_padding")
    else:
        border_pad=20
        _print_default_used_message("border_padding",border_pad)
    if control.is_defined("depth_padding_multiplier"):
        zpad = control.get_double("depth_padding_multiplier")
    else:
        zpad=1.2
        _print_default_used_message("depth_padding_multiplier",zpad)
    if control.is_defined("taper_length_turning_rays"):
        taper_length=control.get_double("taper_length_turning_rays")
    else:
        taper_length=2.0
        _print_default_used_message("taper_length_turning_rays",taper_length)
    if control.is_defined("recompute_weight_functions"):
        rcomp_wt=control.get_bool("recompute_weight_functions")
    else:
        rcomp_wt=True
        _print_default_used_message("recompute_weight_functions",rcomp_wt)
    if control.is_defined("weighting_function_smoother_length"):
        nwtsmooth=control.get_long("weighting_function_smoother_length")
    else:
        nwtsmooth=10
        _print_default_used_message("weighting_function_smoother_length",nwtsmooth)
    if control.is_defined("slowness_grid_deltau"):
        dux=control.get_double("slowness_grid_deltau")
    else:
        dux=0.01
        _print_default_used_message("slowness_grid_deltau",dux)
    if control.is_defined("ray_trace_depth_increment"):
        dz=control.get_double("ray_trace_depth_increment")
    else:
        dz=1.0
        _print_default_used_message("ray_trace_depth_increment",dz)

    result.put("use_3d_velocity_model",use_3d_vmodel)
    result.put("use_grt_weights",use_grt_weights)
    result.put("stack_only",stack_only)
    result.put("border_padding",border_pad)
    result.put("depth_padding_multiplier",zpad)
    result.put("taper_length_turning_rays",taper_length)
    result.put("recompute_weight_functions",rcomp_wt)
    result.put("weighting_function_smoother_length",nwtsmooth)
    result.put("slowness_grid_deltau",dux)
    result.put("ray_trace_depth_increment",dz)
    # these have no defaults
    result.put("maximum_depth",control["maximum_depth"])
    result.put("maximum_time_lag",control["maximum_time_lag"])
    result.put("data_sample_interval",control["data_sample_interval"])
    return result;

def BuildSlownessGrid(g,source_lat, source_lon, source_depth,model='iasp91',phase='P'):
    model=TauPyModel(model=model)
    Rearth=6378.164
    svm=SlownessVectorMatrix(g.n1,g.n2)
    for i in range(g.n1):
        for j in range(g.n2):
            stalat=g.lat(i,j)
            stalon=g.lon(i,j)
            georesult=gps2dist_azimuth(source_lat,source_lon,stalat,stalon)
            # obspy's function we just called returns distance in m in element 0 of a tuple
            # their travel time calculator it is degrees so we need this conversion
            dist=kilometers2degrees(georesult[0]/1000.0)
            arrivals=model.get_travel_times(source_depth_in_km=source_depth,
                            distance_in_degree=dist,phase_list=phase)
            ray_param=arrivals[0].ray_param
            umag=ray_param/Rearth    # need slowness in s/km but ray_param is s/radian
            baz=georesult[2]   # The obspy function seems to return back azimuth
            az=baz+180.0
            # az can be > 360 here but all known trig function algs handl this automatically
            ux=umag*math.sin(math.radians(az))
            uy=umag*math.cos(math.radians(az))
            u=SlownessVector(ux,uy,0.0)
            svm.set_slowness(u,i,j)
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
    query={'source_id': source_id, "gridid" : gridid}
    collection=db[collection]
    return collection.find(query)
def _set_incident_slowness_metadata(d,svm):
    """
    Internal function used in map to set metadata fields in
    mspass Seismogram, d, from incident wave slowness data stored in
    the SlownessVectorMatrix object svm.

    """
    i=d['ix1']
    j=d['ix2']
    # not sure this is in the bindings but idea is to fetch a SlownessVector for i,j
    slowness=svm.get_slowness(i,j)
    # Warning:  these keys must be consistent with C++ function migrate_one_seismogram
    # corresponding get
    d['ux0']=slowness.ux
    d['uy0']=slowness.uy
    return d

def _add_fieldata(f1,f2):
    """
    Applies += operator and returns f1+f2.  If geometries of f1 and f2
    differ the returned field will have the geometry of f1
    """
    f1 += f2
    return f1

def _migrate_component(cursor,db,parent,TPfield,VPsvm,Us3d,Vp1d,Vs1d,control):
    """
    This is an intermediate level function used in the mspass version of
    pwmig.  It will migrate one plane wave component of data created by
    pwstack and return the result as a GCLvectorfield3d object.
    The caller may either immediately stack the plane wave components
    or save them and run a seperate stacker later.  In either case the
    expectation is this function appears as the function object in a
    map operator driven by a bag of cursors. The cursors are assumed
    define a correct and consistent set of data.  No checking is made
    here to verify the consistency of the inputs.

    The code first sets up the raygrid geometry based on input slowness
    data passed through the VPsvm object (here a GCLvectorfield but originally
    this was a smaller thing read from the old pwstack to pwmig file structure).

    """

    zmax=control.get_double("maximum_depth")
    tmax=control.get_double("maximum_time_lag")
    dt=control.get_double("data_sample_interval")

    # old pwmig had a relic calculation of a variable ustack and deltaslow here
    # Replacing it here with slowness vectors for each grid point in the
    # function called in the fold operation below.  Note deltaslow is
    # invariant anyway and can be extracted from the metadata of each Seismogram
    # also skipping coherence grid stuff as that was found earlier to not be
    # worth the extra compute time and it would be better done in his framework
    # as an indepndent processing step
    VPVSmax=2.0   # Intentionally large as this is just used to avoid infinite loops where used
    # This sigature of this function has changed from old pwmig - depricated
    # consant u mode
    raygrid=Build_GCLraygrid(parent,VPsvm,Vs1d,zmax,VPVSmax*tmax,dt*VPVSmax)
    pwdgrid=PWMIGfielddata(raygrid)
    for doc in cursor:
        seis = dask.delayed(db.read_data)(doc)
        seis = _set_incident_slowness_metadata(seis, VPsvm)
        migseis = dask.delayed(migrate_one_seismogram)(seis,parent,raygrid,TPfield,Us3d,Vp1d,Vs1d,control)
        pwdgrid.accumulate(migseis)
        
    pwdgrid = dask.compute(*pwdgrid)    
    return pwdgrid

def pwmig_verify(db,pffile="pwmig.pf",GCLcollection='GCLfielddata',
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
    algorithm="pwmig"
    pf=AntelopePf(pffile)
    # TODO  this does not exists and is subject to design changes from github discussion
    #pfverify(algorithm,pf)
    # this also doesn't exists.  Idea is to verify all model files are
    # present and valid
    #pwmig_verify_files(pf)
    # now verify the database collection and print a report
    # TODO;  need to implement this fully - this is rough
    # 1  verify all gclfeield and vmodel data
    # print a report for waveform inputs per event

def migrate_event(db,source_id,pf,collection='GCLfielddata'):
    """
    Top level map function for pwmig.   pwmig is a "prestack" method
    meaning we migrate data from individual source regions (always best
    done as the output of stacked data produced by running telecluster
    followed by RFeventStacker.)  This function creates a 3d image
    volume from migration of one ensemble of data assumed linked to
    a single source region.

    :param db:  database handle used to manage data (mspass Database handle).
    :param source_id:  ObjectId of the source associated with each of waveform
      to use for imaging. This assumes each wf_Seismogram entry has this
      field set with the key "source_id".
    :param pf:  AntelopePf object containing the control parameters for
      the program (run the pwmig_pfverify function to assure the parameters
      in this file are complete before starting a complete processing run.)
    :param collection:  collection name where 3d modes are defined
      (default is GCLfielddata which is the default for the dbsave
       methods of the set of gclgrid objects.)

    """
    gclcollection=db[collection]
    # Freeze use of source collection for source_id consistent with MsPASS
    # default schema.
    doc=db.source.find_one({'_id' : source_id})
    if doc == None:
        # This is fatal because expected use is parallel processing of multiple event
        # if any are not defined the job should abort.  Also a preprocess
        # checker is planned to check for such problems
        raise MsPASSError("migrate_event:  source_id="+str(source_id)+" not found in database",ErrorSeverity.Fatal)
    source_lat=doc['lat']
    source_lon=doc['lon']
    source_depth=doc['depth']
    #source_time=doc['time']

    # This function is the one that extracts parameters required in
    # what were once inner loops of this program.  As noted there are
    # so many parameters it makes the code more readable to pass just
    # this control metadata container around.  The dark side is if any
    # new parameters are added changes are required in this function,
    control = _build_control_metadata(pf)
    
    # This builds the image volume used to accumulate plane wave 
    # components.   We assume it was constructed earlier and saved
    # to the database.  This parameter is outside control because it 
    # is only used in this function
    imgname = pf.get_string("stack_grid_name")
    imggrid = GCLdbread_by_name(db,imgname)
    migrated_image = GCLvectorfield3d(imggrid,5)
    del imggrid
    migrated_image.zero()
    
    # This function extracts parameters passed around through a Metadata
    # container (what it returns).   These are a subset of those extracted
    # in this function.  This should, perhaps, be passed into this function
    # but the cost of extracting it from pf in this function is assumed tiny
    base_message="class pwmig constructor:  "
    border_pad = pf.get_long("border_padding")
    # This isn't used in this function, but retain this for now for this test
    # This test will probably be depricated when we the pfmig_verify functions is
    # completed as it is planned to have a constraints on the data a parameter allows
    zpad = pf.get_double("depth_padding_multiplier")
    if zpad>1.5 or zpad<=1.0:
        message='Illegal value for depth_padding_multiplier={zpad}\nMust be between 1 and 1.5'.format(zpad=          zpad)
        raise MsPASSError(base_message+message,ErrorSeverity.Invalid)
    # these were used in file based io - revision uses mongodb but some of this
    # may need to be put into the control Metadata container
    #fielddir=pf.get_string("output_field_directory");
    #if os.path.exists(fielddir):
    #    if not os.path.isdir():
    #        message='fielddir parameter defined in parameter file as {}\n'.format(          fielddir)
    #        message+='File exists but is not a directory as required'
    #        raise MsPASSError(base_message+message,ErrorSeverity.Invalid)
    #else:
    #     os.mkdir(fielddir)
    #dfilebase=pf.get_string("output_filename_base");
    # Following C++ pwmig the output fieldnames, which become file names
    # defined with a dir/dfile combo i a MongoDB collection, are dfilebase+'_'+source_id
    # This is now always true - TODO:  verify that and when sure delete this next line
    #use_depth_variable_transformation=pf.get_bool("use_depth_variable_transformation")
    zmax=pf.get_double("maximum_depth")
    tmax=pf.get_double("maximum_time_lag")
    #dux=pf.get_double("slowness_grid_deltau")
    dt=pf.get_double("data_sample_interval")
    zdecfac=pf.get_long("Incident_TTgrid_zdecfac")
    #duy=dux
    #dumax=pf.get_double("delta_slowness_cutoff")
    #dz=pf.get_double("ray_trace_depth_increment")
    #rcomp_wt=pf.get_bool("recompute_weight_functions")
    #nwtsmooth=pf.get_long("weighting_function_smoother_length")
    #if nwtsmooth<=0:
    #    smooth_wt=False
    #else:
    #    smooth_wt=True
    #taper_length=pf.get_double("taper_length_turning_rays")
    # Parameters for handling elevation statics.
    # These are depricated because we now assume statics are applied with
    # an separate mspass function earlier in the workflow
    #ApplyElevationStatics=pf.get_bool("apply_elevation_statics")
    #static_velocity=pf.get_double("elevation_static_correction_velocity")
    #use_grt_weights=pf.get_bool("use_grt_weights")
    #use_3d_vmodel=pf.get_bool("use_3d_velocity_model")
    # The C++ version of pwmig loaded velocity model data and constructed
    # slowness grids from that.  Here we always use a 3D slowness model
    # loaded from the database using the new gclgrid library that
    # is deriven by mongodb.  Conversion from velocity to slowness and
    # building on from a 1d model is considered a preprocessing step
    # to assure the following loads will succeed.
    # WARNING - these names are new and not in any old pf files driving C++ version
    up3dname=pf.get_string('Pslowness_model_name')
    us3dname=pf.get_string('Sslowness_model_name')
    Up3d=GCLdbread_by_name(db,up3dname)
    Us3d=GCLdbread_by_name(db,us3dname)
    # Similar for 1d models.   The velocity name key is the pf here is the
    # same though since we don't convert to slowness in a 1d model
    # note the old program used files.  Here we store these in mongodb
    Pmodel1d_name=pf.get_string("P_velocity_model1d_name");
    Smodel1d_name=pf.get_string("S_velocity_model1d_name");
    Vp1d=vmod1d_dbread_by_name(db,Pmodel1d_name)
    Vs1d=vmod1d_dbread_by_name(db,Smodel1d_name)
    # Now bring in the grid geometry.  First the 2d surface of pseudostation points
    parent_grid_name=pf.get_string("Parent_GCLgrid_Name");
    # CAUTION:  name may not be the right key
    query={'name' : parent_grid_name}
    doc=gclcollection.find_one(query)
    parent=GCLdbread(db,doc)
    # This functions is implemented in python because we currently know of
    # no stable and usable, open-source, travel time calculator in a lower
    # level language.  The performance hit doesn't seem horrible anyway
    # since we only compute this once per event
    svm0=BuildSlownessGrid(parent,source_lat, source_lon, source_depth)
    TPfield=ComputeIncidentWaveRaygrid(parent,border_pad,
       Up3d,Vp1d,svm0,zmax*zpad,tmax,dt,zdecfac,True)
    del Up3d
    
    # The loop over plane wave components is driven by a list of cursors
    # created in this MongoDB incantation
    query={'source_id' : source_id}
    gridid_list=db.wf_Seismogram.find(query).distinct('gridid')
    for gridid in gridid_list:
        cursor = query_by_id(gridid, db, source_id)
        migrated_data = _migrate_component(cursor,db,parent,TPfield,svm0,
                                 Us3d,Vp1d,Vs1d,control)
        migrated_image += migrated_data
        
    return migrated_image
