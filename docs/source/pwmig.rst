.. _pwmig.rst:

pwmig concepts
*******************
Overview
============
Running *pwmig* is the ultimate objective of this package.
All the many preceding steps can be viewed as only data preparation.
If you have gotten here you no doubt realize that preparation is lengthy
with many steps.   You can think of the last steps needed as the
process of linking all that data as input to the actual main processing
function that has the name *pwmig_dataset*.   There are three final
steps to do that described in sections below:
1.  Edit the parameter file that defines linkages to auxiliary data
    and run the python function *pwmig_verify* to validate the input.
2.  Estimate memory use and configure your batch job script to
    assure the processing will run within the memory constraints of your system.
3.  Actually run *pwmig_dataset* with the proper incantation.

.. _pwmig.pf:

pf-file configuration
=======================

TODO:   this section is incomplete.  The default pf seems to lack
some new parameters added for parallel opions.

As noted elsewhere a legacy feature of this package is the use of
Antelope "parameter files" (pf-file) as a means to configure a complex
algorithm like pwmig.  Note you should always edit this file and not
remove any of the keys even if your run will not require them.  The reason
is that *pwmig_dataset* creates an internal control structure from the pf-file
and it currently doesn't work around optional keys.

A version of pf file is the following grouped
by concept discussed in subsections below:

.. code-block:: python

   # parameters linking previously prepared auxiliary data
   P_velocity_model1d_name ak135_P
   S_velocity_model1d_name ak135_S
   use_3d_velocity_model false
   P_velocity_model3d_name ak135_Pvelocity
   S_velocity_model3d_name ak135_Svelocity
   # alternative - see below for details
   #P_slowness_name ak135_Up
   #S_slowness_name ak135_Us
   Parent_GCLgrid_Name all48
   stack_grid_name all48
   source_collection telecluster
   #
   # imaging algorithm control parameters
   #
   use_grt_weights true
   stack_only false
   border_padding  10
   depth_padding_multiplier 1.2
   taper_length_turning_rays 1.5
   recompute_weight_functions true
   weighting_function_smoother_length 10
   slowness_grid_deltau    0.01
   ray_trace_depth_increment       1.0
   maximum_depth   1000.0
   maximum_time_lag        120.0
   data_sample_interval 0.05


   # parameters used for computing incident P wave travel time 3d raygrid
   Incident_TTgrid_zdecfac 10
   depth_padding_multiplier 1.2
   # note several parameters used for incident P raygrid are also used
   # in S raygrid - they are defined above in the control group
   # hence it is in the group loaded as control

   # Run control parameters
   parallel true
   sliding_window_size auto
   accumulate true
   save_components true
   clear_scratch_data true
   save_component_directory /N/scratch/pavlis/pwmigtest

Auxiliary Linking Data Linking Parameters
----------------------------------------------

The parameters in this section can be grouped by two concepts
you had to deal with earlier:

1.  *Velocity Model Definition*.    (TODO:   not sure how I'm going to structure that page(s))
    What exactly is required is first controlled by the boolean *use_3d_velocity_model*.
    As the name implies when it is set True the application will require a 3d
    earth model to run.  When it is False the pf-file keys related to 3d models
    are ignored.  (Note the key must still be present in pf file even if the
    value is meaningless.)  Two keys related to velocity models are
    always required: *P_velocity_model1d_name* and *S_velocity_model1d_name*.
    They must resolve to allow retrieving the 1d velocity model data you
    stored in the working database earlier.  When using a 3D model you
    must also define either *P_velocity_model3d_name* and
    *S_velocity_model3d_name* or the alternative pair
    *P_slowness_name* and *S_slowness_name*.   As the names, imply which
    to use depends upon how to built your 3d velocity model (as a grid of velocity
    or slowness values).   Note the code first checks for slowness data and
    reverts to velocity only if the slowness names are not defined.  That means
    if you specify both the slowness data will be used and the velocity data
    ignored.
2.  *Image grid geometry*.  Two parameters relate to the way you created
    the image grid that pwmig uses to accumulate a 3d image:

    - *Parent_GCLgrid_name* must define the name keyword to load the
      original, 2D, surface grid used to define the pseudostation
      points when you ran *pwstack*.  The reason this grid geometry is
      required is pwmig creates a 2d field of incident wave slowness
      vectors it uses for computing ray grids (see below).
    - *stack_grid_name* is should define the name keyword you used when
      you ran *makegclgrid* to create what I called the "image grid".
      What *pwmig* does is load that grid geometry and clones it to
      produce what I call a `GCLvectorfield3d` object that holds
      migrated data from each event.

There is one parameter in the *pwmig.pf* file you should never change
called *source_collection*.   That parameter exists in that file only to
support the standard test run you may have already used to validate
the package is functional.

Imaging algorithm parameters
----------------------------------

Incident travel-time grid geometry parameters
-------------------------------------------------

TODO:   I think I can make this section clearer if I add an auxiliary page
accessible by an internal link discussing the idea of a "raygrid".   I'll
assume that will be created and for now use the tag "_raygrid"

Two parameters are used ONLY IF the you want to use a 3d model
for imaging (i.e. *use_3d_velocity_model* is set to "true"):
*Incident_TTgrid_zdecfac* and *depth_padding_multiplier*.   You
normally should not need to change these, but if you do continue
with the rest of this subsection to understand what they do.

Travel-time calculations are expensive and can be challenging for a fully 3D
model.  This package uses the same approach for addressing this as the
original *pwmig* package.  It uses a :ref:`_raygrid` computed like
the ones used for image backprojection, but the grid stores
P wave travel time perturbations relative to the 1D reference model
instead of migrated data.  The geometry of the raygrid generated in side *pwmig*is
illustrated in Figure :ref:`_ttraygrid`. (TODO:   figure form one of the pwmig papers showing travel time surfaces)
Key things to note about how the generic geometry of that figure ties to your
project are:

1. The surface grid geometry is defined by the grid geometry you load
   with the parameter *Parent_GCLgrid_Name*.
2. The rays that form the n3 coordinates of the raygrid generalized
   coordinates are computed using the 1D reference model with the
   incident wave slowness computed from that same model.  The source-receiver
   geometry is then used to rotate the ray path to point toward the
   source from each surface grid point. (i.e. the "back azimuth" or the standard
   MsPASS Metadata key "seaz" angle.)  The rays are computed to the
   depth defined by the product *maximum_depth* x *depth_padding_multiplier*.

The grid is first sampled in n3 (depth) direction using the same
values you set for the imaging with parameter *ray_trace_depth_increment*.
Early in the development of *pwmig* I learned that sampling a P wave travel-time grid
at the same interval as the imaging grids, which need to be consistent with the
data sample interval, was inefficient for computing and memory use.
The fundamental reason is  that (a) tomography models are always smooth relative
to the scale of converted wave images, and (b) travel-times integrate the
slowness field so are the integration of a field or already smooth values.
For that reason the incident P wave field should always be decimated
relative to the image grid geometry.  That decimation is controlled by
the *Incident_TTgrid_zdecfac* parameter.   It should be an integer that
defines the decimation factor for creating the actual incident P wave
travel time grid.  The bottom of the decimated grid used is defined by
the first decimated point below the value defined by the *maximum_depth* parameter.
The default for *Incident_TTgrid_zdecfac* is known to work well with the default
value for *ray_trace_depth_increment*.   I recommend you change it only if
you elect to change *ray_trace_depth_increment*.

.. note::

  I emphasize that if you use a 3D model for imaging it must be in
  the form of velocity or slowness perturbations.  If you run
  *pwmig_verify* as described below and read the output for the report
  you are unlikely to make this error but using absolute velocity or
  slowness as input will guarantee garbage output of a run time failure.
  The input model should also be relative to the same
  1d reference model defined by *P_velocity_model1d_name* or there will be
  systematic depth errors in your results.

Establish run parameters
===========================
*pwmig* is both memory and cpu intensive.   I have found it essential to
estimate the memory requirements for components of the dask cluster you
will run the application on.  If, as is most likely, you will be running
the application on a large cluster shared by many users you will also need
to have an estimate of how long the job will require.   I'll address these
issues in separate subsections below.   Before continuing, however,
you should realize that both scale by the number of workers,
which I will refer to with the symbol :math:`N_w` below.  A key factor
is that for most clusters :math:`N_w<N_{core}` where :math:`N_{core}`
is the number of cores per cluster node.  The reason is that *pwmig* is a
memory pig because it stores data in multiple large 3d grids.
The code can still use all cores in effectively for reasons
best address elsewhere.  The overall strategy a present should, however, be to
make :math:`N_w` as large as possible within memory constraints.

.. note::

  Work is in progress to use the standard threading package called
  `OpenMP<https://www.openmp.org/>`__
  to further improve the performance of *pwmig*.  When that is resolved
  the strategy for setting :math:`N_w` will change.

Memory use estimation
-----------------------
I have created a tool to help you estimate memory use.  I implemented it
as a python class with the (verbose) name `worker_memory_estimator`.
To use the tool you can use this partial code fragment as a starting point:

.. code:: python

   from pwmigpy.pwmig.pwmig import worker_memory_estimator

   memtool = worker_memory_estimator(db,sid,pf)
   memtool.report()

which prints a report that gives you initial estimates of the
memory requirements for each worker.

The required arguments are:

1.  `db` is an instance of the MsPASS Database class.  It is normally produced
    by a variation of this standard incantation to start any MsPASS workflow:

.. code::  python

   from mspasspy.client import Client
   mspass_client = Client()
   db = mspass_client.get_database("my_database_name")

2.  `pf` is the image of the "parameter file" (normally "pwmig.pf") you
    modified using guidance from earlier in this section.   To load your
    instance you can use the code box below if you put "pwmig.pf" in your
    run directory.

.. code:: python

   from mspasspy.ccore.utility import AntelopePf
   pf = AntelopePf("pwmig.pf")

3.  `sid` is the most challenging value to acquire.   `sid` is expected
    to be a key to describe a "source id".   The default, which should be
    used unless you are an expert in using this package, is to use the
    key `telecluster_id`.  A value for `telecluster_id` is inserted by
    `pseudosource_stacker` in each stacked `Seismogram` object it saves
    and then copied to all outputs of *pwstack*.
    That id is needed to group data from each event that is to be migrated
    by *pwmig*.   The outermost loop in *pwmig* is a loop over plane
    wave component estimates created by *pwstack* and linked to
    a particular value of `telecluster_id`.   A complication, however, is
    that because *pwmig* always has to handle incomplete data the
    number of actual Seismogram objects it needs to handle depends upon
    `telecluster_id`.   Consequently, what you need to establish for memory
    estimation is what event has the highest coverage which translates into
    the count of the number inputs.  The following block of code is
    a template you can use to establish an appropriate value for `sid`:

.. code:: python

   # assume an instance of db is already defined
   idlist = db.wf_Seismogram.distinct("telecluster_id")
   counts = dict()
   print("telecluster_id  N_data")
   for sid in idlist:
       # this is default data_tag value or pwstack output
       # if you change that change this
       query = {"data_tag" : "pwstack_data",
                "telecluster_id" : sid}
       n = db.wf_Seismogram.count_documents(query)
       print(sid,n)
       counts[sid] = n
   max_item = max(counts.items(), key=lambda k: k[1])
   print(f"telecluster_id={max_item[0]} has {max_item[1]} Seismgrams")
   sid2use = max_item[0]

    Note when you get that to run you should use the value `sid2use` in the
    constructor for `worker_memory_estimator`.

When you have successfully run the `worker_memory_estimator.report` method
read the output carefully.  The most important number in the report
is the line in the report under the heading "estimate memory use per worker".
I will use the symbolic shorthand :math:`M_{worker}` for convenience.
How to use these numbers?   The answer depends on the configuration you set
up for your virtual cluster.   The simplest case to consider is workers
run on secondary nodes on an HPC system where the only competition for memory
is the operating system and the container overhead.   If the memory size of
such a worker node is :math:`M_{node}` and we let :math:`M_{os}` represent
the memory consumed by the operating system and container software, then the
maximum number of workers that can be run on that node is

.. math::

   N_w = \frac{M_{node} - M_{os}}{M_{worker}}

On a desktop, the primary node on an HPC system, or
cloud system, any workers have to
complete for memory with the other required MsPASS services.  That is,
we need an estimate of:

1.  The maximum memory used by the MongoDB database server, which
    I define as :math:`M_{db}`.
2.  The maximum memory used by the dask scheduler, which I define as
    :math:`M_{scheduler}`.

Both of those numbers are dynamic and data dependent.  I suggest if you want
precise estimates, then you should consult a number of internet pages that
address this issue.   In my experience a good, simple upper bound for
both is around 1 GB.

In addition, `worker_memory_estimator.report` prints two additional
memory estimates that are required on the primary node.  The
first is a sum of memory required for all the major grid objects
insider the driver function, which I will call :math:`M_{base}` in
reference to the descriptive line "Base size without worker data ... ".
The second is much smaller but not neglible.  Each worker computes a
large "raygrid" that it returns to the primary node where it is accumulated
into the master 3d grid holding the results.   The scheduler has to
allocate a buffer to hold that amount of data, which I will defined
as :math:`M_{buffer}`, for each worker.

In any case, with those number an upper bound on the number of workers
that can be run on the primary node is

.. math::

  N_w =
  \frac{M_{node} - M_{os} - M_{db} - M_{scheduler} - M_{base} - c*M_{buffer}}
  {M_worker}

where c is a guess of the number of objects in transit at any give time.
That term is small and a small number like 2 o 3 has worked for me.
The best advice is to be conservative in your initial estimates and use
the memory monitoring option, `monitor_memory=True`, in initial runs to
refine the number of workers to run on the primary node.   For pure worker
nodes there is far less uncertainty.

Run time estimates
--------------------

Estimating run time is challenging without getting into a long and
complicated discussion of the known scaling relationships of this code.
My advice is to use an empirical approach with a few loose guardrails.
That is, use some reasonable initial guess on how long it will take to
process a single event, multiply that by the number of "pseudoevents"
you processed with *pwstack*, and request that length of time.   That
is a workable solution because the *pwmig_dataset* function, which I
describe below, has a `restart` option.   With that feature if your
first run aborts on a wall clock limit, you can just verify how much
was completed, and estimate how much more time will be needed to finish
data not yet processed.

To help you set those "guardrails" check the time it took on your system
to run the validation test data using (TODO:   this needs to be a link to a notebook)
Two scaling factors you can use to translate that to your data are:

1.  An approximation to adjust for different numbers of workers used to run
    the test and your production work is to just use the ratio of two
    numbers to scale the time per event up or down as appropriate.
2.  The compute time scales approximately linearly with the size of the
    imagegrid.   Look at the number of grid points in the test data
    and the imagegrid you defined for your data.  Use the ratio to
    scale the test processing time to get a rough guess for your data set.

I you don't want deal with all that just try 2 hours per event see how far
you get.

Runtime options
================

I recommend most users use the function loaded with this import to run
pwmig:

.. code:: python

  from pwmigpy.pwmig.dataset import pwmig_dataset

This function automates the messy process of setting up the runtime
environment to allow the workers to access the database server
and to allow the primary python interpreter to submit work to the
cluster.

The call signature for that function is the following:

.. code::python

   def pwmig_dataset(
      mspass_client,
      dbname,
      pfname="pwmig.pf",
      source_collection="telecluster",
      pwstack_data_tag="pwstack_data",
      base_fieldname="pwmigdata",
      outdir=None,
      restart=False,
      minimum_data=10000,
      verbose=False,
      initialize_workers=True,
      monitor_memory=False,
   ):

The required arguments are:

1.  `mspass_client` is an instance of a `mspasspy.client.Client` object.
    It is normally created with the standard incantation shown near the beginning
    of this section.  It contains the data needed to do that "messy process" noted above.
2.  `dbname` is the name of the database that you have used to manage the data
    for your project.   That name is necessary to set up the database client
    connection for each worker.

Implicitly required is that the file referenced by the `pfname` argument
exists and contains the configuration parameters you worked out earlier.
Note the default assumes that data is in a file called "pwmig.pf" in the
run directory.   If you put the parameter file somewhere else this argument
should be a path name string.

The following options are likely of use but not set by default:

- `restart` is a boolean used as described above.  *pwmig* essentially uses
  the MongoDB database to checkpoint progress.   When set true, as the
  function starts it will check for completed data and skip source ids
  found to have already been processed.  Setting this True, as noted,
  is a way to pick up processing if an earier job aborts on a time limit
  or some other reason.
- `monitor_memory` when set True enables a series of print statements
  to monitor system memory at various stages of processing.  The output is
  not that verbose and the overhead is tiny
  so I advise you always set this True for the first few times you run this
  package on a new dataset.  Note the monitor ONLY check the primary node
  memory and not workers.   It is useful because of the uncertainties in
  memory use by MsPASS services noted above.  The numbers spit out by this
  option can help you identify a possible memory problem on the primary node.
- `minimum_data` should be used to automatically discard data with
  poor spatial coverage.   The program queries the database to find the
  total number of `Seismogram` objects from *pwstack* stored in the database.
  If that number is less than `minimum_data` the `pwmig_dataset` function
  posts a warning message and skips all the data from that event.
  A reasonable size for this parameter is data dependent.  My advice is
  to keep this value small initially and decide later if you want to exclude the
  result from the final stack produced with `gridstacker`.   A reasonable
  initial setting is 10% of the total number of pseudostations times
  the number of plane wave components computed by *pwstack*.
- `outdir` can be used to change the directory where the large grid
  file data are stored.  The default sets that to run directory.
  Because *pwmig* generates one file per input pseudosource, it is good practice
  to set this argument to some appropriate directory name to avoid cluttering
  an already cluttered run directory.
- `verbose` defines the common use that when set true causes more information
  to be printed during runtime.  The verbose output isn't that extreme so
  I recommend setting it True for initial runs.
- `base_fieldname` is use to define a unique name tag for each 3d grid
  created as output.   That tag is used in two ways:  (a) to generate a
  unique file name to hold the sample data for each grid, and (b) as
  a unique value for the "name" attribute stored in the MongoDB database.
  sample data for the migrated grid output from each pseudosource group.
  The actual name used to uniquely define each result is
  `base_fieldname` + "_str(telecluster_id)".   That is, the "telecluster_id"
  is a unique id for each pseudosource which has a unique string
  representation.  The most important thing to realize is that if you
  rerun *pwmig* with different input parameters and save the results to
  the same database as used earlier, you MUST change this argument or
  it will be difficult to distinguish the results.

I do not recommend changing the defaults for the following unless
you have a deep understanding of what they do:  `source_collection`,
`pwstack_data_tag`, and `initialize_workers`.
