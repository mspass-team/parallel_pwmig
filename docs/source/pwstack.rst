.. _pwstack.rst:

pwstack concepts
*******************
Overview
============
At this point you have a set of what would be called "shot gathers"
in seismic reflection processing.  *pwmig* is a "prestack migration" method
meaning imaging is applied to the shot gathers before they are averaged
to produce a final image.  In addition, the "*pw*" prefix
in the name *pwmig* is a shorthand for "plane wave".
*pwmig* evolved from an earlier idea described in a pair of publications
by Neal and Pavlis (citation links via DUI here).
Scott Neal's work focused on using this idea to improve deconvolution,
but Christian Poppeliers and I (citation and links via dui' here) morphed
that into the concepts that became the foundations for *pwmig*.

*pwstack* has a fair number of input parameters.   The parameters directly
used by the algorithm need to be defined in the parameter file *pwstack.pf*.
The first section of this page describes which of those may require custom specifications
and guidelines on how to set them.  The final two sections
describe the python interface to run *pwstack*.

.. _pwstack_pf:

pwstack control parameters
============================

Plane wave component definition
---------------------------------

*pwstack* aims to decompose the input into a set of plane wave components,
The direction and speed of propagation of a plane wave is parameterized in
*pwmig* by a discrete set of slowness vectors.  A slowness vector fills the
same role as a wave vector (:math:`k`) for a single frequency plane wave but
is a more useful representation than a wavenumber for a signal made up of \
multiple frequencies since slowness :math:`u = \frac{1}{v} = \frac{k}{\omega}`.
i.e. :math:`u` is the same for all frequencies (:math:`\omega`) while :math:`k`
depends on frequency via that relationship.

*pwstack* uses what we call a *RectangularSlownessGrid* object.
The geometry of that grid is illustrated in Figure :ref:`_slowness_grid_figure`

.. figure:: figures/slowness_grid.tif

   :alt:  Slowness grid figure
   :align: center
   :name: _slowness_grid_figure

   **Figure 1**.
   Slowness space figure showing how the RectangularSlownessGrid vectors
   are created by summation with the incident wave slowness vector.
   :math:`p_0` is the incident wave slowness, :math:`\delta p` is the
   perturbation added for each slowness grid value, and
   :math:`p` is their vector sum.  The red grid illustrates a 13X13 grid
   of points defining the range of :math:`\delta p`.


The geometry of that grid is defined in attributes to the &Arr
group with the key *RectangularSlownessGrid*.   This example from
the default pf file is a helpful way to describe in words what is required:

..code-block:: python

  RectangularSlownessGrid &Arr{
    Slowness_Grid_Name  test
    dux 0.01
    duy 0.01
    nux 11
    nuy 11
    uxlow       -0.05
    uylow       -0.05
  }

This example defines a grid of slowness vectors with *nux=11* points
along the x-axis (positive local east direction) and *nuy=11*
points along the y-axis (positive local north direction). If drawn as
slowness space like Figure (cross refrerence to figure) the lower
left corner of the slowness grid is the pair *(uxlow, uylow)*.   The
slowness vectors used for plane wave phasing are spaced at intervals
defined by *dux* for x and *duy* for y.   For this example that is
{-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05} or both
the x and y axes.

*Slowness_Grid_Name* is a tag that could be used to store different grid
names.  It is, however, ignored at present.   It is a legacy from the C++ code.

The default slowness grid is a good starting point for most teleseismic data.
A few things to keep in mind:

1.  Although the definition allows otherwise it would be irrational to have
different range of slowness component values in x and y.
2.  Note as the figure shows the slowness vectors defined by this grid
are ADDED to the incident wave slowness.   *pwstack* will handle arbitrary
phasing but *pwmig* will throw a lot of error warnings about turning rays
if the range is made too large.   You must recognize that the magnitude of
each slowness vector computed as

.. math::

   \mathbf{u} = \mathbf{u}_0 + \delta \mathbf{u}^{ij}

with

.. math::

   \delta \mathbf{u}^{ij}
   =
   \left [
   \begin{array}{cc}
   \delta u_x^{ij} \\
   \delta u_y^{ij}
   \end {array}
   \right ]

Since Snell's law can be written as :math:` \| \mathbf{u} \| = \frac{ sin \theta }{v}`
large magnitude slowness vectors scale to increasing angles of incidence.
You should design the grid to never create a turning raw for a P to S conversion.
That means, means you should assure
:math:`\| \mathbf{u} \| < \frac{1}{V_s^{max}}`
where :math:`V_S^{max}` is the largest shear wave velocity in the reference
earth model.   Consider the default.   Normally we limit data for
converted wave imaging to events at a distance greater than 30 degrees.
The ray parameter (slowness) for a P wave from an event at 30 degrees is
approximately XXX.   For the default grid the largest increment in slowness is
:math:`0.05 \sqrt{2} ` so the largest slowness used in imaging with that grid is
YYY s/km.  ADD WORDS about iasp91 or ak135 S velocities.   On the other hand,
you want to make the range as large as possible to maximize dip resolution.
A turning ray has a 90 degree dip resolution at the depth it turns, but
can produce artifacts from truncation inside the image volume.
3.   You may want to decrease the slowness spacing values *dux* and *duy*
with parallel changes to *nux* and *nuy* but keep in mind that will produce
an $N^2$ increase in compute time for both *pwmig* and *pwstack*.  i.e. if you
double *nux* and *nuy* the compute time will increase by a factor of 4.
4. If you want to revert to so called "CCP stacking" set the grid to use only
a tiny phasing around with the minimum number of values.  Here is an example:

..code-block:: python

  RectangularSlownessGrid &Arr{
    Slowness_Grid_Name  ccp
    dux 0.0001
    duy 0.0001
    nux 2
    nuy 2
    uxlow       -0.0001
    uylow       -0.0001
  }


Windowing parameters
----------------------

Data that are received by the *pwstack* function are windowed in two stages.
First, the raw data are trimmed to the (relative) time range
defined by *data_time_window_start* to *data_time_window_end*.
The defaults are -25 and 120 respectively which is taken to mean
25 s before the P arrival pulse to a maximum lag of 120 s.
Each datum is then muted with a geometry defined by the
*Data_Top_Mute* parameter.   That parameter is defined by
an *&Arr* block that contains secondary parameters that define
the geometry.   The secondary parameters are more easily explained
with an example from the default *pwstack.pf* file:

.. code-block:: python

    Data_Top_Mute   &Arr{
     end_time    3.0
     time_reference_type relative
     zero_end_time       1.0
    }

This example applies a top mute that zeros all data prior to *zero_end_time* seconds
and then applies a linear ramp in amplitude between the time *zero_end_time*
(1.0 s for this example) and time *end_time*.   Samples at times after
*end_time* are not changed.   Note the *time_reference_time* for this case is
baggage that should never be changed.   It exists because the same top mute
code can be applied to data with a UTC time standard, but that feature is not
used in *pwstack*.   All times in *pwstack* are assumed to be times relative to
a P wave arrival time.

A secondary stage of windowing is defined by a time range defined
by the parameters *stack_start_time* to *stack_end_time*.   The plane
wave estimates emitted by the the *pwstack* are windowed by that (relative) time
range before being returned by the stacking function.   They are also
muted using the top mute geometry defined by the parameter
*Stack_Top_Mute*.   *Stack_Top_Mute* must be linked to an &Arr
block in the pf file with the same structure as *Data_Top_Mute*.
This, for example, is the default:

.. code-block:: python

   Stack_Top_Mute  &Arr{
     end_time    3.0
     time_reference_type relative
     zero_end_time       1.0
   }

The secondary parameters *zero_end_time* and *end_time* are used
the same way as for *Data_Top_Mute*.  i.e. this mute zeros data before
relative time 1, applies a linear taper to 3 s, and leaves the remainder
of each stack unaltered.

Spatial aperture parameters
-----------------------------

As pointed out by Neil and Pavlis (year) *pwstack* can be thought of as a
spatial filter.   Individual plane wave estimates are horizontally smoothed
averages of the recorded wavefield.   The smoother in *pwstack* is always
a 2D Gaussian function with a width that can and usually does depend on
lag (time relative to P wave arrival time).  Figures :ref:`_pseudostation_mapfigure`
and :ref:`_pwstack_aperture` may help you understand the concepts.
Figure :ref:`_pseudostation_mapfigure` illustrates how aperture parameters
select stations inside a circular map area.  Figure :ref:`_pwstack_aperture`
illustrates how data within a circular regions get projected into the subsurface
for a single pseudostation point.  It also shows the difference between
variable aperture methods and constant aperture discussed further below.

.. figure:: figures/pseudostation_mapfigure.tif

   :alt: pseudostation map figure
   :align: center
   :name: _pseudostation_mapfigure

   **Figure 2**.   Map example of how *pwstack* assembles data in
   circular regions it uses to derive plane wave estimates.   The
   map illustrates Earthscope TA stations in the Pacific Northwest
   with triangle symbols.   The inner circle shows the radius of
   one sigma distance for the Gaussian smoother.  The outer
   circle illustrates the concept of a *cutoff aperture* - stations
   outside that circle are not used to create a stack referenced to
   the virtual source point at the center of the circular region.



There are two different ways to
define the lag-dependent smoother width.

1.  The *Fresnel zone method* is the default and recommended for most use.
    It is enabled by setting *use_fresnel_aperture* to "true".
    When selected the width of the Gaussian as a function of lag is computed
    from an analytic formula for the Fresnel zone size computed as

    .. math::

       \sigma = V_s \sqrt {
        \left [ \frac{V_p}{V_p - V_s} t + \frac{\tau}{2} \right ]^2
        - \left [ \frac{V_p}{V_p - V_s} t \right ]^2
      }

    where :math:`\sigma` is the scale factor for 2D-Gaussian function.
    Parameters in the pf map to the symbols in the equation above as
    follows: :math:`V_s = `*fresnel_vs*, :math:`V_p = `*fresnel_vp*, and
    :math:`\tau = `*fresnel_period*.
    For efficiency *pwstack* also requires the
    related parameter *fresnel_cutoff_multiplier*.   For each pseudosource
    position, *(lat0,lon0)* the *pwstack* algorithm forms a gather of
    all stations within a distance of :math:`\sigma_{max} * ` *fresnel_cutoff_multiplier*
    where :math:`\sigma_{max}` is the largest value of the :math:`\sigma(t)`.
    For the Fresnel formula that always means the maximum computed lag for the
    depth dependent aperture.   That is defined by two parameters with
    somewhat self-descriptive names:
    (1) *fresnel_lag_time_sampling_interval* and (2)*fresnel_number_lag_samples*.
    Their product defines the maximum lag.

2.  *manual method*.  You may also specify the pseudostation aperture as a
    function of lag manually by setting *use_fresnel_aperture* to false.
    When set false, *pwstack* expects to find  a list of
    values under the pf &Tbl with the tag
    *depth_dependent_aperture*.   Here is an example from the default
    *pwmig.pf* file:

    .. code-block:: python


       depth_dependent_aperture        &Tbl{
         0.0 75.0 150.0
         100.0 75.0 150.0
       }

    This defines a constant width aperture with :math:`\sigma = 75` km
    and the cutoff distance at 150 km.  The first number in each row
    of the Tbl is S-P lag in seconds.  If the values varied with
    lag intermediate values would be determined by a linear interpolation
    between points.  This example has constant aperture.

    Note you can manually specify a lag-dependent aperture with
    this option by specifying more than two lines in the
    *depth_dependent_aperture* &Tbl block.  If you do that, however,
    you MUST make sure the cutoff values increase with lag.
    If not you can create artifacts in the final image related to
    the discussion of the *aperture_taper_length* parameter below.
    It would also be a bit irrational to do that given the theory
    behind why the Fresnel zone method is the default.

Two additional spatial aperture related parameter are *stack_count_cutoff*
and *centroid_cutoff*.
*pwstack* counts the number of data with a circular region defined by
the largest aperture cutoff distance.  Any pseudostation point where the
count is less than *stack_count_cutoff* is dropped and treated as a data
gap.  If the *stack_count_cutoff* is exceeded it then computes the
centroid of the actual stations inside that circular region.   If the
centroid distance form the pseudostation point (center of the circle)
exceeds *centroid_cutoff* the point will also be treated as a data gap.
*centroid_cutoff* is important to prevent artifact at edge as it
reduces artificial duplications on edges.   *centroid_cutoff* should be set to
about the same size as the pseudostation spacing in the regular image
grid you create with *makegclgrid*.

A more subtle requirement when using a time-variable aperture like
that with the Fresnel zone method, is a need to taper some of the data.
Early in the development of the pwmig algorithm I discovered that
data for stations near the outer edge of the circular aperture
near the cutoff distance (see Figure :ref:`_pseudostation_mapfigure`)
would produce artifacts in the output without a feature
defined by the parameter *aperture_taper_length*.   This issue happens
only when using a lag-dependent aperture like the (default) Fresnel ()
zone method.   Data for stations inside a circles defined by the
minimum cutoff distance (0 lag for the Fresnel zone case) to
the maximum cutoff distances (normally the end of the input data range)
will have an edge at the lag where the actual distance from the reference
point equals the aperture cutoff distance.   To handle this problem
the lag where that occurs is computed and all samples before that lag
are zeroed.   A linear taper is then applied for the
next *aperture_taper_length* seconds.   You can think of this as a
linear top mute like that defined by *Data_Top_Mute*
but applied from the zero cutoff lag point with
a taper length of *aperture_taper_length* seconds.

.. _pwstack_dataset:

pwstack_dataset function
=============================

Function usage
----------------

Most users will most likely want to run *pwstack* with the top-level
processing function called *pwstack_dataset* defined in the
python module `pwmigpy.pwmig.dataset`.  This section is an
extension of the docstring for that function that explains features
difficult to document without the background of this section.

The function declaration for *pwstack_dataset* is this:

.. code-block:: python

    def pwstack_dataset(
      mspass_client,
      dbname,
      pfname="pwstack.pf",
      wf_query=None,
      data_tag="pseudosource_stacks",
      pseudosource_stacker_algorithm="weighted_average",
      source_collection="telecluster",
      parallel=True,
      initialize_workers=True,
      storage_mode="file",
      output_data_tag="pwstack_data",
      outdir="pwstack_output",
      verbose=False,
      restart=False,
    ):

Arguments Requiring Attention
-------------------------------

This shows the code has two required arguments with the symbols in the
declaration *mspass_client* and *dbname*.  They are used as follows:

-  The *pwmig* python package was built as an extension of the
   MsPASS (Massively Parallel Analysis System for Seismology) framework.
   MsPASS defines a single client that is used to hold handles to
   required MsPASS services.   In *pwstack_dataset* the *mspass_client* is
   assumed to be an instance of `mspasspy.client.Client`.   It is used
   by *pwstack_dataset* to manage connections to the MongoDB database
   server and the dask cluster.   When the function runs it first
   sets up those connections.   If any of that setup fail it will abort
   immediately.
-  *dbname* is a secondary parameter used by the startup process using
   the *mspass_client*.   As implied by the symbol's name this should be
   the "database name" - a name chosen by you that is descriptive of your
   project.   It defines the name of the database containing the deconvolved
   data that are to be used as inputs to *pwstack*.

The behavior of the initialization section of this function
using *mspass_client* and *dbname* is controlled by two important
optional parameter:

1.  *initialize_workers* is a boolean argument that when set True (default)
    causes the function to create a database client on each running worker
    that can be reused by every task submited to each worker.   Without
    that feature a new database client would need to be instantiated for
    each "pseudostation point" handled by *pwstack*.   That would slow the
    processing significantly.   Although it might work (I haven't tried it)
    I would not recommend ever setting *initialize_workers* to False if
    running *pwstack* via the *pwstack_dataset* function.
2.  *restart* is another boolean, but this one you will likely want to
    change sometimes.   Because *pwstack* can run for a very long time
    it is very easy, particularly on an HPC cluster, to have the processing
    aborted prematurely.   Setting *restart* True (the default is False)
    is a way to recover from a previous run that aborted.   The default
    assumes you want to process the entire data set.

.. note::

   When using the restart feature (setting it True) you should always
   first run the convenience function `pwmigpy.pwstack.checkpoint_report`
   (See it's docstring for usage) interactively.   Read the output carefully
   as it will suggest known potential problems.   The most important one
   you may have to deal with is that if a job was aborted one group of
   data (normally defined by a `telecluster_id` value) may have only been
   partially processed.  The *restart* feature cannot recover such data.
   I recommend manually editing the database if this happens to remove
   output of partially processed groups.


Arguments often requiring changes
----------------------------------

Parameters you might need to change at times in the order of likelihood
from my experience:

1.  The *pseudosource_stacker* tool normally computes multiple estimates of
    "pseudosource stacks".   Each algorithm inserts a unique Metadata tag
    on it's output that defines itself.   The *pseudosource_stacker_algorithm*
    argument should be used to select which of those output should be used
    for input to *pwstack*.   The default is "weighted_average", which means
    is uses a stack using some version of a set of weighting options for
    the "pseudosource stacks".   Current alternatives are "average", "median",
    "robust_dbxcor".   *pwstack_dataset* does not check the validity of
    the value you send it.  If you use a value for *pseudosource_stacker_algorithm*
    that does not match any of those keywords the algorithm will pass
    through the dataset but get only Null results for all database queries and
    will fail for all "pseudosources" producing a potentially confusing output.
    I may eventually add a check for allowed values of this argument, but
    *pseudosource_stacker* may evolve so currently there is no sanity check
    on the *pseudosource_stacker_algorithm* value.
2.  Experienced uses will often produce multiple deconvolution algorithm outputs
    to use as inputs to *pwstack*.   One way MsPASS supports such multiple
    estimates is through a "data_tag" value, but there could be others.
    If your processing workflow creates multiple deconvolution estimates
    you may need to set the parameter *wf_query* to a python dictionary
    defining a MongoDB query to select only the data you want.
    Most instances can handle this by using an different value for
    the "data_tag" value on the output of *pseudsource_stacker*.  In that
    case set the *data_tag* argument to that alternative value.   i.e.
    use the *data_tag* argument if you want to select only data from a
    secondary run of *pseudosource_stacker* that used a different
    output data_tag value from the default "pseudosource_stacks".
3.  *minimum_input_data* should be used to remove data from pseudosource
    outputs with low coverage.  Earthquakes globally are highly clustered
    and there are usually some groups of events created with telecluster that
    that have very limited data. Including such data can be counterproductive
    as the result passed into pwmig can produce streaking artifacts from
    migration from a limited part of the study area.  You should create a
    report of the number of outputs from *pseudsource_stacker* for each
    `telecluster_id` value.  If you have some ids with notably lower values
    set the value of this parameter appropriately.  Note it is zero by
    default which means the program will attempt to handle all data.
    There is no other rational default as a working value is highly
    data dependent.
4.  *pfname* may change if you choose to organize your data files differently
    from the default setup.   That is, for convenience I set up the default for
    pf files for all pwmig components to be the run directory.   To reduce
    clutter it is often better to put all pf files in a separate directory.
    Change this parameter to a relative or absolute path if you choose to do
    that.  e.g. if you put pf file in a subdirectory to the run directory called
    "pf", you use `pfname=./pf/pwstack.pf`.
5.  When working with a new dataset you often might find it useful to
    set `verbose=True`.   The output in this mode is currently not
    that voluminous so turning in on initially is a good idea.

Arguments you may want to change
----------------------------------

Two parameters related to the output data organization may require changes:
*output_data_tag*, and *outdir*.
 The most common reason for wanting to do so is if you rerun the
*pwstack_dataset* function with different parameters and want to save
both outputs in the same database.  In that case, you would likely
want to change the value of *output_data_tag* from the default "pwstack_data"
so you can use a database query to run tell *pwmig* which of alternative
outputs it should use.   If you run this function multiple times I would also
recommend you change the value of *outdir* to cleanly separate the output files.
*pwstack* writes scratch files to *outdir* with a set of internally generated
file names.  If you rerun *pwstack* without changing *outdir* it will work
but the second run will have data appended to some or all files found
in that directory.


Arguments you should not change
--------------------------------

I do not recommend changing any of the following arguments unless you have
good reasons for doing so:  *source_collection, parallel, initialize_workers*,
and *storage_mode*.

pwstack function
==================

The *pwstack* function is best thought of as a lower-level alternative
to *pwstack_dataset*.   The main reason you might consider running this
algorithm instead of *pwstack_dataset*
is if you need to run the code in a serial setting because of
hardware limits or for testing.  The only other reason would be to
change some of the optional parameters used by *pwstack* that are
fixed when running *pwstack_dataset*.

This is the function definition:

.. code:: python

   def pwstack(
    db,
    pf,
    source_query=None,
    wf_query=None,
    minimum_input_data=None,
    source_collection="telecluster",
    slowness_grid_tag="RectangularSlownessGrid",
    data_mute_tag="Data_Top_Mute",
    stack_mute_tag="Stack_Top_Mute",
    storage_mode="gridfs",
    outdir=None,
    output_data_tag="pseudostation_stacks",
    run_serial=True,
    dask_client=None,
    restart=False,
    verbose=False,
  ):

If you compare this to *pwstack_dataset* you will see these fundamental
differences:

1.  *db* (arg0) here is required to be an instance of a MsPASS `Database` that
    references the desired data.  In *pwstack_dataset* that is acquired
    from an instance of `mspasspy.client.Client`.   *pwstack_dataset*, in fact,
    passes the instance of *db* it fetches from the MsPASS client to run
    *pwstack*.
2.  *pf* (arg1) here is required to be an instance of a
    `mspasspy.ccore.utility.AntelopePf` object.   *pwstack_dataset* creates
    an instance of an `AntelopePf` by reading a "pf-file" using a file
    name you sent it via it's *pfname* argument.  That instance is
    passed to *pwstack* inside *pwstack_dataset*.

The following arguments keywords are the same as those in *pwstack_dataset*.
Some have different defaults but how they are used is described
above:  *wf_query*, *minimum_input_data*, *storage_mode*, *outdir*, *output_data_tag*,
*restart*, and *verbose*.

Two arguments to *pwstack* are closely linked but handled in a completely
different with in *pwstack_dataset*.   Note that the default for the
boolean *run_serial* here is True.   That default emphasizes that the main
use of this function is for serial processing.   If *run_serial* is set False,
which is what *pwstack_dataset* always does, the *dask_client* argument is
required.   Be warned that if you run *pwstack* directly, as opposed to
indirectly with *pwstack_dataset*, your run script will need to initialize
the workers with something similar to these lines form *pwstack_dataset*:

.. code:: python

   dbplugin = MongoDBWorker(mspass_client)
   dask_client.register_plugin(dbplugin)

Noting this the same instance of `dask_client` should be that sent
to *pwstack*.

The following are options to *pwstack* that are hard wired into
*pwstack_dataset*.  I do not recomend you ever change them:
*slowness_grid_tag*, *data_mute_tag*, *stack_mute_tag*,
*source_query*, and *source_collection*.


TODO Debris:  instance, save_history need to be removed from pwstack.py

TODO:  add minimum_input_data option to pwstack_dataset.
