.. _gridstacker:

*gridstacker* Concepts
**********************
Overview
=========
The *pwmig* method is a "prestack migration" method, meaning data from
individual sources are transformed from time-domain,
three-component, signals to an image of the subsurface in physical
space dimensions.   The python function *gridstacker* can be used to
average all the results from all events to produce a unified image
of the subsurface.  It is important to realize *gridstacker* only works
with data preprocessed with *pseudosource_stacker* driven by *telecluster*
because it requires access to a "telecluster" MongoDB collection.

As noted multiple times in this manual, *pwmig* has to
deal with the issue of irregular signal-to-noise ratio.
The mathematical approach used by *gridstacker* is a set of options
to computed weighted averages of a set of 3d image volumes.
Most of the configuration
parameters for the algorithm control how those weights are computed.
I would also note that the current implementation is a research problem.
All the methods other than a simple average should be viewed
skeptically as experimental prototypes.   I consider the issue of
how to compute an optimal solution from a set of image volumes
produced by *pwmig* to be an unresolved
research problem.  I encourage all users of the package to consider
creative alternatives to what is currently available in
*gridstacker*.   In addition, you can expect updates from me for
other approaches using robust estimators.

Common features
==================
Solid Angle Normalization
---------------------------
*pwmig* uses the so-called Generalized Radon Transform Inversion (GRTI)
to obtain an estimate of what we
(`Poppeliers and Pavlis (2003a)< https://doi.org/10.1029/2001JB000216>`__ and
TODO:   having trouble finding monograph paper `Pavlis, 20`)
called "scattering potential" at each grid point.
The GRTI formula, which can be found in the paper noted above,
is a integral equation.   The integral is performed over a range of
scattering angles that the publications above denote
with the symbol :math:`\Omega`.   As I noted in the AGU monograph
paper each grid cell from each event is then a weighted sum of
the data - the GRTI is a linear operator.   A practical problem, however,
is that the solid angles coverage varies inside any image volume.
That variation can be thought of as a variation in illumination
direction.  Each event is illuminated from one direction in a limited
range of views.   Stacking data from many directions can be thought of
as a way to illuminate the subsurface from all directions.

The biggest illumination variation for individual events is
that coverage always drops to zero as you move
outside the area covered by seismic stations.  You can also get coverage
holes inside irregular arrays.   More subtle is the fact that source
regions with few events often have incomplete coverage compared to the
areas that have more events that can be merged with *pseudosource_stacker*.
Such events often have entire regions blank where other regions with more
sources have coverage.

I would assert the nonuniform coverage problem is a fundamental, unresolved
issue for scattered wave imaging with any tool like *pwmig*.  At present
I address this problem only through normalization by the range of solid
angles the data support for each grid cell.  Specifically, the *pwmig*
algorithm accumulates the solid angle range as a simple sum

.. math::

   \Omega_{total} = \sum_{i=0}{N_{hits}} \Delta \Omega_i

where :math:`N_hits` is the number of plane wave components that map into
that that grid cell and :math:`\Delta \Omega_i` is the computed solid
angle for that cell that is applied to the GRTI integration (sum) for
that cell.

The point of this is that *gridstacker* always normalizes the data in
each cell by :math:`\frac{1}{\Omega_{total}}` for that cell.   That step
is essential to preserve relative amplitudes in the normal situation where
the GRTI integral has irregular coverage within he image volume.
In addition, practical experience using the earlier version of this algorithm
taught us that it was essential to exclude cells where the size of
:math:`\Omega_{total}` was too small.   There are two reasons for that.
First, because we normalize by the reciprocal of :math:`\Omega_{total}`
small values give bad data too much weight.   Second, cells with small
:math:`\Omega_{total}` by definition are averaging smaller fractions of the
data so are prone to higher uncertainty from reduced averaging of noise.
The *solid_angle_cutoff* parameter described below is used to provide
some control over that problem.

I will close this discussion by noting a potentially useful enhancement of *pwmig*
that need to studied.  That is, *gridstacker* is currently blind to
the actual range of the :math:`\Omega` angles for different data.
It simply stacks results from migrating different "pseudoevents"
without considering how the ranges of :math:`\Omega` from each pseudoevent.
That means, some scattering angles are covered by multiple events and that
coverage is irregular.  The impact this has on the results is not clear.
This same issue occurs in all migration including the fanciest ones used
in seismic reflection processing.   The huge difference in seismic reflection
data is that the data support is designed to be nearly constant so that
although illumination is incomplete, it is uniform throughout the primary
target of a well designed survey.   I have not addressed this problem
because it would require a redesign of the entire implementation to
do the bookkeeping to define the actual 3d range of :math:`\Omega`.
Furthermore, even if I computed that range it is not clear how you would
use it or if it would help.   The right solution is probably to unify what
is now done in *migrate_dataset* with *gridstacker* to produce a GRTI
inversion that merges all the data in a single operation.   That is,
however, a very challenging research problem I leave for others to solve.

Signal-to-Noise Weighting
----------------------------
The current options in *gridstacker* are pair of weighted stack
options that are enabled by including the two keyword "azimuth_weighting"
and "bin_weighting" in the `methods` list.
Both build on results from
`Poppeliers and Pavlis (2003b)<https://doi.org/10.1029/2001JB001583>`__.
In that paper we found
*pwmig* has a tendency map noise into dipping artifacts with the dip
direction controlled by the incident wave amplitude.   We found
results could be improved by azimuthal weighting to balance noise
from multiple directions.   The "azimuth_weighting" option implements
a 3D version of what was implemented in
`Poppeliers and Pavlis (2003b)<https://doi.org/10.1029/2001JB001583>`__
with some options added to produce a more flexible weighting scheme using
a generic power law function (see below).

The optional method enabled with the tag "bin_weighting" is experimental.
It uses a finer scale weighting where each bin defined with
*telecluster*, which translates to a different "pseudosource",
is given a different weight for the stack.   That contrasts with
"azimuth_weighting" where all the results from pseudsources at a
particular azimuth of the telecluster radial grid receive a common weight.

Both weighted methods use a common, generic power law weighting funtion
described in the next subsection.

Power Law Weighting Function
--------------------------------
The fundamental concept behind the weighting methods used
in this version of *gridstacker* is the universal principle that
averaging suppresses noise.  Averaging more data always improves
signal-to-noise ratio as long as any common signal is present.
A standard theorem in statistics is that averaging Gaussian noise
reduces the variance by :math:`\sqrt{N}` where :math:`N` is the
number of data being averaged.   The error model for *pwmig* is
known to not be Gaussian as seismic noise is always colored
(see examples in
`Poppeliers and Pavlis (2003b)<https://doi.org/10.1029/2001JB001583>`__
paper).  Although I have no idea
what the actual loss rate is for migrated data, in *gridstacker*
one can vary the exponent of the assumed dependence on :math:`N` from
1/2 to some other power.   The default is the 1/2 for all methods
and I will state that I have no idea if other values for the exponent
will improve the results from any given data set.  Users are welcome
to experiment with this feature.

The way the power law is implemented can be visualized in the
graph shown in Figure :ref:`_power_law_weight_figure`.

.. figure:: figures/gridstack_weight.png

   :alt:  Power Law Weighting Figure
   :align: center
   :name: _power_law_weight_figure

   **Figure 1**.  Form of power law weight curve for variable exponent
   and fixed clip level.   The exponents corresponding to each curve
   are shown in the legend.  All use a fixed floor of 0.01.   Note the
   plot is a log-log plot so different exponents yield lines with
   different slopes.   The x-axis, which is labeled "Relative count",
   is defined in the text.

All the methods use a nondimensional form to make the weighting formula
scale-invariant.  For example, for the "bin_weighting" method
*gridstacker* first queries the *telecluster* collection to determine
the number :math:`N_j` of events stacked by *pseudosource_stacker* in
each source cell (the index :math:`j`).   It then computes the
"Relative count" (the x-axis if Figure :ref:`_power_law_weight_figure`)
as

.. math::

   N_{nd} = \frac{N_j}{max ( N_j)}

As the figure shows that cause the cell with the maximum number of
events to have a weight of 1 while all other cells have a smaller weight.
The weight for any cell, however, is not allowed to fall below the
floor value (0.01 in the Figure :ref:`_power_law_weight_figure`,
which is also the default).

The azimuth weighting method uses the same formula, but the counts
are assembled from a common azimuth instead of all bins.   i.e. :math:`N_j` is the
count for azimuth :math:`j` with :math:`j=\{0, 1, ... , N_{azimuth_bins} \}`
and :math:`N_{azimuth_bins}`  being the number of azimuths defined
for the radial grid used by *telecluster*.

A key point is both "bin_weighting" and "azimuth_weighting" methods
use weights that increase with the number of events averaged to produce
that image or, in the case of "azimuth_weighting", that group of
grids.   In all cases the final stack is normalized by the sum of
the weights.  The "average" method weights all data equally and normalizes
by the total number of grids summed.

*gridstacker* function
=========================

The *gridstacker* function has this signature:

.. code-block:: python

   def gridstacker(
      doclist_or_cursor,
      db,
      control=None,
      methods=["average", "azimuth_weighting", "bin_weighting"],
      output_base_name="stack",
      pfname="gridstacker.pf",
      verbose=False,
   ):

The *pwmig_dataset* function saves the data from each migrated event
to files managed by a MongoDB collection with the fixed name
"GCLfieldata".  The same collection is used to store other data used
by *pwmig*.  In particular, that same collection is used to manage the
image grid you created and 3D velocity model data if you used that option.
As a result, the workflow used to drive *gridstacker* always needs to
contain a query that defines a match to all the *pwmig* outputs.
The package was designed to do this using the keyword "name" defined for
each document in the "GCLfielddata" collection.   If you used *pwmig_dataset*
those names are generated using this line of code found in *pwmig_dataset*:

.. code-block:: python

   fieldname = base_fieldname + "_" + str(sid)

where `base_fieldname` is an argument of the *pwmig_dataset* function.
`sid` is the `ObjectId` of a the parent document from the *telecluster*
collection containing source metadata for the pseudosource used to
assemble the data input to each run of the *migrate_event* function.
e.g. for the default value of `base_fieldname="pwmigdata"` a typical
"name" value would be something "pwmigdata_69a03a728c310f3f6d5dd76c",
where the random junk after the "_"  is the unique string generated from
the parent "telecluster_id" as a cross-reference throughout this package.
With that structure, the simplest way to assemble the input to
*gridstacker* via arg0 is a variant of the following:

.. code-block:: python

   base_fieldname="pwmigdata"  # change if you change the default in pwmig_dataset
   exstr = f"^{base_fieldname}_"
   query = {"name" : {"$regex" : exstr}}
   cursor = db.GCLfieldata.find(query)
   # best to turn this into a list as it is always small
   # function can accept the cursor, but this is safer
   doclist=list(cursor)
   cursor.close()   # we have learned this is good practice
   gridstacker(doclist,db)

Noting that is the uses the default for all the `**kwargs` values.
The default uses the default name for the parameter file
(the `pfname` argument) to construct the control structure
(`control` option) and runs all supported stacking methods.
See the docstring for additional guidance on optional parameters.

.. _gridstacker.pf:

Configuration
==================

Like most of the components of this package, *gridstacker* uses an
`AntelopePf` object for configuration.   The default content of that pf
is the following:

.. code-block:: python

   solid_angle_cutoff 0.1
   clip_level 5.0
   enable_cell_weighting true   # TODO:  not used except loaded to control -fix
   save_weight_data false
   output_directory fielddata
   azimuthal_weighting_exponent 0.5
   azimuthal_weighting_floor 0.1
   binned_weighting_exponent 0.5
   binned_weighting_floor 0.1

The  *solid_angle_cutoff*  parameter is the value used to exclude
data in cells where the range of view angles is too small.  As noted
above, that is commonly a problem at the edges of station coverage.
Data in cells where :math:`\int d \Omega` is less than this value
are excluded from all averages.   The units is square radians, which
I admit is a little weird.   The default of 0.1 is equivalent to a
spherical patch approximately 9 degrees on a side.  I've found that a
useful default, but encourage users to experiment with different choices.

*clip_level* is used to fix an unresolved numerical problem(s) inside the
*pwmig* algorithm.   There is an unresolved numerical instability that
causes cells to occasionally contain very large values.   These seem
to most commonly happen near edges and/or in the vicinity of known
singularities near turning rays.   The *clip_level* solves the problem by
setting large values to a computed clip level.   It is critical to note that
*clip_level* is a scaling factor NOT and value that defines the clip level.
*gridstacker* computes the median absolute difference (MAD) of all
valid (the implementation uses numpy masked arrays to define invalid)
data samples.  The amplitude level of the clip is computed as
the MAD times the value set as *clip_level*.   e.g. if the computed MAD
is 0.0001 and you use the default *clip_level=5* the data clip level is
0.0005.  That level is computed for each input image to
*gridstacker* and can produce minor artifacts when that data is
averaged in the output stacks. 

Set the *save_weight_data* boolean to "true" if you want to save a 3d scalar
field of the sum of the weights used to normalize the stack at each cell.
That can be helpful if your data has major coverage holes inside the
receiver array.   Such holes can create artifacts that can be appraised by
viewing the sum of weights field on the same slice as that through the
stacked image.   The default (false) throws the weight data away.

Change *outdir* if you want the stacked output to appear in a directory
with a different name than the default of "fielddata".   Note if the
path defined by *outdir* does not exist, *gridstacker* will attempt to
create it.

Finally, the parameter *binned_weighting_floor*, *binned_weighting_exponent*,
*azimuthal_weighting_floor*, and *azimuthal_weighting_exponent* define the
exponent and floor values
for the signal-to-noise ratio weighting methods described above.   As the
names imply they are relevant only to the "azimuth_weighting" and
"bin_weighting" solutions.   I assume the name association is obvious.
