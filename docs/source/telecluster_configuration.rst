.. _telecluster_configuration:

Telecluster Configuration
**************************
Setup
======

The *telecluster* application defines a grid of cells to bin global earthquakes
into groups it will treat as equivalent source positions.   The assumption is that the plane wave
response of the Earth under the array is the same for all events from each cell.   How good that
assumption is depends upon the geometry of those cells. The *telecluster* application defines the
geometry but appraising the validity of what you use is a research problem.  What I do supply
is a jupyter notebook you can use to design your source cell grid.  To use this section effectively
I suggest you open a jupyter session with the pwmig container and open the notebook
in the repository called *telecluster_setup.ipynb*.  Use this section as a reference
for what parameters need to be changed.

Edit pf file
=============

The behavior of *telecluster*, like all the tools in the pwmig package, is mainly driven by an Antelope parameter file.  The default file name is "telecluster.pf".   You could copy the master version of this file from the repository (use GitHub or clone a local copy) into the working directory for the data you are working with.  Peruse the following, edit the file as needed, and use the jupyter notebook to verify it produces what you need.

Required changes
-----------------
Two parameter should be changed for each data set.   *origin_latitude* and *origin_longitude*, as the verbose key names suggest, are the geographic coordinates of the point that is to be used as the origin of a `RadialGrid` object it uses.  The best way to understand what a `RadialGrid` means is to look at the maps displayed in the notebook you should have open.   As the maps there show the objective of telecluster is to group events in your data set into common source regions that will be stacked in the next program in the sequence called `pseudosource_stacker`.   The origin should be a point near the center of the array of stations  that will be used in your data set.   With the scale of teleseismic data you don't need to be too exact about your choice.

Although not essential it is good practice to set the parameter *gridname* to some meaningful tag for your data and this choice of the `RadialGrid` geometry.  It is used only as a tag for documents stored in the *telecluster* collection by `telecluster`.

Optional changes
-------------------
What telecluster expect is control by a boolean with the key `use_regular_grid`.   Subsections below describe usage for when that parameter is true or false

use_regular_grid True
^^^^^^^^^^^^^^^^^^^^^^
A `RadialGrid` can be thought of as grid points in polar coordinates where the radius is great circle path distance from the origin and azimuth is the back azimuth to the source region.  With a regular grid the azimuth grid geometry is set by the parameters *grid_minimum_azimuth*, *grid_maximum_azimuth*, and *number_of_grid_points_for_azimuth*.   The verbose names tell their function.   The comparable parameters for the distance (delta) axis are *grid_minimum_delta*, *grid_maximum_delta*, and *number_grid_points_for_delta*.  In both cases the units are assumed to be degrees.  The range is split into equal sized cells defined by the "number_of..." parameter.  Note that the grid uses lower-left registration so the nodes points that define each cell are always on the low side of each independent variable.   The range for azimuth should never be changed from the default 0 to 360 as it would be strange to exclude some source back azimuths.  For azimuth the grid is always forced to wrap correctly at the 360 degree azimuth.   If you are confused by that experiment edit the file and use the notebook to examine the changes.

use_regular_grid False
^^^^^^^^^^^^^^^^^^^^^^^^
In this case the parameters for the True cause are completely ignored and `telecluster` will instead expect to see two parameters with the keys *delta_grid_points* and *azimuth_grid_points*.   As the default pf file shows these should define a list of values to use for cell boundaries for distance (delta) and azimuth respectively (units again in degrees).  Note for azimuth the grid will be automatically extended from the last point to 360.  For distance the last point is the outer edge of the grid.
