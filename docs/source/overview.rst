PWMIG User Manual
*******************

*Prof. Gary L. Pavlis*
***********************

What is pwmig?
##########################

The name "PWMIG" is a typical computer program abbreviation of a longer
name tag:  "Plane-Wave MIGration".  "migration" is a jargon term
from seismic reflection processing.   Migration is a numerical algorithm
that maps recorded seismic data into an image the subsurface.   It has
a long history as a fundamental component of seismic imaging.   It is
absolutely true that without migration algorithms the world would have
run short of petroleum decades ago.   Migration in the world of seismic
reflection aims to image sources of backscattered (reflected) P-waves from
subsurface geologic structures.   The data *pwmig* aims to "migrate"
is not backscattered P-waves, but forward scattered P to S conversions.
More on these fundamental concepts can be gleaned from a pair of
overview papers I wrote in `year of Leading Edge article<>`__
and `year of AGU monograph paper<>`__.

Some properties of this package that make it unique are:

- The package handles data from nonuniformly sampled, 2D arrays.
  A large fraction of publications using converted wave imaging
  depend upon densely sampled linear arrays.
- The package always aims to produce a 3D image of the subsurface.
  i.e. it directly manipulates data from a 2D array to produce a 3D
  image with all data projected to their true position in space.
  That includes the nontrivial step of depth and direction dependent
  coordinate transformations of the input vector data.   To the
  best of my knowledge no other implementation has handled that
  problem correctly.
- The package can be efficiently run on modern computer clusters to
  utilize an arbitrarily large number of CPUs to make the challenging
  calculations more feasible.  For a perspective the each of largest images
  handled by the earlier version of this package (cite Wang papers)
  would have taken of the order of two months to compute
  if they had been run serially.
  That is, at best, a marginally feasible calculation without parallel
  processing.  Be aware, however, that it remains expensive to run
  even with this new version.   This package improved parallel performance
  by exploiting the `dask<>`__ scheduling package through the
  `MsPASS framework<https://www.mspass.org`__.  Parallel processing, however,
  does not reduce the operation count for the algorithm.  It simply
  reduces elapsed wall time by exploiting many cpus to do the work.
- *pwmig* is a "prestack migration method".   That means the standard
  workflow applies a function called *pseudosource_stacker* to first
  compute a set of simple averages to improve the estimates of the impulse
  response for each station.   The parallel with reflection imaging comes
  in the migration algorithm that is implemented as a function called
  *pwstack* followed immediately by *migrate_event*.   That pair of
  functions migrate the data from individual events to produce a 3d
  image of the subsurface.  The set of images produced from all
  "pseudosources" is averaged by the final core function called
  *gridstacke*.   TODO:   each of above should have :ref: links to
  sections in this manual.
- *pwmig* uses a "Inverse Generalized Radon Transform"
  (IGRT) for an analytic inversion of
  for the scattering potential in each grid cell of what I call an
  "image volume".   Note the way *pwmig* uses the IGRT differs from
  the method developed around the same time by
  `Bostock and students<cite bostock paper>`__.   As described in
  more detail in my `AGU monograph paper<cite that paper>`__ the
  two are closely related with some fundamental differences:
  *  Bostock's IGRT attempts to extract more from the data.  It aims to
     estimate velocity perturbations from a background to produce a
     velocity field.   *pwmig* only aims to image what is commonly
     called "scattering potential" in this context.  Scattering potential is roughly
     equivalent conceptually to conversion coefficients for
     P to S scattering.   It is well known from the reflection literature that
     velocity model inversion from seismic reflection data is far more
     sensitive to noise than the direct migration output.   There is
     every reason to believe that is equally true for converted waves
     for a long list of reasons.  In fact, it is likely worse because
     of the widely variable quality of data from different earthquakes.
  *  The IGRT used in Bostock's formulation is best thought of as a
     form of simultaneous inversion in 2D.  I suspect strongly
     that a basic reason no one, to my knowledge,
     has successfully adapted that algorithm to fully 3D problems is
     the numerical problem of fitting all that data into a computer's memory
     space.   The implicit divide and conquer approach of a prestack method
     uses computer memory more efficiently.   On the other hand, it bloats
     what needs to be handled enormously by using the plave decomposition
     (*pwstack*) as an intermediate result, but this package handles that
     provided sufficient scratch storage is available.
- This implementation should be viewed as an extension of the
  `MsPASS framework<https://www.mspass.org`__. As noted that enables more
  efficient parallel processing, but also has significant additional
  advantages including:
  *  Running *pwmig* can be treated as the end produced of a long,
     reproducible workflow to reduce a massive data set to a final
     subsurface image.  My hope is this will allow users to publish
     the jupyter notebooks that define their workflow that will
     allow others to reproduce the work.
  *  The integrated database functionality of MsPASS provided a way in
     this package to simplify a lot of operations that required some
     rather ugly manipulations in the C++ version.  An example is
     that this package uses MongoDB geospatial queries to assemble data for
     inputs to the Gaussian smoother used in *pwstack* while the
     older version required a weird, nonstandard relational database table
     implemented originally with the `Antelope software<https://www.brtt.com`__.
  *  Implementing the package as a python package makes it more
     appropriate as a research package.  Readers should realize there
     are lots of ways this package could be improved
     (see limitations immediately below).  Building the package
     on python makes it much more hackable to create experimental version
     to test new ideas.   Furthermore, few in our community can write effective
     C++ code today, while most young scientists are very proficient in python.
  *  As a fully open-source package *pwmig* has hope of living on past my
     finite lifetime.  I fully expect this implementation of*pwmig*
     to outlive me by many years.
- The package has a mechanism to output the 3D result to be displayed with
  one of the most common scientific visualization packages called
  `paraview<https://www.paraview.org`__

Some limitations of the package are:

- The propagators used for imaging are based on ray theory.   That is
  demonstrably wrong, but as in seismic reflection imaging is known to
  still produce surprisingly reliable results even from simulation
  data (see e.g. the simulation results in `Pavlis <doi of C&G paper>`__).
- Although this implementation scales with the number of cpus far far
  better than the original C++ version, it has some serious limitations
  in the current form.  More work is needed to parallelize the code at
  a finer level of grandularity to make more effective use of newer
  hardware with 60 or more CPUs per node.  This version has a memory
  bottleneck that usually limits the number of workers that can be assigned
  to each node to less than the number of cores per node.
- In common with all converted wave imaging methods, there is a fundamental
  problem that all current deconvolutions operators are flawed.
  I gave an oral presentation in `year <cite SSA abstract where I gave tha ttalk`__
  arguing that all deconvolution is based on an assumption the earth inside
  the entire imaging volume is transparent.   In addition, all methods used
  to separate P and S components are, at best, an approximation.   That means
  that all inputs are prone to producing artifacts from deconvolution errors.
- All scattered wave imaging methods need a stronger focus on error analysis.
  This version has very little that can address that fundamental problem
  effectively.  High on my development agenda is implementing a
  nonparametric (jackknife and/or bootstrap) to appraise noise levels in
  a final image.   That is highly feasible in this implementation of
  gridstacker, but is a nontrivial programming exercise.  A similar
  approach might be valuable with *pseudosource_stacker* to improve
  inputs to *pwstack*  The later is a research problem that could fail.
  The former is guaranteed to be helpful.   The research problem there is
  how to visualize the data to emphasize the most reliable features.



Development History
########################

The concepts that led to the development of this package are the result
of a collaboration with a group of student over a period of nearly 30 years.
Key milestones were:

- `Scott Neal<cite dissertation>`__ and I developed a core idea of
  using a spatial smoother for passive array data processing.
  Neal's work focused on the preprocessing
  step of deconvolution of teleseismic earthquake data to estimate the
  impulse response of the medium to an incident P wave, which is the
  main input data for this package.   An important insight of his work
  published in a `GJI paper in 2<>`__ is that for linear operations like
  deconvolution the Gaussian window function could be viewed as a spatial
  filter.   In *pwmig* that is a fundamentally important property as
  the function in this package called *pwstack* generates plane wave
  estimates that should be viewed as processed with an anti-alias filter.
- Around the same time `Chenliang Fan<cite dissertation>`__ and I
  developed the concepts behind the grid library available in this
  python package in the module `pwmigpy.ccore.gclgrid`.   That module,
  in fact, contains minor variations on the C++ code we published
  in a paper in `Computers and Geosciences<cite doi>`__, which serves
  as a primary reference for the concepts of that library.   In *pwmig*
  the `pwmigpy.ccore.gclgrid` library is used as the primary container to
  hold the 3D grid objects produced by the *pwmig* and summed by
  *gridstacker*.  The objects it implements are fundamental components of
  *migrate_events*  that allow the package to correctly handle geometry
  with large scale 3d grids where the spherical geometry is essential.
  Some portions of that algorithm would be problematic if we had
  chosen to use the flattening transformation to handle the radius dependence.
- The core concepts of *pwmig* were developed in collaboration with
  Christian Poppeliers.   The foundation for *pwmig* are in a pair of
  papers first submitted in 2001 but not published in JGR till 2003
  due to one of those unfortunately publication delays that happen to
  some papers:  `Poppeliers and Pavlis (2003a)<https://doi.org/10.1029/2001JB000216>`__
  and `Poppeliers and Pavlis (2003b)<https://doi.org/10.1029/2001JB001583>`__.
  Both should be considered essential reading by all users of this package.
  Note all the examples in those papers were 2D using a similar approach to
  `Bostock<>`__ but with a different approach for handling variations in
  incident wavefield direction relative to a profile line.
- Poppeliers noted in his `dissertation<>`__ that handling the 3D problem
  was not even computationally feasible with the prototype matlab code he
  had developed for his research.  At that point I had become proficient
  enough in C++ that I elected to implement our ideas into a 3D code.
  The thought was that a compiled code could make up for known performance
  issues with the matlab prototype to make a 3D algorithm feasible.  It took
  me several years to convert Poppeliers prototype into the working 3D
  code described in an article in
  `Computers and Geosciences<https://doe.org/10.1016/j.cageo.2010.11.015>`
  published in 2011.  There were multiple reasons why that work took that
  long that are, I think, a useful practical lesson to young scientist
  contemplating a major software development task like that was:
  1.  As a full time faculty member my time was fragmented.  Coding something
      as complex as the original pwmig package in that situation was
      challenging because I had to leave what was done on the shelf for
      long periods.  The context switch to restore the cache of knowledge
      needed to make progress took a long time to load.
  2.  I had to solve multiple problems that surfaced in 3D that
      Poppeliers did not need to address with the 2D approximation.
  3.  A lesson I learned was that handling the nasty problem where a program
      runs without errors but produces garbage output is a serious challenge
      to debug for any 3D problem.   The barrier is both the size of
      data volumes you need to sift through and visualizing intermediate
      results to figure out where the problem is.
- Once we had a functional 3D implementation the package was refined an
  improved in collaboration with two outstanding PhD students I had the
  pleasure to work with during that period:   `Ian (Yinzhi) Wang<cite dissertation>`__
  who used the package to image data from the Earthscope Transportable
  Array, and `Xiaotao Yang<cite dissertation>`__ who used the package with
  data from the OIINK experiment for a high resolution study of the
  central US.  Their work provide the most outstanding existing examples
  of the what *pwmig* can produce.   See the reference list (TODO: need to make
  that and create a cross reference)  Both helped find and solve typical bugs
  that surface in development of any complex piece of software.

Implementation History
########################
The current version is implemented
as a python package.   The python package was created by a nearly
complete rewrite of the original implementation
described in a
`the 2011 paper mentioned earlier <https://doi.org/10.1016/j.cageo.2010.11.015>`__.
As noted above, the original implementation was written in C++.
The very first versions had a dependency on the
`Antelope<https://www.brtt.com`__ package.   Although Antelope
is readily available to U.S. universities, installing it on the HPC
cluster I was developing the code was problematic.  Further,
that dependence was a barrier to colleagues outside the U.S.  Consquently, after the
2011 paper was published I modified the code to remove all dependencies
on Antelope.  The source code for the older package is still on GitHub
and can be found `here<https://github.com/pavlis/pwmig>`__.

For this new implementation I extracted
some of the core numerical functions and created python bindings to them
via `pybind11<https://pybind11.readthedocs.io/en/stable/basics.html>`__.
Much of the rest of the package was rewritten in python using the
`MsPASS framework<https://www.mspass.org/>`__.   This package, in fact,
should be viewed as an extension of MsPASS.  The "parallel" tag for
this implementation, `parallel_pwmig<https://github.com/mspass-team/parallel_pwmig>`__,
emphasizes that this version is designed to exploit parallel processing
to run on large HPC or Cloud cluster.  Following the MsPASS model
the package was bundled in a docker container that is constructed on top
of the container for MsPASS.   The *parallel_pwmig* package contains
some additional large packages not found in MsPASS.  The two largest are the
`python vtk libraries<https://docs.vtk.org/en/latest/api/python.html>`__,
which are the core libaries used
by `paraview<https://www.paraview.org>`__, and
`pygmt<https://www.pygmt.org/dev/install.html>`__
that is widely used in the seismology community to
create maps.
