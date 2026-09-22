.. _nutshell:

Run Sequence
**************

Step 1:  Assemble Data Set
==============================
Generate Deconvolved Data Set
--------------------------------
The primary input to this package is estimates of the three-component
impulse response of the earth to incident teleseismic P waves.
One version of what that means is so called "Receiver Functions" that
are computed on a station-by-station basis.
A number of algorithms for computing receiver functions exist in
MsPASS through two top-level processing functions:

1.  *RFdecon* found in the module `mspasspy.algorithms.RFdeconProcessor`.
    *RFdecon* is a common interface to run a set of common
    receiver function deconvolution methods.
2.  A new, experimental algorithm called *CNRRFDecon* is available in
    the MsPASS module `mspasspy.algorithms.CNRDecon`.   *CNRFDecon* is
    a frequency domain method that is a variant of the multitaper methods.
    The primary thing it adds is shaping the data to a bandwidth
    consistent with where the input has a detectable signal.  See the
    docstring for the function and mspass documentation for more details.

MsPASS also has a multichannel deconvolution function called
*CNRArrayDecon*.   As the "CNR" prefix implies the algorithm is a
multichannel version of the algorithm used by *CNRRFDecon*.

Because this package is built as an extension of MsPASS the cleanest
way to create input data for the package is to use MsPASS to do the
low-level processing.   I recommend you use a set of instructional
notebooks for a course I have taught for several years for Earthscope.
In particular,
`this notebook from the 2025 course<https://github.com/mspass-team/mspass_tutorial/blob/master/Earthscope2025/tacc_files/RFprocessing.ipynb>`_
is a good starting point.  It is a complete recipe to go from raw data
to a set of `Seismogram` objects that can be used as input to this package.

If you are importing data from another package or previously computed
estimates in some stock format like SAC, you will need to convert that
data be visible as a MsPASS dataset.   There are some important constraints
you will need to satisfy if you import the data from externally.

1.  The data must be packaged up into MsPASS `Seismogram` object.
    See the MsPASS documentation including the
    `tutorial repository<https://github.com/mspass-team/mspass_tutorial>`_
    for help in doing that.   Also your favorite AI can probably help
    you sort out what is needed to do that.
2.  The `Seismogram` objects are assumed to be oriented in what MsPASS
    calls "cardinal".   You can guaranteed that in MsPASS by making sure
    all inputs have been passed through the MsPASS processing function
    called `rotate_to_standard`.
    (TODO:  should alter the _migrate_component_parallel function to
    force call rotate_to_standard always.  Change this when that is done.)
3.  You must have a database built with the following collections defined
    and internally consistent:   (1) wf_Seismogram, (2) source, and (3) site.
    By "internally consistent" I mean the source and site collections can
    be used to "normalize" data loaded from wf_Seismogram to set source
    and receiver metadata for each datum.

I stress that for anything but a tiny dataset the process of creating
the `Seismogram` inputs is likely to be a major challenge.
That preprocessing has much in common with seismic reflection processing
and shares some of the same algorithms used in reflection processing.
Prior to our development of MsPASS this problem was even more formidable.
With MsPASS the data management and workflow run process is much cleaner,
but there is a large learning curve to master all the components.  A key
point is to use this package you will need to first learn to use
MsPASS one way or the other.  i.e. either as the engine to do your
processing or as the tool to import data you created through other packages.  

Assemble control files
------------------------
The *pwmig* package evolved from a set of C++ program
that utilized a relational database in the software framework
called `Antelope<https://www.brtt.com>`__.
Antelope has a special format for handling parametric input they calle "pf files".
Pf files can be mapped one-to-one into yaml or xml.
I chose to retain the pf format for use in this package
for backward compatibility with workflows we had developed for the older C++ package.
For most steps in using pwmig you will need to edit master pf files
from either a previous run of the package
or from the master copies you can download from
`GitHub here<https://github.com/mspass-team/parallel_pwmig/tree/master/data/pf>`__.
Best practice is to make a copy of each of the following pf files
in the run directory for the data set you are processing:
*telecluster.pf, makegclgrid.pf, pseudosource_stacker.pf, pwstack.pf, pwmig.pf,*
and *gridstacker.pf*.   The default run line for all the applications that
use them look for a file by that name in the run directory.
A better organization is to but the copies of those files in a "pf" directory.
Then, however, you will need to change the defaults for all of those steps.

Step 2:  Run telecluster
==========================
Edit control parameter file
----------------------------
Edit the file *telecluster.pf* using guidance from :ref:`telecluster_configuration`.
Those instructions include an interactive jupyter notebook
you can use to design and run `telecluster`.
If you are an experienced pwmig use you can edit the pf file
and run the command line tool in a shell running on the pwmig container:

.. code-block:: python

   telecluster mydb

where "mydb" is the name of the database that you are using for this data set.
Use the `--h` flag and/or the docstring for the telecluster function for additional options.

Step 3: Run pseudosource_stacker
================================
Edit control parameter file
----------------------------
Edit the file *pseudosource_stacker.pf* with guidance from :ref:`pseudosource_stacker_configuration`__.

Run application
--------------------
*pseudosource_stacker* is implemented as a command line tool.
Since the pwmig package uses the MsPASS model it is normally
run inside a docker container with a jupyter server as the "frontend".
I have always run it from an interactive node on HPC or directly
from a desktop run.   In that context, command line tools are best run
by creating a "Terminal" tab within jupyter lab and entering commands to
run *pseudosource_stacker* in that terminal window.

A typical example run for *pseudosource_stacker* is the following:

.. code-block::python

   pseudosource_stacker mydbname --parallel --verbose

where `mydbname`, as the symbol name suggests, is the name you used to
define your database for your project.

If you want to run *pseudosource_stacker* within a jupyter notebook,
which would be the easiest way to run *pseudosource_stacker* in a
batch mode, use the standard jupyter "magic" incantations to execute a
command line tool within a code box.  e.g. the same line as above would
appear in a jupyter code box as:

.. code-block::python

   %%bash
   pseudosource_stacker mydbname --parallel --verbose

This approach has the added advantage of saving the output to the notebook.
Runs in a terminal window will not save output unless you use unix
shell output redirect to a file.

Step 4:  Create the Image Volume Grid
=======================================
Edit control parameter file
---------------------------------
Edit the file *makegclgrid.pf* with guidance from :ref:`imagegrid_configuration`__.

Run makegclgrid
------------------
Like telecluster this tool is easiest to run from the command line
using a jupyter terminal window.   The run line is similar to `telecluster`:

.. code-block:: python

   makegclgrid mydb

where, again, you should substitute your database name for "mydb".
You can also use the magic `%%bash` trick noted as well to run it from
a jupyter notebook.

TODO: need to copy a master imagegrid.pf file from test area.

Step 5:  Create Earth model interface
========================================
Overview
----------
*pwmig* is a "depth migration" method
which means the output on the vertical axis is a true position not a time axis.
Since the input you have at this stage is `Seismogram` objects
that have an implicit time axis
it should be clear that some Earth model with P and S wave velocities is
required to do that mapping correctly.  You can run the package with
a radially symmetric (1D model) or with a 3D model.   This section
describes how to set up either option.

1D Earth Model
-----------------
In *pwmig* a radially symmetric Earth model needs to be
defined to at least the depth of the bottom of the image volume you created in Step 4.
The *pwmig* package requires a 1D velocity model to be packaged up
into a python object with the verbose but descriptive name `VelocityModel_1d`.
You will need to create one or more such objects and save them
to the MongoDB database following more extensive directions found in the section
:ref:`VelocityModel_1D`.

3D Earth Model (optional)
--------------------------
Today most studies will have access to one or more 3D velocity models
that you may want to use with *pwmig*.   The details of how that model
is produced and merits for use with *pwmig* is up to the user to appraise.
What is critical, is that to use any 3D model in *pwmig* it must be cast
or recast with three constraints:

1.  The model must be perturbations relative to a 1d reference model.  If the model you are using has absolute velocities you must convert it to perturbatations relative the 1D reference model you loaded and defined above.   If the model is already perturbations (e.g. most teleeseismic, body wave, travel time inversions) you should be sure the 1d model you use is consistent with the reference model used in that inversion.

2. The model must specify both P and S velocities.   If only P or S is available you will need to use a fixed VPVS ratio to generate an approximation for the missing half of the pair.

3. *pwmig* demands a 3D model be cast into what is commonly called a "structured grid" in the 3d graphics world.  The implementation in *pwmig* is based on the "GCLgrid library" based on a now old paper by
`Fan and Pavlis<https://doi.org/10.1016/j.cageo.2005.07.001>`_.   Some useful tools for handling some cases are found in the module `pwmigpy.utility.earthmodel`.   e.g. most earth models downloadable from Earthscope's EMC page (URL) are easily converted for use with pwmig using that module in combination with `pwmigpy.paraview.netcdf_reader.py`.

The complexities of integrating the huge variations in data formats produced by different colleagues and the subleties of inversion make this a topic outside the scope of *pwmig* so I do not have a more verbose set of guidance on this topic.  I stress, however, that using the right velocity model can make huge difference in the final image volume you create with *pwmig*.   For most uses this will evolve to a research topic for each study area and use.

Step 6:  Run pwstack
======================
Edit control parameters
-----------------------
You first need to obtain and edit a copy of the control file with
the default name of *pwstack.pf*.   If you haven't used the package before
you can find a default file in the *data/pf* directory in
the parallel_pwmig repository or download it directly from GitHub at
`this url<https://github.com/mspass-team/parallel_pwmig/blob/master/data/pf/pwstack.pf>`__.
Edit that file based on guidance in
:ref:`this section of the page on pwstack<pwstack_pf>` and save the copy
in your run directory for this dataset.

pwstack_dataset
------------------
Although *pwstack* can be run serial that should be viewed
as only useful for testing.  Even laptops today have multiple cores
that will allow *pwstack* to run much faster than serial.
Serial processing is only supported with the lower level function
called *pwstack*.   I recommend strongly you use the simpler
top-level function *pwstack_dataset* for normal use with dask.

I recommend you first estimate the number of dask workers you can
use for the application within the memory constraints of machine(s) on which
it will be run.   Guidance on that configuration can be found
:ref:`this section of the manual <pwstack_memory>`.

You then need to decide on how to set the arguments to run the
function *pwstack_dataset*.  An example run line for default usage is:

.. code-block:: python

   pwstack_dataset(mspass_client,"my_database_name")

Where "my_database_name" should be changed appropriately for your project .
See :ref:`pwstack_dataset` for guidance on additional run options.

An important option if your job is aborted by the job scheduler is the `restart` boolean.   Read the guidance on validation of output state in :ref:`pwstack_usage` and you can restart an aborted job using:

.. code-block:: python

   pwstack_dataset(mspass_client,"my_database_name",restart=True)

When completed you should run this basic sanity check on the output.
This is a starting point and I recommend you evaluate the result
critically before proceeding to the next step:

.. code-block:: python

   data_tag = "pseudostation_stacks"  # change if you changed default for output_tag_tag
   tcids = db.telecluster.distinct("_id")
   print("telecluster_id N_Seismogram")
   for id in tcids:
       query = {"data_tag" : data_tag, "telecluster_id" : id}
       n=db.wf_Seismogram.count_documents(query)
       print(id,n)

Step 7:  Run pwmig
=====================
Estimate memory use
--------------------


Edit control parameters
--------------------------
Text noting need to edit pwmig.pf.  Link to a page describing details about the control parameters and how to set them.

pwmig_dataset
------------------
Run line showing example run with pwmig_dataset.   Link to more verbose page describing control and use of the sliding window feature and system tuning.  definitely needs to link to memory.rst page - maybe even an appropriate subsectionn of that page.

Step 8:  Run gridstacker
=========================
Edit control parameters
--------------------------
Text noting need to edit gridstacker.pf.  Link to a page describing details about the control parameters and how to set them.

run application
------------------
Run line showing example runline.   Link to more verbose page describing algorithm choices.

Step 9:  Visualize results
===============================
Create vts file
------------------
The output of *gridstacker* is a single 3d image volume.  *gridstacker*
normally saves it's result to the database as described above.   To
be of scientific interest that volume needs to be digested in way humans can
understand.   That is a component of the now generic area of computing
called "scientific visualization".   Many books have been written on
visualization concepts that can be used for understanding the content
of things like the output of *gridstacker*.  I have been a user of
a now standard visualization package called *paraview* since
some of the earliest versions in the early 2000s.   This package has
python functions that can be used to export data to *paraview*.  If you
are using any other package, the odds are good there is an import option
for handling the only currently supported output - something KitWare calls a
"vts-file".   If you need details on the format, consult the *paraview*
documentation
`found here<https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf>`__.

Creating a vts is a three-step process:
1.  Load the final image to be visualized using a database-driven reader.
2.  Convert the pwmig image volume structure to a standard VTK data structure.
3.  Use a VTK writer to save that data to a file.

This example from the test notebook should be used as a pattern:

.. code-block:: python

   from pwmigpy.db.database import GCLdbread_by_name
   from pwmigpy.paraview.vtk_converters import GCLfield2vtksg,vtkFieldWriter
   # step 1 - load the pwmig format grid from the database
   gridname = "pwmigtest"    # name used in test - change for your project
   pwmiggrid = GCLdbread_by_name(db,gridname)
   # step 2 - convert to a vtk field
   # These help identify the content of each vector component
   # They are useful in paraview to know which component to view
   compnames=["R","T","L","omega","sumwt"]
   # important - gridstacker absorbs components 3 and 4 and saves only
   # the migrated data as a 3-vector at each node point.  For gridstacker
   # output use:
   # compnames=["R","T","L"]
   vtkdata = GCLfield2vtksg(pwmiggrid,gridname=gridname,fieldnames=compnames)
   # these are large so best to clear this immediately
   del pwmiggrid
   # step 3 - save to a file you can load into paraview
   outfile = "pwmigtest.vts"   # change to match your project
   vtkFieldWriter(vtkdata,outfile)

Noting that the test data example handles the output from migrate_one_event.
The *gridstacker* function, as noted above, saves only the stacked data as
a three-component vector field.

Visualize with paraview
------------------------
Paraview is a large package that is now very well documented.  A simple
web search will yield a long list of sources for the documentation and
a large number of tutorials.   You can find examples of the kinds of
graphical visualizations possible with paraview and pwmig from
figures in (cite Wang, Yang, and Bauer publications)
