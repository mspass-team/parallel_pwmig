.. _installation:

Installation
***************
*Prof. Gary L. Pavlis*
*************************

HPC Cluster
##############
Build container file
----------------------
My assumption is most users of this package will be running it on a
modern High Performance Computing (HPC) cluster.   The computational
demands and data storage demands make that the most rational choice
for anything but the smallest data set and image volume.

The only recommended way to install *parallel_pwmig* is with
the docker container package.   The standard tool at present
for running docker container applications on HPC system is a
piece of software called `apptainer<https://apptainer.org/>`__.
All HPC clusters I have used use a software environment manager
to enable packages like apptainer.  At both Indiana University and
TACC, where I have worked most recently, you would enable
apptainer by entering this shell command:

.. code-block:: bash

   module load apptainer

You should then create a directory to hold the file apptainer
will produce for you in a moment.   A common suggestion is a "containers"
directory in home directory.   For that example, you can create a
file apptainer can use to launch *parallel_pwmig* with the following:

.. code-block:: python

   cd ~/containers
   apptainer build pwmig.sif docker:ghcr.io/mspass-team/parallel_pwmig:dev

Install mspass_launcher
----------------------------------
MsPASS has a standard python tool to run that package on HPC systems
that is part of a package called
`mspass_launcher<https://github.com/mspass-team/mspass_launcher>`_.
I recommend you use that to run *parallel_pwmig*.  If you do not already
have the `mspass_launcher<https://github.com/mspass-team/mspass_launcher>`__
package installed on your system you
will want to do so.  Most readers of this page will need to have
set up MsPASS earlier to assemble their working data set.  If that is
you can skip to the next section on configuring mspass_launcher to run
with parallel_pwmig.

I strongly encourage use of virtual environments
with conda or pyenv.  Best practice is to create a special environment
to run mspass_launcher.   For conda use something like this to create
and environment and install the launcher:

.. code-block:: bash

   conda create --name pwmig
   conda activate pwmig
   conda install mspass_launcher

and with pyenv

.. code-block:: bash

   pyenv virtualenv pwmig
   pyenv activate pwmig
   pip install mspass_launcher

Note that all job scripts will then need to run the appropriate
"activate" command to allow your job script to find mspass_launcher.

Configure mspass_launcher
--------------------------
The process to configure mspass_launcher is described in the
MsPASS User Manual
`here<https://www.mspass.org/getting_started/HPCClusterLauncher_configuration.html#hpc-cluster-configuration>`__.
The only significant thing different is you will need to
change the *container* attribute in the file "HPCClusterLauncher.yaml"
to point at the *parallel_pwmig* container instead of the MsPASS container.
For example, if you created the pwmig.sif file with apptainer as
suggested above that line would be:

.. code-block:: python

   # Note I recommend you use an actual path of your home director
   # instead of the ~ shell shorthand. mspass_launcher may not handle ~ correctly
   container ~/containers/pwmig.sif

You may want to set up multiple configuration files to run different
components of *pwmig*.  The reason is several of the components you
will need to run are appropriate as interactive jobs or batch jobs run
on a single node.
The heavy computational work is centered on the two python functions
`pwmigpy.pwmig.workflow.pwstack_dataset` and
`pwmigpy.pwmig.workflow.pwmig_dataset`.   Furthermore, the two have
different memory footprints so you will face additional issues on
the HPC job scheduler to assure you have sufficient memory and cpu
resource consistent with the configuration you set up.   Variations in
system configuration and software setups make this a problem you will
need to solve for your local setup.

Desktop
########
There are a few special situations where running *parallel_pwmig* on
a desktop can be helpful.  A case in point is that if you are new to
this package the first thing I recommend you do is run the test
notebooks found in the "tutorials" directory of the repository found
`here<https://github.com/mspass-team/parallel_pwmig>`_.  The notebooks
are self documenting so you need only launch the container in an
interactive mode and you can run them.  The simplest way to do that is
run the `mspass_desktop` command line tool that is part of
the `mspass_launcher` package.   For it to work correctly, you
will need to edit a different yaml file called "DesktopLauncher.yaml".
The desktop tool is designed to run with a different container management
package called `docker`.   If you don't already have docker installed on
the desktop system you want to use you will need to do so.
Stock installation procedures can be found `here<https://www.docker.com/get-started/<`_.
With docker installed you will need a similar incantation to that you
used for apptainer:

.. code-block:: bash

   docker pull ghcr.io/mspass-team/parallel_pwmig:dev

You will then need to copy the master "DesktopLauncher.yaml" for editing.
I recommend you copy it to a "yaml" directory and edit that copy.
The only change from the `setup used for mspass<>`_ is you need to
change every occurrence of the "image" key to:

.. code-block:: yaml

   image:  ghcr.io/mspass-team/parallel_pwmig:dev

You can then launch and run *parallel_pwmig* with the same procedures
used for running MsPASS described `here<https://www.mspass.org/getting_started/mspass_desktop.html>`_.
Note the title in the top banner of the GUI will be MsPASS but by
changing the "image" attribute you will be running the *parallel_pwmig*
container instead of the MsPASS one.  
