.. _pseudosource_stacker_configuration:

Configuration
**************************
Concepts
=========
The purpose of *pseudosource_stacker* is to create "pseudosource" gathers
created by stacking (averaging) data from sources assumed to be spaced closely
enough together that the impulse responses for P to S scattered arrivals being
imaged will add constructively.   The recipe for which seismograms are stacked
comes from the output MongoDB collection created by the *telecluster* algorithm
and stored with the collection name *telecluster*.   The basic idea is that
sources are binned by *telecluster* and for each source bin common station
gathers are stacked to ideally produce one Seismogram per station per source bin.
In practice there are always empty source bins and many stations have no
data from some source bins.

In our earlier work we used a different version of this algorithm written
in C++.   Previously published results used mainly simple averages from
the equivalent of this algorithm.   This new version adds a number of new
features that should be viewed as experimental.   The main addition is a
set of optional weighting methods that aim to downweight poor quality data.
In this context, "poor quality" is handled by weights used event magnitude
and/or signal-to-noise ratio.   There are also to options that use a
version of a "robust stack";  a *median* and *robust_dbxcor* option.
More on what these do and how to configure them is the subject of
subsections below.

Parameter File Structure
==========================
It is best to think of the parameters in the file *pseudosource_stacker.pf*
in one of two categories:  (1) The Arr section with the tag *stacking_parameters*,
and (2) everything else.   A further complexity is that some of the "everything else"
parameters are global and always relevant but some are only applicable for
some algorithms.   I would note that is a typical reason for using a
format like pf or yaml to define the configuration for a complicated algorithm
like this one:  not all algorithms need to reference all entries in the
configuration.

Global parameters
-------------------

 *wf_Seismogram_base_query*.  The *pseudosource_stacker* application always
 expects to be reading data from a MongoDB collection called "wf_Seismogram".
 Because that collection is used in MsPASS to store the index for all
 three component data sets (`Seismogram` objects), it usually contains
 other data you will not want *pseudosource_stacker* to use.  The recommended
 way to handle that is to alway use a unique "data_tag" attribute for
 a particular collection of deconvolved data.   You can, for example,
 compute estimates from multiple deconvolution operators but you would then
 want to usually handle the data differently for the output from different
 deconvolution methods.  A "data_tag" is a simple way to do that, but
 not the only way.  Hence, this argument is defined as a "base query"
 but that query currently has limited capabilities.  Specifically it only
 works with a query that is an equality match.  For example, the default is

.. code-block:: python

     wf_Seismogram_base_query &Arr{
     data_tag CNRRFDecon_data_raw
   }

which the code translates to a dictionary for the query which
in json format is:

.. code-block:: python

     {"data_tag" : "CNRRFDecon_data_raw"}

*stack_add2keepers_list* should never be changed. It contains a list of
metadata keys used internally by the application.  It is part of the pf
only to make it easier to adapt the code to possible future extensions.

*stacking_parameters* section
------------------------------
The data inside the &Arr{} it encloses contain common parameters
for each of the possible stacking algorithms *pseudosource_stacker* can
run.  Each &Arr{} block inside that has an algorithm name as a keyword.
The current default is this:

.. code-block:: python

   stacking_parameters &Arr{
    average &Arr{
        enable true
    }
    weighted_average &Arr{
        enable true
        weight_key weight
        undefined_weight 1.0
    }
    median &Arr{
        enable true
        timespan_method ensemble_inner
        pad_fraction_cutoff 0.05
    }
    robust_dbxcor &Arr{
        enable false
        timespan_method ensemble_inner
        pad_fraction_cutoff 0.05
        residual_norm_floor 0.01
    }
   }

Notice the keywords "average", "weighted_average", "median", and
"robust_dbxcor" are the list of currently supported algorithm.  Each
Arr block associated with those keys has an "enable" boolean.
The default above tells *pseudosource_stacker* to run the
"average", "weighted_average", "median", and but not run
"robust_dbxcor".

All but the "average" algorithm have additional parameters that can change their
behavior.  Since some are common I'll describe the concept of each
below in the order of the defaults above:

*weight_key*   the default value of "weight" should never be changed.
As described below a weighted stack can mean a lot of different things.
This key is used internally for the composite weight computed from one or
more weighting methods (see below).

*undefined_weight*  With any weighted average you need to have
a default way to handle data for which there is no way to compute the weight.
This application never discards inputs but sets undefined results to this
value.   The default is 1.0.  If you want undefined values ignored change
this to a small value.

*timespan_method* and *pad_fraction_cutoff* are needed to handle a robust
estimator properly.  With a linear average (weighted is also linear)
irregular start and end times in a group are not a major issue provided
the result is trimmed to remove edge effects.  For *median* and *robust_dbxcor*
the recommended option is the default "ensemble_inner".   That means the
stack is trimmed to the latest start time to the earliest end time of
each ensemble.   The idea of *pad_fraction_cutoff* and *timespan_method* can
be found by reading the docstring for the mspass function
`robust_stack<docstingurl>`_.

weighting parameters
----------------------
Two groups of parameters with these keys apply only to the
*weighted_average* algorithm.   They are: (a) *snr_weighting* and
(b) *magnitude_weighting*.   Currently, the *snr_weighting*
is syonymous with *weighted_average* while *magnitude_weighting* is
optional.   That is why there is no *enable* boolean for
*snr_weighting* but there is for *magnitude_weighting*.
If *magnitude_weighting* is enabled the magnitude
weights are merged with the ones computed from snr using the
generic concepts described below.

Both *magnitude_weighting* and *snr_weighting* use a power law
weighting function.

TODO:  this isn't right.  should allow either alone or both.   I need
to also look up and remember how multiple weights are mixed to
produce a composite weight.   This section also may need a notebook
to illustrate the power law weight formula for magnitude and
snr.
