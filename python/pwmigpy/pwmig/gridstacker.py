#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 05:47:47 2026

@author: pavlis
"""
import numpy as np
import scipy
from pwmigpy.ccore.gclgrid import (
    GCLgrid3d,
    GCLvectorfield3d,
    GCLscalarfield3d,
    extract_data_array,
    load_numpy_data,
)
from pwmigpy.db.database import GCLdbread, GCLdbsave
from mspasspy.ccore.utility import AntelopePf

import psutil
import os


def report_memory_use():
    """
    Simple little function to print a report of process memory use
    at a particular point in the code.
    """
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024**2:.2f} MB")


class gridstacker_control:
    """
    Control data structure for gridstacker.

    This class in more-or-less a simple data structure holding control
    parameters that defines the mode(s) gridstacker should be run in.  It exists
    largely to simplify adding features as the application evolves.
    That is, the application is starting out bare-bones implementing
    what we found worked well with the older C++ version.   There are
    features in the older version and new ideas, however, that I hope to
    add to gridstacker.  By putting them all in this data structure it
    will allow expansion without changing the arg list to main processing
    function(s).

    Note also that because python class attributes are normally public
    a CLI tool can easily change values defined by a pf to then just
    always use the default pf from the distribution to initialzied.
    Then the tool can change attributes with arg overrides.  For that
    reason the construction is just an interface for the content of a pf
    file.
    """

    def __init__(self, pf_or_pfname="gridstacker.pf"):
        if isinstance(pf_or_pfname, str):
            pf = AntelopePf(pf_or_pfname)
        elif isinstance(pf_or_pfname, AntelopePf):
            pf = pf_or_pfname
        else:
            raise ValueError(
                "gridstacker_control constructor:  illegal type for ar0 - must be a pf file name or pf object"
            )

        self.solid_angle_cutoff = pf.get_double("solid_angle_cutoff")
        self.clip_level = pf.get_double("clip_level")
        self.enable_source_cell_weighting = pf.get_bool("enable_cell_weighting")
        self.save_weight_data = pf.get_bool("save_weight_data")
        self.dir = pf.get_string("output_directory")
        self.azwt_power = pf.get_double("azimuthal_weighting_exponent")
        self.azwt_floor = pf.get_double("azimuthal_weighting_floor")
        self.binwt_power = pf.get_double("binned_weighting_exponent")
        self.binwt_floor = pf.get_double("binned_weighting_floor")


def count_events_by_azimuth(db, iaz) -> dict:
    """
    Count the number of events in cells with specified azimuth index.
    Counts are returned in a dictionary keyed by the distance index
    value.   To get teh total events on a particular azimuth range
    sum the values of the returned dictionary.
    :param db:  database handle with telecluster collection to be scanned.
    :param iaz: azimuth index.  If there are not data on this azimuth
      the function silently returns and ampty dictionary.

    :return:  dictionary with distance index as key and counts in that
      cell as the value.
    """
    query = {"gridcell.azimuth_index": iaz}
    result = dict()
    cursor = db.telecluster.find(query)
    for doc in cursor:
        celldoc = doc["gridcell"]
        dindex = celldoc["distance_index"]
        evlist = doc["events"]
        result[dindex] = len(evlist)
    return result


def _azimuth_of_cells(db, iaz):
    """
    RadialGrid does not store a center point for each grid cell but only
    the azimuth range.  Rounding error makes the range values slightly
    variable but here assume that variation is tiny and only return the center
    azimuth computed from the first item found.

    This is a kind of trivial function for internal use in this module.
    """
    query = {"gridcell.azimuth_index": iaz}
    doc = db.telecluster.find_one(query)
    subdoc = doc["gridcell"]
    azmin = subdoc["azimuth_minimum"]
    azmax = subdoc["azimuth_maximum"]
    return (azmin + azmax) / 2


def report_counts(db):
    azindlist = db.telecluster.distinct("gridcell.azimuth_index")
    for iaz in azindlist:
        az = _azimuth_of_cells(db, iaz)
        print(f"Counts for azimuth={az} which is azimuth index={iaz}")
        print("distance_index number_events")
        counts = count_events_by_azimuth(db, iaz)
        totalevents = 0
        for k in counts:
            print(f"{k}  {counts[k]}")
            totalevents += counts[k]
        print("Total number of events on this azimuth=", totalevents)


def source_cell_weights(db, tcidlist, power=0.5, floor=0.05) -> np.ndarray:
    # this could be slow but telecluster is always small and this
    # function is only run once so should not ber a concern
    weights = np.zeros(len(tcidlist))
    i = 0
    for tcid in tcidlist:
        # telecluster_id is the id of this collection
        doc = db.telecluster.find_one({"_id": tcid})
        if doc is None:
            message = "source_cell_weights:   telecluster_id={} defined in pwmig output image number {} not found in telecluster table\n".format(
                tcid, i
            )
            message += (
                "Likely error in use of options for telecluster_id use in pwstack/pwmig"
            )
            raise RuntimeError(message)
        evlist = doc["events"]
        weights[i] = float(len(evlist))  # use weights as a work space for now
        i += 1

    maxcount = np.max(weights)
    scale = 1.0 / maxcount
    weights *= (
        scale  # this scaling makes power and floor independent of total data size
    )
    for i in range(len(weights)):
        w = weights[i] ** power
        if w < floor:
            w = floor
        weights[i] = w
    return weights


def azimuth_weights(db, tcidlist, power=0.5, floor=0.05) -> np.ndarray:
    azindlist = db.telecluster.distinct("gridcell.azimuth_index")
    # holds totals for each azimuth keyed by int azimuth index
    totals = dict()
    for iaz in azindlist:
        counts = count_events_by_azimuth(db, iaz)
        this_az_total = 0
        for k in counts:
            n = counts[k]
            this_az_total += n
        totals[iaz] = this_az_total
    # fix max count
    maxcount = 0  # counts are always positive so this works as initial value
    for k in totals.keys():
        maxcount = max(maxcount, totals[k])
    azwts = dict()
    for k in totals.keys():
        wt = totals[k] / maxcount
        wt = wt**power
        if wt < floor:
            wt = floor
        azwts[k] = wt
    # now return the vector of weights for each image that is to be stacked
    # note this algorithm has all data from a common azimuth with the same
    # weight
    N = len(tcidlist)
    weights = np.zeros(N)
    for i in range(N):
        tcid = tcidlist[i]
        # telecluster_id values are the id keys for teleclustet collection
        query = {"_id": tcid}
        doc = db.telecluster.find_one(query)
        if doc is None:
            message = "azimuth_weights:   telecluster_id={} defined in pwmig output image number {} not found in telecluster table\n".format(
                tcid, i
            )
            message += (
                "Likely error in use of options for telecluster_id use in pwstack/pwmig"
            )
            raise RuntimeError(message)
        gdcell = doc["gridcell"]  # this is a subdocument
        azind = gdcell["azimuth_index"]
        # with the logic of this code this key must be present in azwts
        # so we don't test if itis valid.
        weights[i] = azwts[azind]
    return weights


def normalize_by_solid_angle(imagevolume, cutoff) -> tuple:
    """
    The raw outputs of pwmig needs to be normalized by the range of
    scattering angles covered in each cell.   That is essential or the
    data from different events will not be balanced nor represent anything
    close to true relative (within the 3d volume that is) amplitudes.
    The solid angle sum is accumulated in each GCLvectorfield3d as component
    3 of the vector in each cell.   We normalize each cell by 1/solid angle.
    The "cutoff" argument, however, sets the smallest solid angle allowed.
    cells with limited azimuthal coverage (small solid angle coverage)
    will give misleading results and experience has shown discarding data
    from the stack when the coverage is too low improves the results.

    This function exploits a more obscure by very useful feature of numpy
    called masked arrays.   We apply cutoff by using the `masked_less`
    function in numpy.ma.   That allows this function to be vectorized
    and the normalizations step becomes a remarkably short, albeit
    obscure application of the : operator and subarrays.

    :param imagevolume:  4 dimensional number array extracted from an
       individual output from migrate_event.
    :param cutoff:  solid angle cutoff for zeroing cells (see above)

    :return:  tuple with 0 containing the normalized content of imagevolume
    with the vector size reduced from 5 to 3.   component 1 is a
    numpy masked array of the weights (1/sum(omega)) values for each
    cell.  The masked array has masked cells (less than cutoff value for
    omega) zeroed.  Corresponding cells have the vectors in that cell
    zeroed on the return in component 0.
    """
    image = imagevolume[:, :, :, 0:3].copy()
    omega = imagevolume[:, :, :, 3].copy()
    # TODO:  something in pwmig is causing NaNs in output - we need to fix that
    # this is a workaround until that is resolved
    # debugging shows that if one of the 3 components is NaN they all are so use this construct
    imgmask = np.isnan(image[:, :, :, 0])
    image = np.nan_to_num(image)
    # not component 4 of each data vector contains the sume of grt weights
    # they are applied to each 3c vector in the C++ PWMIGfielddata.accumulate
    # method   domega is also applied there but the sum of the grt weights is
    # not useful while sum of omegas is the focus of this function.
    [N1, N2, N3, N4] = imagevolume.shape
    omega_inv = np.ma.masked_less(omega, cutoff)
    # combine with above mask - works as mask is a logical array
    combined_mask = omega_inv.mask | imgmask
    omega_inv.mask = combined_mask  # if this works this should be shortened
    omega_inv = 1.0 / omega_inv  # safely computes 1/sum(omega) values
    # this returns a regular array with the masked values set to 0
    # helpful to assure null data cells are returned as zeros
    # note teh mask of omega_inv provides unambiguous null definition
    scaling_matrix = omega_inv.filled(0.0)
    # now apply scaling_matrix - works because zeroed masked values
    # range is 0,1,2 because image extracts only teh 3c vector data
    for k in range(3):
        image[:, :, :, k] = image[:, :, :, k] * scaling_matrix
    return image, omega_inv


def stack_data(imagelist, cutoff, clip_level, weights=None, verbose=False):
    """
    Stacks data with optional weights in imagelist.
    """
    # if weights are used make sure it is the same length as imagelist
    if weights is not None:
        weighted_stack = True
        if len(imagelist) != len(weights):
            message = "stack_data:  received a weight array of length={}\n".format(
                len(weights)
            )
            message += "Does not match length of array of image volumes={}".format(
                len(imagelist)
            )
            raise RuntimeError(message)
    else:
        weighted_stack = False

    [N1, N2, N3, NV] = imagelist[0].shape
    sum_images = np.zeros(shape=[N1, N2, N3, 3])
    sumwts = np.zeros(shape=[N1, N2, N3])
    if verbose:
        print("Image volume number of components=", N1 * N2 * N3 * NV)
    for i in range(len(imagelist)):
        nancount = np.isnan(imagelist[i]).sum()
        if verbose and nancount > 0:
            print(f"Image number {i} has {nancount} NaN values")
            for k in range(NV):
                x = imagelist[i][:, :, :, k]
                c = np.isnan(x).sum()
                print(f"Number of NaNs in component {k}={c}")
        img, omega_inv = normalize_by_solid_angle(imagelist[i], cutoff)
        if verbose:
            nmasked = omega_inv.mask.sum()
            mask_fraction = nmasked / img.size
            print(
                f"Fraction of image number {i} masked in nbormalize_by_solid_angle function={mask_fraction}"
            )

        # form a matrix of 1s where the data are valid and 0s where not
        # note normalize_by_solid angle already sets invalid values of
        # omega_inv to 0.   Here we only need to change all valid values
        # to 1 - very obscure syntax with the python not operator on the mask
        wtmatrix = (~omega_inv.mask).astype(float)
        # simple way to apply a weight to all values is to scale wtmagtrix
        if weighted_stack:
            img *= weights[i]
            wtmatrix *= weights[i]
        madlist = list()
        madavg = 0.0
        for k in range(3):
            # make a regular array with NaNs in mask to allow computing mad correctly
            dtmp = np.ma.masked_array(img[:, :, :, k], mask=omega_inv.mask)
            dtmp = dtmp.filled(np.nan)
            nancount = np.isnan(dtmp).sum()
            dtmp = dtmp.flatten()
            nancount = np.isnan(dtmp).sum()
            mad = scipy.stats.median_abs_deviation(dtmp, nan_policy="omit")
            madlist.append(mad)
            madavg += mad
        madavg /= 3.0
        clip_value = clip_level * madavg
        if verbose:
            print(f"Clipping image {i} at level={clip_level}")
        img = np.clip(img, -clip_value, clip_value)
        sum_images += img
        sumwts += wtmatrix
    # normally testing for zero like this would be problematic but it
    # will not because invalid data are always created as sums of zeros
    # that always yield float zeros
    sumwt_masked = np.ma.masked_equal(sumwts, 0.0)
    # nsmall=(sumwt_masked<0.1).sum()
    for k in range(3):
        # This construct is needed because the / operator on numpy
        # arrays ignores a mask.   The parenthesis exploit the fact the mask
        # is retained so when the filled method is applied to the result it
        # zeros all masked values defined by sumwt_masked in lhs
        sum_images[:, :, :, k] = (sum_images[:, :, :, k] / sumwt_masked).filled(0.0)
    return sum_images, sumwt_masked


def apply_mask(imagedata, masker, nullvalue=0.0):
    """
    Apply the mask defined in the 3 array sumwt to the components of the
    4d array imagedata.  i.e. the mask is applied in a loop over the last
    index of imagedata.   Masked valued are set to nullvalue using the
    filled method.  Returns imagegrid with masked values set to nullvalue.
    """
    # could hard code this but this does this right
    [N1, N2, N3, NV] = imagedata.shape
    for k in range(NV):
        d = np.ma.masked_array(imagedata[:, :, :, k], mask=masker.mask)
        imagedata[:, :, :, k] = d.filled(nullvalue)
    return imagedata


def save_results(db, mastergrid, stack, sumwt, control, nametag_base, algorithm):
    """ """
    gclstack = GCLvectorfield3d(mastergrid, 3)
    gclstack.name = nametag_base + "_" + algorithm
    stack = apply_mask(stack, sumwt)
    # calling the filled method to zero undefined values.
    # TODO:  probably should save the mask. For now it could be extracted from
    # the weight array saved below if it is
    gclstack = load_numpy_data(gclstack, stack)
    GCLdbsave(db, gclstack, dir=control.dir)
    if control.save_weight_data:
        gclsumwt = GCLscalarfield3d(mastergrid)
        # note since weights are alwasy >= 0 setting undefined values negative
        # could be used to defined masked values.
        # TODO:   not implemented.  Needs design for how to interact with
        # paraview
        gclsumwt = load_numpy_data(gclsumwt, sumwt.data.filled(-1.0))
        gclsumwt.name = nametag_base + "_sumwt_" + algorithm
        GCLdbsave(db, gclsumwt, dir=control.dir)


def gridstacker(
    doclist_or_cursor,
    db,
    control=None,
    methods=["average", "azimuth_weighting", "bin_weighting"],
    output_base_name="stack",
    pfname="gridstacker.pf",
    verbose=False,
):
    """
    Driver function to do the stack phase of prestack migration with pwmig.

    pwmig is a "prestack migration method", which means it migrates data
    from individual (pseudo)events first and then stacks the migrated
    3d images.  This function does the final stacking part of the
    algorithm with multiple choices for the how to do that stacking.
    The current implementation supports three stacking recipes defined
    by the "methods" argument:
        1. The "average" method is a straight summation with all inputs
           getting an equal weight.
        2. The "azimuth_weighting" aims to reduce illumination artifacts from
           unbalanced inputs a limited azimuth range.   It is similar in
           concept to the approach used in the original paper on pwmig
           by Poppeliers and Pavlis.
        3. The "bin_weighting" algorithm takes the somewhat opposite
           perspective of azimuth_weighting giving higher emphasis to
           data from pseodosource bins with high fold.
    All method except "average" should be treated as experimental.
    All the published work to date other than that in the original
    Poppeliers and Pavlis paper used the equivalent of "average".

    All stacks use an important normalization.   That is the imaging
    algorithm in pwstack is the so called "generalized Radon transform"
    inverse.  Although the implementation makes it far from obvious, the
    scattering potential at each image point is computed from an integration
    formula.  That formula includes a d_omega term where omega is a solid
    angle.  All the stacking methods applied by this function first normalize
    each image point by the solid angle range (sum of d_omega values).
    That is done with one important implementation detail.   That is,
    experience from previous work showed it was important to not include
    image cells that had limited viewing angles meaning cells where the
    total solid angle was below a threshold.

    This function currently is not set up to run in parallel but attempts to
    load all image volumes it is to stack into memory.   With current generation
    HPC hardware that model is feasible, but future developments may
    profit from the use of dask arrays to distibute the individual images
    across multiple compute nodes.   That currently, however, does
    not seem to be essential when 256G memory nodes are considered routine,

    :param doclist_or_cursor: list of python dictionaries used to define the grids
      to be stacked.  The recommended use is to pass this arg as a list of
      documents retrieved from the GCLfielddata collection but it should
      work if you pass this as a MongoDB command cursor (output of find).
      The list is tiny compared to size of any expected image volume so
      use of a list instead of cursor is wise to reduce the chances of
      a cursor timeout.
    :param dbname_or_handle: instance of mspasspy.db.Database with
      pwmig data stored in the GCLfield collection.   Note this function
      is currently serial so we use the regular client.
    :param control:  instance of gridstacker_control defined in this module.
      It is a data structure containing the control parameters for this
      algorithm.  If set None (default) the pfname argument is used to
      attempt to construct this data structure (object) from the pf
      file defined by pfname.
    :param methods:  list of keywords defining which algorithm(s) to run.
      Currently one or more of the following:  "average", "azimuth_weighting",
      or "bin_weighting".   Default is all three which means all three will
      be computed and saved.  Can be as short as one but function will
      silently do nothing if the list is empty.  Invalid keywords will
      cause a warning to be printed but it will not raise an exception.
    :param output_base_name:   is used as a prefix for the name tag
      generated for the outputs.  That is, it is the prefix for the name
      atribute of the GCLfield objects holding the stack and (optionally)
      the scalar sum of omega coverage result.
    :param pfname:  file name of "pf-file" containing control parameters
      to drive this function.   Contents are loaded into a
      gridstacker_control object defined at the top of this module.
      Default is "gridstacker.pf" and assumes a file by that name exists
      in the current directory and has the set of required parameters.
    """
    if control is None:
        control = gridstacker_control(pfname)
    if len(methods) == 1 and "average" in methods:
        require_weights = False
    else:
        require_weights = True
    # large memory model - may want to convert to dask array to reduce memory footprint
    count = 0
    arraylist = list()
    if require_weights:
        xref = list()
    for doc in doclist_or_cursor:
        if require_weights:
            if "telecluster_id" in doc:
                xref.append(doc["telecluster_id"])
            else:
                message = "gridstacker:  document number {} retrieved from GCLfielddata collection is missing required key='telecluster_id'\n".format(
                    count
                )
                message += "That key is required for azimuthal or binned weighting pwmig/pwstack. "
                raise ValueError(message)
        if verbose:
            print("Loadig grid with name=", doc["name"])
        fdata = GCLdbread(db, doc)
        data_array = extract_data_array(fdata)
        if verbose:
            print("Largest value in array=", np.max(data_array))
        if count == 0:
            mastergrid = GCLgrid3d(fdata)
        else:
            if (
                (fdata.n1 != mastergrid.n1)
                or fdata.n2 != (mastergrid.n2)
                or (fdata.n3 != mastergrid.n3)
            ):
                message = "Size mismatch of inputs from doclist - check query that generated list and try again"
                raise RuntimeError(message)
        arraylist.append(data_array)
        if verbose:
            print(f"{count} grids loaded")
            report_memory_use()
        count += 1
    if require_weights:
        if "azimuth_weighting" in methods:
            azwts = azimuth_weights(db, xref, control.azwt_power, control.azwt_floor)
            if verbose:
                print("using azimuth weighting method")
                report_counts(db)
                print("azimuth_index weight")
                for i in range(len(azwts)):
                    print(i,azwts[i])
        if "bin_weighting" in methods:
            binwts = source_cell_weights(
                db, xref, control.binwt_power, control.binwt_floor
            )
            if verbose:
                i=0
                print("telecluster_id weight")
                for tcid in xref:  # xref is a dict keyed by telecluster_id
                    print(tcid,binwts[i])
                    i += 1
    for alg in methods:
        if verbose:
            print("Computing stack with algorithm=", alg)
        match alg:
            case "average":
                stack, sumwt = stack_data(
                    arraylist, control.solid_angle_cutoff, control.clip_level
                )
            case "azimuth_weighting":
                stack, sumwt = stack_data(
                    arraylist, control.solid_angle_cutoff, control.clip_level, azwts
                )
            case "bin_weighting":
                stack, sumwt = stack_data(
                    arraylist, control.solid_angle_cutoff, control.clip_level, binwts
                )
            case _:
                print("Unsupported key specified for method arg=", alg)
                print("See docstring for allowed options")
                print("Nonfatal - trying any remaining values for methods list")
                continue
        print("Finished stacking with method=", alg)
        number_masked = sumwt.mask.sum()
        masked_fraction = number_masked / stack.size
        print("Fraction of cells set null=", masked_fraction)
        save_results(db, mastergrid, stack, sumwt, control, output_base_name, alg)
