import numpy as np


def _polar_to_cart(r, theta, center):

    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y


def extract_stack(img, disk_attr, num_slices):
    """Extract a stack of radial slices from a disk.

    Parameters
    ----------
    img : numpy.ndarray
        Image containing a full (solar) disk.
    disk_attr : tuple of ints
        Center coordinates and radius of the disk contained in `img` (x,y,r).
    num_slices : int
        Number of equally spaced radial slices to extract from the disk.

    Returns
    -------
    stack: numpy.ndarray
        Stack of width `disk_attr[2]` (disk radius) and height `num_slices`.
        Slices are kept in order

    Notes
    -----
    The slices are stacked in order, going clockwise from positive horizontal.
    No interpolation is performed.

    TODO
    ----
    Accuracy/"straightness" could potentially be improved by rounding rather
    than truncating the output of the polar to Cartesian transformation.

    """
    x, y, r = disk_attr

    theta, radial = np.meshgrid(np.linspace(0, 2*np.pi, num_slices),
                                np.arange(0, r))

    x_cart, y_cart = _polar_to_cart(radial, theta, (x, y))
    x_cart = x_cart.astype(int)
    y_cart = y_cart.astype(int)

    if img.ndim == 3:
        stack = img[y_cart, x_cart, :]
        stack = np.reshape(stack, (r, num_slices, 3))
    else:
        stack = img[y_cart, x_cart]
        stack = np.reshape(stack, (r, num_slices))

    return stack.swapaxes(0, 1)


def clean_stack(stack, m=1.):
    """Reject outlier slices (noisy rows) from a stack.

    Parameters
    ----------
    stack : numpy.ndarray
        Radial slices stacked as rows.
    m : int, optional
        Exclusion threshold. See notes below for more information.

    Returns
    -------
    stack : numpy.ndarray
        Version of the input stack with outliers removed.

    Notes
    -----
    Uses http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    through http://stackoverflow.com/questions/11686720/is-there-a-numpy-
    builtin-to-reject-outliers-from-a-list

    TODO
    ----
    Consider user selectable mode (MAD & percentile).

    """
    avg = stack.mean(axis=1)  # Mean average of each slice (row).
    ad = np.abs(avg - np.median(avg))  # Absolute deviation from median.
    mad = np.median(ad)
    s = ad/mad if mad else 0.
    stack = stack[s < m]

    #  Alternative percentile approach kept for future testing:
    # avg = stack.mean(axis=1)
    # percentiles = np.percentile(avg, (30, 70))
    # mask = np.logical_and(avg >= percentiles[0], avg <= percentiles[1])
    return stack


def compress_stack(stack, inner_region=0.2):
    """Derive an average intensity profile (slice) from the entire stack.

    Parameters
    ----------
    stack : np.ndarray
        Stack of intensity slices from which to create the average profile.
    inner_region : float, optional
        Fraction of the profile (from disk center) that should be used in
        estimating the center intensity.

    Returns
    -------
    profile : numpy.ndarray
        An average intensity profile from the sun's center to its limb.

    Notes
    -----
    Replacing the center value is inspired by the approach used by the
    Global H-alpha network, and is done to counteract the fact that the
    center value is unique (same in all slices, therefore not affected by
    stack averaging). If left untouched the center could take any value
    between the local minima and maxima, which would be likely to lead to
    skewed results for subsequent analysis (e.g. model fitting).

    TODO
    ----
    Center value should arguably be substituted through a weighted average
    since variance is lower towards the center?

    """
    profile = np.median(stack, axis=0)
    slice_size = len(profile)
    inner = round(slice_size * inner_region)
    profile[0] = np.median(stack[:, 1:inner])

    return profile

