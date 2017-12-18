import numpy as np
import cv2


def create_slice_stack(img, disk_attr, num_slices):
    """Takes radial slices from a disk's center to its limb, returning a sequential stack of the slices.
    
    The slices are taken at (360/num_slices) degrees offset to each other, starting with the horizontal slice from
    (x,y) to (x+r, y+r), and continuing clockwise (the disk is rotated counter-clockwise).
    
    Args:
        img (np.ndarray): Solar image. 
        disk_attr (tuple): Disk attributes. Tuple of the x/y coordinates of the disk center, and disk radius. ((x, y), r)
        num_slices (int): Number of radial slices to collect into the returned slice stack. To minimize the number of 
            rotations, `num_slices` must be a multiple of 4 to allow all straight line slices to be taken for each rotation.

    Returns:
        Slice stack: np.ndarray of the radial slices in order.

    """
    x, y, r = disk_attr

    if num_slices % 4 != 0:
        raise ValueError("The number of slices must be divisible by four.")

    if len(img.shape) > 2:
        stack = np.zeros((num_slices, r, img.shape[2]), img.dtype)
    else:
        stack = np.zeros((num_slices, r), img.dtype)

    section_size = int(num_slices / 4)
    for i in range(section_size):
        matrix = cv2.getRotationMatrix2D(center=(x, y), angle=(i * (360 / num_slices)), scale=1)
        rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

        stack[i, :] = rotated[y, x:x+r]  #center to right
        stack[i + section_size, :] = rotated[y:y+r, x]  #center to bottom
        stack[i + section_size * 2, :] = rotated[y, x:x-r:-1]  #center to left
        stack[i + section_size * 3, :] = rotated[y: y-r:-1, x] #center to top
        # cv2.imwrite("out/rotation/rotated{}.png".format(i), rotated)  # KEEP: can be used to create animations
        # plt.plot(stack[i, :])
    return stack


def create_slice_stack_2(img, disk_attr, num_slices):

    p = np.zeros((round(2*np.pi*disk_attr[2]), disk_attr[2]), dtype=np.uint8)
    polar = cv2.linearPolar(img, (disk_attr[0], disk_attr[1]), disk_attr[2], cv2.INTER_CUBIC | cv2.WARP_FILL_OUTLIERS, p)
    cv2.imwrite("out/debug/polar.jpg", polar)
    return create_slice_stack(img, disk_attr, num_slices)


def reject_outliers(stack, m=1.):
    """ 
    Uses http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm through 
    http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    
    Args:
        stack: 
        m:

    Returns:

    """
    avg = stack.mean(axis=1)  # mean average of each slice
    ad = np.abs(avg - np.median(avg))  # absolute deviation of each average slice from the median average slice
    mad = np.median(ad)  # median absolute deviations
    s = ad/mad if mad else 0.  # median absolute deviations scaled by the median absolute deviation
    stack = stack[s < m]

    #  Alternative percentile approach:
    # avg = stack.mean(axis=1)
    # percentiles = np.percentile(avg, (30, 70))
    # mask = np.logical_and(avg >= percentiles[0], avg <= percentiles[1])
    return stack


def compress_stack(stack, inner_region=0.2):
    """Creates a modified average profile (1-d array) from a slice stack.
    
    TODO: update to reflect use of median instead of mean
    
    After the slices of the stack are averaged the first value (sun center) is replaced with the average of the 
    surrounding `inner_region`. This is the same approach used by the Global H-alpha network, and is used to counteract
    the fact that the center pixel is unique (same in all slices, therefore not a product of stack averaging). If left
    untouched the center pixel could take any value between the local minima and maxima, which ultimately would likely 
    lead to skewed results for model fitting later. 
    
    Args:
        stack (np.ndarray): stack of slices from which to create the average profile. 
        inner_region (float): fraction of the profile (from disk center) that should be averaged to replace the first value.

    Returns:
        An average intensity profile (np.ndarray) from the sun's center to its limb.
    """
    profile = np.median(stack, axis=0)
    slice_size = len(profile)
    inner = round(slice_size * inner_region)

    profile[0] = np.median(stack[:, 1:inner])
    return profile

