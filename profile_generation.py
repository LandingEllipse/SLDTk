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
        # cv2.imwrite("out/rotation/rotated{}.png".format(i), rotated)  # can be used to create animations
        # plt.plot(stack[i, :])
    return stack


def clean_stack(stack):
    return stack


def compress_stack(stack, inner_region=0.1):
    """Creates a modified average profile (1-d array) from a slice stack.
    
    After the slices of the stack are averaged the first value (sun center) is replaced with the awerage of the 
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
    mean = np.mean(stack, axis=0)
    slice_size = len(mean)
    inner = round(slice_size * inner_region)

    mean[0] = mean[1:inner].mean()
    return mean


