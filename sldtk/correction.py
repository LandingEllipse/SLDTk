import numpy as np


def correct_disk(img, disk_attr, bias, model):
    """Perform a flat field correction on a solar disk.

    Parameters
    ----------
    img : numpy.ndarray
        An image containing a solar disk.
    disk_attr : tuple of 3 ints
        The x, y and r properties of the solar disk present in the image.
    bias : int or float
        Brightness level of the disk's centre.
    model : limb_model.LimbModel
        Model used for radius-based flat field generation.

    Returns
    -------
    numpy.ndarray
        Flat field corrected version of the input image.

    Raises
    ------
    TypeError
        If the image is multichannel (i.e. color).

    Notes
    -----
    The floating point range resulting from the flat field
    correction is rescaled to 8-bit precision through centering around
    `bias` and clipping overflowing values. This can result in the loss
    of contrast of and within faclula, and is primarily done to increase
    umbra/penumbra distinction.

    """
    if len(img.shape) > 2:
        raise TypeError("`img` appears to be a color image. Currently only "
                        "grayscale images can be flat-field corrected.")

    d_x, d_y, d_r = disk_attr

    xx, yy = np.ogrid[0:2*d_r, 0:2*d_r]
    distances = np.sqrt(np.square(xx-d_r)+np.square(yy-d_r)) / d_r
    distances = np.ma.masked_array(distances, mask=(distances >= 1))

    flat = model.eval(distances, absolute=True)

    disk = img[d_y-d_r:d_y+d_r, d_x-d_r:d_x+d_r].copy()
    disk = (disk / flat) * bias

    img[d_y-d_r:d_y+d_r, d_x-d_r:d_x+d_r] = np.clip(disk, 0, 255).round()

    return img
