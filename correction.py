import math


def correct_disk(img, disk_attr, profile=None, model=None):
    """Corrects a solar disk for limb darkening using an intensity profile or intensity modelling function.
    Args:
        img (numpy.ndarray): source image to be corrected. The current implementation modifies the source image.
        disk_attr (tuple): disk attributes in the form of a `(x, y, radius)` tuple.
        profile (numpy.ndarray): a single dimensional array representing the intensity profile of the disk
                                 from center to limb. Must be at least as long as `r` in `disk_attr`.
        model (Model): alternative to profile, a model able to evaluate a floating point distance, returning
                        a uint8 intensity value.
    Returns:
        image (numpy.ndarray): image corrected for limb darkening in the solar disk
    Throws:
        TypeError: if input image is a colour image (only grayscale is currently supported)
        TypeError: if either none or both of profile and model are provided (they are mutually exclusive)
    """
    if len(img.shape) > 2:
        raise TypeError("`img` appears to be a color image. Currently only grayscale images can be flat-field corrected.")

    if profile is None and model is None \
            or profile is not None and model is not None:
        raise TypeError("Must provide either an intensity profile OR a model function")

    dx, dy, dr = disk_attr

    # Loop over every pixel in a bounding square centered on the disk
    for y in range(dy - dr, dy + dr):
        for x in range(dx - dr, dx + dr):

            dist = math.sqrt((dy - y)**2 + (dx - x)**2)  # Absolute distance from center
            dist_rounded = round(dist)

            if dist_rounded < dr:  # Only correct pixels within the disk's radius
                if profile is not None:
                    if dist_rounded + 1 < len(profile):  # Interpolate if possible
                        flat = (profile[dist_rounded] - profile[dist_rounded+1]) * (dist - int(dist)) + profile[dist_rounded]
                    else:
                        flat = profile[dist_rounded]
                else:  # model
                    flat = model.evaluate(dist)

                img[y, x] = min(((img[y, x] / flat) * 127).round(), 255)  # TODO: take bias (127) as argument

                # Below: attempt at colour flattening (NOP!)
                # tmp = (img[y,x,:]/ flat) * 127
                # tmp[tmp > 255] = 255
                # img[y, x, :] = tmp.astype(img.dtype)
    return img
