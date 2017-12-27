"""
TODO
----
The module should choose backend based on availability (see #22).

"""
import cv2
import numpy as np


def detect_disk(img, threshold):
    """Determine the center and radius of a solar disk in an image.

    Parameters
    ----------
    img : numpy.ndarray
        Greyscale image containing a full, single solar disk against a
        background that is below `threshold`.
    threshold : int
        Minimum brightness threshold to be considered part of the solar disk.
    
    Returns
    -------
    disk attributes tuple of ints
        Center coordinates and radius of the largest disk found in `img`.

    Raises
    ------
    TypeError
        If `img` is not a single channel numpy.ndarray image.
    RuntimeError
        If no disk is found in `img`.

    TODO
    ----
    Should maybe return None instead of raising an exception.

    """
    if not isinstance(img, np.ndarray) or img.ndim > 2:
        raise TypeError("Expected single channel (grayscale) image.")

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.inRange(blur, threshold, 255)
    img_mod, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    # Determine and use the biggest contour found.
    x = None
    y = None
    r = 0
    for cnt in contours:
        (c_x, c_y), c_r = cv2.minEnclosingCircle(cnt)
        if c_r > r:
            x = c_x
            y = c_y
            r = c_r
    # print("Number of contours found: {}".format(len(contours)))
    # cv2.imwrite("out/disk_analyzer/mask.png", mask)
    # cv2.imwrite("out/disk_analyzer/circled_contours.png", img)
    if x is None:
        raise RuntimeError("No disk detected in the image.")
    return round(x), round(y), round(r)

