import numpy as np
import cv2


def detect_disk(img, threshold):
    """Finds the center and radius of a single solar disk present in the supplied image.
    
    Uses cv2.inRange, cv2.findContours and cv2.minEnclosingCircle to determine the centre and 
    radius of the solar disk present in the provided image.
    
    Args:
        img (numpy.ndarray): greyscale image containing a full, single solar disk against a background that is below `threshold`.
        threshold (int): threshold of min pixel value to include in the solar disk
    
    Returns:
        disk attributes: ((x (int), y (int)), r (int)) - center coordinates as an x,y tuple, and the radius.
    """
    if len(img.shape) > 2:
        raise TypeError("Expected single channel (grayscale) image.")

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.inRange(blur, threshold, 255)
    img_mod, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find and use the biggest contour
    r = 0
    for cnt in contours:
        (c_x, c_y), c_r = cv2.minEnclosingCircle(cnt)
        # cv2.circle(img, (round(c_x), round(c_y)), round(c_r), (255, 255, 255), 2)
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

