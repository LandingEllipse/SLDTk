# import numpy as np
import cv2


def analyze_disk(img, threshold=10):
    """Finds the center and radius of a single solar disk present in the supplied image.
    
    Uses cv2.inRange, cv2.findContours and cv2.minEnclosingCircle to determine the centre and 
    radius of the solar disk present in the supplied image.
    
    Args:
        img (numpy.ndarray): greyscale image containing a full, single solar disk against a background that is below `threshold`.
        threshold (int): threshold of min pixel value to include in the solar disk
    
    Returns:
        tuple: center coordinates in x,y form (int) 
        int: radius
    """
    if img is None:
        raise TypeError("img argument is None - check that the path of the loaded image is correct.")

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
        raise RuntimeError("No disks detected in the image.")

    return (round(x), round(y)), round(r)


if __name__ == "__main__":
    path = "tests/images/20170315_130000_4096_HMIIC_-watermark.jpg"
    path2 = "tests/images/20170315_aberystwyth_combined.jpg"
    path5 = "tests/images/LimbDark.png"
    path7 = "tests/images/20170420_4096_HMIIC.jpg"
    path8 = "tests/images/20170420_4096_HMIIC_-watermark.jpg"

    image = cv2.imread(path7)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    center, radius = analyze_disk(img=gray, threshold=20)

    print("min enclosing circle x,y: {},{}".format(center[0], center[1]))
    print("min enclosing circle radius: {}".format(radius))

    cv2.circle(image, center, radius, (0, 0, 255), 1)
    cv2.rectangle(img=image, pt1=(center[0] - 2, center[1] - 2), pt2=(center[0] + 2, center[1] + 2), color=(0, 0, 255), thickness=-1)
    cv2.imwrite("out/disk_analyzer/circle_superimposed.png", image)
