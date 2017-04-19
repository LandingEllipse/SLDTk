# import numpy as np
import cv2


def resize_to_fit(img, size):
    """Resizes and image in numpy.ndarray form to fit within the supplied bounds.
    
    Aspect ratio is preserved.
    
    Args:
        img (numpy.ndarray): the image to be resized
        size (int): desired size of the resulting longest edge
        
    Returns:
        numpy.ndarray: the resized image, or the original if it was already sufficiently small
        float: the scaling factor applied
    """
    in_h, in_w = img.shape

    if in_h > size or in_w > size:
        if in_h == in_w:
            dsize = (size, size)
            scale = size / in_h
        elif in_h > in_w:
            dsize = (size, round(size / (in_h / in_w)))
            scale = size / in_h
        else:  # in_w > in_h
            dsize = (round(size / (in_w / in_h)), size)
            scale = size / in_w

        return cv2.resize(img, dsize, interpolation=cv2.INTER_AREA), scale

    else:
        scale = 1.0
        return img, scale


def analyze_disk(img, threshold=60, resize_target=1024):
    """Finds the center and radius of a single disk present in the supplied image.
    
    Uses cv2.Canny, cv2.findContours and cv2.minEnclosingCircle to determine the centre and 
    radius of the solar disk present in the supplied image.
    
    Args:
        img (numpy.ndarray): greyscale image containing a single disk in high contrast to the background.
        threshold (int): threshold used by Canny()
    
    Returns:
        tuple: center coordinates (int) 
        int: radius
    """
    if len(img.shape) > 2:
        raise TypeError("Expected grayscale image")  # TODO: just do grayscale transformation internally?

    img, scale = resize_to_fit(img, resize_target)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, threshold, threshold * 2)
    img_mod, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (x, y), r = cv2.minEnclosingCircle(contours[0])  # use the first/primary contour detected

    # FIXME: debug output
    # cv2.drawContours(image, contours, -1, (0, 0, 255), -1)
    cv2.imwrite("out/disk_analyzer_standalone_edges.png", edges)

    return (round(x/scale), round(y/scale)), round(r/scale)


if __name__ == "__main__":
    path = "tests/images/20170315_130000_4096_HMIIC_-watermark.jpg"
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    center, radius = analyze_disk(gray, 60)

    print("min enclosing circle x,y: {},{}".format(center[0], center[1]))
    print("min enclosing circle radius: {}".format(radius))

    cv2.circle(image, center, radius, (0, 0, 255), 4)
    cv2.rectangle(img=image, pt1=(center[0] - 2, center[1] - 2), pt2=(center[0] + 2, center[1] + 2), color=(0, 0, 255), thickness=-1)
    cv2.imwrite("out/disk_analyzer_standalone.png", image)
