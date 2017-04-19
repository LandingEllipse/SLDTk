# import numpy as np
import cv2


def analyze_disk(img, threshold=10):
    """Finds the center and radius of a single disk present in the supplied image.
    
    Uses cv2.Canny, cv2.findContours and cv2.minEnclosingCircle to determine the centre and 
    radius of the solar disk present in the supplied image.
    
    Args:
        img (numpy.ndarray): greyscale image containing a single disk in high contrast to the background.
        threshold (int): threshold of min pixel value to include in the solar disk
    
    Returns:
        tuple: center coordinates (int) 
        int: radius
    """
    if len(img.shape) > 2:
        raise TypeError("Expected grayscale image")  # TODO: just do grayscale transformation internally?

    # NOTE: we've moved away from resizing images (which was done when Canny was used for contour detection)
    # TODO: clean up by removing the resize function and references to scale

    # img, scale = resize_to_fit(img, resize_target)
    scale = 1.0

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.inRange(blur, threshold, 255)
    img_mod, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # TODO: remove debug output
    print("Number of contours found: {}".format(len(contours)))
    cv2.imwrite("out/disk_analyzer/mask.png", mask)
    cv2.drawContours(img, contours, -1, (0, 0, 255), -1)
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cv2.circle(img, (round(x), round(y)), round(r), (0, 0, 255), 2)
    cv2.imwrite("out/disk_analyzer/circled_contours.png", img)


    (x, y), r = cv2.minEnclosingCircle(contours[0])  # use the first/primary contour detected
    return (round(x/scale), round(y/scale)), round(r/scale)


# NOTE: this function is not currently needed.
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
    # print("in_h: {}".format(in_h))
    # print("in_w: {}".format(in_w))

    if in_h > size or in_w > size:
        if in_h == in_w:
            dsize = (size, size)
            scale = size / in_h
        elif in_h > in_w:
            dsize = (round(size / (in_h / in_w)), size)
            scale = size / in_h
        else:  # in_w > in_h
            dsize = (size, round(size / (in_w / in_h)))
            scale = size / in_w

        out = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        # print("out_h: {}".format(out.shape[0]))
        # print("out_w: {}".format(out.shape[1]))
        # print("scale: {}".format(scale))
        return out, scale

    else:
        scale = 1.0
        return img, scale


if __name__ == "__main__":
    path = "tests/images/20170315_130000_4096_HMIIC_-watermark.jpg"
    path2 = "tests/images/20170315_aberystwyth_combined.jpg"

    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    center, radius = analyze_disk(img=gray, threshold=20)

    print("min enclosing circle x,y: {},{}".format(center[0], center[1]))
    print("min enclosing circle radius: {}".format(radius))

    cv2.circle(image, center, radius, (0, 0, 255), 1)
    cv2.rectangle(img=image, pt1=(center[0] - 2, center[1] - 2), pt2=(center[0] + 2, center[1] + 2), color=(0, 0, 255), thickness=-1)
    cv2.imwrite("out/disk_analyzer/circle_superimposed.png", image)
