import cv2
import numpy as np
import pandas as pd


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s < m

if __name__ == "__main__":
    path1 = "tests/images/slice_stack_sunspot_1870x100.jpg"
    path2 = "tests/images/slice_stack_sunspot_1870x360.jpg"

    img = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise TypeError("img is none...")

    avg = img.mean(axis=1)
    percentiles = np.percentile(avg, (30, 70))
    print(percentiles)
    mask = np.logical_and(avg >= percentiles[0], avg <= percentiles[1])

    percentile = img[mask]
    cv2.imwrite("out/debug/stack_percentile.jpg", percentile)

    outliers = img[reject_outliers(avg, m=1.)]
    cv2.imwrite("out/debug/stack_outliers.jpg", outliers)

    ###########################################################################

    # blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # canny = cv2.Canny(blurred, 150, 300)
    # cv2.imwrite("out/debug/stack_canny.jpg", canny)

    # img_mod, contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # contours_img = np.zeros((img.shape[0], img.shape[1], 3))
    # colour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # cv2.drawContours(colour, contours, -1, (0, 0, 255), thickness=1)

    # cv2.imwrite("out/debug/stack_contours.jpg", colour)

