import numpy as np
from scipy.ndimage.interpolation import rotate


def radial_slice(img, center, radius, slices=20):
    # radius = int(radius * 1.05)  # give us some headroom to ensure the limb isn't clipped
    radius = radius + 10  # TODO: determine radius to use based on min of radius*1.05 and center->img_edge (up/down/left/right)
    stack = np.zeros((slices, radius), img.dtype)

    # TODO: optimize to look both left and right from center for each rotation - place cnt->r in stack[i] and cnt->-r in stack[-i]?
    for i in range(slices):
        matrix = cv2.getRotationMatrix2D(center=(center[0], center[1]), angle=(i * (360 / slices)), scale=1)
        rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        # rotated = rotate(input=img, angle=(i * (360 / slices)), reshape=False)
        stack[i, :] = rotated[center[0], center[1]:center[1] + radius]  # FIXME: verify x/y -> h/w
        # cv2.imwrite("out/radial_slicer/rotation/rotated{}.png".format(i), rotated)  # can be used to create animations
        # plt.plot(stack[i, :])

    return stack


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import disk_analyzer

    path1 = "tests/images/20170315_130000_4096_HMIIC_-watermark_small.jpg"
    path2 = "tests/images/20170315_130000_4096_HMIIC_-watermark_small_offset.jpg"
    path3 = "tests/images/20170315_aberystwyth_combined.jpg"
    path4 = "tests/images/20170315_aberystwyth_combined_square.jpg"

    image = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

    center, radius = disk_analyzer.analyze_disk(img=image.copy(), threshold=20)

    print("min enclosing circle x,y: {},{}".format(center[0], center[1]))
    print("min enclosing circle radius: {}".format(radius))

    cv2.imwrite("out/radial_slicer/input_image.png", image)

    from timeit import default_timer as timer
    start = timer()
    image_sliced = radial_slice(img=image, center=center, radius=radius, slices=20)
    end = timer()
    print(end - start)

    median = np.median(image_sliced, axis=0)
    plt.plot(median)
    mean = np.mean(image_sliced, axis=0)
    plt.plot(mean)
    plt.show()

    cv2.imwrite("out/radial_slicer/slices.png", image_sliced)
