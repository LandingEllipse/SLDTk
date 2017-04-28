import numpy as np
import cv2


def create_slice_stack(img, disk_attr, num_slices):
    x, y, r = disk_attr

    if len(img.shape) > 2:
        stack = np.zeros((num_slices, r, img.shape[2]), img.dtype)
    else:
        stack = np.zeros((num_slices, r), img.dtype)

    # TODO: optimize to look both left and right from center for each rotation - place cnt->r in stack[i] and cnt->-r in stack[-i]?
    for i in range(num_slices):
        matrix = cv2.getRotationMatrix2D(center=(x, y), angle=(i * (360 / num_slices)), scale=1)
        rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        stack[i, :] = rotated[y, x: x + r]
        # cv2.imwrite("out/radial_slicer/rotation/rotated{}.png".format(i), rotated)  # can be used to create animations
        # plt.plot(stack[i, :])
    return stack


def clean_stack(stack):
    return stack  # TODO: implement sunspot rejection


def compress_stack(stack):
    mean = np.mean(stack, axis=0)
    slice_size = len(mean)
    inner = round(slice_size * 0.1)

    mean[0] = mean[1:inner].mean()
    return mean

