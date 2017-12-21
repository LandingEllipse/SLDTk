import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


def polar2cart(r, theta, center):
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y


def img2polar(img, center, final_radius, initial_radius=None, phase_width=3000):

    if initial_radius is None:
        initial_radius = 0

    theta, R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    # print('Xcart:')
    # print(Xcart)
    # print('Ycart:')
    # print(Ycart)

    if img.ndim == 3:
        polar_img = img[Ycart, Xcart, :]
        polar_img = np.reshape(polar_img, (final_radius-initial_radius, phase_width, 3))
    else:
        polar_img = img[Ycart, Xcart]
        polar_img = np.reshape(polar_img, (final_radius-initial_radius, phase_width))

    return polar_img


if __name__ == "__main__":
    # img = cv2.imread('images/LimbDark_marked.png')
    img = cv2.imread('images/20140704_022325_4096_HMII.jpg')

    start = time.time()
    pol = img2polar(img, (2057, 2051), 1870, )
    end = time.time()
    print(end-start)

    cv2.imwrite("tmp/ferrari_pol_hmi.jpg", pol.swapaxes(0, 1))
