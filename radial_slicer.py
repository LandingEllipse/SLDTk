from pylab import imshow, figure, zeros, plot
from scipy.misc import imread
from scipy.ndimage.interpolation import rotate
from numpy import savetxt
import numpy as np
import matplotlib.pyplot as plt

a = imread("testing/20170315_130000_4096_HMIIC_-watermark.jpg", flatten=True)
#imshow(a)
shape = a.shape
print(shape[1])

slices = 20

stack = zeros((int(shape[1]/2), slices))  # create an array to save the slices
total = stack[:, 0]  # create an array to save the average slice

for i in range(slices):  # take ten slices
    print(i)
    stack[:, i] = rotate(input=a, angle=(i * (360/slices)), reshape=False)[int(shape[1]/2), int(shape[1]/2):]  # TODO: just get img centre as arg to class...
    plt.plot(stack[:, i])
    total += stack[:, i]
    # print(total)

total = total / (slices + 1)
plot(total)

plt.savefig("out/intensity_graph_orange")
plt.show()
savetxt("out/jefferson-v2.dat", total)  # save the data to a file
