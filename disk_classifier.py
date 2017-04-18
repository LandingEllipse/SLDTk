# import the necessary packages
import numpy as np
import cv2

path = "testing/20170315_130000_4096_HMIIC_-watermark_small.jpg"

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(path)
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect circles in the image
circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=1.4, minDist=100, minRadius=int(len(image)/4))


# Try contours method
thresh = 60
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, thresh, thresh*2)
contoured_img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
cv2.drawContours(image, contours, -1, (0, 255, 0), -1)

#centroid_x = M10/M00 and centroid_y = M01/M00
width, height, depth = image.shape
M = cv2.moments(cnt)
x = int(M['m10']/M['m00'])
y = int(M['m01']/M['m00'])
print("centroid moment x/y: {},{}".format(x, y))

(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(image, center, radius, (0, 0, 255), 2)
cv2.rectangle(img=image, pt1=(center[0] - 2, center[1] - 2), pt2=(center[0] + 2, center[1] + 2), color=(0, 0, 255), thickness=-1)
print("min enclosing circle x/y: {},{}".format(center[0], center[1]))
print("min enclosing circle radius: {}".format(radius))


# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        print("Center: {}, {}".format(x, y))
        print("Radius: {}px".format(r))
        cv2.circle(img=output, center=(x, y), radius=r, color=(0, 255, 0), thickness=2)
        cv2.rectangle(img=output, pt1=(x - 2, y - 2), pt2=(x + 2, y + 2), color=(0, 255, 0), thickness=-1)

    # show the output image
    cv2.imwrite("out/disk_classifier.png", np.hstack([image, output]))
