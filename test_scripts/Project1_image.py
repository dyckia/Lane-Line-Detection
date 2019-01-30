# import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

# define helper functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, \
    max_line_gap)
    # only keep the longest left line and longest right line
    long_lines = longest_lines(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # extend longest lines
    extend_lines(line_img, long_lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def longest_lines(lines):
    max_len_left=max_len_right = 0
    lx1=ly1=lx2=ly2=rx1=ry1=rx2=ry2= 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 < 480:
                if (x2-x1)**2 + (y2-y1)**2 > max_len_left:
                    max_len_left = (x2 - x1) ** 2 + (y2 - y1) ** 2
                    lx1 = x1
                    ly1 = y1
                    lx2 = x2
                    ly2 = y2
            else:
                if (y2-y1)**2 + (x2-x1)**2 > max_len_right:
                    max_len_right = (x2 - x1) ** 2 + (y2 - y1) ** 2
                    rx1 = x1
                    ry1 = y1
                    rx2 = x2
                    ry2 = y2
    return np.array([[(lx1,ly1,lx2,ly2),(rx1,ry1,rx2,ry2)]], dtype=np.int32)

def extend_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 - x2 == 0:
                x_top = x_bottom = x1
            else:
                k = (y2-y1)/(x2-x1)
                b = y1 - k * x1
                x_bottom = int((540-b)/k)
                x_top = int((340-b)/k)
            cv2.line(img, (x_bottom, 540), (x_top, 340), color, thickness)

# set some parameters
low_threshold = 50
high_threshold = 150
vertices = np.array([[(440,340),(100, 540), (910,540), (540, 340)]], dtype=np.int32)
kernel_size = 3
rho = 1
theta = np.pi/180
threshold = 25
min_line_len = 17
max_line_gap = 3

# load the images in the directory
imgs = os.listdir('test_images/')
num = len(imgs)
for i in range(num):
    # ignore '.DS_Store' file
    if imgs[i] != '.DS_Store':
        # read each image
        img = mpimg.imread('test_images/'+imgs[i])

        # convert to grayscale
        img_gray = grayscale(img)

        # apply gaussian_blur
        gau_gray = gaussian_blur(img_gray, kernel_size)

        # apply canny transform
        cann_gray = canny(gau_gray, low_threshold, high_threshold)

        # apply mask on image
        mask = region_of_interest(cann_gray, vertices)

        # apply hough transform
        lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)

        # combine image
        combine = weighted_img(lines, img)

        # save combined image
        mpimg.imsave('test_images/detected_' +imgs[i], combine)