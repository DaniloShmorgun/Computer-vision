import colorgrad

from skimage import io
from skimage import filters
from skimage import color
from skimage.filters import _gaussian

from matplotlib import pyplot as plt

import numpy as np

import scipy.ndimage as spi

img1 = io.imread('pic1.png')

img1_to_hsv = color.rgb2hsv(img1)
img1_to_luv = color.rgb2luv(img1)
img1_to_lab = color.rgb2lab(img1)

# fig, (ax0, ax1, ax2) = plt.subplots(ncols = 3)

# ax0.imshow(img1[:, :, 0])
# ax1.imshow(img1[:, :, 1])
# ax2.imshow(img1[:, :, 2])


# fig, (ax0, ax1, ax2) = plt.subplots(ncols = 3)

# ax0.imshow(img1_to_hsv[:, :, 0])
# ax1.imshow(img1_to_hsv[:, :, 1])
# ax2.imshow(img1_to_hsv[:, :, 2])

# plt.show()

# fig, (ax0, ax1, ax2) = plt.subplots(ncols = 3)

# ax0.imshow(img1_to_lab[:, :, 0])
# ax1.imshow(img1_to_lab[:, :, 1])
# ax2.imshow(img1_to_lab[:, :, 2])

# plt.show()



# fig, (ax0, ax1, ax2) = plt.subplots(ncols = 3)

# ax0.imshow(img1_to_luv[:, :, 0])
# ax1.imshow(img1_to_luv[:, :, 1])
# ax2.imshow(img1_to_luv[:, :, 2])

# plt.show()

####################################

img2_old = io.imread('pic2.png')
img2 = io.imread('pic2.png')

img2[:, :, 0] = spi.gaussian_filter(img2[:, :, 0], sigma=1.5)
img2[:, :, 1] = spi.gaussian_filter(img2[:, :, 1], sigma=1.5)
img2[:, :, 2] = spi.gaussian_filter(img2[:, :, 2], sigma=1.5)


f, axes_array = plt.subplots(2,1)

for axs in axes_array:
    axs.axis('off')

axes_array[0].set_title('RGB before gaussian filter')
axes_array[1].set_title('RGB after gaussian filter')

axes_array[0].imshow(img2_old)
axes_array[1].imshow(img2)

plt.show()







