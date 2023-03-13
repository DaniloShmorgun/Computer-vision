import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
imgs = [Image.open(f"pic{i}.jpg")  for i in range(1,10)]

# def inverse(img):
#     L = 255 
#     img1_array = np.array(imgs[0])
#     img1_array = L - img1_array
#     return Image.fromarray(img1_array)

# res1 = inverse(imgs[0])
# res1.show()
# res1.save(r".\converted\img1.jpg")

###########################################
 
# def log_transform(img, с, base):
#     img2_array = np.array(img)
#     img2_array = (с * np.log(1 + img2_array) / np.log(base)).astype(np.uint8)
#     return Image.fromarray(img2_array)

# res2 = log_transform(imgs[1], 1000, 10)

# res2.show()
# res2.save(r".\converted\img2.jpg")
    
###########################################

# def exp_transform(img, c, gamma):
#     img3_array = np.array(img)
#     img3_array = (255 * (c * np.power(img3_array/255, gamma))).astype(np.uint8)
#     return Image.fromarray(img3_array)

    
# res3 = exp_transform(imgs[2], 1, 0.3)
# res3.show()
# res3.save(r".\converted\img3.jpg")

###########################################

# def contrast_stretching(img, m, E):
#     img4_array = np.array(img) / 255
#     img4_array = (255/(1 + np.power((m/img4_array), E))).astype(np.uint8)
#     return Image.fromarray(img4_array)

# res4 = contrast_stretching(imgs[3], 1/2, 6)

# res4.show()
# res4.save(r".\converted\img4.jpg")

# plt.plot(res4.histogram())
# plt.plot(imgs[3].histogram())
# plt.show()

###########################################

# img5_array = np.array(imgs[4])
# hist = np.array(imgs[4].histogram())
# n = hist.sum()
# table = hist.cumsum() / n

# hist_stretch = (table[img5_array] * 255).astype(np.uint8)
# res5 = Image.fromarray(hist_stretch)
# res5.show()
# res5.save(r".\converted\img5.jpg")

###########################################

# def mean_filt(img, w):
#     img6_array = np.array(img)
#     filter = np.ones((w, w))/(w**2)
#     res = sp.convolve2d(img6_array, filter).astype(np.uint8)
#     return Image.fromarray(res)


# res6 = mean_filt(imgs[5], 8)
# res6.show()
# res6.save(r".\converted\img6.jpg")

###########################################

def der_filr(img):
    img_arr = np.array(img)
    filter = np.array([
    [0, 1 ,0],
    [1, -4, 1],
    [0, 1, 0]
    ])
    
    laplas = convolve2d(img_arr,filter)[1:-1, 1:-1]
    res = np.clip(img_arr - laplas, 0, 255).astype(np.uint8)
    return res
    # return Image.fromarray(res)
 
res7 = der_filr(imgs[6])
res7.show()
res7.save(r".\converted\img7.jpg")

###########################################

# def shiness_filter(img):
#     img_arr = np.array(img)
#     filter_x = np.array([
#     [1, 0 ,-1],
#     [2, 0, -2],
#     [1, 0, -1]
#     ])

#     filter_y = np.array([
#     [1, 2 , 1],
#     [0, 0, 0],
#     [-1, -2, -1]
#     ])
    
#     x_trans = convolve2d(img_arr,filter_x)[1:-1, 1:-1]
#     y_trans = convolve2d(img_arr,filter_y)[1:-1, 1:-1]
    
#     grad = np.sqrt(x_trans**2 + y_trans**2)
#     res = np.clip(grad, 0, 255).astype(np.uint8)
    
#     return Image.fromarray(res)
 
# res7 = shiness_filter(imgs[7])
# res7.show()
# res7.save(r".\converted\img7.jpg")

###########################################

# def der_filr(img, size):
#     img_arr = np.array(img)
    
#     img_arr = median_filter(img_arr, size)
#     res = np.clip(img_arr, 0, 255).astype(np.uint8)
    
#     return Image.fromarray(res)
 
# res8 = der_filr(imgs[8], 8)
# res8.show()
# res8.save(r".\converted\img9.jpg")









