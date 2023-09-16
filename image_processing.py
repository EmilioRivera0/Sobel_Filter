# necessary imports -------->
import cv2 
import numpy as np
import matplotlib.pyplot as plt

# variables declaration and initialization -------->
# Sobel Filters (kernels)
HORIZONTAL_LINE_KERNEL = np.array(
    [
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])

VERTICAL_LINE_KERNEL = np.array(
    [
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])
# original image in gray scale
ORIGINAL_IMG = cv2.imread('TF2Merchs.png', cv2.IMREAD_GRAYSCALE)

# program execution -------->
# apply Sobel kernels to the image
image_x_filtered = cv2.filter2D(src=ORIGINAL_IMG,kernel=HORIZONTAL_LINE_KERNEL,ddepth=-1)
image_y_filtered = cv2.filter2D(src=ORIGINAL_IMG,kernel=VERTICAL_LINE_KERNEL,ddepth=-1)
# combine both images
combined_image = cv2.bitwise_or(image_y_filtered,image_x_filtered)

# show original, filtered and combined images
# all 4 images in a signle window
fig = plt.figure()

fig.add_subplot(2,2,1)
plt.title('Original Image')
plt.imshow(ORIGINAL_IMG,'gray')
plt.axis('off')

fig.add_subplot(2,2,2)
plt.title('Horizontal Line Image')
plt.imshow(image_x_filtered,'gray')
plt.axis('off')

fig.add_subplot(2,2,3)
plt.title('Vertical Line Image')
plt.imshow(image_y_filtered,'gray')
plt.axis('off')

fig.add_subplot(2,2,4)
plt.title('Combined Image')
plt.imshow(combined_image,'gray')
plt.axis('off')

plt.show()
