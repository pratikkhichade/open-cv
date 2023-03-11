#!/usr/bin/env python
# coding: utf-8

# # Basic Image Manipulations
# 
# In this notebook we will cover how to perform image transformations including:
# 
# * Accessing and manipulating images pixels 
# * Image resizing
# * Cropping
# * Flipping

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# ### <font style="color:rgb(8,133,37)">Original checkerboard image</font>

# In[3]:


# Read image as gray scale.
cb_img = cv2.imread("Face.png",0)

# Set color map to gray scale for proper rendering.
plt.imshow(cb_img, cmap='gray')
print(cb_img)


# ## Accessing Individual Pixels
# Let us see how to access a pixel in the image.
# 
# For accessing any pixel in a numpy matrix, you have to use matrix notation such as matrix[r,c], where the r is the row number and c is the column number. Also note that the matrix is 0-indexed. 
# 
# For example, if you want to access the first pixel, you need to specify matrix[0,0]. Let us see with some examples. We will print one black pixel from top-left and one white pixel from top-center.

# In[4]:


# print the first pixel of the first black box
print(cb_img[0,0])
# print the first white pixel to the right of the first black box
print(cb_img[0,6])


# ## <font style="color:black">Modifying Image Pixels</font>
# We can modify the intensity values of pixels in the same manner as described above.

# In[12]:


cb_img_copy = cb_img.copy()
cb_img_copy[2,2] = 200
cb_img_copy[2,3] = 200
cb_img_copy[3,2] = 200
cb_img_copy[3,3] = 200

# Same as above
# cb_img_copy[2:3,2:3] = 200

plt.imshow(cb_img_copy, cmap='gray')
print(cb_img_copy)


# # Cropping Images
# 
# Cropping an image is simply achieved by selecting a specific (pixel) region of the image.

# In[13]:


img_BT_bgr = cv2.imread("Face.jpg",cv2.IMREAD_COLOR)
img_BT_rgb = img_BT_bgr[:,:,::-1] 

plt.imshow(img_BT_rgb)


# ### <font color="green">Crop out the middle region of the image</font>

# In[16]:


img_Tj_rgb = cv2.imread("Face.jpg",cv2.IMREAD_COLOR)
cropped_region = img_Tj_rgb[200:500, 300:600]
plt.imshow(cropped_region)


# # Resizing Images
# 
# The function **`resize`** resizes the image `src` down to or up to the specified size. The size and type are derived from the `src`,`dsize`,`fx`, and `fy`.
# 
# ### <font style="color:rgb(8,133,37)">Function Syntax </font>
# ``` python
# dst = resize( src, dsize[, dst[, fx[, fy[, interpolation]]]] )
# ```
# 
# `dst`: output image; it has the size dsize (when it is non-zero) or the size computed from src.size(), fx, and fy; the type of dst is the same as of src.
# 
# The function has **2 required arguments**:
#     
# 1. `src`:	input image
# 2. `dsize`: output image size
# 
# Optional arguments that are often used include:
# 
# 1. `fx`: Scale factor along the horizontal axis; when it equals 0, it is computed as (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜
# 2. `fy`: Scale factor along the vertical axis; when it equals 0, it is computed as (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜
# 
# The output image has the size `dsize` (when it is non-zero) or the size computed from `src.size()`, `fx`, and `fy`; the type of dst is the same as of src.
# 
# ### <font style="color:rgb(8,133,37)">OpenCV Documentation</font>
# 
# **`resize():`** https://docs.opencv.org/4.5.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d

# ### <font color="green"> Method 1: Specifying Scaling Factor using fx and fy
# 

# In[18]:


resized_cropped_region_2x = cv2.resize(cropped_region,None,fx=2, fy=2)
plt.imshow(resized_cropped_region_2x)


# ### <font color="green">Medthod 2: Specifying exact size of the output image</font>

# In[19]:


desired_width = 100
desired_height = 200
dim = (desired_width, desired_height)

# Resize background image to sae size as logo image
resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)


# ### <font color="green">Resize while maintaining aspect ratio</font>

# In[20]:


# Method 2: Using 'dsize'
desired_width = 100
aspect_ratio = desired_width / cropped_region.shape[1]
desired_height = int(cropped_region.shape[0] * aspect_ratio)
dim = (desired_width, desired_height)

# Resize image
resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)


# ### <font style="color:rgb(8,133,37)">Let's actually show the (cropped) resized image.</font>

# In[21]:


# Swap channel order
resized_cropped_region_2x = resized_cropped_region_2x[:,:,::-1]

# Save resized image to disk
cv2.imwrite("resized_cropped_region_2x.png", resized_cropped_region_2x)

# Display the cropped and resized image
Image(filename='resized_cropped_region_2x.png') 


# In[22]:


# Swap channel order
cropped_region = cropped_region[:,:,::-1]

# Save cropped 'region'
cv2.imwrite("cropped_region.png", cropped_region)

# Display the cropped and resized image
Image(filename='cropped_region.png') 


# # Flipping Images
# 
# 
# The function **`flip`** flips the array in one of three different ways (row and column indices are 0-based):
# 
# ### <font style="color:rgb(8,133,37)">Function Syntax </font>
# ``` python
# dst = cv.flip( src, flipCode )
# ```
# 
# `dst`: output array of the same size and type as src.
# 
# The function has **2 required arguments**:
#     
# 1. `src`:	input image
# 2. `flipCode`: a flag to specify how to flip the array; 0 means flipping around the x-axis and positive value (for example, 1) means flipping around y-axis. Negative value (for example, -1) means flipping around both axes.
# 
# 
# ### <font style="color:rgb(8,133,37)">OpenCV Documentation</font>
# 
# **`flip:`** https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441

# In[23]:


img_BT_rgb_flipped_horz = cv2.flip(img_BT_rgb, 1)
img_BT_rgb_flipped_vert = cv2.flip(img_BT_rgb, 0)
img_BT_rgb_flipped_both = cv2.flip(img_BT_rgb, -1)

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(141);plt.imshow(img_BT_rgb_flipped_horz);plt.title("Horizontal Flip");
plt.subplot(142);plt.imshow(img_BT_rgb_flipped_vert);plt.title("Vertical Flip");
plt.subplot(143);plt.imshow(img_BT_rgb_flipped_both);plt.title("Both Flipped");
plt.subplot(144);plt.imshow(img_BT_rgb);plt.title("Original");


# ## Thank You!
