#!/usr/bin/env python
# coding: utf-8

# ##  <font style="color:black">Import Libraries</font>

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import Image


# In[2]:


#Display image directly
# Display 18x18 pixel image.
Image(filename='Face_18x18.png') 


# In[5]:


# Display 80x80 pixel image.
Image(filename='Face.jpg')


# In[6]:


# Display 18x18 pixel image.
Image(filename='Face.jpg') 


# In[7]:


# Display 12x12 pixel image.
Image(filename='Face.png')


# In[3]:


# Read image as gray scale.
cb_img = cv2.imread("Face.png",0)

# Print the image data (pixel values), element of a 2D numpy array.
# Each pixel value is 8-bits [0,255]
print(cb_img)


# In[ ]:





# In[ ]:





# ## Display Image attributes

# In[9]:


# print the size  of image
print("Image size is ", cb_img.shape)

# print data-type of image
print("Data type of image is ", cb_img.dtype)


# ## Display Images using Matplotlib

# ### What happened?
# Even though the image was read in as a gray scale image, it won't necessarily display in gray scale when using `imshow()`. matplotlib uses different color maps and it's possible that the gray scale color map is not set.

# In[8]:


# Display image.
plt.imshow(cb_img)


# In[12]:


# Set color map to gray scale for proper rendering.
plt.imshow(cb_img, cmap='gray')


# ## Another example

# In[13]:


# Read image as gray scale.
cb_img_fuzzy = cv2.imread("India_fuzzy_18x18.jpg",0)

# print image
print(cb_img_fuzzy)

# Display image.
plt.imshow(cb_img_fuzzy,cmap='gray')


# ## Working with Color Images
# Until now, we have been using gray scale images in our discussion. Let us now discuss color images.

# In[14]:


#working with color image
# Read and display Coca-Cola logo.
Image("sprite-logo.png")


# ## Read and display color image
# Let us read a color image and check the parameters. Note the image dimension.

# In[7]:


#Read and display color image
# Read in image
coke_img = cv2.imread("sprite-logo.png",1)

# print the size  of image
print("Image size is ", coke_img.shape)

# print data-type of image
print("Data type of image is ", coke_img.dtype)

print("")


# ## Display the Image

# In[ ]:





# In[16]:


#Displlay the images
plt.imshow(coke_img)
#  What happened?


# The color displayed above is different from the actual image. This is because matplotlib expects the image in RGB format whereas OpenCV stores images in BGR format. Thus, for correct display, we need to reverse the channels of the image. We will discuss about the channels in the sections below.

# In[17]:


coke_img_channels_reversed = coke_img[:, :, ::-1]
plt.imshow(coke_img_channels_reversed)


# ## Converting to different Color Spaces
# 
# 
# **`cv2.cvtColor()`** Converts an image from one color space to another. The function converts an input image from one color space to another. In case of a transformation to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.
# 
# ### <font style="color:rgb(8,133,37)">Function Syntax </font>
# ``` python
# dst = cv2.cvtColor( src, code )
# ```
# 
# `dst`: Is the output image of the same size and depth as `src`.
# 
# The function has **2 required arguments**:
# 
# 1. `src` input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision floating-point.
# 2. `code` color space conversion code (see ColorConversionCodes). 
# 
# ### <font style="color:rgb(8,133,37)">OpenCV Documentation</font>
# 
# **`cv2.cvtColor:`** https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
# **`ColorConversionCodes:`** https://docs.opencv.org/4.5.1/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0

# ## Splitting and Merging Color Channels
# 
# 
# **`cv2.split()`** Divides a multi-channel array into several single-channel arrays.
# 
# **`cv2.merge()`** Merges several arrays to make a single multi-channel array. All the input matrices must have the same size.
# 
# ### <font style="color:rgb(8,133,37)">OpenCV Documentation</font>
# 
# https://docs.opencv.org/4.5.1/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a
# 

# In[10]:


#Splitting and Merging Color Channels
# Split the image into the B,G,R components
img_Tj_bgr = cv2.imread("Face.jpg",cv2.IMREAD_COLOR)
b,g,r = cv2.split(img_Tj_bgr)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(r,cmap='gray');plt.title("Red Channel");
plt.subplot(142);plt.imshow(g,cmap='gray');plt.title("Green Channel");
plt.subplot(143);plt.imshow(b,cmap='gray');plt.title("Blue Channel");

# Merge the individual channels into a BGR image
imgMerged = cv2.merge((b,g,r))
# Show the merged output
plt.subplot(144);plt.imshow(imgMerged[:,:,::-1]);plt.title("Merged Output");


# ### Changing from BGR to RGB

# In[11]:


#Changing from BGR to RGB
# OpenCV stores color channels in a differnet order than most other applications (BGR vs RGB).
img_Tj_rgb = cv2.cvtColor(img_Tj_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_Tj_rgb)


# ### Changing to HSV color space

# In[12]:


#Changing to HSV color space
img_hsv = cv2.cvtColor(img_Tj_bgr, cv2.COLOR_BGR2HSV)
# Split the image into the B,G,R components
h,s,v = cv2.split(img_hsv)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h,cmap='gray');plt.title("H Channel");
plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel");
plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel");
plt.subplot(144);plt.imshow(img_Tj_rgb);plt.title("Original");


# ## Modifying individual Channel

# In[16]:


#Modifying individual channel
h_new = h+10
img_Tj_merged = cv2.merge((h_new,s,v))
img_Tj_rgb = cv2.cvtColor(img_Tj_merged, cv2.COLOR_HSV2RGB)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h,cmap='gray');plt.title("H Channel");
plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel");
plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel");
plt.subplot(144);plt.imshow(img_Tj_rgb);plt.title("Modified");


# In[13]:


#Saving Image
# save the image
cv2.imwrite("Face.png", img_Tj_bgr)

Image(filename='Face.png') 


# In[17]:


# read the image as Color
img_Tj_bgr = cv2.imread("Face.png", cv2.IMREAD_COLOR)
print("img_Tj_bgr shape is: ", img_Tj_bgr.shape)

# read the image as Grayscaled
img_Tj_gry = cv2.imread("New_Zealand_Lake_SAVED.png", cv2.IMREAD_GRAYSCALE)
print("img_Tj_gry shape is: ", img_Tj_gry.shape)


# ## Thank You!
