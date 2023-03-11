#!/usr/bin/env python
# coding: utf-8

# # <font style="color:black">Annotating Images</font>
# 
# In this notebook we will cover how to annotate images using OpenCV. We will learn how to peform the following annotations to images.
# 
# * Draw lines 
# * Draw circles
# * Draw rectangles
# * Add text
# 
# These are useful when you want to annotate your results for presentations or show a demo of your application. Annotations can also be useful during development and debugging.

# In[1]:


# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (9.0, 9.0)
from IPython.display import Image


# In[2]:


# Read in an image
image = cv2.imread("Appollo_moon_11.jpg", cv2.IMREAD_COLOR)

# Display the original image
plt.imshow(image[:,:,::-1])


# ## <font style="color:black">Drawing a Line</font>
# 
# 
# Let's start off by drawing a line on an image. We will use cv2.line function for this.
# 
# ### <font style="color:rgb(8,133,37)">Function Syntx</font>
# ``` python
#     img = cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
# ```
# 
# `img`: The output image that has been annotated.
# 
# The function has **4 required arguments**:
# 
# 1. `img`:   Image on which we will draw a line
# 2. `pt1`:   First point(x,y location) of the line segment
# 3. `pt2`:   Second point of the line segment
# 4. `color`: Color of the line which will be drawn
#     
# Other optional arguments that are important for us to know include:
# 
# 1. `thickness`: Integer specifying the line thickness. Default value is 1.
# 2. `lineType`:  Type of the line. Default value is 8 which stands for an 8-connected line. Usually, cv2.LINE_AA (antialiased or smooth line) is used for the lineType.
#     
# ### <font style="color:rgb(8,133,37)">OpenCV Documentation</font>
# 
# **`line:`**https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
# 
# Let's see an example of this.

# In[3]:


imageLine = image.copy()

# The line starts from (200,100) and ends at (400,100)
# The color of the line is YELLOW (Recall that OpenCV uses BGR format)
# Thickness of line is 5px
# Linetype is cv2.LINE_AA

cv2.line(imageLine, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA);

# Display the image
plt.imshow(imageLine[:,:,::-1])


# ## <font style="color:black">Drawing a Circle</font>
# 
# 
# Let's start off by drawing a circle on an image. We will use cv2.circle function for this.
# 
# ### <font style="color:rgb(8,133,37)">Functional syntx</font>
# ``` python
# img = cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
# ```
# 
# `img`: The output image that has been annotated.
# 
# The function has **4 required arguments**:
# 
# 1. `img`:    Image on which we will draw a line
# 2. `center`: Center of the circle
# 3. `radius`: Radius of the circle
# 4. `color`:  Color of the circle which will be drawn
#     
# Next, let's check out the (optional) arguments which we are going to use quite extensively.
# 
# 1. `thickness`: Thickness of the circle outline (if positive). 
# If a negative value is supplied for this argument, it will result in a filled circle.
# 2. `lineType`:  Type of the circle boundary. This is exact same as lineType argument in **cv2.line**
#     
# ### <font style="color:rgb(8,133,37)">OpenCV Documentation</font>
#     
# **`circle:`** https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
# 
# Let's see an example of this.

# In[4]:


# Draw a circle
imageCircle = image.copy()

cv2.circle(imageCircle, (400,200), 100, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA);

# Display the image
plt.imshow(imageCircle[:,:,::-1])


# ## <font style="color:black">Drawing a Rectangle</font>
# 
# 
# We will use **cv2.rectangle** function to draw a rectangle on an image. The function syntax is as follows.
# 
# ### <font style="color:rgb(8,133,37)">Functional syntx</font>
# 
#     img = cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
# 
# `img`: The output image that has been annotated.
# 
# The function has **4 required arguments**:
# 
# 1. `img`: Image on which the rectangle is to be drawn.
# 2. `pt1`: Vertex of the rectangle. Usually we use the **top-left vertex** here.
# 3. `pt2`: Vertex of the rectangle opposite to pt1. Usually we use the **bottom-right**              vertex here.
# 4. `color`: Rectangle color
#     
# Next, let's check out the (optional) arguments which we are going to use quite extensively.
# 
# 1. `thickness`: Thickness of the circle outline (if positive). 
#     If a negative value is supplied for this argument, it will result in a filled rectangle.
# 2. `lineType`: Type of the circle boundary. This is exact same as lineType argument in 
#     **cv2.line**
#     
# ### <font style="color:rgb(8,133,37)">OpenCV Documentation Links</font>
# 
# **`rectangle:`**https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
#     
# Let's see an example of this.

# In[5]:


# Draw a rectangle (thickness is a positive integer)
imageRectangle = image.copy()

cv2.rectangle(imageRectangle, (500, 300), (400,100), (150, 0, 150), thickness=5, lineType=cv2.LINE_8);

# Display the image
plt.imshow(imageRectangle[:,:,::-1])


# ## <font style="color:black">Adding Text</font>
# 
# 
# Finally, let's see how we can write some text on an image using **cv2.putText** function.
# 
# ### <font style="color:rgb(8,133,37)">Functional syntx</font>
# 
#     img = cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
# 
# `img`: The output image that has been annotated.
# 
# The function has **6 required arguments**:
# 
# 1. `img`: Image on which the text has to be written.
# 2. `text`: Text string to be written.
# 3. `org`: Bottom-left corner of the text string in the image.
# 4. `fontFace`: Font type
# 5. `fontScale`: Font scale factor that is multiplied by the font-specific base size.
# 6. `color`: Font color
#  
# Other optional arguments that are important for us to know include:
# 
# 1. `thickness`: Integer specifying the line thickness for the text. Default value is 1.
# 2. `lineType`: Type of the line. Default value is 8 which stands for an 8-connected line. Usually, cv2.LINE_AA (antialiased or smooth line) is used for the lineType.
# 
# ### <font style="color:rgb(8,133,37)">OpenCV Documentation</font>
# 
# **`putText:`**https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
#     
# Let's see an example of this.

# In[6]:


imageText = image.copy()
text = "Apollo moon V launch"
fontScale = 1.6
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (0,200, 0)
fontThickness = 2

cv2.putText(imageText, text, (200, 500), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

# Display the image
plt.imshow(imageText[:,:,::-1])


# ## Thank You!

# In[ ]:




