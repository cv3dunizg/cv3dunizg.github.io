---
layout: page
title: Second Laboratory Exercise
description: Instructions for the second laboratory exercise.
nav_exclude: true
---

# Second laboratory exercise: corner and edge detection

Feature detection is one of the basic computer vision tasks. A feature is typically defined as an "interesting" part of an image.
Feature detection is necessary for solving many complex problems in computer vision such as: tracking, image retrieval, image matching and others.
Interesting parts of an image often correspond with corners and edges of the objects in the scene.
Therefore, we consider two algorithms in this exercse: Harris corner detection algorithm ([paper](http://www.bmva.org/bmvc/1988/avc-88-023.pdf), [wiki](https://en.wikipedia.org/wiki/Harris_corner_detector), [opencv](https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html)) and Canny edge detection algorithm ([paper](https://ieeexplore.ieee.org/abstract/document/4767851), [wiki](https://en.wikipedia.org/wiki/Canny_edge_detector), [opencv](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)).
Implementations of these algorithms are available in many computer vision libraries. However, in order to understand them better, during this exercise we will develop our own implementations.

These instructions are focused on Python implementation. We suggest reading the original papers and studying the links above in order to prepare and understand the algorithms better from the theoretical viewpoint. In our implementation we will not rely on opencv, but the following libraries will be helpful:
[PIL](https://pillow.readthedocs.io/en/stable/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/) te [numpy](https://numpy.org/).

During the development, we will test our algorithms on two images: a simple image of FER logo and on a picture of a house that is popular in the computer vision community.

<img src="../../assets/images/lab2/fer_logo.jpg" alt="house" width="400"/>
<img src="../../assets/images/lab2/house.jpg" alt="house" width="300"/>

Download both images and store them locally on your computer.
After you have implemented both algorithms, demonstrate the results on both images.

## Harris corner detector

We will divide the implementation of the Harris algorithm into a series of smaller steps.

### 1. Image loading

The following code snippet loads the image into the multidimensional [numpy array](https://numpy.org/doc/stable/reference/arrays.html):
```
from PIL import Image
import numpy as np
img = np.array(Image.open("path/to/image"))
```
Tasks:
- Check the dimension of the tensor `img`. What are the width and the height of the image?
- If the image is in the RGB format, convert it to grayscale by averaging across the channels.
- Find the minimum and the maximum value of the pixel intensity in the image.
- Print the pixel intensities inside the upper-left patch of the image, if the size of the patch is equal to $$10\times10$$ pixels. Use *slicing* instead of the for loops.
- Execute the following command: `print(img.dtype)`. What is the datatype of the `img` array? In order to avoid the overflow in next operations, convert the `img` to `float` type (use [ndarray.astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)). Once again, check the datatype of the array `img`. 

### 2. Gaussian smoothing

We will use [gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html) function from `scipy.ndimage` in order to smooth the image. 

Tasks:
- Show the results of the Gaussian smoothing for different values of the argument `sigma`. How does the value of `sigma` affects the results visually? In order to show the image, you can use the function `imshow` from `matplotlib.pyplot`.

An example of the Gaussian smoothing for `sigma=5`:

<img src="../../assets/images/lab2/fer_logo_sigma5.jpg" alt="FER logo zaglađane sa sigma=5." width="400"/>

### 3. Image gradient computation

We will use the convolution function [ndimage.convolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html) in order to compute the image gradients.
Initialize the values of the convolutional kernels for the computation of image gradients $$\mathrm{I}_x$$ and $$\mathrm{I}_y$$  according to the [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator).

Tasks:
- Show the image gradients $$\mathrm{I}_x$$ and $$\mathrm{I}_y$$.
- Compute the second moments of the gradient: $$\mathrm{I}_x^2$$, $$\mathrm{I}_x \mathrm{I}_y$$ te $$\mathrm{I}_y^2$$.

Examples of the gradient $$\mathrm{I}_x$$ and $$\mathrm{I}_y$$:

<img src="../../assets/images/lab2/fer_grad_x.jpg" alt="FER logo gradijent po x-u." width="300"/>
<img src="../../assets/images/lab2/fer_grad_y.jpg" alt="FER logo gradijent po y-u." width="300"/>


### 4. Gradient summation inside the local neighborhood

Prior the computation of the Harris response, we should compute the characteristic matrix $$\mathbf{G}$$ in each pixel $$\mathbf{q}$$:

$$
  \begin{equation}
  \mathbf{G}(\mathbf{q})=
    \left [\begin{array}{cc}
     \sum_W \mathrm{I}_x^2   & 
     \sum_W \mathrm{I}_x \mathrm{I}_y\\ 
     \sum_W \mathrm{I}_x \mathrm{I}_y & 
     \sum_W \mathrm{I}_y^2 
    \end{array} \right]=
    \left [\begin{array}{cc}
     a&c\\ c&b
    \end{array} \right]
  \end{equation}
$$

Note that the elements of the matrix are not equal to the moment value in a particular pixel, but to the sum of the moments inside the local neighborhood of the corresponding pixel.
The size of that neighborhood is one of the parameters of the algorithm.
One way to compute that sum in each pixel is to use slicing and sum reduction inside the for loop.
However, notice that this is actually equal to the convolution with a specific kernel.

Tasks:
- Initialize the convolutional kernel appropriately and use the `ndimage.convolve` function to compute the sums of the second moments of the gradients inside the local neighborhood of each pixel.

### 5. Harris response calculation

Harris response in each pixel is equal to the difference between the image determinant and the squared trace of the matrix scaled with constant $$k$$ which is the parameter of the algorithm. 
According to the previous matrix definition we can write:

$$r(\mathbf{q})=a b -c^2 -k(a+b)^2$$

Tasks:
- Compute the Harris response in each pixel using the results of the previous step.
- Show the Harris responses in an image.

An example of the Harris response:

<img src="../../assets/images/lab2/fer_logo_harris_odziv.jpg" alt="FER logo harrisov odziv." width="400"/>


### 6. Non-maximum suppression

In this step, we want to supress the responses which might cause the false positive detections. We will do this in two steps.

First, we will set all responses smaller than a certain threshold to zero.
This threshold paramater is image dependent, so it requires tuning for every image.

Second, we will supress all responses which do not correspond to the maximal value inside their local neighborhood.
Size of the local neighborhood is another paramater of the algorithm.
Similarly to convolution, we can implement this by sliding the window centered at the corresponding pixel,
and setting the pixel to zero if its value is smaller than the maximal value inside the window.

Tasks:
- Implement the Harris response thresholding without using the for loop.
- Implement the non-maximum supression inside the local neighborhood by using at most two nested for loops.

### 7. Selection top-k responses

The last step of the algorithm is selecting the top-k responses.

Tasks:
- Find all coordinates where Harris response is not zero using the function [numpy.nonzero](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html).
- Find the pixel coordinates with k largest responses.
- Visualize the results of the algorithm by drawing circles on top of the image at the detected corners.

Example of the detected Harris corners:

<img src="../../assets/images/lab2/fer_logo_harris_corners.jpg" alt="FER logo Harrisovi kutovi." width="500"/>

Algorithm parameters: sigma=1, threshold=1e10, k=0.04, topk=100, neighborhood_size_for_summation=(5, 5), neighborhood_size_for_nonmaximum_suprression=(14, 14)


<img src="../../assets/images/lab2/house_harris_corners.jpg" alt="House Harrisovi kutovi." width="400"/>

Algorithm parameters: sigma=1.5, threshold=1e0, k=0.04, topk=100, neighborhood_size_for_summation=(5, 5), neighborhood_size_for_nonmaximum_suprression=(32, 32)

## Canny edge detector

We will divide the implementation of the Canny edge detection algorithm into a series of steps as well.
The first three steps are the same as in the Harris corner detection algorithm.

### 1. Image loading
### 2. Gaussian smoothing
### 3. Image gradient computation

### 4. Gradient magnitude and angle computation
In this step, we will compute the gradient magnitude $$|G| = \sqrt{I_x^2 + I_y^2}$$ and the gradient angle $$\theta = \arctan(I_y/I_x)$$ in each pixel.

Tasks:
- Compute the gradient magnitude and the gradient angle in each pixel according to the formulas.
- Normalize the magnitude array to the following interval $$[0-255]$$. You can do that by dividing each element with the maximum value in the array and then multiplying it with 255.
- Visualize the normalized magnitudes in an image.


An example of gradient magnitude visualization:

<img src="../../assets/images/lab2/house_magnitudes.jpg" alt="House Canny gradient magnitudes." width="400"/>


### 5. Non-maximum suprression

The goal of this step is to supress the responses in pixels which do not belong to any edges within the image.
Canny algorithm proposes supressing the non-maximum responses inside a neighborhood defined by the direction of the gradient.
We do that in a following way. First, we determine the gradient angle in each pixel.
Then, we define the discrete direction of the gradient based on that angle.
The neighborhood is then consisted of the two opposite neighbor pixels that correspond to the gradient direction.
For example, according to the image below,
if an angle $$\theta$$ belongs to one of the following intervals $$22.5^{\circ} < \theta < 67.5^{\circ}$$ or $$-157.5^{\circ} < \theta < -112.5^{\circ}$$,
then the neighborhood is consisted of the upper-right and bottom-left pixels.

In your implementation you should take care about the return value interval of the inverse tangens function, and the fact that the row indices increase from top to bottom.

<img src="../../assets/images/lab2/canny_angles.jpg" alt="House Canny gradient magnitudes." width="400"/>

Finally, the corresponding pixel will "survive" only if its magnitude is larger than the magnitudes of the neighboring pixels.
Otherwise, its magnitude is set to zero.

This type of non-maximum supression results in an image with "thinned" edges.

Tasks:
- Implement the non-maximum supression procedure described above.
- Visualize the gradient magnitudes after the non-maximum supression.


An example of the gradient magnitudes after the non-maximum supression:

<img src="../../assets/images/lab2/house_magnitudes_nms.jpg" alt="House Canny gradient magnitudes after NMS." width="400"/>

### 6. Hysteresis thresholding

In the last step of the algorithm we have to make the decision for each pixel whether it is an edge or not.
We do that by comparing the magnitude with two thresholds - high threshold max_val, and the low threshold min_val.
This procedure is known as hysteresis thresholding.

Pixels with the magnitude larger than the high threshold are considered as strong edges.

Pixels with the magnitude smaller than the low threshold are discarded and considered as not edges.

Pixels with the magnitude value between the two thresholds are considered as weak edges.

The final decision for weak edges is based on their connectivity.
Weak edges that are neighbors to any of the strong edges are considered as edges. 
Weak edges that are not neighbors to any of the strong edges are discarded and considered as not edges.

Tasks:
- Implement the described procedure of hysteresis thresholding.
- Visualize only strong edges.
- Visualize the final result of the algorithm.

An example of only strong edges:

<img src="../../assets/images/lab2/house_strong_edges.jpg" alt="House Canny strong edges." width="400"/>

An example of Canny algorithm edge detection results:

<img src="../../assets/images/lab2/house_edges.jpg" alt="House Canny edges." width="400"/>

These results are achieved with the following algorithm paramaters: sigma=1.5, min_val=10, max_val=90