---
layout: page
title: First Laboratory Exercise
description: Instructions for the first laboratory exercise.
nav_exclude: true
---


# Exercise 1: geometric image deformations

In computer vision, we often need to modify images 
with various geometric deformations
such as scaling, rotation, or cropping.
This lab exercise explores backward warping deformations
that are commonly used in practice.
Let's denote the input image as $$I_s$$,
the output image as $$I_d$$,
the integer pixel coordinate vector as $$\mathbf{q}$$,
and the parameterized coordinate transformation as $$\mathbf{T}_p$$.
Then, we can formulate the backward warping deformation
of the image with the following expression:

$$I_d (\mathbf{q}) = I_s (\mathbf{T}_p(\mathbf{q})) \ .$$

As images typically have two geometric axes,
the domain and codomain of coordinate transformations 
correspond to the Euclidean plane: 
$$\mathbf{T}_p : \mathbb{R}^2 \rightarrow \mathbb{R}^2$$.
While various types of geometric transformations are used
in practice, the most common ones include
affine, projective, and radial transformations.
Affine and projective transformations
preserve collinearity of points,
while radial transformations
do not affect the distance from 
the origin of the coordinate system.

## Affine transformations

Let's denote the initial 2D vector
of real coordinates as $$\mathbf{q}_s$$,
the final 2D vector of real coordinates as $$\mathbf{q}_d$$,
the linear planar mapping as $$A$$,
and the 2D translation as $$b$$.
Then, an affine transformation of the plane
can be represented by the following equation:

$$\mathbf{q}_d = \mathbf{T}_p(\mathbf{q}_s) = \mathbf{A} \cdot \mathbf{q}_s + \mathbf{b} \ .$$

The table shows a hierarchical list of types of affine transformations,
where each subsequent type generalizes the previous one
($$\mathbf{I}$$=$$\mathrm{diag}(1,1)$$,
 $$\mathbf{R}$$ is a rotation matrix for 2D data): 

| *transformation*                | *degreees of freedom* | *invariants* | *constraints* |
| ----------------                | --------------------- | ------------ | --------- |
| translation                     |           2           | orientation  | $$\mathbf{A}=\mathbf{I}$$ |
| rigid body transformation       |           3           | distance     | $$\mathbf{A}=\mathbf{R}$$, $$\mathbf{R}^\top\mathbf{R}=\mathbf{I}$$ |
| similarity                      |           4           | angles       | $$\mathbf{A}=s\mathbf{R}$$, $$\mathbf{R}^\top\mathbf{R}=\mathbf{I}$$ |
| general affine transformation   |           6           | parallelism  | None |

If the relationship between two images 
can be described by an affine deformation,
then the parameters of the coordinate transformation
can be extracted from correspondences.
Let's have the source image $$I_s$$ 
with given points $$\mathbf{q}_{di}$$.
In the destination image $$I_d$$,
we have corresponding points $$\mathbf{q}$$<sub>si</sub>.
Then, for each correspondence pair,
the following holds:

$$\eqalign{
a_{11} q_{si1} + a_{12} q_{si2} + b_1 &= q_{di1}\\  
a_{21} q_{si1} + a_{22} q_{si2} + b_2 &= q_{di2}}$$

We can rearrange these two equations to
express the 6 parameters of the affine transformation
in matrix form as follows:

$$ {\left\lbrack \matrix{q_{si1} & q_{si2} & 0 & 0 & 1 & 0\cr 0 & 0 & q_{si1} & q_{si2} & 0 & 1} \right\rbrack} 
\cdot \left\lbrack \matrix{a_{11} \cr a_{12} \cr a_{21} \cr a_{22} \cr b_{1} \cr b_{2}} \right\rbrack
= \left\lbrack \matrix{q_{di1} \cr q_{di2}} \right\rbrack
$$

If we add two more correspondences,
we will obtain a system of size $$6\times 6$$ 
that has a unique solution, 
except when the correspondences are collinear. 
The desired deformation will be determined
by solving this system.

## Projection transformations

We can represent planar projective transformations
with the following equation:

$$\mathbf{q}_d = \mathbf{T}_p(\mathbf{q}_s) = 
  \frac{\mathbf{A} \cdot \mathbf{q}_s + \mathbf{b}}
       {\mathbf{w}^\top\mathbf{q}_s + w_0} \ .$$

Note that the numerator in the equation is a vector,
and the denominator is a scalar.
The matrix $$\mathbf{A}$$,
vectors $$\mathbf{b}$$ and $$\mathbf{w}$$,
and the scalar $$w_0$$
are the parameters of the projective transformation.
We can determine the parameters
of a projective transformation from correspondences
in a very similar way to affine transformations.
However, in this case,
each correspondence contributes to
two _homogeneous_ linear constraints
on the transformation parameters. We obtain
these constraints by moving the denominator
to the left side of the equation.
This can be done whenever correspondences
 are finite because the denominator
 is guaranteed to be nonzero in that case. 
If we gather $$n$$ correspondences,
we will have a homogeneous linear system
with $$2n$$ equations and nine unknowns 
of the form $$\mathbf{M}\mathbf{x}=\mathbf{0}$$.
As an exercise,
write down the coefficients
of the linear system for a single correspondence!

Given that our equations are homogeneous,
we can determine the projective parameters
only up to an arbitrary multiplicative constant
$$\lambda\neq 0$$. 
From this, we conclude that a projective transformation 
has only eight degrees of freedom.
Therefore, it is not surprising that the system $$\mathbf{M}$$ 
will have exactly one nontrivial solution
if we gather only four correspondences,
assuming that no triplet of correspondences is collinear.
The solution of the system corresponds to the
[right singular vector](https://en.wikipedia.org/wiki/Singular_value_decomposition#Solving_homogeneous_linear_equations)
of the matrix $$\mathbf{M}$$
corresponding to the singular value of zero. 
If we have more constraints
(more than 4 correspondences),
the optimal solution in the algebraic sense
is obtained as the right singular vector
corresponding to the smallest singular value
of the matrix $$\mathbf{M}$$.

Note that we can approach this problem
even if we choose a projective representation
of correspondences. In that case,
it would be shown that a projective mapping
can be expressed by a linear transformation
of the homogeneous coordinates of points
in the projective plane.
The projective representation would lead
to three equations for each correspondence,
and two of these three equations
would be the same as above.
This is not a significant drawback
because we could still use only
two equations since the third equation
corresponds to a linear combination of the first two.
One advantage of the projective approach
is the ability to easily incorporate
correspondences at infinity,
but this is not crucial here
because we assume that all correspondences
are within the image.

## Image interpolation in real coordinates

Earlier, we announced that we formulate 
the backward deformation of the image
with the following expression:

$$I_d (\mathbf{q}) = I_s (\mathbf{T}_p(\mathbf{q})) \ .$$

Notice that the coordinate transformation is not discrete, 
i.e., the 2D vector $$\mathbf{T}_p(\mathbf{q})$$
has real coordinates.
This means that for the destination pixel
$$\mathbf{q}$$
you should write the element 
from the source image that is "between" its pixels.
The approximated sampling of discrete images
in real coordinates is called interpolation.
There are several interpolation approaches,
and here, we will introduce
nearest-neighbor interpolation and bilinear interpolation.

### Nearest neighbour interpolation

Nearest-neighbor interpolation simply
takes the value of the nearest pixel.
The coordinates of the nearest pixel
are obtained by rounding the real coordinates
to the nearest integer values:

$$I(r+\Delta r, c+\Delta c)_{NN} = I(\lfloor r+\Delta r+0.5 \rfloor, \lfloor c+\Delta c+0.5 \rfloor )$$

### Bilinear interpolation

Bilinear interpolation assumes
that the elements of the image
at real coordinates linearly depend
on the distance from known discrete pixels.
The image shows that
the interpolated image element is obtained
as a linear combination of the four neighboring pixels,
with more weight given to the pixels
closer to the real coordinates.
The interpolated element is obtained
as the blue area times I(r, c)
plus the green area times I(r, c+1)
plus the orange area times I(r+1, c)
plus the red area times I(r+1, c+1).

![Image interpolation in real coordinates](../assets/images/bilin2a.svg)

The depicted relationships shown can be
 expressed mathematically with the following equation:
$$I_{BL}(r+\Delta r, c+\Delta c) = 
  I(r,c)     \cdot (1-\Delta r)(1-\Delta c) + 
  I(r,c+1)   \cdot (1-\Delta r)\Delta c + 
  I(r+1,c)   \cdot \Delta r(1-\Delta c) + 
  I(r+1,c+1) \cdot \Delta r\Delta c 
$$

It's worth noting that the same equation
would be obtained if we were to mirror the vertical coordinate,
or if the row indices were to increase upwards.
For comparison, nearest-neighbor interpolation
would be equal to the lower-left pixel because
it is the closest to the given real coordinates.
This situation reflects the fact that the orange rectangle is larger
than both the blue and green, and red rectangles.

A more detailed explanation of bilinear interpolation,
as well as an efficient implementation guide,
can be found in the final paper by
[Petra Bosilj](http://www.zemris.fer.hr/~ssegvic/project/pubs/bosilj10bs.pdf).

## Exercise 1: interpolation

Write a code that loads an image
and applies a random affine transformation to it using
i) nearest-neighbor interpolation and
ii) bilinear interpolation.

Instructions:

- use images `scipy.misc.ascent()` and `scipy.misc.face()`
- define the matrix $$\mathbf{A}$$ for the random affine transformation as follows: `A = .25*np.eye(2) + np.random.normal(size=(2, 2))`
- define the vector $$\mathbf{b}$$ for the random affine transformation so that the central pixel of the source image maps to the central pixel of the destination image
- write a function `affine_nn(Is, A,b, Hd,Wd)` that deforms the source image `Is` according to parameters `A` and `b` and returns a target image with resolutioin `Hd`$$\times$$`Wd`; destination pixels falling outside the source image should be black; the function should use nearest-neighbor interpolation and work for grayscale and color images
- write a function `affine_bilin(Is, A,b, Hd,Wd)` that does the same as `affine_nn`, but with bilinear interpolation 
- set the target resolution to `Hd`$$\times$$`Wd` = 200$$\times$$200
- characterize the deviation of corresponding pixels obtained with `affine_bilin` and `affine_nn`  (hint: print the root mean square deviation of pixels between the two images)
- have your main program correspond to the following code:

```
import matplotlib.pyplot as plt
import scipy.misc as misc
import numpy as np

Is = misc.face()
Is = np.asarray(Is)

Hd,Wd = 200,200
A,b = recover_affine_diamond(Is.shape[0],Is.shape[1], Hd,Wd)

Id1 = affine_nn(Is, A,b, Hd,Wd)
Id2 = affine_bilin(Is, A,b, Hd, Wd)
# ADD: print out out the standard deviation 

fig = plt.figure()
if len(Is.shape)==2: plt.gray()
for i,im in enumerate([Is, Id1, Id2]):
  fig.add_subplot(1,3, i+1)
  plt.imshow(im.astype(int))
plt.show()
```

## Exercise 2: estimate affine transformation parameters from correspondences

Write a function `recover_affine_diamond(Hs,Ws, Hd,Wd)`
that returns the parameters of an affine transformation that maps
pixels at _the centers of the sides_ of a source image with dimensions Hs$$\times$$Ws 
to the _corners_ of a target image with dimensions Hd$$\times$$Wd
Test your parameters by performing the deformation from the first exercise.

instructions:
- use `np.linalg.solve` to solve the system of equations
- the image shows the input and output of the deformation subroutine if you choose `misc.face()` as the source image

![Input and the desired output for exercise 2.](../assets/images/face_warp_diamond.png)

## Exercise 3: estimate projection transformation parameters from correspondences

Write a function `recover_projective(Qs, Qd)`
that returns the parameters of the projective transformation
given source image points `Qs` and target image points `Qd`.
Instructions:
- use `np.linalg.svd` to solve the homogeneous system of linear equations
- set the destination points to the corners of the destination image
- display the results for various randomly selected source points
