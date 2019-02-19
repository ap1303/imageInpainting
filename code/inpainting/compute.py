## CSC320 Winter 2019 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    row = psiHatP.row()
    col = psiHatP.col()
    radius = psiHatP.radius()

    conf = copyutils.getWindow(confidenceImage, (row, col), radius)
    filled = copyutils.getWindow(confidenceImage, (row, col), radius) / 255
    mask = conf * filled

    c = np.sum(mask) / ((2 * radius + 1) ** 2)
    
    return c

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#


def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    indicator_image = filledImage / 255
    indicator_window, indicator_filled = copyutils.getWindow(indicator_image, (psiHatP.row(), psiHatP.col()), psiHatP.radius())

    gray_image = cv.cvtColor(inpaintedImage, cv.COLOR_BGR2GRAY)
    patch, valid = copyutils.getWindow(gray_image, (psiHatP.row(), psiHatP.col()), psiHatP.radius())

    mask = patch * indicator_window

    sobel_x = cv.Sobel(mask, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(mask, cv.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    index_1, index_2 = np.unravel_index(magnitude.argmax(), magnitude.shape)

    Dy = sobel_y[index_1][index_2]
    Dx = sobel_x[index_1][index_2]

    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#
def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    fronts = np.sum(fillFront)
    if fronts == 255:
        Ny = None
        Nx = None
        return Ny, Nx

    front_window, front_fill = copyutils.getWindow(fillFront, (psiHatP.row(), psiHatP.col()), psiHatP.radius())
    front_window = front_window / 255
    front = front_window * psiHatP.pixels()

    finite_diff_vertical = np.gradient(front, axis=0)
    slope_vertical = finite_diff_vertical[psiHatP.row()][psiHatP.col()]

    finite_diff_horizontal = np.gradient(front, axis=1)
    slope_horizontal = finite_diff_horizontal[psiHatP.row()][psiHatP.col()]

    magnitude = (slope_vertical ** 2 + slope_horizontal ** 2) ** 0.5

    Nx = - slope_vertical / magnitude
    Ny = slope_horizontal / magnitude

    return Ny, Nx
