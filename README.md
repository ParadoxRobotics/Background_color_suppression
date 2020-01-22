# Background_color_suppression

Simple color based object segmentation (with pose) in industrial context.

The code use OpenCV 4.0.2 on python 3.6.

The aim of this application is to remove a dominant color in a image using HSV histogram separation. 
Given a "background" ROI we estimate the H,S and V mean and standard deviation in the image; the following map are then
used to create a weighted mask on a grayscale conversion of the initial image.
The masked grayscale image is filtered using a morphological opening (erosion + dilatation) to extract the object.
We finally found the approximate contour (rotated rectangle) and the (pose + angle) of the following contour in the camera pixel space.

This code can be use in an assembly line to detect successive object on a carrier belt.
