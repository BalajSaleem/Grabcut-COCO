# coco,py
A simple program to run grabcut on various categories of COCO dataset (2017 validation dataset) and calculate the IOU.
It basically compares the accuracy of grabcut on different images and objects vs the human segmented, ground truth. 
It also presents some images ground truth and grabcut output.]

run it using coco.py

You must have the folder structure according to coco guidelines in a folder called COCOdataset2017.

# grabcut.py

A simple program for interactively removing the background from an image using
the grab cut algorithm and OpenCV.

This code was derived from the Grab Cut example from the OpenCV project but is
hopefully more usable for day-to-day tasks.

See the [OpenCV GrabCut Tutorial](
  https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html) for more
information.

## Examples

![Orignal Image](example/buildbotics_cnc_controller-orig.jpg)
![Result Image](example/buildbotics_cnc_controller.png)
![Orignal Image](example/forest_cat-orig.jpg)
![Result Image](example/forest_cat-final.png)

## Usage

    grabcut.py <input> [output]

## Instructions
1. After seeing the input and output windows, draw a rectangle arround the object using middle mouse button and wait for segmentation in output window.
2. Press ctrl+P to see the control pannel with the options of Mark Forgeground / Background, Reset and Save.
3. Click the relevant option and start marking the regions with the brush using the left mouse button, you may change the thickness of the brush, from input window.
4. Reset or Save the final annotation.

## Keys
  * 0 - Select areas of sure background
  * 1 - Select areas of sure foreground
  * 2 - Select areas of probable background
  * 3 - Select areas of probable foreground
  * n - Update the segmentation
  * r - Reset the setup
  * s - Save the result
  * q - Quit
