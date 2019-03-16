'''
blah
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import math
import os

# Global parameters

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 1  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 1  # ditto for top edge of trapezoid
trap_height = 0.5  # height of the trapezoid expressed as percentage of image height

trap_bottom_width_e = 0.9  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width_e = 0.85  # ditto for top edge of trapezoid
trap_height_e = 0.2  # height of the trapezoid expressed as percentage of image height

# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 5	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 20	# maximum gap in pixels between connectable line segments


# Helper functions
def grayscale(img):
	"""Applies the Grayscale transform
	This will return an image with only one color channel
	but NOTE: to see the returned image as grayscale
	you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
def canny(img, low_threshold, high_threshold):
	"""Applies the Canny transform"""
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def region_of_interest_e(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
	
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	mask = cv2.bitwise_not(mask)
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def filter_colors(image):
	"""
	Filter the image to include only yellow and white pixels
	"""
	# Filter white pixels
	white_threshold = 200 #130
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)

	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([90,100,100])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

	# Combine the two above images
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 0., 0.)

	return image2

def filter_yellow(image):
	"""
	Filter the image to include only yellow and white pixels
	"""

	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([80,70,70])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

	return yellow_image

def annotate_image_array(image_in):
	""" Given an image Numpy array, return the annotated image as a Numpy array """
	# Only keep white and yellow pixels in the image, all other pixels become black
	image = filter_colors(image_in)
	
	# Read in and grayscale the image
	gray = grayscale(image)

	# Apply Gaussian smoothing
	blur_gray = gaussian_blur(gray, kernel_size)

	# Apply Canny Edge Detector
	edges = canny(blur_gray, low_threshold, high_threshold)

	# Create masked edges using trapezoid-shaped region-of-interest
	imshape = image.shape
	vertices = np.array([[\
		((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
		((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
		, dtype=np.int32)
	masked_edges_n = region_of_interest(edges, vertices)
	
	#My contribution
	vertices_e = np.array([[\
		((imshape[1] * (1 - trap_bottom_width_e)) // 2, imshape[0]),\
		((imshape[1] * (1 - trap_top_width_e)) // 2, imshape[0] - imshape[0] * trap_height_e),\
		(imshape[1] - (imshape[1] * (1 - trap_top_width_e)) // 2, imshape[0] - imshape[0] * trap_height_e),\
		(imshape[1] - (imshape[1] * (1 - trap_bottom_width_e)) // 2, imshape[0])]]\
		, dtype=np.int32)
	return region_of_interest_e(masked_edges_n, vertices_e)

def annotate_image(input_file, output_file):
	""" Given input_file image, save annotated image to output_file """
	annotated_image = annotate_image_array(mpimg.imread(input_file))
	plt.imshow(annotated_image)
	plt.show()
	plt.imsave(output_file, annotated_image)

# End helper functions


# Main script
if __name__ == '__main__':
	from optparse import OptionParser

	# Configure command line options
	parser = OptionParser()
	parser.add_option("-i", "--input_file", dest="input_file",
					help="Input video/image file")
	parser.add_option("-o", "--output_file", dest="output_file",
					help="Output (destination) video/image file")
	parser.add_option("-I", "--image_only",
					action="store_true", dest="image_only", default=False,
					help="Annotate image (defaults to annotating video)")

	# Get and parse command line options
	options, args = parser.parse_args()

	input_file = options.input_file
	output_file = options.output_file
	image_only = options.image_only

	annotate_image(input_file, output_file)
