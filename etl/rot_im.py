import sys
import os
import numpy as np
#import cv2
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from scipy.misc import imsave


#Load input image
arg = sys.argv

fname = arg[1]

step = int(float(arg[2]))

flip_choice = arg[3]

name = os.path.splitext(fname)[0]

print fname

# Read fits
def readFITS(fname):
	hdu_list = fits.open(fname) # Load the file
	image_data = hdu_list[0].data
	return image_data

# Clipper function
def clip(data,lim):
	data[data<lim] = 0.0
	return data

def cut_rot(image,new_width):
	new_img = image.copy()
	new_height = new_width
	width,height = image.shape
	left = (width - new_width)/2
	top = (height - new_height)/2
	right = (width + new_width)/2
	bottom = (height + new_height)/2
	new_img = new_img[left:right,top:bottom]
	return new_img


image_data = readFITS(fname)

img = np.copy(image_data)

idx = np.isnan(img)
img[idx] = 0

# Estimate stats
mean, median, std = sigma_clipped_stats(img, sigma=3.0, iters=10)



for i in range(0,359,step):
	rot = ndimage.rotate(img,i, reshape=False)
	fname1 = name +'-'+str(i)+'.png'
	rot = cut_rot(rot, 150)
	# Clip off n sigma points
	rot = clip(rot,std*3)
	imsave(fname1,rot)

if flip_choice == '1':
	print 'Flip version enabled'
	# Flipped version
	#flpd = cv2.flip(img_clip,0)
	flpd = np.fliplr(img)

	for i in range(0,359,step):
		rot1 = ndimage.rotate(flpd,i)
		fname1 = name +'-flip-'+str(i)+'.png'
		rot1 = cut_rot(rot1, 150)
		rot1 = clip(rot1,std*3)
		imsave(fname1,rot1)