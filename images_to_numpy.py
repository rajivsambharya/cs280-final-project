import sys
import os
# import numpy as np
# import scipy as sp
# from cvxpy import *

import json
from pprint import pprint

from pytube import YouTube
import ffmpy

import skimage as sk
import skimage.io as skio

import matplotlib.pyplot as plt


def main(argv):
	# run as 
	# python images_to_numpy.py vehicles_auto/0/
	return images_to_numpy(argv[0])

def images_to_numpy(folder_of_images):
	images = sorted(os.listdir(folder_of_images), key=lambda v: int(v[:-4]))
	numpyized_images = []
	for i in range(len(images)):
		curr_image = images[i]
		numpy_image = preprocess(folder_of_images + curr_image)
		numpyized_images.append(numpy_image)
		#plt.imshow(numpy_image)
		#plt.show()
	return numpyized_images


def preprocess(imname):
	# read in the image
	im = skio.imread(imname)
	# convert to double (might want to do this later on to save memory)    
	im = sk.img_as_float(im)
	return im


if __name__ == '__main__':
	main(sys.argv[1:])
