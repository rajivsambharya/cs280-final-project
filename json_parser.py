import sys
# import numpy as np
# import scipy as sp
# from cvxpy import *

import json
from pprint import pprint

from pytube import YouTube
import ffmpy




def main(argv):
	# with open('train_val_videodatainfo.json') as data_file:
	# 	data = json.load(data_file)

	# pprint(data)

	yt = YouTube("http://www.youtube.com/watch?v=Ik-RsDGPI5Y")
	title = 'Dancing Scene from Pulp Fiction'
	yt.set_filename(title)

	# # Once set, you can see all the codec and quality options YouTube has made
	# # available for the perticular video by printing videos.

	# print(yt.get_videos())

	video = yt.get('mp4', '720p')
	video.download('')

	timestamp = '00:00:59'
	output_filename = 'out_59.jpg'

	ff = ffmpy.FFmpeg(inputs={title + '.mp4': None}, outputs={output_filename: '-ss ' + timestamp + '  -vframes 1'})
	print ff.cmd
	ff.run()

if __name__ == '__main__':
	main(sys.argv[1:])