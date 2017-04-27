import sys
import os
# import numpy as np
# import scipy as sp
# from cvxpy import *

import json
from pprint import pprint

from pytube import YouTube
import ffmpy




def main(argv):
	with open('train_val_videodatainfo.json') as data_file:
		data = json.load(data_file)

	videos = data["videos"]

	print 'length', len(videos)


	COOKING_CATEGORY = 17


	for i in range(len(videos)):
		curr_video = videos[i]
		if curr_video['category'] == COOKING_CATEGORY:
			try:
				curr_video = videos[i]

				time_start = curr_video['start time']
				time_end = curr_video['end time']
				duration = time_end - time_start
				yt = YouTube(curr_video['url'])

				title = 'video' + str(i)
				yt.set_filename(title)
				video = yt.get('3gp', '144p')
				newpath = '%d/' %(i) 
				if not os.path.exists(newpath):
				    os.makedirs(newpath)
				video.download(newpath)

				ff = ffmpy.FFmpeg(inputs={newpath + title + '.3gp': None}, outputs={newpath + 'out%d.jpg': '-ss ' + str(time_start) + ' -t ' + str(duration) +  ' -vf fps=1 '})
				print ff.cmd
				ff.run()
			except:
				print 'SOME ERROR OCCURRED'

			



if __name__ == '__main__':
	main(sys.argv[1:])