import numpy as np
import matplotlib as plt
#from PIL import Image
import os

for video in os.listdir('generated_video'):
    print(video)
    frames = np.load('generated_video/'+video)[0]
    for i in range(frames.shape[0]):
        #img = Image.fromarray(frames[i], 'RGB')
        #img.show()
        plt.imshow(frames[i])
        plt.show()
