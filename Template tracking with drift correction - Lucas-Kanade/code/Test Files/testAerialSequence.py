import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

import SubtractDominantMotion


carseq_data = np.load("../data/aerialseq.npy")
frame_num = carseq_data.shape[2]

for i in range(frame_num-1):

    # Load the template and the input images
    It = carseq_data[:,:,i]
    It1 = carseq_data[:,:,i+1]

    print(i)
    
    # find out the mask that shows the main motion
    mask = SubtractDominantMotion.SubtractDominantMotion(It, It1)

    # plot the tracking
    fig = plt.figure()
    plt.imshow(It1)
    locs = np.transpose(np.nonzero(mask))
    plt.plot(locs[:,1], locs[:,0], 'b.')
    plt.draw()

    # save the tracking
    file_name = 'md_%s.png' % (str(i+1))
    plt.savefig(file_name)
    plt.close()
