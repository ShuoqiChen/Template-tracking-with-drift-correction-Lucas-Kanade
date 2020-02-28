import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

import LucasKanadeBasis
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
sylv_data= np.load("../data/sylvseq.npy")
frame_num = sylv_data.shape[2]
# print(sylv_data.shape) 

img_w = sylv_data.shape[1]
img_h = sylv_data.shape[0]

sylv_bases = np.load("../data/sylvbases.npy")
bases = sylv_bases
# print(sylv_bases.shape)

# Input the first rect patch
rect = np.asarray([101, 61, 155, 107])

# use for record the rectangle and save them
sylvseqrects = np.zeros((frame_num,4))
sylvseqrects[0,:] = rect

# Re-draw the rectangle
# ------------------------------------------------------

# open_sylvseqrects = np.load('../code/sylvseqrects_no_bases_v4.npy')
# rect = open_sylvseqrects[430, :]


# # width 54, height f46

# p0 = np.zeros(2)


for i in range(frame_num-1):
# for i in range(430, frame_num-1):
# for i in range(10):
# for i in range(2):

    # Print the the frame number in process
    print(i)
    # print(rect)

    # Load the template and the input images
    It = sylv_data[:,:,i]
    It1 = sylv_data[:,:,i+1]


    p0 = np.zeros(2)

    # Choose to use appearance basis or not
    # ------------------------------------------------------------------------------------------------
    p = LucasKanadeBasis.LucasKanadeBasis(It = It, It1= It1, rect=rect, bases = bases)
    # p = LucasKanade.LucasKanade(It = It, It1= It1, rect=rect, p0 = p0)


    # update p0, but only update the translation
    p0 = np.zeros(2)
    p0[0] = p[4]
    p0[1] = p[5]

    # 
    tmplt_pts, warp_p = LucasKanadeBasis.init_affine(p0=p0, rect=rect)

    # Update square
    # --------------------------------------------------------------------------------
    # Compose tranformation
    W = np.zeros((2,3))

    # first row
    W[0][0] = p[0] + 1
    W[0][1] = p[2]
    W[0][2] = p[4]

    # second row
    W[1][0] = p[1]
    W[1][1] = p[3] + 1
    W[1][2] = p[5]

    # Use W to findout the new corners
    new_tmplt_pts = np.zeros(tmplt_pts.shape)
    for k in range(4):
        u = tmplt_pts[k,0]
        v = tmplt_pts[k,1]
        xy = np.matmul(W, np.asarray([[u],[v],[1]])).T
        new_tmplt_pts[k, :] = xy

    min_X = np.min(new_tmplt_pts[:,0])
    max_X = np.max(new_tmplt_pts[:,0])
    min_Y = np.min(new_tmplt_pts[:,1])
    max_Y = np.max(new_tmplt_pts[:,1])

    # # Update to the newest track rectangle
    rect = np.asarray([min_X, min_Y, max_X, max_Y])

    # rect = np.floor(rect)     # Round float number to the closet integer
    # rect = rect.astype(int)   # this is the new rect converted to integer

    # put the new values into the carseqrect array we are going to save later
    sylvseqrects[i+1,:] = rect   # remember there should be n (n = 415 in this case) rects

    # save image files when needed
    # if i == 1 or i == 100 or i == 200 or i == 300 or i == 400:
    if i %10 == 0:
        file_name = 'sylv_%s.png' % (str(i+1))

        fig,ax = plt.subplots(1)
        ax.imshow(sylv_data[:,:,i])
        rect_patch = patches.Rectangle((rect[0],rect[1]), (rect[2] - rect[0])+1, (rect[3] - rect[1])+1, linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect_patch)
        plt.savefig(file_name)
        plt.close()
        # Incremental save
        np.save('sylvseqrects_wo_basis.npy', sylvseqrects)

# save rect coordinate one last time when all the frames are processed
np.save('sylvseqrects_wo_basis.npy', sylvseqrects)

# Just for test. Load the rect coordinate file
# open_sylvseqrects = np.load('../code/sylvseqrects_wo_basis.npy')
# print(open_sylvseqrects)
# print(open_sylvseqrects.shape)
