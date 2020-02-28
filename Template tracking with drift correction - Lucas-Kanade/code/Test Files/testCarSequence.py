import nuy * x_primempy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation

import LucasKanade
import cv2   # for sobel filter


carseq_data = np.load("../data/carseq.npy")
frame_num = carseq_data.shape[2]

# initialize the tracing rectangle by hard-coding its corner coordinates
rect = np.asarray([59, 116, 145, 151])   # This is the very first square that defines the frame to use

# The p0 to be used in the first time
p0 = np.zeros(2)

# Initialize the variable to be saved at the very end
carseqrects = np.zeros((frame_num,4))
carseqrects[0,:] = rect

# The width and height of the tracking patch
rect_width = 145 - 59 + 1  
rect_height = 151 - 116 + 1

# The width and height of the whole images to be processed
img_h = carseq_data.shape[0]
img_w = carseq_data.shape[1]

# Plot the very first rectangle
file_name = 'lk_%s.png' % (str(0))
fig,ax = plt.subplots(1)
ax.imshow(carseq_data[:,:,0])
rect_patch = patches.Rectangle((rect[0],rect[1]), (rect[2] - rect[0] + 1), (rect[3] - rect[1] + 1),linewidth=1,edgecolor='r',facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect_patch)
plt.savefig(file_name)
plt.close()

for i in range(frame_num-1):
# for i in range(5):

    # Print the the frame number in process
    # print(i)

    # Load the template and the input images
    It = carseq_data[:,:,i]
    It1 = carseq_data[:,:,i+1]

    p = LucasKanade.LucasKanade(It = It, It1= It1, rect=rect, p0 = p0)

    # update p0, but only update the translation
    p0 = np.zeros(2)
    p0[0] = p[4]
    p0[1] = p[5]

    # 
    tmplt_pts, warp_p = LucasKanade.init_affine(p0=p, rect=rect)

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


    # Update to the newest track rectangle
    rect = np.asarray([min_X, min_Y, max_X, max_Y])

    # put the new values into the carseqrect array we are going to save later
    carseqrects[i+1,:] = rect   # remember there should be n (n = 415 in this case) rects



    # save image files when needed
    # if i == 1 or i == 100 or i == 200 or i == 300 or i == 400:
    if i %10 == 0:
        file_name = 'lk_%s.png' % (str(i+1))

        fig,ax = plt.subplots(1)
        ax.imshow(carseq_data[:,:,i])
        rect_patch = patches.Rectangle((rect[0],rect[1]), (rect[2] - rect[0])+1, (rect[3] - rect[1]+1),linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect_patch)
        plt.savefig(file_name)
        plt.close()
        # incremental save
        np.save('carseqrects.npy', carseqrects)


# save rect coordinate one last time when all the frames are processed
np.save('carseqrects.npy', carseqrects)


