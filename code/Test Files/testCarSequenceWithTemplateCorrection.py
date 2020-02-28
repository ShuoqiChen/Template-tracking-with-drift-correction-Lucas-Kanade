import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
import LucasKanade
import LucasKanadeTemplateCorrection

carseq_data = np.load("../data/carseq.npy")
frame_num = carseq_data.shape[2]

# initialize the tracing rectangle by hard-coding its corner coordinates
rect = np.asarray([59, 116, 145, 151])   # This is the very first square that defines the frame to use

# The p0 to be used in the first time
p0 = np.zeros(2)                # The very first, initial guess of p
T0 = carseq_data[:,:,0]         # The very first template patch

# Define a small error threshold epsilon
epsilon = 100

# Initialize the variable to be saved at the very end
carseqrects_wcrt = np.zeros((frame_num,4))
carseqrects_wcrt[0,:] = rect

# The width and height of the tracking patch
rect_width = 145 - 59    
rect_height = 151 - 116

# The width and height of the whole images to be processed
img_h = carseq_data.shape[0]
img_w = carseq_data.shape[1]



# # Plot the very first rectangle
# file_name = 'lk_%s.png' % (str(0))
# fig,ax = plt.subplots(1)
# ax.imshow(carseq_data[:,:,0])
# rect_patch = patches.Rectangle((rect[0],rect[1]), (rect[2] - rect[0]), (rect[3] - rect[1]),linewidth=1,edgecolor='b',facecolor='none')
# # Add the patch to the Axes
# ax.add_patch(rect_patch)
# plt.savefig(file_name)
# plt.close()


# --------------------------------------------------------

# mannual_starting_point = 112

# open_carseqrects = np.load('../code/carseqrects-wcrt.npy')
# rect = open_carseqrects[mannual_starting_point-1,:]
# rect = rect.astype(int)
# --------------------------------------------------------


for i in range(frame_num-1):

    # Print the the frame number in process
    print(i)

    # Load the template and the input images
    It = carseq_data[:,:,i]
    It1 = carseq_data[:,:,i+1]

    # Use Lucas-Kanade algorithm to update the new p
    # p = LucasKanade.LucasKanade(It = It, It1= It1, rect=rect, p0 = np.zeros(2))

    # Track it a first time with Tn
    p1 = LucasKanade.LucasKanade(It = It, It1= It1, rect=rect, p0 = p0)

    # update p0
    p0 = np.zeros(2)
    p0[0] = p1[4]
    p0[1] = p1[5]
    # print(p0)

    # Track it a second time with T0
    It0 = carseq_data[:,:,0]
    rect0 = np.asarray([59, 116, 145, 151])
    rect1 = rect

    # Use either to track the car with template drift correction
    # The first option utilizes the subpixel information
    # The second one rounded off to the nearest pixel
    # ---------------------------------------------------------------------------------------------------
    # p2 = LucasKanade.LucasKanade_template_drift(It0 = It0, It1 = It1, rect0 = rect0, rect1 = rect1, p0 = p0)
    p2 = LucasKanadeTemplateCorrection.LucasKanade_template_correction(It0 = It0, It1 = It1, rect0 = rect0, rect1 = rect1, p0 = p0)

    p0 = np.zeros(2)
    p0[0] = p2[4]
    p0[1] = p2[5]

    # print('------------')
    # print('p2 p1 error')
    # print(np.linalg.norm(p2 - p1))

    if (np.linalg.norm(p2 - p1)) <= epsilon:
        p = p2
        # print('use p2')
    else:
        p = p1
        # print('use p1')

    # print('------------')

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
    rect = np.round(rect)     # Round float number to the closet integer
    rect = rect.astype(int)   # this is the new rect converted to integer

    # put the new values into the carseqrect array we are going to save later
    carseqrects_wcrt[i+1,:] = rect   # remember there should be n (n = 415 in this case) rects

    # ---------------------------------------------------------------
    # save image files when needed
    # if i == 1 or i == 100 or i == 200 or i == 300 or i == 400:
    if i %10 == 0:
        file_name = 'lk_%s.png' % (str(i+1))

        fig,ax = plt.subplots(1)
        ax.imshow(carseq_data[:,:,i])
        rect_patch = patches.Rectangle((rect[0],rect[1]), (rect[2] - rect[0]), (rect[3] - rect[1]),linewidth=1,edgecolor='b',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect_patch)
        plt.savefig(file_name)
        plt.close()
        # incremental save
        np.save('carseqrects-wcrt.npy', carseqrects_wcrt)

    # save rect coordinates in termittently
    # if i%10 == 0:
    #     np.save('../code/carseqrects-wcrt.npy', carseqrects_wcrt)

# save rect coordinate one last time when all the frames are processed
np.save('carseqrects-wcrt.npy', carseqrects_wcrt)

# Just for test. Load the rect coordinate file
# open_carseqrects = np.load('../code/carseqrects-wcrt.npy')
# print(open_carseqrects)

