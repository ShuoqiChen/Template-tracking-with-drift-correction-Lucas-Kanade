import numpy as np
import cv2   # for sobel filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import matplotlib
matplotlib.use('TkAgg')

global car_seq, bounding_boxes, bounding_boxes_2


def show_animation(i):

    global car_seq, bounding_boxes, bounding_boxes_2

    frame = car_seq[:, :, i]
    rect = bounding_boxes[i]
    x, y, height, width = get_rect_params(rect)

    ax1.set_data(frame)
    box.set_xy((x, y))
    box.set_width(width)
    box.set_height(height)

    rect_2 = bounding_boxes_2[i]

    x_2, y_2, height_2, width_2 = get_rect_params(rect_2)

    box_2.set_xy((x_2, y_2))
    box_2.set_width(width_2)
    box_2.set_height(height_2)

    
    return [ax1, box, box_2]

def get_rect_params(rect):

    x = rect[0]
    y = rect[1]
    height = rect[3] - rect[1]
    width = rect[2] - rect[0]

    return x, y, height, width

# Load data
car_seq = np.load('../data/carseq.npy')

# import bounding boxes data
bounding_boxes = np.load('../data/carseqrects.npy')
bounding_boxes_2 = np.load('../data/carseqrects-wcrt.npy')

# set bouding boxes
frame = 400
rect = bounding_boxes_2[frame]
x, y, height, width = get_rect_params(rect)
rect_2 = bounding_boxes[frame]
x_2, y_2, height_2, width_2 = get_rect_params(rect_2)

# Prepare plot
fig,ax = plt.subplots(1)
ax1 = ax.imshow(car_seq[:,:,frame])
box = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(box)

box_2 = patches.Rectangle((x_2, y_2), width_2, height_2, linewidth=1, edgecolor='b', facecolor='none')
ax.add_patch(box_2)

# Show animation
# ani = animation.FuncAnimation(fig, show_animation, frames=car_seq.shape[2], blit=True, repeat=False, interval=20)
plt.show()

