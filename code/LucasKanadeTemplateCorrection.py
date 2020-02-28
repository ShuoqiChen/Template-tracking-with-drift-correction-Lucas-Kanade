import numpy as np

from scipy.interpolate import RectBivariateSpline


import cv2
import io
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import patches as patches


from numpy.linalg import inv


def LucasKanade_template_correction(It0, It1, rect0, rect1, p0):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the car
    #   (top left, bot right coordinates)
    #   p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #   p: movement vector [dp_x, dp_y]
    
    # Put your implementation here
    # ============================================================================================================

    p = p0

    # Set up a error threshold
    threshold = 0.008


    # get the template image
    # Use this:
    # ------------------------------------------------------------
    tmplt0 = It0[rect0[1]:rect0[3]+1, rect0[0]:rect0[2]+1]
    tmplt1 = It1[rect1[1]:rect1[3]+1, rect1[0]:rect1[2]+1]
    # ------------------------------------------------------------

    # Common initialisation
    # P0 starts with all 0s
    tmplt_pts, warp_p = init_affine(p0=p0, rect=rect0)

    # Pre-computation
    # -------

    # Compute image gradients - will warp these images later
    # !!! NOTE: I think we are warping the input image, but I am not 100% sure
    img_dx, img_dy = gradient_affine(It1)

    # Evaluate Jacobian - constant for affine warps
    # print(rect)

    It1_w = np.round(rect1[2] - rect1[0]) + 1
    It1_h = np.round(rect1[3] - rect1[1]) + 1
    dW_dp = jacobian_affine(It1_w, It1_h)

    # Lucas-Kanade, Forwards Additive Algorithm -------------------------------

    # Set a big error to start with
    new_rms_error = 1
    old_rms_error = 1

    N_p = 6

    counter = 0

    while new_rms_error > threshold:

        counter = counter + 1

        # 1) Compute warped image with current parameters
        IWxp = warp_affine(img=It1, p=warp_p, tmplt=tmplt1, rect=rect1)

        # 2) Compute error image
        error_img = tmplt0 - IWxp
            
        # -- Save current fit parameters --
        new_warp_p = warp_p;
        new_rms_error = np.sqrt(np.mean(error_img**2))
        # -- Really iteration 1 is the zeroth, ignore final computation --
        if (counter >= 100):
             break

        if old_rms_error < new_rms_error:
            break
        else:
            old_rms_error = new_rms_error

        # 3b) Evaluate gradient
        nabla_Ix = warp_affine(img=img_dx, p=warp_p, tmplt=tmplt1, rect=rect1)
        nabla_Iy = warp_affine(img=img_dy, p=warp_p, tmplt=tmplt1, rect=rect1)
        
        # 4) Evaluate Jacobian - constant for affine warps. Precomputed above

        # 5) Compute steepest descent images, VI_dW_dp
        VI_dW_dp = steepest_descent_images(dW_dp, nabla_Ix, nabla_Iy, It1_h, It1_w)
        
        # 6) Compute Hessian and inverse
        H = hessian(VI_dW_dp, It1_w)

        # H_inv = np.linalg.inv(H)
        H_inv = np.linalg.pinv(H)

        # 7) Compute steepest descent parameter updates
        sd_delta_p = steepest_descent_update(VI_dW_dp, error_img, It1_w)

        # 8) Compute gradient descent parameter updates
        delta_p = np.matmul(H_inv, sd_delta_p)

        # 9) Update warp parmaters
        warp_p = update_p(warp_p, delta_p)

    return warp_p

def init_affine(p0, rect):

    # Common initialisation things for all affine algorithms


    if p0.shape[0] != 6:                                     # default p has 6 parameters
        if p0.shape[0] != 2:                          # can also accept p that only has 2 parameters
            print('Number of warp parameters incorrect')
        else:
            # print('warning: length of p = 2')
            p_init = np.zeros(6)
            p_init[-2:] = p0
    else:
        p_init = p0

    # Initial warp parameters
    warp_p = p_init

    # make warp_p a column vector
    warp_p = warp_p[:, np.newaxis]


    # Template verticies, rectangular [minX minY; minX maxY; maxX maxY; maxX minY]
    min_X = rect[0]
    min_Y = rect[1]

    max_X = rect[2]
    max_Y = rect[3]

    # the sequences of the four corners are: top_left, bottom left, bottom right, top, right (counter clockwise)
    tmplt_pts = np.asarray([[min_X, min_Y], [min_X, max_Y], [max_X, max_Y], [max_X, min_Y]])


    return tmplt_pts,warp_p

def gradient_affine(img):
    img_dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)   # Have to use cv2.CV_64F for this...
    img_dy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    return img_dx, img_dy

def jacobian_affine(nx, ny):

    # dW_dp will have a size of of 

    jac_x = np.kron([i for i in range(nx)], np.ones((ny, 1)))
    
    arg_1 = np.asarray([i for i in range(ny)])
    arg_1 = arg_1[:, np.newaxis]
    jac_y = np.kron(arg_1, np.ones((1, nx)))

    jac_zero = np.zeros((ny, nx))
    jac_one = np.ones((ny, nx))


    # dW_dp = np.asarray[[jac_x, jac_zero, jac_y, jac_zero, jac_one, jac_zero], [jac_zero, jac_x, jac_zero, jac_y, jac_zero, jac_one]]
    dW_dp_row_1 = np.hstack((jac_x, jac_zero, jac_y, jac_zero, jac_one, jac_zero))
    dW_dp_row_2 = np.hstack((jac_zero, jac_x, jac_zero, jac_y, jac_zero, jac_one))

    dW_dp = np.vstack((dW_dp_row_1, dW_dp_row_2))

    return dW_dp


def warp_affine(img, p, tmplt, rect):

    W = np.zeros((2,3))

    # first row
    W[0][0] = p[0] + 1
    W[0][1] = p[2]
    W[0][2] = p[4]

    # second row
    W[1][0] = p[1]
    W[1][1] = p[3] + 1
    W[1][2] = p[5]


    # Find out the interpolation relationship of all the pixels in the input image
    rect_spline = RectBivariateSpline(np.array(range(0, img.shape[0])), np.array(range(0, img.shape[1])), img)  # img should be the input image
    
    # Template size
    h = tmplt.shape[0]
    w = tmplt.shape[1]

    wimg = np.zeros((h, w))

    for i in range(rect[1], rect[3]):  # for each row
        for j in range(rect[0], rect[2]): # for each col

            xy = np.matmul(W, np.asarray([[j],[i],[1]]))   # the wrapped coordinates

            x = xy[1]  # col index, corresponding to x 
            y = xy[0]  # row index, corresponding to y
    
            wimg[i-rect[1]][j-rect[0]] = rect_spline.ev(x,y)  # interpolate (get the pixel intensity) of the input patch
                                                              # ...after being warped

    return wimg

def steepest_descent_images(dW_dp, nabla_Ix, nabla_Iy, h, w):

    # number of parameters in p should be 6
    N_p = 6

    VI_dW_dp = np.zeros((nabla_Ix.shape[0], nabla_Ix.shape[1]*N_p))

    for p in range(1,7): 

        Tx = np.multiply(nabla_Ix, dW_dp[0:h,((p-1)*w):((p-1)*w)+w])
        Ty = np.multiply(nabla_Iy, dW_dp[h:,((p-1)*w):((p-1)*w)+w])

        VI_dW_dp[:,((p-1)*w):((p-1)*w)+w] = Tx + Ty

    return VI_dW_dp

def hessian(VI_dW_dp, w):

    # default N_p  =6
    N_p = 6

    H = np.zeros((N_p, N_p))


    for i in range(1, N_p+1):

        h1 = VI_dW_dp[:,((i-1)*w):((i-1)*w)+w];

        for j in range(1, N_p+1):
            h2 = VI_dW_dp[:,((j-1)*w):((j-1)*w)+w]
            H[j-1, i-1] = np.sum(np.sum((np.multiply(h1, h2))))


    return H

def steepest_descent_update(VI_dW_dp, error_img, w):

    # default number of parameters in p is 6

    N_p = 6

    sd_delta_p = np.zeros((N_p, 1))


    for p in range(1, N_p+1):
        h1 = VI_dW_dp[:,((p-1)*w):((p-1)*w)+w];
        sd_delta_p[p-1] = np.sum(np.sum(np.multiply(h1, error_img)))


    return sd_delta_p

def update_p(warp_p, delta_p):

    # delta_p_force is here to force the p1 - p4 to be 0 and only keeps p5 and p6
    delta_p_force = np.zeros(delta_p.shape)
    delta_p_force[4] = delta_p[4]
    delta_p_force[5] = delta_p[5]


    # warp_p = warp_p + delta_p
    warp_p = warp_p + delta_p_force

    return warp_p

