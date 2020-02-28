import numpy as np
from scipy.interpolate import RectBivariateSpline

from scipy.ndimage import affine_transform

import cv2

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import patches as patches

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here

    p0 = np.zeros(6)

    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


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
    threshold = 0.02

    # get image width and height in pixel level
    # w = np.round(rect[2] - rect[0])
    # h = np.round(rect[3] - rect[1])

    # get the template image
    # Use this:
    # ------------------------------------------------------------
    # tmplt = It
    # ------------------------------------------------------------
    # Or this:
    # ------------------------------------------------------------
    # # linearly interpolate the whole image
    # tmplt_spline = RectBivariateSpline(np.array(range(0, It.shape[0])), np.array(range(0, It.shape[1])), It)  # img should be the template image
    
    # # create x and y coordinates in subpixel level
    # x_vct = np.linspace(rect[0], rect[2], num=w)
    # y_vct = range(rect[1], rect[3], num=h)

    # # create the x and y coordinates of the in matrix form
    # x_mtx, y_mtx = np.meshgrid(x_vct, y_vct)  # xx, yy sequence is delibrately shifted

    # # get the interpolated image intensity values from the xy coordinates in subpixel level
    # tmplt = tmplt_spline.ev(x_mtx,y_mtx)  # interpolate (get the pixel intensity) of the input patch
    #                                                           # ...after being warped
    # ------------------------------------------------------------

    # x = xy[1]  # col index, corresponding to x 
    # y = xy[0]  # row index, corresponding to y


    # # This is really 
    # h = tmplt.shape[0]
    # w = tmplt.shape[1]


    # Common initialisation
    # P0 starts with all 0s

    tmplt_pts, warp_p = init_affine(p0=p0)

    # Pre-computation
    # -------

    # Compute image gradients - will warp these images later
    # !!! NOTE: I think we are warping the input image, but I am not 100% sure
    img_dx, img_dy = gradient_affine(It1)

    # Evaluate Jacobian - constant for affine warps
    # print(rect)
    w = It1.shape[1]
    h = It1.shape[0]
    dW_dp = jacobian_affine(w, h)

    # Lucas-Kanade, Forwards Additive Algorithm -------------------------------

    # Set a big error to start with
    new_rms_error = 1
    old_rms_error = 1

    N_p = 6
    counter = 0

    # Find out the interpolation relationship of all the pixels in the input image
    It1_spline = RectBivariateSpline(np.array(range(0, It1.shape[0])), np.array(range(0, It1.shape[1])), It1)  # img should be the input image

    while new_rms_error > threshold:

        counter = counter + 1
        # print(counter)

    #     # 1) Compute warped image with current parameters
        M = warp_affine(img=It1, p=warp_p)

        IWxp = affine_transform(It1, M)

        mask = np.ones(It1.shape)
        mask_warped = affine_transform(mask, M)

        # Get the template intensities that overlap with that of input image's
        tmplt = It * mask_warped

    #     # 2) Compute error image
        error_img = tmplt - IWxp
            
    #     # -- Save current fit parameters --
        new_warp_p = warp_p;
        new_rms_error = np.sqrt(np.mean(error_img**2))

        # print(new_rms_error)

    #     # -- Really iteration 1 is the zeroth, ignore final computation --
        if (counter >= 25):
             break

        nabla_Ix = affine_transform(img_dx, M)
        nabla_Iy = affine_transform(img_dy, M)
        
    #     # 4) Evaluate Jacobian - constant for affine warps. Precomputed above

    #     # 5) Compute steepest descent images, VI_dW_dp



        VI_dW_dp = steepest_descent_images(dW_dp, nabla_Ix, nabla_Iy, h, w)
        
    #     # 6) Compute Hessian and inverse
        H = hessian(VI_dW_dp, w)

        # H_inv = np.linalg.inv(H)
        H_inv = np.linalg.pinv(H)

    #     # 7) Compute steepest descent parameter updates
        sd_delta_p = steepest_descent_update(VI_dW_dp, error_img, w)

    #     # 8) Compute gradient descent parameter updates
        delta_p = np.matmul(H_inv, sd_delta_p)

    #     # 9) Update warp parmaters
        warp_p = update_p(warp_p, delta_p)


    return M

def init_affine(p0):

    # Common initialisation things for all affine algorithms

    if p0.shape[0] != 6:                                     # default p has 6 parameters
        if p0.shape[0] != 2:                          # can also accept p that only has 2 parameters
            print('Number of warp parameters incorrect')
        else:
            print('warning: length of p = 2')
            p_init = np.zeros(6)
            p_init[-2:] = p0
    else:
        p_init = p0

    # Initial warp parameters
    warp_p = p_init

    # make warp_p a column vector
    warp_p = warp_p[:, np.newaxis]

    tmplt_pts = 0

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


def warp_affine(img, p):

    # WARP_A - Affine warp the image
    #   WIMG = WARP_A(IMG, P, DST)
    #   Warp image IMG to WIMG. DST are the destination points, i.e. the corners
    #   of the template image. P are the affine warp parameters that project
    #   DST into IMG.

    #   P = [p1, p3, p5
    #        p2, p4, p6];

    # % Convert affine warp parameters into 3 x 3 warp matrix
    # % NB affine parameterised as [1 + p1, p3, p5; p2, 1 + p4, p6]

    M = np.zeros((3,3))

    # first row
    M[0][0] = p[0] + 1
    M[0][1] = p[2]
    M[0][2] = p[4]

    # second row
    M[1][0] = p[1]
    M[1][1] = p[3] + 1
    M[1][2] = p[5]

    M[-1,:] = np.asarray([0,0,1])


    return M

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

    warp_p = warp_p + delta_p

    return warp_p

