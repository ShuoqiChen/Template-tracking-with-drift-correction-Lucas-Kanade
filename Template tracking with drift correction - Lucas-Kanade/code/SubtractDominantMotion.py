import numpy as np
import LucasKanadeAffine
import InverseCompositionAffine

from scipy.ndimage import affine_transform

from scipy.ndimage.morphology import binary_dilation

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt



def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    # Use LucasKanadeAffine
    # ============================================================
    # M = LucasKanadeAffine.LucasKanadeAffine(It=image1, It1=image2)
    # threshold = 0.375
    # ============================================================

    # Use Inverse LucasKanadeAffine
    # ============================================================
    M = InverseCompositionAffine.InverseCompositionAffine(It=image1, It1=image2)
    threshold = 0.325
    # ============================================================

    # warp the first image using the p parametrization
    image1_warped = affine_transform(image1 , M)

    # generate the mask by subtracting the pixels
    mask = image2 - image1_warped

    # Set up the threshold, the intensity above the threshold will be regarded as 1 whereas below will be regarded as 0
    mask = mask > threshold
    mask = 1*mask


    # Do something here to get rid of the edges
    # --------------------------------------------------------

    # initialize the mask to be all ones / trues
    cutoff = np.ones(image1.shape, dtype=bool)

    # warp the first image using the p parametrization
    cutoff_warped = affine_transform(cutoff , M)
    cutoff_subtracted = cutoff_warped < 1

    cutoff_subtracted = 1*cutoff_subtracted
    # # Mannual tuning
    cutoff_subtracted[:,0:50] = 1
    cutoff_subtracted[200:, :] = 1


    mask = mask - cutoff_subtracted

    mask = mask > 0
    mask = 1*mask
    # --------------------------------------------------------


    # Use dilation to make the points look bigger
    mask = binary_dilation(mask)

    # matplotlib.pyplot.imshow(cutoff_subtracted)
    # plt.show()

    # ------------------------------------------------
    return mask
