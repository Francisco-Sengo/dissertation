from photutils.datasets import load_star_image
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch
import numpy as np
from scipy.spatial.transform import Rotation
import starfinder as sf
from photutils.datasets import make_gaussian_sources_image, make_noise_image, make_4gaussians_image
import cv2
from astropy.table import Table
from photutils.datasets import make_gaussian_sources_image
import random

img_list, homo = sf.generate_data(6, 60, 500, 350)

sf.show_telescopes(img_list)

#%%

final_img = []
# Define the telescope 1 as the reference for the stitching
master_img = img_list[0]
warped_list = [master_img]

# Acquire coordinates from the reference image
master_coords = sf.stars_features(master_img)

# Iteratively finde the homography and warp all the telescope images
for tele in range(1, len(img_list)):

    tele_coords = sf.stars_features(img_list[tele])

    # Match the stars between the two images
    matches, master_match, tele_match = sf.stars_matcher(master_coords, tele_coords, 5)
    print(matches)

    if matches == 0:
        print("NOT ENOUGH MATCHES TO CALCULATE HOMOGRAPHY")

    # Estimate the homography using the cv2 library
    H, _ = cv2.findHomography(tele_match, master_match)

    # Warp image base on the estimated homography
    warped_tele = sf.warp_image(img_list[tele], H)
    # Append the warped image from the telescope into a list
    warped_list.append(warped_tele)

# Stack images to form the master image
final_img = sf.join_images(warped_list)

for img in img_list:
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(img, origin='lower', cmap='inferno',
               interpolation='nearest')
    plt.show()

# _, _, _, std = sf.background_noise(img_list[0])
#
# srcs = sf.find_stars(img_list[0], std)
#
# sf.print_sources(srcs)
#
# _, _, _, std = sf.background_noise(img_list[1])
#
# srcs = sf.find_stars(img_list[1], std)
#
# sf.print_sources(srcs)
#
# _, _, _, std = sf.background_noise(img_list[2])
#
# srcs = sf.find_stars(img_list[2], std)
#
# sf.print_sources(srcs)
#
# _, _, _, std = sf.background_noise(img_list[3])
#
# srcs = sf.find_stars(img_list[3], std)
#
# sf.print_sources(srcs)








