from astropy.io import fits
import glob
from pathlib import Path, PureWindowsPath
import starfinder as sf
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

# Turn off warning in case the DAOStarFinder doesn´t identify any stars that fit our criteria
warnings.filterwarnings('ignore')
#close all opened figure
plt.close("all")

norm = ImageNormalize(stretch=SqrtStretch())

'''''''                    SETUP PARAMETERS                                                                      '''''''
# Path to the the images containing the stars divides by telescope .fits files and conversion to the user OS
stars_files = PureWindowsPath("./stars/images/*.fits")
stars_files = Path(stars_files)
stars_files = glob.glob(str(stars_files))

m1 = [140, 200, 100, 150]
m2 = [150, 200, 50, 100]
m3 = [50, 100, 140, 200]

db1 = [m1, m2]
db2 = [m2, m3]

roi_list = [[], [], db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, [], db1, db1, [], [], db1, db1]
# roi_list = [[], [], db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, [], db2, db2, [], [], db2, db2]
out_list = [4,  5,   3,   7,   5,   5,   5,   5,   3,   3,   3,   7,   2.5,  7, 10,   5,   10,  7,  7,  3,  7]
iso_list = [7,  7,   9,   9,   9,   9,   9,   9,   7,   7,   7,   9,   9,   9,   7,   9,   7,   7,  9,   9,  9]
exp_list = [5,  5,   7,   7,   7,   7,   7,   7,   5,   5,   5,   7,   7,   7,   5,   7,  5,   5,  7,   7,  7]


file_counter = 0

image_list = []

# Loop to access all files in fits folder
for fits_file in stars_files:

    images, name, time, _ = sf.open_file(fits_file)

    # name_list.append

    print(name, "(", file_counter + 1, "/", len(stars_files), ")")
    file_counter = file_counter + 1


    if file_counter != 5:
        continue

    for frames in images:
        image_list.append(frames[0])

    break

    for tele in range(images[0]):
        frame_list = []
        for frame in range(images):
            frame_list.append()

avg_img = sf.join_images(image_list)

_, _, _, std = sf.background_noise(avg_img)
stars = sf.find_stars(avg_img, std, exp_fwhm=7.0, flux_min=5, roi=m1)
brightest = sf.main_stars(stars, 1)

#%%
sf.print_sources(stars)
# Isolate the star
star_pixels = sf.stars_box(avg_img, brightest[0], 9)
# Mask star
masked = sf.mask_roi(star_pixels)
amp_max = np.max(masked)
sf.show_stars(avg_img, amp_max)