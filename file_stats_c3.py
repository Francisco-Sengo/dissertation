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
from astropy.table import Table

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
columns_to_remove = ['sharpness', 'roundness1', 'roundness2', 'sky', 'npix']

image_list = []

# Loop to access all files in fits folder
for fits_file in stars_files:

    images, name, time, _ = sf.open_file(fits_file)

    # name_list.append

    print(name, "(", file_counter + 1, "/", len(stars_files), ")")
    print(images.shape)
    file_counter = file_counter + 1


    if file_counter != 5:
        continue

    _, _, _, std = sf.background_noise(images[40][0])
    stars = sf.find_stars(images[40][3], std, exp_fwhm=7, flux_min=0, peak_min=10, roi=[], th=3)
    # sf.print_sources(stars)
    brightest = sf.main_stars(stars, 1)
    # Isolate the star
    star_pixels = sf.stars_box(images[40][0], brightest[0], 9)
    # Mask star
    masked = sf.mask_roi(star_pixels)
    amp_max = np.max(masked)
    sf.show_apertures(images[40][0], stars, amp_max, amp_min=0, name=name, ext='aprt_frame41', save=True, folder_path='./stars/')
    #
    # for frames in images:
    #     image_list.append(frames[0])

    flux_sorted_indices = np.argsort(stars['flux'])[::-1][:15]

    sorted_sources = stars[flux_sorted_indices]
    new_table = sorted_sources[[col for col in sorted_sources.colnames if col not in columns_to_remove]]
    sf.print_sources(new_table)

    new_table.write('stars_table.tex', format='ascii.latex', formats={'X Position': '%.2f', 'Y Position': '%.2f'},
                        latexdict={'preamble': '\\begin{center}', 'tablefoot': '\\end{center}'},
                        caption='Table of identified stars.', names=new_table.colnames, overwrite=True)

    break