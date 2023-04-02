from astropy.io import fits
import glob
from pathlib import Path, PureWindowsPath
import starfinder as sf
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Turn off warning in case the DAOStarFinder doesn´t identify any stars that fit our criteria
warnings.filterwarnings('ignore')
#close all opened figure
plt.close("all")

'''''''                    SETUP PARAMETERS                                                                      '''''''
# Path to the the images containing the stars divides by telescope .fits files and conversion to the user OS
stars_files = PureWindowsPath("./stars/images/*.fits")
stars_files = Path(stars_files)
stars_files = glob.glob(str(stars_files))

# Number of expecteded stars for image assessment
num_stars = 2

m1 = [140, 200, 100, 150]
m2 = [150, 200, 50, 100]
m3 = [50, 100, 140, 200]

db1 = [m1, m2]
db2 = [m2, m3]

#            1   2   3    4    5    6    7    8    9    10   11   12   13   14  15   16   17  18  19   20   21
roi_list = [[], [], db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, [], db1, db1, [], [], db1, db1]
# roi_list = [[], [], db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, [], db2, db2, [], [], db2, db2]
out_list = [4,  5,   3,   7,   5,   5,   5,   5,   3,   3,   3,   7,   2.5,  7, 10,   5,   10,  7,  7,  3,  7]
iso_list = [7,  7,   9,   9,   9,   9,   9,   9,   7,   7,   7,   9,   9,   9,   7,   9,   7,   7,  9,   9,  9]
exp_list = [5,  5,   7,   7,   7,   7,   7,   7,   5,   5,   5,   7,   7,   7,   5,   7,  5,   5,  7,   7,  7]
'''''''                    OTHER SETUPS                                                                          '''''''
# todo 10, 11, 20

file_counter = 0
# Name list for plots
name_list = []
distances = []
stds = []


'''''''-------------------STITCHING ALGORITHM--------------------------------------------------------------------'''''''
# Loop to access all files in fits folder
for fits_file in stars_files:

    images, name, time, _ = sf.open_file(fits_file)

    # name_list.append

    print(name, "(", file_counter + 1, "/", len(stars_files), ")")
    file_counter = file_counter + 1

    if file_counter != 10 and file_counter != 11 and file_counter != 20:
        continue

    # distance_list, std_list = sf.file_distances(images, mask=True, outlier=5, iso_box=7, exp_fwhm=7, flux_min=5,
    #                                             filter=False, roi_list=roi_list[file_counter-1])

    distance_list, std_list = sf.file_distances(images, mask=True, outlier=out_list[file_counter-1],
                                                iso_box=iso_list[file_counter-1],
                                                exp_fwhm=exp_list[file_counter-1], flux_min=0,
                                                filter=False, roi_list=roi_list[file_counter-1])

    sf.plt_file_dist(distance_list, name=name, exp_time=time, error=std_list, ext='dist', ylabel='Distance',
                      xlabel='Time (s)', save=True, folder_path='./stars/dist_img/')

    # for tele in range(len(distance_list)):
    #     sf.plt_tele_stat(distance_list[tele], time, error=std_list[tele], name=name, telescope=tele+1,
    #                      ylabel='Distance', ext='2s',
    #                      save=True, folder_path='./stars/dist2/')

    # break

print("____________________________________________FINISHED___________________________________________________________")
