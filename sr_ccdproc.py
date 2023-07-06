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
import astroalign as aa
import copy
import cv2

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
m4 = [110, 140, 60, 90]

db1 = [m1, m2]
db2 = [m2, m3]
db3 = [m1, m4]

roi_list = [db3, db3, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db3, db1, db1, db3, db3, db1, db1]
roi_list2 = [db3, db3, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db3, db2, db2, db3, db3, db2, db2]



list_roi = [roi_list, roi_list2]

# roi_list = [[], [], db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, [], db2, db2, [], [], db2, db2]
#                 1   2    3    4    5    6    7    8    9   10   11   12   13   14   15   16  17   18  19   20  21
out_list =       [4,  5,   3,   7,   5,   5,   5,   5,   3,   3,   3,   7,   2.5, 7,  10,   5, 10,   7,  7,   3,  7]
iso_list =       [9,  9,   9,   9,   7,   9,   7,   7,   7,   9,   9,   9,   9,   9,   11,   9,  7,   7,  9,   9,  9]
exp_list =       [5,  5,   5,   7,   5,   7,   7,   7,   5,   7,   5,   7,   7,   7,   5,   7,  5,   5,  7,   7,  7]
ana_iso_list =   [7,  7,  11,   9,   7,   9,   9,   9,   7,   9,   9,   9,   9,   9,  7,   9,  7,   7,  9,   9,  9]
th_list =        [2,  2,   3,   3,   3,   3,   3,   3,   3,   2,   2,   3,   3,   3,   2,   3,  3,   2,  3,   3,  3]
sig_list =       [7,  7,   7,   7,   5,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,  7,   7,  7,   7,  7]
ksize_list =     [5,  5,   5,   5,   3,   5,   5,   5,   5,   5,   5,   5,   5,   5,   7,   5,  5,   5,  5,   5,  5]
#                 1   2    3    4    5    6    7    8    9   10   11   12   13   14   15   16  17   18  19   20  21

file_counter = 0
num = 6

image_list = []
# Name list for plots
name_list = []
name2_list = []
# Name list for plots
centroid_x_list = []
centroid_y_list = []
sigma_list = []
amp_list = []
dist_list = []
dist2_list = []
ang_list = []
ang2_list = []

# Loop to access all files in fits folder
for fits_file in stars_files:

    images, name, time, wcs = sf.open_file(fits_file)

    # name_list.append

    print(name, "(", file_counter + 1, "/", len(stars_files), ")")
    file_counter = file_counter + 1


    # if file_counter < 15:
    #     continue

    imgs = sf.ccdproc_sr(images, wcs)
    final_imgs = [imgs[0].data, imgs[1].data, imgs[2].data, imgs[3].data]

    print("Single Telescope Fusion Complete")

    stat_list = []
    ####################################################################################################################
    for tele in range(len(final_imgs)):
        print("-Telescope #", tele + 1)
        tele_stats = []
        _, _, _, std = sf.background_noise(final_imgs[tele])
        stars = sf.find_stars(final_imgs[tele], std, exp_fwhm=5, flux_min=0, roi=m1,
                              th=th_list[file_counter - 1], peak_min=0)
        brightest = sf.main_stars(stars, 1)

        sigma, star_max, val_max, centroid = sf.star_stats_2(final_imgs[tele], brightest[0], std,
                                                             iso_box=ana_iso_list[file_counter - 1], plt_gauss=True)
        sf.show_stars(final_imgs[tele], val_max, name=name, ext='tele' + str(tele), save=True,
                      folder_path='./stars/sr/ccdproc/')
        distance_list = []
        angle_list = []
        for rois in list_roi:
            star_list = []
            for roi in rois[file_counter - 1]:
                stars = sf.find_stars(final_imgs[tele], std, exp_fwhm=5,
                                      flux_min=0, roi=roi, th=th_list[file_counter - 1], peak_min=0)
                star_list.append(sf.main_stars(stars, 1)[0])

            distance = sf.star_distance_2(final_imgs[tele], star_list, std=std,
                                          iso_box=ana_iso_list[file_counter - 1], error=1)

            distance_list.append(distance[0])
            angle_list.append(distance[1])
        if tele == 0:
            amp_max = val_max
        stat_list.append([tele + 1, centroid[0], centroid[1], sigma, star_max, distance_list[0], angle_list[0],
                          distance_list[1], angle_list[1]])
    print(stat_list[0])
    print(stat_list[1])
    print(stat_list[2])
    print(stat_list[3])
    sf.generate_stats_table(stat_list, name=name, ext='Table', folder_path='./stars/sr/ccdproc/')

    # sf.show_telescopes(final_imgs, amp_max, name, save=True, folder_path='./stars/sr/ccdproc/')

    print("All Telescope Features Acquired")
    print("---------------------------------------------")
    ####################################################################################################################
    master_img = sf.final_stitcher(final_imgs, distance=5, exp_fwhm=5, flux_min=0,
                                   th=th_list[file_counter - 1])
    print("Multi Telescope Fusion Complete")
    # sf.show_stars(master_img, np.max(master_img))
    _, _, _, master_std = sf.background_noise(master_img)
    master_stars = sf.find_stars(master_img, master_std, exp_fwhm=5, flux_min=0, roi=m1,
                                 th=th_list[file_counter - 1], peak_min=0)
    # master_pixels = sf.stars_box(master_img, brightest[0], 15)
    # sf.show_stars(master_pixels, np.max(master_pixels))
    # sf.print_sources(master_stars)
    brightest = sf.main_stars(master_stars, 1)
    master_sigma, master_star_max, val_max, master_centroid = sf.star_stats_2(master_img, brightest[0], master_std,
                                                                              iso_box=ana_iso_list[file_counter - 1],
                                                                              plt_gauss=False)
    master_distance_list = []
    master_angle_list = []
    for rois in list_roi:
        star_list = []
        for roi in rois[file_counter - 1]:
            stars = sf.find_stars(master_img, master_std, exp_fwhm=5,
                                  flux_min=0, roi=roi, th=th_list[file_counter - 1], peak_min=0)
            star_list.append(sf.main_stars(stars, 1)[0])

        master_distance = sf.star_distance_2(master_img, star_list, std=master_std,
                                             iso_box=ana_iso_list[file_counter - 1], error=1)

        master_distance_list.append(master_distance[0])
        master_angle_list.append(master_distance[1])

    centroid_x_list.append(master_centroid[0])
    centroid_y_list.append(master_centroid[1])
    sigma_list.append(master_sigma)
    amp_list.append(master_star_max)
    dist_list.append(master_distance_list[0])
    ang_list.append(master_angle_list[0])

    if len(images) != 94:
        name2_list.append(name[14:22])
        dist2_list.append(master_distance_list[1])
        ang2_list.append(master_angle_list[1])

    name_list.append(name[14:22])

    # master_star_pixels = sf.stars_box(master_img, brightest[0], 15)
    # sf.show_stars(master_star_pixels, val_max, amp_min=np.min(master_star_pixels), name=name, ext='final_b', save=True,
    #               folder_path='./stars/sr/ccdproc/')
    sf.show_stars(master_img, val_max, amp_min=0, name=name, ext='master', save=True,
                  folder_path='./stars/sr/ccdproc/')
    # sf.show_stars(final_imgs[0], val_max, name=name, ext='tele1', save=True,
    #               folder_path='./stars/sr/ccdproc/')
    print(master_centroid[0], master_centroid[1], master_sigma, master_star_max, master_distance_list[0],
          master_angle_list[0], master_distance_list[1], master_angle_list[1])
    print("All Telescope Features Acquired")
    print("---------------------------------------------")

####################################################################################################################
# %%
sf.print_fwhm(centroid_x_list, name_list, save=True, name='lucky_w', ext='cx_files', title='Centroid x',
              folder_path='./stars/sr/ccdproc/',
              xlabel='files', ylabel='x centroid', )
sf.print_fwhm(centroid_y_list, name_list, save=True, name='lucky_w', ext='cy_files', title='Centroid y',
              folder_path='./stars/sr/ccdproc/',
              xlabel='files', ylabel='y centroid', )
sf.print_fwhm(sigma_list, name_list, save=True, name='lucky_w', ext='sigma_files', title='Sigma',
              folder_path='./stars/sr/ccdproc/',
              xlabel='files', ylabel='sigma', )
sf.print_fwhm(amp_list, name_list, save=True, name='lucky_w', ext='amp_files', title='Maximum Amplitude',
              folder_path='./stars/sr/ccdproc/',
              xlabel='files', ylabel='maximum amplitude', )
sf.print_fwhm(dist_list, name_list, save=True, name='lucky_w', ext='dist_files', title='Distance Two Brightest Stars',
              folder_path='./stars/sr/ccdproc/',
              xlabel='files', ylabel='distance', )
sf.print_fwhm(ang_list, name_list, save=True, name='lucky_w', ext='ang_files', title='Angle Two Brightest Stars',
              folder_path='./stars/sr/ccdproc/',
              xlabel='files', ylabel='angle', )
sf.print_fwhm(dist2_list, name2_list, save=True, name='lucky_w', ext='dist_files',
              title='Distance Second and Third Brightest',
              folder_path='./stars/sr/ccdproc/',
              xlabel='files', ylabel='distance', )
sf.print_fwhm(ang2_list, name2_list, save=True, name='lucky_w', ext='ang_files',
              title='Angle Second and Third Brightest',
              folder_path='./stars/sr/ccdproc/',
              xlabel='files', ylabel='angle', )

print("----------------------------------FINISHED----------------------------------")

