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

m1 = [150, 170, 120, 140]
m2 = [165, 200, 50, 100]
m3 = [50, 100, 140, 200]
m4 = [110, 140, 60, 90]

db1 = [m1, m2]
db2 = [m2, m3]
db3 = [m1, m4]

roi_list = [db3, db3, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db1, db3, db1, db1, db3, db3, db1, db1]
roi_list2 = [db3, db3, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db2, db3, db2, db2, db3, db3, db2, db2]



list_roi = [roi_list, roi_list2]

"""
NOT WEIGHTED
"""
# #                 1   2    3    4    5    6    7    8    9   10   11   12   13   14   15   16  17   18  19   20  21
out_list =       [4,  5,   3,   7,   5,   5,   5,   5,   3,   3,   3,   7,   2.5,  7,  10,   5,  10,   7,  7,   3,  7]
iso_list =       [9,  9,   9,   9,   8,   9,   7,   7,   7,   9,   9,   9,   9,    9,   11,  9,  7,   7,  9,   9,  9]
exp_list =       [5,  5,   5,   7,   5,   7,   7,   7,   5,   7,   5,   7,   5,    7,   5,  7,   5,   5,  7,   7,  7]
ana_iso_list =   [6,  6,  6,   6,   7,   6,   7,   7,   7,   7,   6,   8,   7,     9,  7,   8,   7,   7,  7,   5,  6]
th_list =        [2,  2,   3,   3,   3,   3,   3,   3,   3,   2,   2,   3,   2.5,  3,   2,   3,  3,   2,  3,   3,  3]
sig_list =       [7,  7,   7,   7,   5,   7,   7,   7,   7,   7,   7,   7,   7,    7,   7,   7,  7,   7,  7,   7,  7]
ksize_list =     [5,  5,   5,   5,   3,   5,   5,   5,   5,   5,   5,   5,   5,    5,   7,   5,  5,   5,  5,   5,  5]


"""
WEIGHTED
"""
#                 1   2    3    4    5    6    7    8    9   10   11   12   13   14   15   16  17   18  19   20  21
# out_list =       [4,  5,   3,   7,   5,   5,   5,   5,   3,   3,   3,   7,   2.5, 7,  10,   5, 10,   7,  7,   3,  7]
# iso_list =       [7,  7,   9,   7,   7,   9,   7,   7,   7,   9,   9,   9,   9,   9,   11,   9,  7,   7,  9,   9,  9]
# exp_list =       [7,  5,   5,   7,   5,   7,   7,   7,   5,   7,   5,   7,   7,   7,   5,   7,  5,   5,  7,   7,  7]
# ana_iso_list =   [7,  9,  11,   9,   7,   9,   9,   9,   7,   9,   9,   9,   9,   9,  11,   9,  7,   7,  9,   9,  9]
# th_list =        [2,  2,   3,   3,   3,   3,   3,   3,   3,   2,   2,   3,   3,   3,   3,   3,  3,   2,  3,   3,  3]
# sig_list =       [13,  7,   7,   7,   5,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,  7,   7,  7,   7,  7]
# ksize_list =     [13,  5,   5,   5,   3,   5,   5,   5,   5,   5,   5,   5,   5,   5,   7,   5,  5,   5,  5,   5,  5]
#                 1   2    3    4    5    6    7    8    9   10   11   12   13   14   15   16  17   18  19   20  21

file_counter = 0
num = 6


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

sigma_list_s = []
amp_list_s = []
dist_list_s = []
dist2_list_s = []
ang_list_s = []
ang2_list_s = []
x_err_list = []
y_err_list = []

# Loop to access all files in fits folder
for fits_file in stars_files:

    images, name, time, wcs = sf.open_file(fits_file)

    # name_list.append

    print(name, "(", file_counter + 1, "/", len(stars_files), ")")
    file_counter = file_counter + 1

    if file_counter < 3 or file_counter > 12:
        continue

    # if file_counter < 11:
    #     continue
    final_imgs = []
    for tele in range(len(images[0])):
        frame_list = []
        for frame in range(len(images)):
            frame_list.append(images[frame][tele])
        final_imgs.append(sf.drizzle_stitcher(frame_list, wcs, time, distance=5, exp_fwhm=exp_list[file_counter - 1],
                                              flux_min=0,
                                              peak_min=0, th=th_list[file_counter - 1], kernel='lanczos3', pixfrac=0.7,
                                              max_features=10))



    print("Single Telescope Fusion Complete")

    # %%
    # _, _, _, std = sf.background_noise(final_imgs[0])
    # stars = sf.find_stars(final_imgs[0], std, exp_fwhm=exp_list[file_counter - 1], flux_min=0, roi=m1, th=th_list[file_counter - 1])
    #
    # brightest = sf.main_stars(stars, 1)
    # # Isolate the star
    # star_pixels = sf.stars_box(final_imgs[0], brightest[0], 9)
    # # Mask star
    # masked = sf.mask_roi(star_pixels)
    # amp_max = np.max(masked)
    # stat_list = []
    # err_list = []
    # # ####################################################################################################################
    # for tele in range(len(final_imgs)):
    #     print("-Telescope #", tele + 1)
    #     tele_stats = []
    #     _, _, _, std = sf.background_noise(final_imgs[tele])
    #     stars = sf.find_stars(final_imgs[tele], std, exp_fwhm=6, flux_min=0, roi=m1,
    #                           th=th_list[file_counter - 1], peak_min=0)
    #     brightest = sf.main_stars(stars, 1)
    #
    #     sigma, star_max, val_max, centroid, sigma_err_, amp_err_ = sf.star_stats(final_imgs[tele], brightest[0], 0,
    #                                                          iso_box=ana_iso_list[file_counter - 1], plt_gauss=False)
    #
    #     distance_list = []
    #     angle_list = []
    #     error_list = []
    #     for rois in list_roi:
    #         star_list = []
    #         #     if len(rois[file_counter - 1]) != 2:
    #         #         stars = sf.find_stars(final_imgs[tele], std, exp_fwhm=5,
    #         #                               flux_min=0, th=th_list[file_counter-1], peak_min=0)
    #         #         star_list = sf.main_stars(stars, 2)
    #         # # Acquire sources in the images
    #         #     else:
    #         for roi in rois[file_counter - 1]:
    #             stars = sf.find_stars(final_imgs[tele], std, exp_fwhm=6,
    #                                   flux_min=0, roi=roi, th=th_list[file_counter - 1], peak_min=0)
    #             star_list.append(sf.main_stars(stars, 1)[0])
    #
    #         distance = sf.star_distance(final_imgs[tele], star_list, std=0,
    #                                     iso_box=ana_iso_list[file_counter - 1], error=0)
    #
    #         distance_list.append(distance[0])
    #         angle_list.append(distance[1])
    #         error_list.append([distance[2], distance[3]])
    #     if tele == 0:
    #         amp_max = val_max
    #     stat_list.append([tele + 1, centroid[0], centroid[1], sigma, star_max, distance_list[0], angle_list[0],
    #                       distance_list[1], angle_list[1]])
    #     err_list.append([tele + 1, sigma_err_, amp_err_, error_list])
    #
    #     sf.show_stars(final_imgs[tele], amp_max, name=name, ext='tele' + str(tele), save=True,
    #                   folder_path='./stars/imgs/drizzle_/')
    #
    # print(stat_list[0])
    # print(stat_list[1])
    # print(stat_list[2])
    # print(stat_list[3])
    # sf.generate_stats_table(stat_list, name=name, ext='Table', folder_path='./stars/imgs/drizzle_/')
    # if file_counter == 5:
    #     print(err_list[0])
    #     print(err_list[1])
    #     print(err_list[2])
    #     print(err_list[3])

    # sf.show_telescopes(final_imgs, amp_max, name, save=True, folder_path='./stars/imgs/drizzle_/')

    print("All Telescope Features Acquired")
#     print("---------------------------------------------")
#     ####################################################################################################################
#     master_img = sf.final_stitcher(final_imgs, distance=5, exp_fwhm=5, flux_min=0,
#                                    th=th_list[file_counter - 1])
#     print("Multi Telescope Fusion Complete")
#     # sf.show_stars(master_img, np.max(master_img))
#     _, _, _, master_std = sf.background_noise(master_img)
#     master_stars = sf.find_stars(master_img, master_std, exp_fwhm=5, flux_min=0, roi=m1,
#                                  th=th_list[file_counter - 1], peak_min=0)
#     # master_pixels = sf.stars_box(master_img, brightest[0], 15)
#     # sf.show_stars(master_pixels, np.max(master_pixels))
#     # sf.print_sources(master_stars)
#     brightest = sf.main_stars(master_stars, 1)
#     master_sigma, master_star_max, val_max, master_centroid = sf.star_stats_2(master_img, brightest[0], master_std,
#                                                                               iso_box=ana_iso_list[file_counter - 1],
#                                                                               plt_gauss=True)
#     master_distance_list = []
#     master_angle_list = []
#     for rois in list_roi:
#         star_list = []
#         # if len(images[0]) != 94:
#         #     stars = sf.find_stars(master_img, master_std, exp_fwhm=5,
#         #                           flux_min=0, th=th_list[file_counter - 1], peak_min=0)
#         #     star_list = sf.main_stars(stars, 2)
#         # # Acquire sources in the images
#         # else:
#         for roi in rois[file_counter - 1]:
#             stars = sf.find_stars(master_img, master_std, exp_fwhm=5,
#                                   flux_min=0, roi=roi, th=th_list[file_counter - 1], peak_min=0)
#             star_list.append(sf.main_stars(stars, 1)[0])
#
#         master_distance = sf.star_distance_2(master_img, star_list, std=master_std,
#                                              iso_box=iso_list[file_counter - 1], error=1)
#
#         master_distance_list.append(master_distance[0])
#         master_angle_list.append(master_distance[1])
#
#     centroid_x_list.append(master_centroid[0])
#     centroid_y_list.append(master_centroid[1])
#     sigma_list.append(master_sigma)
#     amp_list.append(master_star_max)
#     dist_list.append(master_distance_list[0])
#     ang_list.append(master_angle_list[0])
#
#     if len(images[0]) != 94:
#         name2_list.append(name[14:22])
#         dist2_list.append(master_distance_list[1])
#         ang2_list.append(master_angle_list[1])
#
#     name_list.append(name[14:22])
#
#     # master_star_pixels = sf.stars_box(master_img, brightest[0], 15)
#     # sf.show_stars(master_star_pixels, val_max, amp_min=np.min(master_star_pixels), name=name, ext='final_b', save=True,
#     #               folder_path='./stars/imgs/drizzle_/')
#     sf.show_stars(master_img, val_max, amp_min=0, name=name, ext='master', save=True,
#                   folder_path='./stars/imgs/drizzle_/')
#     # sf.show_stars(final_imgs[0], val_max, name=name, ext='tele1', save=True,
#     #               folder_path='./stars/imgs/drizzle_/')
#     print(master_centroid[0], master_centroid[1], master_sigma, master_star_max, master_distance_list[0],
#           master_angle_list[0], master_distance_list[1], master_angle_list[1])
#     print("All Telescope Features Acquired")
#     print("---------------------------------------------")
#
# ####################################################################################################################
# # %%
# sf.print_fwhm(centroid_x_list, name_list, save=True, name='lucky_w', ext='cx_files_', title='Centroid x',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='x centroid', )
# sf.print_fwhm(centroid_y_list, name_list, save=True, name='lucky_w', ext='cy_files_', title='Centroid y',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='y centroid', )
# sf.print_fwhm(sigma_list, name_list, save=True, name='lucky_w', ext='sigma_files_', title='Sigma',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='sigma', )
# sf.print_fwhm(amp_list, name_list, save=True, name='lucky_w', ext='amp_files_', title='Maximum Amplitude',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='maximum amplitude', )
# sf.print_fwhm(dist_list, name_list, save=True, name='lucky_w', ext='dist_files_', title='Distance Two Brightest Stars',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='distance', )
# sf.print_fwhm(ang_list, name_list, save=True, name='lucky_w', ext='ang_files_', title='Angle Two Brightest Stars',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='angle', )
# sf.print_fwhm(dist2_list, name2_list, save=True, name='lucky_w', ext='dist_files_',
#               title='Distance Second and Third Brightest',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='distance', )
# sf.print_fwhm(ang2_list, name2_list, save=True, name='lucky_w', ext='ang_files_',
#               title='Angle Second and Third Brightest',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='angle', )
#
# print("----------------------------------FINISHED----------------------------------")

    print("All Telescope Features Acquired")
    print("---------------------------------------------")
    ####################################################################################################################
    master_img = sf.final_stitcher(final_imgs, distance=7, exp_fwhm=7, flux_min=0,
                                   th=th_list[file_counter - 1], peak_min=0, max_features=7)
    print("Multi Telescope Fusion Complete")
    # sf.show_stars(master_img, np.max(master_img))
    _, _, _, master_std = sf.background_noise(master_img)
    master_stars = sf.find_stars(master_img, master_std, exp_fwhm=5, flux_min=0, roi=m1,
                                 th=th_list[file_counter - 1], peak_min=0)
    # master_pixels = sf.stars_box(master_img, brightest[0], 15)
    # sf.show_stars(master_pixels, np.max(master_pixels))
    # sf.print_sources(master_stars)
    brightest = sf.main_stars(master_stars, 1)[0]
#     master_sigma, master_star_max, val_max, master_centroid, sigma_err, amp_err = sf.star_stats(master_img, brightest[0], 0,
#                                                                               iso_box=ana_iso_list[file_counter - 1],
#                                                                               plt_gauss=True)
#     master_distance_list = []
#     master_angle_list = []
#     master_distance_list_s = []
#     master_angle_list_s = []
#     for rois in list_roi:
#         star_list = []
#         # if len(images[0]) != 94:
#         #     stars = sf.find_stars(master_img, master_std, exp_fwhm=5,
#         #                           flux_min=0, th=th_list[file_counter - 1], peak_min=0)
#         #     star_list = sf.main_stars(stars, 2)
#         # # Acquire sources in the images
#         # else:
#         for roi in rois[file_counter - 1]:
#             stars = sf.find_stars(master_img, master_std, exp_fwhm=5,
#                                   flux_min=0, roi=roi, th=th_list[file_counter - 1], peak_min=0)
#             star_list.append(sf.main_stars(stars, 1)[0])
#
#         master_distance = sf.star_distance(master_img, star_list, std=0,
#                                              iso_box=ana_iso_list[file_counter - 1])
#
#         master_distance_list.append(master_distance[0])
#         master_angle_list.append(master_distance[1])
#         master_distance_list_s.append(master_distance[2])
#         master_angle_list_s.append(master_distance[3])
#
#
#     sigma_list.append(master_sigma)
#     amp_list.append(master_star_max)
#     sigma_list_s.append(sigma_err)
#     amp_list_s.append(amp_err)
#
#
#     if file_counter >= 3 and file_counter <= 12:
#         centroid_x_list.append(master_centroid[0])
#         centroid_y_list.append(master_centroid[1])
#         dist_list.append(master_distance_list[0])
#         ang_list.append(master_angle_list[0])
#         dist_list_s.append(master_distance_list_s[0])
#         ang_list_s.append(master_angle_list_s[0])
#         name2_list.append(name[14:22])
#         dist2_list.append(master_distance_list[1])
#         ang2_list.append(master_angle_list[1])
#         dist2_list_s.append(master_distance_list_s[1])
#         ang2_list_s.append(master_angle_list_s[1])
#
#     name_list.append(name[14:22])
#
#     # master_star_pixels = sf.stars_box(master_img, brightest[0], 15)
#     # sf.show_stars(master_star_pixels, val_max, amp_min=np.min(master_star_pixels), name=name, ext='final_b', save=False,
#     #               folder_path='./stars//imgs/drizzle_/')
#     sf.show_stars(master_img, val_max, amp_min=0, name=name, ext='master', save=True,
#                   folder_path='./stars/imgs/drizzle_/')
#     # sf.show_stars(final_imgs[0], val_max, name=name, ext='tele1', save=False,
#     #               folder_path='./stars//imgs/drizzle_/')
#     print(master_centroid[0], master_centroid[1], master_sigma, master_star_max, master_distance_list[0],
#           master_angle_list[0], master_distance_list[1], master_angle_list[1])
#     print("All Telescope Features Acquired")
#     print("---------------------------------------------")
#
# ####################################################################################################################
# # %%
# sf.print_fwhm(centroid_x_list, name2_list, save=True, name='drizzle_three', ext='cx_files', title='Centroid x',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='x centroid')
# sf.print_fwhm(centroid_y_list, name2_list, save=True, name='drizzle_three', ext='cy_files', title='Centroid y',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='y centroid')
# sf.print_centroid(sigma_list, sigma_list_s, names=name_list, save=True, name='drizzle_three', ext='sigma_files', title='Sigma',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='sigma')
# sf.print_centroid(amp_list, amp_list_s, names=name_list, save=True, name='drizzle_three', ext='amp_files', title='Maximum Amplitude',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='maximum amplitude')
# sf.print_centroid(dist_list, dist_list_s, names=name2_list, save=True, name='drizzle_three', ext='dist_files', title='Distance Two Brightest Stars',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='distance')
# sf.print_centroid(ang_list, ang_list_s, names=name2_list, save=True, name='drizzle_three', ext='ang_files', title='Angle Two Brightest Stars',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='angle')
# sf.print_centroid(dist2_list, dist2_list_s, names=name2_list, save=True, name='drizzle_three', ext='dist_files_',
#               title='Distance Second and Third Brightest',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='distance')
# sf.print_centroid(ang2_list, ang2_list_s, names=name2_list, save=True, name='drizzle_three', ext='ang_files_',
#               title='Angle Second and Third Brightest',
#               folder_path='./stars/imgs/drizzle_/',
#               xlabel='files', ylabel='angle')
#
# print("----------------------------------FINISHED----------------------------------")
    fit_p, gauss2d_fit, star_pixels, masked = sf.fit_star(master_img, brightest, 0,
                                                          iso_box=iso_list[file_counter - 1])

    amp_max = np.max(masked)

    y, x = np.mgrid[:2 * iso_list[file_counter - 1], :2 * iso_list[file_counter - 1]]

    sigma = np.mean([gauss2d_fit.x_stddev.value, gauss2d_fit.y_stddev.value])

    # Acquire max value of the star
    star_max = np.max(gauss2d_fit(x, y))

    # Acquire new centroid
    c_x, c_y = gauss2d_fit.x_mean.value, gauss2d_fit.y_mean.value
    centroid = [brightest[1] - (iso_list[file_counter - 1] - c_x), brightest[2] - (iso_list[file_counter - 1] - c_y)]

    # Get the covariance matrix of the fit result
    cov_matrix = fit_p.fit_info['param_cov']

    # Extract the standard deviation
    stddev_x_err = cov_matrix[1, 1]
    stddev_y_err = cov_matrix[2, 2]

    # if file_counter >= 3 and file_counter <= 12:
    centroid_x_list.append(centroid[0])
    centroid_y_list.append(centroid[1])
    x_err_list.append(np.sqrt(stddev_x_err))
    y_err_list.append(np.sqrt(stddev_y_err))
    # dist_list.append(master_distance_list[0])
    # ang_list.append(master_angle_list[0])
    # dist_list_s.append(master_distance_list_s[0])
    # ang_list_s.append(master_angle_list_s[0])
    # name2_list.append(name[14:22])
    # dist2_list.append(master_distance_list[1])
    # ang2_list.append(master_angle_list[1])
    # dist2_list_s.append(master_distance_list_s[1])
    # ang2_list_s.append(master_angle_list_s[1])

    name_list.append(name[14:22])

    # master_star_pixels = sf.stars_box(master_img, brightest[0], 15)
    # sf.show_stars(master_star_pixels, val_max, amp_min=np.min(master_star_pixels), name=name, ext='final_b', save=False,
    #               folder_path='./stars/imgs/ef/')
    # sf.show_stars(master_img, val_max, amp_min=0, name=name, ext='master', save=True,
    #               folder_path='./stars/imgs/ef/')
    # # sf.show_stars(final_imgs[0], val_max, name=name, ext='tele1', save=False,
    # #               folder_path='./stars/imgs/ef/')
    # print(master_centroid[0], master_centroid[1], master_sigma, master_star_max, master_distance_list[0],
    #       master_angle_list[0], master_distance_list[1], master_angle_list[1])
    # print("All Telescope Features Acquired")
    # print("---------------------------------------------")

####################################################################################################################
# %%
# sf.print_fwhm(centroid_x_list, name2_list, save=True, name='li_three', ext='cx_files', title='Centroid x',
#               folder_path='./stars/imgs/ef/',
#               xlabel='files', ylabel='x centroid')
# sf.print_fwhm(centroid_y_list, name2_list, save=True, name='li_three', ext='cy_files', title='Centroid y',
#               folder_path='./stars/imgs/ef/',
#               xlabel='files', ylabel='y centroid')
sf.print_centroid(centroid_x_list, x_err_list, names=name_list, save=True, name='cent_x', ext='cent_x',
                  title='Centroid x-axis',
                  folder_path='./stars/imgs/drizzle/',
                  xlabel='files', ylabel='centroid x')
sf.print_centroid(centroid_y_list, y_err_list, names=name_list, save=True, name='cent_y', ext='cent_y',
                  title='Centroid y-axis',
                  folder_path='./stars/imgs/drizzle/',
                  xlabel='files', ylabel='centroid y')