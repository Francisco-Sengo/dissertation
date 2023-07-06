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


m1 = [140, 200, 100, 150]
m2 = [150, 200, 50, 100]
m3 = [50, 100, 140, 200]

#            1   2   3    4    5    6    7    8    9    10   11   12   13   14  15   16   17  18  19   20   21
# roi_list = [[], [], m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, [], m1, m1, [], [], m1, m1]
roi_list = [m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1]
out_list = [4,  5,  3,  7,  5,   5,  5,  5,  3,  3,  3,  7, 2.5, 7, 10, 5,  10,  7,  7,  3,  7]
iso_list = [7,  7,  9,  9,  9,   9,  9,  9,  7,  7,  7,  9,  9,  9,   7, 9,  7,  7,  9,  9,  9]
exp_list = [5,  5,  7,  7,  7,   7,  7,  7,  5,  5,  5,  7,  7,  7,   5, 7,  5,  5,  7,  7,  7]
'''''''                    OTHER SETUPS                                                                          '''''''


file_counter = 0

#TODO:

'''''''-------------------STITCHING ALGORITHM--------------------------------------------------------------------'''''''
# Loop to access all files in fits folder
for fits_file in stars_files:

    images, name, time, _ = sf.open_file(fits_file)

    print(name, "(", file_counter + 1, "/", len(stars_files), ")")
    file_counter = file_counter + 1

    # continue
    #
    # if file_counter < 1:
    #     continue

    # pos_x, pos_y, sigma, amp_max, sigma_err, amp_err = sf.file_statistics(images, flux_min=0,
    #                                                                       iso_box=iso_list[file_counter-1],
    #                                                                       exp_fwhm=exp_list[file_counter-1],
    #                                                                       outlier=out_list[file_counter-1], mask=True)
    data_list = sf.file_statistics(images, flux_min=0, iso_box=iso_list[file_counter - 1],
                                   exp_fwhm=exp_list[file_counter - 1], outlier=out_list[file_counter - 1], mask=True, roi=m1)

    # for tele in range(len(pos_x)):
    #     sigma_x = sf.plt_tele_hist(pos_x[tele], name=name, ext='cx', telescope=tele + 1, xlabel='centroid_x',
    #                                save=True)
    #     sigma_y = sf.plt_tele_hist(pos_y[tele], name=name, ext='cy', telescope=tele + 1, xlabel='centroid_y',
    #                                save=True)
    #
    #     sf.plt_tele_stat(pos_x[tele], time, sigma_x, name=name, telescope=tele+1, ylabel='centroid_x', ext='cx',
    #                      save=True)
    #     sf.plt_tele_stat(pos_y[tele], time, sigma_y, name=name, telescope=tele+1, ylabel='centroid_y', ext='cy',
    #                      save=True)
    #
    #     sf.plt_tele_stat(sigma[tele], time, sigma_err[tele], name=name, telescope=tele+1, ylabel='Sigma', ext='sigma',
    #                      save=True)
    #     sf.plt_tele_stat(amp_max[tele], time, amp_err[tele], name=name, telescope=tele + 1, ylabel='Maximum Amplitude', ext='amp',
    #                      save=True)
    # TODO alterar para um for/chamar funcção
    sf.plt_file_stats(data_list, name=name, exp_time=time, save=True, folder_path='./stars/file_stats/')

    # break
'''''''                 DATA ANALYSIS                                                                            '''''''
#%%
print("FINNISH")
# TODO: tirar std do file statistics quebra, mask ou sem mask é parecido. não esta a dar bom fit as estrelas do frame
#melhorar fit para estrelas dos frames, ou arranjar forma de funcionar.... boa sorte sengo