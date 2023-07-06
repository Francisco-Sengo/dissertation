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

from photutils.psf import extract_stars
from astropy.nddata import NDData

from astropy.table import Table

import pmapper

from photutils.psf import EPSFBuilder

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
db_all = [m1, m2, m3]

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

    for frame in images:
        image_list.append(frame[0])

    break


#%%

psfs = []
star1_list = []
star2_list = []
star3_list = []
star_list = [[] for _ in range(len(db_all))]

table_list = []
tele_table = []

epsf_builder = EPSFBuilder(oversampling=2, maxiters=500,
                           progress_bar=True, smoothing_kernel='quadratic')

# TODO oversampling < 3
#

for frame in image_list:
    # image_list.append(frame[0])

    _, _, _, std = sf.background_noise(frame)
    brightest = []
    for db in range(len(db_all)):
        stars = sf.find_stars(frame, std, th=3, exp_fwhm=7, flux_min=0, peak_min=0, roi=db_all[db])
        if stars is None:
            brightest.append(np.nan)
            continue

        main = sf.main_stars(stars, 1)[0]
        # iso_box = 9
        # fit_p, gauss2d_fit, star_pixels, masked = sf.fit_star(frame, main, std, iso_box=iso_box)
        # y, x = np.mgrid[:2 * iso_box, :2 * iso_box]
        # cov_matrix = fit_p.fit_info['param_cov']
        #
        # # Check if fit is good or bad
        # if cov_matrix is None:
        #     continue
        # star_list[db].append(np.array(gauss2d_fit(x, y)))
        star_pixels = sf.stars_box(frame, main, 7)

        # print(len(star_list[db]))
        # plt.imshow(sf.stars_box(frame, main, 9), norm=norm, origin='lower', cmap='inferno')
        # plt.colorbar()
        # plt.show()

        # star_list[db].append(star_pixels)
        brightest.append(main)

    stars_tbl = Table()
    stars_tbl['x'] = []  # create an empty column 'x'
    stars_tbl['y'] = []  # create an empty column 'y'

    for i in range(len(brightest)):
        if brightest[i] is not np.nan:
            stars_tbl.add_row([int(brightest[i][1]), int(brightest[i][2])])

    # print(stars_tbl)

    nddata = NDData(data=frame)
    #TODO size 7, 9n
    stars_ = extract_stars(nddata, stars_tbl, size=7)

    # total = 0
    # for star in stars_.all_good_stars:
    #     total = total + np.max(star)

    # amp = total / len(stars_.all_good_stars)

    # star1_list.append(np.array(stars_[0]))
    # star2_list.append(np.array(stars_[1]))
    # star3_list.append(np.array(stars_[2]))

    for star in range(len(stars_)):
        star_list[star].append(stars_.all_good_stars[star].data)


    # print(len(stars_))

    epsf, fitted_stars = epsf_builder(stars_)

    # plt.imshow(stars_[0], origin='lower', cmap='inferno')
    # plt.colorbar()
    # plt.show()
    psfs.append(epsf.data[2:epsf.data.shape[0]-2, 2:epsf.data.shape[1]-2])
    # psfs.append(epsf.data)
#TODO 100
# pmp = pmapper.PMAP(star_list[0][0], psfs[0])  # "PMAP problem"
# while pmp.iter < 5:  # number of iterations
#     fHat = pmp.step()  # fHat is the object estimate
# plt.imshow(fHat, origin='lower', cmap='inferno')
# plt.title('5 itr')
# plt.colorbar()
# plt.show()
# while pmp.iter < 10:  # number of iterations
#     fHat = pmp.step()  # fHat is the object estimate
# plt.imshow(fHat, origin='lower', cmap='inferno')
# plt.title('10 itr')
# plt.colorbar()
# plt.show()

#######################################################
# for psf in psfs:
#     plt.imshow(psf, norm=norm, origin='lower', cmap='inferno')
#     plt.show()
print("____________________________________________ACQUIRED___________________________________________________________")
img1 = star_list[0]
img2 = star_list[1]
img3 = star_list[2]
epsfs = np.array(psfs)
pmp1 = pmapper.MFPMAP(img1, epsfs)  # "PMAP problem"
pmp2 = pmapper.MFPMAP(img2, epsfs)
pmp3 = pmapper.MFPMAP(img3, epsfs)
#TODO 500
print("_______________________________________________S1______________________________________________________________")
while pmp1.iter < len(img1)*1:  # number of iterations
    fHat1 = pmp1.step()  # fHat is the object estimate

plt.imshow(fHat1, origin='lower', cmap='inferno')
plt.colorbar()
plt.show()
print("_______________________________________________S2______________________________________________________________")
while pmp2.iter < len(img2)*1:  # number of iterations
    fHat2 = pmp2.step()  # fHat is the object estimate

plt.imshow(fHat2, origin='lower', cmap='inferno')
plt.colorbar()
plt.show()
print("_______________________________________________S3______________________________________________________________")
while pmp3.iter < len(img3)*1:  # number of iterations
    fHat3 = pmp3.step()  # fHat is the object estimate

#%%
# print(fHat)
plt.imshow(fHat3, origin='lower', cmap='inferno')
plt.colorbar()
plt.show()

print("____________________________________________FINISHED___________________________________________________________")

# pmapper.RichardsonLucy()
#
# maximum = []
# freq_list = []
#
# # Define the scaling factor for superresolution
# scale_factor = 4
#
# # Compute the size of the high-resolution image
# h_lr, w_lr = 250, 250
# h_hr, w_hr = h_lr * scale_factor, w_lr * scale_factor
#
# # Create a high-resolution filter in the Fourier domain
# f_hr_filter = np.zeros((h_lr, w_lr))
# f_hr_filter[h_lr // 2 - h_hr // 2: h_lr // 2 + h_hr // 2, w_lr // 2 - w_hr // 2: w_lr // 2 + w_hr // 2] = 1
#
# for image in image_list:
#     _, _, _, std = sf.background_noise(image)
#     stars = sf.find_stars(image, std, exp_fwhm=7.0, flux_min=5, roi=m1)
#     if stars is None:
#         continue
#
#     brightest = sf.main_stars(stars, 1)
#     # box = sf.stars_box(image, brightest, 9)
#     _, star_max, _, _, _, _ = sf.star_stats(image, brightest[0], std=std, iso_box=9, plt_gauss=False)
#     if star_max is np.nan:
#         continue
#     maximum.append(star_max)
#     f_input = np.fft.fft2(image)
#     # Shift the zero-frequency component to the center of the spectrum
#     f_input_shifted = np.fft.fftshift(f_input)
#     # Apply the high-resolution filter to the shifted input image
#     f_hr_shifted = f_input_shifted * f_hr_filter
#     # Shift the zero-frequency component back to the corner of the spectrum
#     f_hr = np.fft.ifftshift(f_hr_shifted)
#     freq_list.append(f_hr)
#
# weights = (maximum - np.min(maximum)) / (np.max(maximum) - np.min(maximum))
#
# final_image = 0
#
# for i in range(len(freq_list)):
#
#     final_image += freq_list[i] * weights[i]
#
# avg_img = np.abs(np.fft.ifft2(final_image))

#%%
# avg_img = sf.join_images(image_list)
#
# _, _, _, std = sf.background_noise(avg_img)
# stars = sf.find_stars(avg_img, std, 3, exp_fwhm=7, flux_min=20, peak_min=0)
# brightest = sf.main_stars(stars, 3)
# # _, _, max_star, _, _, _ = sf.star_stats(avg_img, brightest[0], std=std, iso_box=9, plt_gauss=False)
# # sf.show_apertures(avg_img, stars, max_star)
# # sf.print_sources(stars)
#
# #%%
# # from photutils.detection import find_peaks
# # peaks_tbl = find_peaks(avg_img, threshold=50.0, box_size=9)
# # peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output
# # print(peaks_tbl)
#
# # size = 10
# # hsize = (size - 1) / 2
# # x = stars['xcentroid']
# # y = stars['ycentroid']
#
# # x = peaks_tbl['x_peak']
# # y = peaks_tbl['y_peak']
# # mask = ((x > hsize) & (x < (avg_img.shape[1] -1 - hsize)) &
# #         (y > hsize) & (y < (avg_img.shape[0] -1 - hsize)))
# #
#
# stars_tbl = Table()
# stars_tbl['x'] = []  # create an empty column 'x'
# stars_tbl['y'] = []  # create an empty column 'y'
#
# for i in range(len(brightest)):
#
#     stars_tbl.add_row([brightest[i][1], brightest[i][2]])
#
# print(stars_tbl)
#
# # stars_tbl['x'] = x
# # stars_tbl['y'] = y
#
# nddata = NDData(data=avg_img)
#
# # from photutils.psf import extract_stars
# stars_ = extract_stars(nddata, stars_tbl, size=11)
#
# for star in stars_:
#     plt.imshow(star, norm=norm)
#     plt.show()
#
# from photutils.psf import EPSFBuilder
# epsf_builder = EPSFBuilder(oversampling=1, maxiters=100,
#                            progress_bar=True)
#
# epsf, fitted_stars = epsf_builder(stars_)
#
# import matplotlib.pyplot as plt
# norm = ImageNormalize(stretch=SqrtStretch())
# plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
# plt.show()