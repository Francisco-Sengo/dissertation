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


#######################################################

print("____________________________________________FINISHED___________________________________________________________")
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
avg_img = sf.join_images(image_list)

_, _, _, std = sf.background_noise(avg_img)
stars = sf.find_stars(avg_img, std, 3, exp_fwhm=7, flux_min=20, peak_min=0)
brightest = sf.main_stars(stars, 1)
_, _, max_star, _, _, _ = sf.star_stats(avg_img, brightest[0], std=std, iso_box=9, plt_gauss=False)
sf.show_apertures(avg_img, stars, max_star)
sf.print_sources(stars)

#%%
# from photutils.detection import find_peaks
# peaks_tbl = find_peaks(avg_img, threshold=50.0, box_size=9)
# peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output
# print(peaks_tbl)

# size = 10
# hsize = (size - 1) / 2
x = stars['xcentroid']
y = stars['ycentroid']

# x = peaks_tbl['x_peak']
# y = peaks_tbl['y_peak']
# mask = ((x > hsize) & (x < (avg_img.shape[1] -1 - hsize)) &
#         (y > hsize) & (y < (avg_img.shape[0] -1 - hsize)))
#
from astropy.table import Table
stars_tbl = Table()
stars_tbl['x'] = x
stars_tbl['y'] = y

nddata = NDData(data=avg_img)

from photutils.psf import extract_stars
stars_ = extract_stars(nddata, stars_tbl, size=15)

for star in stars_:
    plt.imshow(star, norm=norm)
    plt.show()

from photutils.psf import EPSFBuilder
epsf_builder = EPSFBuilder(oversampling=1, maxiters=100,
                           progress_bar=True, smoothing_kernel='quadratic')
epsf, fitted_stars = epsf_builder(stars_)

import matplotlib.pyplot as plt
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
plt.show()

# extracted = extract_stars(nddata, stars, size=(7,7))

# norm = ImageNormalize(stretch=SqrtStretch(), vmin=np.mean(avg_img))
#
# # for star in stars_:
# #     plt.imshow(star, norm=norm)
# #     plt.show()

# sf.show_apertures(avg_img, stars, max_star)
