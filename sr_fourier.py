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
    # Define the scaling factor for superresolution
    # scale_factor = 4
    #
    # # Compute the size of the high-resolution image
    # h_lr, w_lr = input_image.shape
    # h_hr, w_hr = h_lr * scale_factor, w_lr * scale_factor
    #
    # # Apply Fourier transform to the input image
    # f_input = np.fft.fft2(input_image)
    #
    # # Shift the zero-frequency component to the center of the spectrum
    # f_input_shifted = np.fft.fftshift(f_input)
    #
    # # Create a high-resolution filter in the Fourier domain
    # f_hr_filter = np.zeros((h_lr, w_lr))
    # f_hr_filter[h_lr // 2 - h_hr // 2: h_lr // 2 + h_hr // 2, w_lr // 2 - w_hr // 2: w_lr // 2 + w_hr // 2] = 1
    #
    # # Apply the high-resolution filter to the shifted input image
    # f_hr_shifted = f_input_shifted * f_hr_filter
    #
    # # Shift the zero-frequency component back to the corner of the spectrum
    # f_hr = np.fft.ifftshift(f_hr_shifted)
    #
    # # Apply the inverse Fourier transform to get the high-resolution image
    # img_hr = np.abs(np.fft.ifft2(f_hr))

#######################################################

print("____________________________________________FINISHED___________________________________________________________")

#%%
maximum = []
freq_list = []

# Define the scaling factor for superresolution
scale_factor = 4

# Compute the size of the high-resolution image
h_lr, w_lr = 250, 250
h_hr, w_hr = h_lr * scale_factor, w_lr * scale_factor

# Create a high-resolution filter in the Fourier domain
f_hr_filter = np.zeros((h_lr, w_lr))
f_hr_filter[h_lr // 2 - h_hr // 2: h_lr // 2 + h_hr // 2, w_lr // 2 - w_hr // 2: w_lr // 2 + w_hr // 2] = 1

for image in image_list:
    _, _, _, std = sf.background_noise(image)
    stars = sf.find_stars(image, std, exp_fwhm=7.0, flux_min=5, roi=m1)
    if stars is None:
        continue

    brightest = sf.main_stars(stars, 1)
    # box = sf.stars_box(image, brightest, 9)
    _, star_max, _, _, _, _ = sf.star_stats(image, brightest[0], std=std, iso_box=9, plt_gauss=False)
    if star_max is np.nan:
        continue
    maximum.append(star_max)
    f_input = np.fft.fft2(image)
    # Shift the zero-frequency component to the center of the spectrum
    f_input_shifted = np.fft.fftshift(f_input)
    # Apply the high-resolution filter to the shifted input image
    f_hr_shifted = f_input_shifted * f_hr_filter
    # Shift the zero-frequency component back to the corner of the spectrum
    f_hr = np.fft.ifftshift(f_hr_shifted)
    freq_list.append(f_hr)

weights = (maximum - np.min(maximum)) / (np.max(maximum) - np.min(maximum))

final_image = 0

for i in range(len(freq_list)):

    final_image += freq_list[i] * weights[i]

img_hr = np.abs(np.fft.ifft2(final_image))

_, _, _, std = sf.background_noise(img_hr)
stars = sf.find_stars(img_hr, std, exp_fwhm=7.0, flux_min=5, roi=m1)
brightest = sf.main_stars(stars, 1)

#%%
sf.print_sources(stars)
# Isolate the star
star_pixels = sf.stars_box(img_hr, brightest[0], 9)
# Mask star
masked = sf.mask_roi(star_pixels)
amp_max = np.max(masked)
sf.show_stars(img_hr, amp_max)


plt.imshow(img_hr, norm=norm, origin='lower')
plt.show()