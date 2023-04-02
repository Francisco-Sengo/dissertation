import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import cv2
from photutils.aperture import CircularAperture
from astropy.stats import SigmaClip
from photutils.background import Background2D, SExtractorBackground
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from math import sqrt
import astropy.units
import os
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.optimize import OptimizeWarning

from scipy.optimize import curve_fit
from scipy import exp
from astropy.table import Table
from photutils.datasets import make_gaussian_sources_image
import random
from drizzle import drizzle
from astropy.wcs import WCS
from photutils.centroids import centroid_com

#TODO Divide library into packages

'''''''                    SETUP PARAMETERS                                                                      '''''''
# Width and Height for each telescope in original .fits file
min_h = 0
max_h = 250
t1_min = 0
t1_max = t2_min = 250
t2_max = t3_min = 500
t3_max = t4_min = 750
t4_max = 1000


# Setup for Background Estimator
bkg_estimator = SExtractorBackground()
sigma_clip = SigmaClip(sigma=3.0)

# Setup for find_stars()
mask_min = 25
mask_max = 225


# Setup for mask
thresh_type = cv2.THRESH_BINARY
# Create kernels for masking process
kernel_erode = np.ones((5, 5), 'uint8')
kernel_dilate = np.ones((5, 5), 'uint8')


'''''''                    ACQUIRE DATA FROM FITS FILE                                                           '''''''


def div_files(fits_files):
    """
    Function that opens every .fits file in the fits folder and divides the images based on telescope and frame. When
    opening the new .fits, the data is organized in an array where the first position denotes the frame and the second
    position the telescope
    
    :param fits_files: path to where the fits files are located

    Example:
    
    fits_files = PureWindowsPath("./stars/fits/*.fits")
    
    fits_files = Path(fits_files)
    
    fits_files = glob.glob(str(fits_files))
    """
    # Iterates between every .fits file in the fits folder
    for fits_file in fits_files:
        # Name of the new file
        nname = os.path.splitext(os.path.basename(fits_file))[0]

        # Open .fits file and acquire data
        gravity_file = fits.open(fits_file)
        data = gravity_file[4].data
        tele_list = []

        # Iterates between every frame of a fits file
        for image in data[:][:]:
            # Divide every frame by telescope, storing while maintaining frame order
            t_list = [image[min_h:max_h, t1_min:t1_max], image[min_h:max_h, t2_min:t2_max],
                      image[min_h:max_h, t3_min:t3_max], image[min_h:max_h, t4_min:t4_max]]

            tele_list.append(t_list)

        # Create Header and add important information
        header = fits.Header()
        header = gravity_file[4].header

        # Create a .fits file containing only the telescope images of the stars
        hdu_list = fits.PrimaryHDU(data=tele_list, header=header)

        hdu_list.writeto('./stars/images/{}_{}.fits'
                         .format(nname, 'pt'))

        print("FILE: ", nname)

        # Close .fits file
        gravity_file.close()


def open_file(fits_file):
    """
    Function to open fits file and acquire name from the path (For fits files after being divided by div_files)

    :param fits_file: fits file path
    :return: images and name of file
    """
    # Open the fits file
    stars_file = fits.open(fits_file)

    # Get name to name files
    name = os.path.splitext(os.path.basename(fits_file))[0][:29]

    # Extract data from .fits file
    images = stars_file[0].data
    exp_time = stars_file[0].header['EXPTIME']
    wsc = WCS(stars_file[0].header, naxis=2).sub(2)
    wsc.array_shape = (250, 250)

    # Close the fits file
    stars_file.close()

    return images, name, exp_time, wsc


'''''''                    BACKGROUND DETECTION AND REMOVAL                                                      '''''''


def background_noise(img):
    """
    Function to estimate background image and acquire mean, median and standard deviation of the original image
    with the background removed.
    
    :param img:  image to acquire background data
    :return: image, mean, median and standard deviation of background
    """
    # Aquire background estimation
    bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    # Acquire mean, median and standard deviation of original image without background
    mean, median, std = sigma_clipped_stats(img - bkg.background, sigma=3.0)

    return bkg.background, mean, median, std


def prepare_stars(img, noise):
    """
    Funtion to prepare the image for further analysis, removing the background

    :param img: image to remove noise
    :param noise: image of the noise to be removed
    :return: image with noise removed
    """
    # Subtract the noise
    img = img - noise

    return img


def normalize_img(original):
    """
    Function to normalize the image to apply the median blur

    :param original: original image to normmalize
    :return: normalized image
    """

    # Copy image so to not modify it
    img = original.copy()
    # Maximum and Minimum values for the image
    min_val = np.min(img)
    max_val = np.max(img)

    # Normalize image
    img_normalized = (img - min_val) / (max_val - min_val)
    img_normalized *= 255
    img_normalized = img_normalized.astype(np.uint8)

    return img_normalized


def mask_roi(roi):
    """
    Function to filter noise to acquire stats of a star

    :param roi: region of interest where the star resides
    :return: roi with a mask applied, leaving only the star visible
    """
    # Copy image so to not modify it
    img = roi.copy()

    # Normalize image to apply Median Blur
    norm_roi = normalize_img(img)
    blured = cv2.medianBlur(norm_roi, 5)

    # Create threshold and max value to filter from blured image
    threshold = np.mean(blured)
    max_value = np.max(blured)
    # Filter pixeis
    binary = cv2.threshold(blured, threshold, max_value, thresh_type)[1]
    # Apply Opening method to create mask
    opening = cv2.erode(binary, kernel_erode, iterations=1)
    opened = cv2.dilate(opening, kernel_dilate, iterations=1)

    # Apply mask to region of interest
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if opened[y, x] == 0:
                img[y, x] = 0

    return img


'''''''                    STAR IDENTIFICATION                                                                   '''''''


def find_stars(img, std, exp_fwhm=7.0, flux_min=5, peak_min=45, roi=[]):
    """
        Function used to identificate and qualify stars in an image 

        :param img: image to find stars
        :param std: standard deviation of the background
        :param exp_fwhm: expected fwhm for the stars (optional)
        :param flux_min : minimum flux for the 'normal' star_finder to consider a star (optional)
        :param peak_min: minimum flux for the 'stitched' star_finder to consider a star (optional)
        :return: list containing all the stars that meet the criteria 
    """
    # Setup for Star Finder
    daofind = DAOStarFinder(fwhm=exp_fwhm, threshold=3. * std)

    # Create mask for region of interest
    mask = np.zeros(img.shape, dtype=bool)
    mask[:][:] = True
    if len(roi) != 4:
        mask[mask_min:mask_max, mask_min:mask_max] = False
    else:
        mask[roi[0]:roi[1], roi[2]:roi[3]] = False

    # Find sources in an image, cataloging them
    stars = daofind(img, mask=mask)
    # In case no stars are identified that fit our criteria
    if type(stars) == astropy.units.decorators.NoneType:
        return stars
    # Filter stars by flux
    stars = stars[stars['flux'] > flux_min]
    stars = stars[stars['peak'] > peak_min]

    return stars


def fit_star(img_o, src, std=0, mask=True, iso_box=9):

    img = img_o.copy()
    # Isolate the star
    star_pixels = stars_box(img, src, iso_box)
    # Mask star
    masked = mask_roi(star_pixels)

    if not mask:
        masked = star_pixels

    # Estimate initial parameters for 2D Gaussian fit
    amp = np.max(masked) - np.min(star_pixels)
    # Centroid of the box containing the star
    x_mean, y_mean = iso_box, iso_box
    # Initial guess for standard deviation
    if std == 0:
        x_stddev = y_stddev = np.std(masked)
    else:
        x_stddev = y_stddev = std
    # Initial guess for rotation angle (radians)
    theta = 0.0
    init_params = {'amplitude': amp, 'x_mean': x_mean, 'y_mean': y_mean,
                   'x_stddev': x_stddev, 'y_stddev': y_stddev, 'theta': theta}
    gauss2d_init = Gaussian2D(**init_params)

    # Fit 2D Gaussian model to star pixels
    fit_p = LevMarLSQFitter(calc_uncertainties=True)
    y, x = np.mgrid[:2 * iso_box, :2 * iso_box]
    gauss2d_fit = fit_p(gauss2d_init, x, y, masked)

    return fit_p, gauss2d_fit, star_pixels, masked


def star_stats(img_o, src, std=0, name='Default Name', iso_box=9, plt_gauss=True, save=False):
    """
    Function to calculate Full Width Half Maximum of a star

    :param img_o: image to acquire the star
    :param src: coordinates of the centroid of the star
    :param std: standard deviation used to acquire stars, if std=0, calculated from the isolated star pixels
    :param name: name of the file
    :param iso_box: expected odd fwhm for the star (optional)
    :param plt_gauss: True -> plt brighest star and fitted gaussian |
                      False -> don't plt brighest star and fitted gaussian
    :param save: True -> Save the image in "brightest" folder |
                 False -> Don't save the image
    :return: full width half maximum of the star, the standard deviation of the position of the star and the maximum
    amplitude of the star
    """
    fit_p, gauss2d_fit, star_pixels, masked = fit_star(img_o, src, std, iso_box=iso_box)

    amp_max = np.max(masked)

    y, x = np.mgrid[:2 * iso_box, :2 * iso_box]

    sigma = np.mean([gauss2d_fit.x_stddev.value, gauss2d_fit.y_stddev.value])

    # Acquire max value of the star
    star_max = np.max(gauss2d_fit(x, y))

    # Acquire new centroid
    c_x, c_y = gauss2d_fit.x_mean.value, gauss2d_fit.y_mean.value
    centroid = [src[1]-(iso_box-c_x), src[2]-(iso_box-c_y)]

    # Get the covariance matrix of the fit result
    cov_matrix = fit_p.fit_info['param_cov']

    if cov_matrix is None or sigma < 1 or star_max < 1:
        return np.nan, np.nan, np.nan, [np.nan, np.nan], np.nan, np.nan

    # Extract the standard deviation
    stddev_x_err = cov_matrix[1, 1]
    stddev_y_err = cov_matrix[2, 2]

    # Calculate the uncertainty of sigma
    sigma_err = np.sqrt(np.mean([stddev_x_err, stddev_y_err]))

    star_max_err = np.sqrt(cov_matrix[0, 0])

    if 5*star_max < star_max_err:
        return np.nan, np.nan, np.nan, [np.nan, np.nan], np.nan, np.nan

    if plt_gauss is True:
        # Create a figure and subplots
        fig, axs = plt.subplots(nrows=1, ncols=2)

        # Plot each image in a subplot
        axs[0].imshow(star_pixels, origin='lower', cmap='inferno', vmax=amp_max)
        axs[1].imshow(gauss2d_fit(x, y), origin='lower', cmap='inferno')

        # Idetify telescope in subplot
        axs[0].set_title('Original Star')
        axs[1].set_title('Fitted Gaussian')

        plt.tight_layout()
        if save is True:
            plt.savefig('./stars/brightest/{}_{}.png'.format(name, 'brightest_star'))
        plt.show()

    return sigma, star_max, amp_max, centroid, sigma_err, star_max_err


def file_statistics(img_list, iso_box=9, exp_fwhm=7, flux_min=5, outlier=3, mask=True, roi=[]):
    """
    Function that evaluates every frame in a file, creating a list containing every frame that detectes at least "num"
    of stars. It also removes the background from the frames.

    :param img_list: list of frames to be qualified
    :param num: position of the desired star. Example: For the brightest star = 1, second brightest star = 2 (...)
    :param exp_fwhm: expected fwhm for the stars (optional)
    :param flux_min: minimum flux for the 'normal' star_finder to consider a star (optional)
    :return: return a list for the usable frames organized by telescope
    """
    pos_x_list = []
    pos_y_list = []
    sigma_list = []
    amp_list = []
    sigma_err_list = []
    amp_err_list = []
    # Iterate between all frames and all telescopes
    for tele in range(len(img_list[0])):

        tele_x = []
        tele_y = []
        tele_sigma = []
        tele_amp = []
        tele_sigma_err = []
        tele_amp_err = []
        for frame in range(len(img_list)):
            img = img_list[frame][tele]

            # Acquire background and standard deviation of the frame
            bg, _, _, std = background_noise(img)

            # Acquire sources in the images

            stars = find_stars(img, std, exp_fwhm, flux_min, roi=roi)

            if stars is None:
                tele_sigma.append(np.nan)
                tele_x.append(np.nan)
                tele_y.append(np.nan)
                tele_amp.append(np.nan)
                tele_sigma_err.append(np.nan)
                tele_amp_err.append(np.nan)
                continue

            brightest_star = main_stars(stars)
            # if frame > 10 and (len(stars) < 1 or abs(brightest_star[0][1]-np.nanmedian(tele_x)) > 10):
            #     print("fodeu")
            #     tele_sigma.append(np.nan)
            #     tele_x.append(np.nan)
            #     tele_y.append(np.nan)
            #     tele_amp.append(np.nan)
            #     tele_sigma_err.append(np.nan)
            #     tele_amp_err.append(np.nan)

            sigma, star_max, _, centroid, sigma_err, amp_err = star_stats(img, brightest_star[0],
                                                                          iso_box=iso_box, std=std,
                                                                          plt_gauss=False)
            tele_sigma.append(sigma)
            tele_x.append(centroid[0])
            tele_y.append(centroid[1])
            tele_amp.append(star_max)
            tele_sigma_err.append(sigma_err)
            tele_amp_err.append(amp_err)

        mean_x = np.nanmedian(tele_x)
        mean_y = np.nanmedian(tele_y)
        for i in range(len(tele_x)):
            #TODO HERE
            if abs(tele_x[i] - mean_x) > outlier or abs(tele_y[i] - mean_y) > outlier or tele_sigma[i] > exp_fwhm\
                    or tele_sigma_err[i] > tele_sigma[i]:
                tele_x[i] = np.nan
                tele_y[i] = np.nan
                tele_sigma[i] = np.nan
                tele_amp[i] = np.nan
                tele_sigma_err[i] = np.nan
                tele_amp_err[i] = np.nan

        pos_x_list.append(tele_x)
        pos_y_list.append(tele_y)
        sigma_list.append(tele_sigma)
        amp_list.append(tele_amp)
        sigma_err_list.append(tele_sigma_err)
        amp_err_list.append(tele_amp_err)

    return pos_x_list, pos_y_list, sigma_list, amp_list, sigma_err_list, amp_err_list


def stat_filter(tele_stats, outlier, exp_fwhm):
    mean_x = np.nanmedian(tele_stats[0])
    mean_y = np.nanmedian(tele_stats[1])
    for i in range(len(tele_stats[0])):
        if abs(tele_stats[0][i] - mean_x) > outlier or abs(tele_stats[1][i] - mean_y) > outlier or tele_stats[3][i] > exp_fwhm:
            tele_stats[0][i] = np.nan
            tele_stats[1][i] = np.nan
            tele_stats[2][i] = np.nan
            tele_stats[3][i] = np.nan
            tele_stats[4][i] = np.nan
            tele_stats[5][i] = np.nan

    return tele_stats[0], tele_stats[1], tele_stats[2], tele_stats[3], tele_stats[4], tele_stats[5]


def star_distance(img_o, srcs, std=0, mask=True, iso_box=9):
    centroid_list = []
    stddev_list = []
    for src in srcs:
        fit_p, gauss2d_fit, star_pixels, masked = fit_star(img_o, src, std, mask, iso_box)

        y, x = np.mgrid[:2 * iso_box, :2 * iso_box]
        # Acquire new centroid
        c_x, c_y = gauss2d_fit.x_mean.value, gauss2d_fit.y_mean.value
        centroid_list.append([src[1] - (iso_box - c_x), src[2] - (iso_box - c_y)])

        cov_matrix = fit_p.fit_info['param_cov']

        # Evaluate the fit
        residuals = masked - gauss2d_fit(x, y)
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((masked - np.mean(masked)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Check if fit is good or bad
        if (r_squared < 0.7 and rmse > 0.5) or cov_matrix is None:
            # stddev_list.append([np.nan, np.nan])
            return np.nan, np.nan

        stddev_list.append([cov_matrix[1, 1], cov_matrix[2, 2]])

    distance = sqrt((centroid_list[0][0] - centroid_list[1][0]) ** 2 + (centroid_list[0][1] - centroid_list[1][1]) ** 2)

    if stddev_list[0][0] != np.nan and stddev_list[1][0] != np.nan and \
       stddev_list[0][1] != np.nan and stddev_list[1][1] != np.nan:
        # Calculate the errors associated with the centroids
        # delta_1 = np.sqrt(stddev_list[0][0] ** 2 + stddev_list[0][1] ** 2)
        # delta_2 = np.sqrt(stddev_list[1][0] ** 2 + stddev_list[1][1] ** 2)

        # Calculate the error associated with the distance using error propagation
        # delta_distance = np.sqrt(
        #     ((centroid_list[1][0] - centroid_list[0][0]) / distance) ** 2 * (delta_1) ** 2 + ((centroid_list[1][1] - centroid_list[0][1]) / distance) ** 2 * (delta_1) ** 2 + (
        #                 (centroid_list[0][0] - centroid_list[1][0]) / distance) ** 2 * (delta_2) ** 2 + ((centroid_list[0][1] - centroid_list[1][1]) / distance) ** 2 * (delta_2) ** 2)

        delta_distance = np.sqrt((centroid_list[0][0] - centroid_list[1][0]) ** 2 * stddev_list[0][0] /
                                 ((centroid_list[0][0] - centroid_list[1][0]) ** 2 + (centroid_list[0][1]
                                                                                      - centroid_list[1][1]) ** 2)
                                 + (centroid_list[1][0] - centroid_list[0][0]) ** 2 * stddev_list[1][0] /
                                 ((centroid_list[1][0] - centroid_list[0][0]) ** 2 + (centroid_list[1][1]
                                                                                      - centroid_list[0][1]) ** 2)
                                 + (centroid_list[0][1] - centroid_list[1][1]) ** 2 * stddev_list[0][1] /
                                 ((centroid_list[0][1] - centroid_list[1][1]) ** 2 + (centroid_list[0][0]
                                                                                      - centroid_list[1][0]) ** 2)
                                 + (centroid_list[1][1] - centroid_list[1][0]) ** 2 * stddev_list[1][1] /
                                 ((centroid_list[1][1] - centroid_list[0][1]) ** 2 + (centroid_list[1][0]
                                                                                      - centroid_list[0][1]) ** 2))

    else:
        delta_distance = np.nan

    return distance, delta_distance


# def distance_filter(tele_dist, outlier):
#     mean_d = np.nanmean(tele_dist[0])
#     for i in range(len(tele_dist[0])):
#         if abs(tele_dist[0][i] - mean_d) > outlier:
#             tele_dist[0][i] = np.nan
#             tele_dist[1][i] = np.nan
#
#     return tele_dist[0], tele_dist[1]

# TODO alterar a cena do roi aqui
def file_distances(img_list, mask=True, iso_box=9, exp_fwhm=7, flux_min=5, outlier=10, filter=False, roi_list=[]):

    file_distance = []
    dist_err = []
    for tele in range(len(img_list[0])):

        tele_distance = []
        tele_std = []

        for frame in range(len(img_list)):
            img = img_list[frame][tele]

            _, mean, _, std = background_noise(img)

            flag = False
            star_list = []
            if len(roi_list) != 2:
                stars = find_stars(img - mean, std, exp_fwhm, flux_min)
                star_list = main_stars(stars, 2)

            else:
                for cnt in range(2):

                    # Acquire sources in the images
                    stars = find_stars(img-mean, std, exp_fwhm, flux_min, roi=roi_list[cnt])

                    if stars is None:
                        flag = True
                        break

                    star_list.append(main_stars(stars, 1)[0])

            if len(star_list) < 2 or flag:
                tele_distance.append(np.nan)
                tele_std.append(np.nan)
                continue

            distance, d_std = star_distance(img, star_list, std=std, mask=mask, iso_box=iso_box)

            tele_distance.append(distance)
            tele_std.append(d_std)

        # if filter:
        #     tele_distance, tele_std = distance_filter([tele_distance, tele_std], outlier)

        mean_d = np.nanmean(tele_distance)
        for i in range(len(tele_distance)):
            if abs(tele_distance[i] - mean_d) > outlier or tele_std[i] > outlier:
                tele_distance[i] = np.nan
                tele_std[i] = np.nan

        file_distance.append(tele_distance)
        dist_err.append(tele_std)

    return file_distance, dist_err


def main_stars(src, num=1):
    """
    Function that decomposes "num" of main stars into an list

    :param src: table from photutils contaning the sources identified by the find_stars function
    :param num: number of stars desired
    :return: return a list containing the info about the stars filtered
    """
    # Function to acquire position of "num" brightest stars in the sources catalog of the image
    positions = src['flux'].argsort()[::-1][:num]
    star_data = []
    # For the "num" brightest stars, append relevant data into a list
    for i in range(0, len(positions)):
        star = [src['id'][positions[i]], src['xcentroid'][positions[i]], src['ycentroid'][positions[i]],
                src['peak'][positions[i]], src['flux'][positions[i]], src['mag'][positions[i]]]
        star_data.append(star)

    return star_data


'''''''                    COORDINATE FUNCTIONS                                                                  '''''''


def get_features(b_src):
    """
    Function to append the coordinates of a list of stars into another list for easier access

    :param b_src: sources to acquire coordinates (usually the filteres brightest sources)
    :return: list contaning only the coordinates of the stars
    """
    stars_coords = []
    for i in range(0, len(b_src)):
        coord = [b_src[i][1], b_src[i][2]]
        stars_coords.append(coord)
    return stars_coords


def stars_features(img, exp_fwhm=7, flux_min=5, roi=[]):
    """
    Function that identifies the stars and acquires their coordinates for later matching

    :param img: image to acquire stars
    :param exp_fwhm: expected fwhm for the stars (optional)
    :param flux_min_normal : minimum flux for the 'normal' star_finder to consider a star (optional)
    :return: list containing the coordinates of the identified stars
    """
    # Acquire standard deviation
    _, mean, _, std = background_noise(img)
    # Find sources in the image
    img_src = find_stars(img-mean, std, exp_fwhm, flux_min, roi=roi)

    # Filter to only obtain the three brightest stars
    # Acquire the coordinates of said stars for easier manipulation
    # Append x and y coordinate respectively onto a list
    stars_ft = get_features(img_src)

    return stars_ft


def stars_matcher(src_coords, dst_coords, distance=5):
    """
    Function that matches the three brightest of two telescopes through the coordinates

    :param src_coords: coordinates of the source image
    :param dst_coords: coordinates of the destination image
    :param distance: distance between stars to be considered the same
    :return: a list contaning all the matches grouped, a list contaning only the matched coordinates of the source image
    and a list contaning only the matched coordinates of the destination image
    """
    matches_list = []
    match_1 = []
    match_2 = []
    # Match the coordinates of the same star between the two images
    for c1 in range(len(src_coords)):
        for c2 in range(len(dst_coords)):
            # Calculate distance between two stars in different images
            d = sqrt((src_coords[c1][0] - dst_coords[c2][0]) ** 2 + (src_coords[c1][1] - dst_coords[c2][1]) ** 2)
            # If the distance is lower than 10 px, the two stars match
            if d < distance:
                match = [[src_coords[c1][0], src_coords[c1][1]], [dst_coords[c2][0], dst_coords[c2][1]]]
                matches_list.append(match)
                match_1.append([src_coords[c1][0], src_coords[c1][1]])
                match_2.append([dst_coords[c2][0], dst_coords[c2][1]])

    if len(matches_list) < 4:
        return 0, 0, 0

    return matches_list, np.array(match_1), np.array(match_2)


def stars_box(img, coords, box_size):
    """
    Function to isolate the a star for further analysis

    :param img: image to acquire the star
    :param coords: coordinates of the centroid of the star
    :param box_size: size of the box to fit the brightest star
    :return: "box" containing the star
    """
    star_pixels = img[int(coords[2]) - box_size:int(coords[2]) + box_size,
                      int(coords[1]) - box_size:int(coords[1]) + box_size]

    return star_pixels


'''''''                    MANIPULATION OF IMAGES                                                                '''''''


def warp_image(src_img, H):
    """
    Function that warps images given the estimated homography

    :param src_img: image to be applied the homography estimate
    :param H: homography estimate
    :return: warped image
    """
    # Warp the source image to the destination image using the estimated homography
    # Flags:
    #       cv2.INTER_LINEAR: Linear interpolation is used, which produces smoother results than the default method
    #                       (nearest-neighbor interpolation)
    #       cv2.BORDER_CONSTANT: Border mode used when resampling image, meaning that pixels outside the input image are
    #       set to a constant value.
    warped_img = cv2.warpPerspective(src_img, H, (src_img.shape[1], src_img.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
    
    return warped_img


def join_images(img_list):
    """
    Function that joins a list of images and divides the result by the median of the result

    :param img_list: list of the images to be stacked
    :return: stacked image
    """
    list_sum = np.sum(img_list, axis=0)
    final_img = list_sum / np.mean(list_sum)

    return final_img


'''''''                    QUALITY CHECK                                                                         '''''''


def frame_quality(img_list, num, exp_fwhm=7, flux_min=5):
    """
    Function that evaluates every frame in a file, creating a list containing every frame that detectes at least "num"
    of stars. It also removes the background from the frames.

    :param img_list: list of frames to be qualified
    :param num: number of stars required for the frame to be deemed usable
    :param exp_fwhm: expected fwhm for the stars (optional)
    :param flux_min: minimum flux for the 'normal' star_finder to consider a star (optional)
    :return: return a list for the usable frames organized by telescope
    """
    frame_list = []
    # Iterate between all frames and all telescopes
    for tele in range(len(img_list[0])):
        tele_list = []
        for frame in range(len(img_list)):
            img = img_list[frame][tele]
            # Acquire background and standard deviation of the frame
            bg, _, _, std = background_noise(img)
            # Prepare frame for identification of stars
            img = prepare_stars(img, bg)
            # Acquire sources in the images
            stars = find_stars(img, std, exp_fwhm, flux_min)
            # In case no stars are identified that fit our criteria
            if type(stars) == astropy.units.decorators.NoneType:
                continue
            if len(stars) >= num:
                tele_list.append(img)

        frame_list.append(tele_list)

    return frame_list


def stitch_quality(stitched_img, num, exp_fwhm=7, flux_min=25, peak_min=45):
    """
    Function that evaluates every stacked image for all 4 telescopes, creating a list containing every image that
    detectes at least "num" stars. It also removes the background from the stacked images.

    :param stitched_img: master image
    :param num: number of stars required for the master image to be considered a success
    :param exp_fwhm: expected fwhm for the stars (optional)
    :param flux_min_stitch: minimum flux for the 'stitched' star_finder to consider a star (optional)
    :param peak_min_stitch: minimum flux for the 'stitched' star_finder to consider a star (optional)
    :return: True -> if the master image is considered good
             False -> if the master image is considered not good
    """
    # Acquire background and standard deviation of the frame
    bg, _, _, std = background_noise(stitched_img)
    # Prepare frame for identification of stars
    stitched_img = prepare_stars(stitched_img, bg)
    # Acquire sources in the images
    stars = find_stars(stitched_img, std, exp_fwhm, flux_min, peak_min)
    # In case no stars are identified that fit our criteria
    if type(stars) == astropy.units.decorators.NoneType:
        return False
    if len(stars) >= num:
        return True
    else:
        return False


'''''''                    STITCH FUNCTIONS                                                                      '''''''


def tele_stitcher(img_list):
    """
    Function to stack frames by telescope after frames ahave been qualified

    :param img_list: list for the frames of the telescope
    :return: stacked imaged of the telescope
    """
    # Stack the frames together to form one single image per telescope
    tele_list = []
    for tele in img_list:
        # Add all usable frames together
        res_tele = join_images(tele)
        # After joining all frames of an telescope, append the final image to a list
        tele_list.append(res_tele)

    return tele_list


def tele_stacker(img_list, wcs, exp_time):
    tele_list = []
    for tele in range(len(img_list[0])):
        # Add all usable frames together
        driz = drizzle.Drizzle(outwcs=wcs)
        for frame in range(len(img_list)):
            driz.add_image(img_list[frame][tele], wcs, expin=exp_time)
        # After joining all frames of an telescope, append the final image to a list
        tele_list.append(driz.outsci)

    return tele_list


def final_stitcher(img_list, distance=5, exp_fwhm=7, flux_min=5):
    """
    Function to find homography and stitch all the telescope perspectives into a master image
    :param img_list: list of images of the four telescopes
    :param distance: approximated distance between the stars of the reference telescope and the other telescopes
    :return: master image
    """
    final_img = []
    # Define the telescope 1 as the reference for the stitching
    master_img = img_list[0]
    warped_list = [master_img]

    # Acquire coordinates from the reference image
    master_coords = stars_features(master_img, exp_fwhm, flux_min)

    # Iteratively finde the homography and warp all the telescope images
    for tele in range(1, len(img_list)):

        tele_coords = stars_features(img_list[tele], exp_fwhm, flux_min)

        # Match the stars between the two images
        matches, master_match, tele_match = stars_matcher(master_coords, tele_coords, distance)

        if matches == 0:
            print("NOT ENOUGH MATCHES TO CALCULATE HOMOGRAPHY")
            return final_img

        # Estimate the homography using the cv2 library
        H, _ = cv2.findHomography(tele_match, master_match)

        # Warp image base on the estimated homography
        warped_tele = warp_image(img_list[tele], H)
        # Append the warped image from the telescope into a list
        warped_list.append(warped_tele)

    # Stack images to form the master image
    final_img = join_images(warped_list)

    return final_img


'''''''                    PLT FUNCTIONS                                                                         '''''''


def show_stars(img, amp_max, name='Default Name', ext='img', save=False, folder_path='./stars/stitched/'):
    """
    Plot a single image of the stars

    :param img: image to be ploted
    :param amp_max: maximum amplitude of the brightest star
    :param name: title of the plot and name of the file
    :param ext: extention for the file
    :param save: If 'True' save file to relative path: ./stars/stitched/
    """

    # Setup Image Normalization for plot
    norm = ImageNormalize(stretch=SqrtStretch(), vmax=amp_max, vmin=np.mean(img))

    plt.imshow(img, norm=norm, origin='lower', cmap='inferno',
               interpolation='nearest')
    plt.title(name, fontweight='bold')
    plt.tight_layout()
    if save is True:
        full_path = os.path.join(folder_path, '{}_{}.png'.format(name, ext))
        plt.savefig(full_path)
    plt.show()


def show_apertures(img, src, amp_max, name='Default Name', ext='aprt', save=False, folder_path='./stars/stitched/'):
    """
    Plot apertures for a single image of the stars

    :param img: image to be ploted
    :param src: sources of the image to be identified
    :param amp_max: maximum amplitude of the brightest star
    :param name: title of the plot and name of the file
    :param save: If 'True' save file to relative path: ./stars/stitched/
    """
    # Apply circles in the positions of the sources
    positions = np.transpose((src['xcentroid'], src['ycentroid']))
    apertures = CircularAperture(positions, r=4.0)

    # Setup Image Normalization for plot
    norm = ImageNormalize(stretch=SqrtStretch(), vmax=amp_max, vmin=np.mean(img))

    # Plot image
    plt.imshow(img, cmap='inferno', origin='lower', norm=norm,
               interpolation='nearest')
    apertures.plot(color='white', lw=1.5, alpha=0.5)
    plt.title(name, fontweight='bold')
    plt.tight_layout()
    if save is True:
        full_path = os.path.join(folder_path, '{}_{}.png'.format(name, ext))
        plt.savefig(full_path)
    plt.show()


def show_telescopes(img_list, amp_max=0, name='Default Name', ext='telescopes', save=False,
                    folder_path='./stars/stitched/'):
    """
    Plot all stacked images for the telescopes into a single image

    :param img_list: list of images of all four telescopes
    :param amp_max: maximum amplitude of the brightest star
    :param name: name of the file
    :param save: If 'True' save file to relative path: ./stars/stitched/
    """
    # Setup Image Normalization for plot
    if amp_max == 0:
        amp_max = np.max(img_list[0])
    norm = ImageNormalize(stretch=SqrtStretch(), vmax=amp_max, vmin=np.mean(img_list[0]))

    # Create a figure and subplots
    fig, axs = plt.subplots(nrows=2, ncols=2)

    # Plot each image in a subplot
    axs[0, 0].imshow(img_list[0], norm=norm, origin='lower', cmap='inferno',
                     interpolation='nearest')
    axs[0, 1].imshow(img_list[1], norm=norm, origin='lower', cmap='inferno',
                     interpolation='nearest')
    axs[1, 0].imshow(img_list[2], norm=norm, origin='lower', cmap='inferno',
                     interpolation='nearest')
    axs[1, 1].imshow(img_list[3], norm=norm, origin='lower', cmap='inferno',
                     interpolation='nearest')

    # Idetify telescope in subplot
    axs[0, 0].set_title('Telescope #1')
    axs[0, 1].set_title('Telescope #2')
    axs[1, 0].set_title('Telescope #3')
    axs[1, 1].set_title('Telescope #4')

    plt.tight_layout()
    if save is True:
        full_path = os.path.join(folder_path, '{}_{}.png'.format(name, ext))
        plt.savefig(full_path)

    plt.show()


def print_centroid(positions, stdev_est, names, title='Default Name', name='Default Name', ext='centroids',
                   xlabel='file', ylabel='value', save=False, folder_path='./stars/graphs/'):
    """
    Function to plot the centroid of the star with the standard deviation

    :param positions: list of one axis of the coordinates for the centroid
    :param stdev_est: standard deviation of the position of the star
    :param names: list of names for the files
    :param title: title of the ploted image
    :param name: name for the saved file
    :param xlabel: label for the x axis
    :param ylabel: label for the y axis
    :param save: If 'True' save file to relative path: ./stars/graphs/
    """
    # Plot image
    plt.figure()
    plt.title(title)
    plt.errorbar(np.arange(len(names)), positions, yerr=stdev_est, fmt='.', ecolor='g',
                 capsize=5)
    plt.xticks(np.arange(len(names)), names, rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    plt.tight_layout()
    if save is True:
        full_path = os.path.join(folder_path, '{}_{}.png'.format(name, ext))
        plt.savefig(full_path)

    plt.show()


def print_fwhm(positions, names, save=False, ext='fwhm_files', folder_path='./stars/graphs/'):
    """
    Function to plot the centroid of the star with the standard deviation

    :param positions: list of one axis of the coordinates for the centroid
    :param names: list of names for the files
    :param save: If 'True' save file to relative path: ./stars/graphs/
    """
    median = np.median(positions)
    # Plot image
    plt.figure()
    plt.title('FWHM per file')
    plt.errorbar(np.arange(len(names)), positions, fmt='.', ecolor='g', capsize=5)
    # Plot median line
    plt.axhline(median, color='r', linestyle='--')
    plt.xticks(np.arange(len(names)), names, rotation=90)
    plt.ylabel('FWHM')
    plt.xlabel('files')
    plt.grid()
    plt.tight_layout()
    if save is True:
        full_path = os.path.join(folder_path, '{}.png'.format(ext))
        plt.savefig(full_path)

    plt.show()


def print_sources(src):
    """
    Function to print the catalog of the stars in an image

    :param src: sources identifies in an image
    """
    # For consistent table output
    for col in src.colnames:
        src[col].info.format = '%.8g'

    print(src)


def plt_tele_stat(pos_list, exp_time=1, error=[], name='Default Name', ext='cx', telescope=1, ylabel='value'
                  , save=False, folder_path='./stars/graphs/'):

    x = np.arange(0, len(pos_list) * exp_time, exp_time)

    # Plot the positions of the centroids of the brightest star
    median = np.nanmedian(pos_list)
    # stdev = np.nanstd(pos_list, ddof=1)
    plt.figure()
    plt.title('{} Telescope#{}'.format(name, telescope))
    # plt.errorbar(arr, pos_x, yerr=error, fmt='.', ecolor='g',
    #              capsize=5)
    # for i in range(len(pos_list)):
    #     if not np.isnan(pos_list[i]):
    if len(error) > 0:
        # plt.fill_between(x, pos_list - error, pos_list + error, alpha=0.3)
        plt.errorbar(x, pos_list, yerr=error, fmt='o', ecolor='black',
                     capsize=5)
    else:
        plt.plot(x, pos_list, 'b.')
    plt.ylabel(ylabel)
    plt.xlabel('Time (s)')
    plt.axhline(median, color='r', linestyle='--')
    plt.grid()
    plt.tight_layout()
    if save is True:
        # plt.savefig('./stars/graphs/{}_{}.{}.png'.format(name, ext, telescope))
        full_path = os.path.join(folder_path, '{}_{}.{}.png'.format(name, ext, telescope))
        plt.savefig(full_path)

    plt.show()


def pos_std(data_list):
    mask = np.isnan(data_list)
    hist_data = np.array(data_list)[np.logical_not(mask)]

    hist, bin_edges = np.histogram(hist_data)
    hist = hist / sum(hist)

    n = len(hist)
    x_hist = np.zeros(n, dtype=float)
    for ii in range(n):
        x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2

    y_hist = hist

    mean = sum(x_hist * y_hist) / sum(y_hist)
    sigma = sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)

    return hist_data, sigma, mean, x_hist, y_hist


def plt_tele_hist(data_list, name='Default Name', ext='Data', telescope=1, xlabel='frequency', save=False,
                  folder_path='./stars/graphs/'):

    # for value in data_list:
    #     if isinstance(value, np.float64):
    #         hist_data.append(value)
    #
    # hist_data = np.delete(data_list, np.where(data_list == np.nan))

    # mask = np.isnan(data_list)
    # hist_data = np.array(data_list)[np.logical_not(mask)]
    #
    # hist, bin_edges = np.histogram(hist_data)
    # hist = hist / sum(hist)
    #
    # n = len(hist)
    # x_hist = np.zeros(n, dtype=float)
    # for ii in range(n):
    #     x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2
    #
    # y_hist = hist
    #
    # mean = sum(x_hist * y_hist) / sum(y_hist)
    # sigma = sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)

    hist_data, sigma, mean, x_hist, y_hist = pos_std(data_list)

    def gaus(X, C, X_mean, sigma):
        return C * exp(-(X - X_mean) ** 2 / (2 * sigma ** 2))

    # Gaussian least-square fitting process
    param_optimised, param_covariance_matrix = curve_fit(gaus, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=5000)

    # STEP 4: PLOTTING THE GAUSSIAN CURVE -----------------------------------------
    x_hist_2 = np.linspace(np.min(x_hist), np.max(x_hist), 500)
    plt.plot(x_hist_2, gaus(x_hist_2, *param_optimised), 'r.:', label='Gaussian fit')
    plt.legend()

    # Normalise the histogram values
    weights = np.ones_like(hist_data) / len(hist_data)
    plt.hist(hist_data, weights=weights)

    # setting the label,title and grid of the plot
    plt.title('{} Telescope#{}'.format(name, telescope))
    plt.xlabel(xlabel)
    plt.ylabel('Probability')
    plt.grid()
    plt.tight_layout()
    if save is True:
        full_path = os.path.join(folder_path, '{}_hist_{}.{}.png'.format(name, ext, telescope))
        plt.savefig(full_path)
    plt.show()

    return sigma


def plt_file_dist(data_list, name='Default Name', exp_time=1, error=[], ext='Data', ylabel='', xlabel='', save=False,
                   folder_path='./stars/graphs/'):

    x = np.arange(0, len(data_list[0]) * exp_time, exp_time)

    fig, axs = plt.subplots(nrows=len(data_list), ncols=2, figsize=(8.3, 11.7))
    fig.suptitle(name)

    hist_list = []
    weights = []
    mean_list = []

    total = data_list

    if len(error) > 0:
        total = np.array(data_list) + np.array(error)

    mask_nan = np.isnan(total)
    new_err = np.array(total)[np.logical_not(mask_nan)]

    max_y = np.max(new_err)
    min_y = np.min(new_err)

    for tele in range(len(data_list)):

        hist_data, _, _, _, _ = pos_std(data_list[tele])
        mean_list.append(np.nanmean(data_list[tele]))
        hist_list.append(hist_data)
        
    for k in range(len(data_list)):
        weights.append(np.ones_like(hist_list[k]) / len(hist_list[k]))
    
    if len(error) > 0:
        
        for i in range(len(data_list)):
            axs[i, 0].hist(hist_list[i], weights=weights[i])
            axs[i, 1].errorbar(x, data_list[i], yerr=error[i], fmt='o', ecolor='black',
                           capsize=5)

    else:
        for i in range(len(data_list)):
            axs[i, 0].hist(hist_list[i], weights=weights[i])
            axs[i, 1].plot(x, data_list[i], 'b.')

    for i in range(len(data_list)):
        axs[i, 0].set_title('Telescope #{}'.format(i + 1))
        axs[i, 1].set_title('Telescope #{}'.format(i + 1))
        axs[i, 1].set_ylabel(ylabel)
        axs[i, 1].set_xlabel(xlabel)
        axs[i, 0].set_xlabel(ylabel)
        axs[i, 0].set_ylabel('Probability')

        axs[i, 1].axhline(mean_list[i], color='r', linestyle='--')

        axs[i, 1].set_ylim(min_y, max_y)
        axs[i, 0].set_xlim(min_y, max_y)

    plt.tight_layout()

    if save is True:
        full_path = os.path.join(folder_path, '{}_hist_{}.png'.format(name, ext))
        plt.savefig(full_path)
    plt.show()


def plt_file_stats(data_list, name='Default Name', exp_time=1, save=False, folder_path='./stars/graphs/'):

    x = np.arange(0, len(data_list[0][0]) * exp_time, exp_time)
    stat_name = ['c_x', 'c_y', 'sigma', 'max_a']
    y_label = ['Centroid x', 'Centroid y', 'Sigma', 'Amplitude']

    for num in range(4):
        if num < 2:
            ncols=2
        else:
            ncols=1

        fig, axs = plt.subplots(nrows=len(data_list[0]), ncols=ncols, figsize=(8.3, 11.7))
        fig.suptitle(name)

        if num < 2:

            max_y = 0
            min_y = 100000

            for tele in range(len(data_list[num])):
                hist_data, sigma, mean, x_hist, y_hist = pos_std(data_list[num][tele])

                def gaus(X, C, X_mean, sigma):
                    return C * exp(-(X - X_mean) ** 2 / (2 * sigma ** 2))

                # Gaussian least-square fitting process
                param_optimised, param_covariance_matrix = curve_fit(gaus, x_hist, y_hist,
                                                                     p0=[max(y_hist), mean, sigma], maxfev=5000)

                # STEP 4: PLOTTING THE GAUSSIAN CURVE -----------------------------------------
                x_hist_2 = np.linspace(np.min(x_hist), np.max(x_hist), 500)
                axs[tele, 0].plot(x_hist_2, gaus(x_hist_2, *param_optimised), 'r.:', label='Gaussian fit')
                axs[tele, 0].legend()

                weight = np.ones_like(hist_data) / len(hist_data)

                axs[tele, 0].hist(hist_data, weights=weight)
                axs[tele, 1].errorbar(x, data_list[num][tele], yerr=sigma, fmt='o', ecolor='black',
                                   capsize=5)
                axs[tele, 1].axhline(mean, color='r', linestyle='--')

                axs[tele, 0].set_title('Telescope #{}'.format(tele + 1))
                axs[tele, 1].set_title('Telescope #{}'.format(tele + 1))
                axs[tele, 1].set_ylabel(y_label[num])
                axs[tele, 1].set_xlabel('Time (s)')
                axs[tele, 0].set_xlabel(y_label[num])
                axs[tele, 0].set_ylabel('Probability')

                total = np.array(data_list[num][tele]) + sigma

                maxi = np.nanmax(total)
                mini = np.nanmin(total)

                if max_y < maxi:
                    max_y = maxi

                if min_y > mini:
                    min_y = mini

            for i in range(len(data_list[num])):
                axs[i, 1].set_ylim(min_y, max_y)
                axs[i, 0].set_xlim(min_y, max_y)

        else:

            # total = np.array(data_list[num]) + np.array(data_list[num+2])
            # mask_nan = np.isnan(total)
            # new_err = np.array(total)[np.logical_not(mask_nan)]
            #
            # max_y = np.max(new_err)
            # min_y = np.min(new_err)

            for tele in range(len(data_list[num])):

                median = np.nanmedian(data_list[num][tele])
                # stdev = np.nanstd(pos_list, ddof=1)

                axs[tele].set_title('Telescope #{}'.format(tele + 1))
                # plt.errorbar(arr, pos_x, yerr=error, fmt='.', ecolor='g',
                #              capsize=5)
                # for i in range(len(pos_list)):
                #     if not np.isnan(pos_list[i]):
                    # plt.fill_between(x, pos_list - error, pos_list + error, alpha=0.3)
                axs[tele].errorbar(x, data_list[num][tele], yerr=data_list[num+2][tele], fmt='o', ecolor='black',
                                 capsize=5)
                axs[tele].set_ylabel(y_label[num])
                axs[tele].set_xlabel('Time (s)')
                axs[tele].axhline(median, color='r', linestyle='--')
                # axs[tele].set_ylim(min_y, max_y)
                plt.grid()

        plt.tight_layout()

        if save is True:
            full_path = os.path.join(folder_path, '{}_stat_{}.png'.format(name, stat_name[num]))
            plt.savefig(full_path)
        plt.show()

'''''''                    DATA GENERATION                                                                       '''''''


def generate_homography():
    # Define the range of allowable homography parameters
    shift_range = 250*0.1  # allow up to 10% shift in x and y direction
    rotation_range = np.pi / 4  # allow up to 45 degrees of rotation
    scale_range = 0.1  # allow up to 10% scaling

    # Generate a random homography
    shift = np.random.uniform(-shift_range, shift_range, size=2)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1 + scale_range, size=2)
    matrix = np.array([
        [scale[0] * np.cos(rotation), -scale[1] * np.sin(rotation), shift[0]],
        [scale[1] * np.sin(rotation), scale[0] * np.cos(rotation), shift[1]],
        [0, 0, 1]
    ])

    return matrix


def create_image(positions, amplitudes, fwhm=5, shape=(250, 250), bg_level=100, bg_noise_level=10):

    # Create a source table
    sources = Table()
    sources['x_mean'] = [pos[0] for pos in positions]
    sources['y_mean'] = [pos[1] for pos in positions]
    sources['amplitude'] = amplitudes
    sources['x_stddev'] = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    sources['y_stddev'] = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))

    image = make_gaussian_sources_image(shape, sources)

    background = np.random.normal(bg_level, bg_noise_level, shape)
    background_image = image + background

    return background_image


def new_centroids(positions, homography):
    new_list = []
    for point in positions:
        # Add homogeneous coordinate to the point
        point_hom = np.array([point[0], point[1], 1]).reshape((3, 1))

        new_point_hom = np.dot(homography, point_hom)
        new_point_hom = np.delete(new_point_hom, -1)

        new_list.append(new_point_hom)

    return new_list


def generate_sources(num, shape, focus, amp_max, amp_min):
    position_list = [[random.randint(focus, shape-focus), random.randint(focus, shape-focus)]]
    amp_list = [random.randint(amp_min, amp_max)]

    while len(position_list) < num:
        new_point = (random.randint(focus, shape-focus), random.randint(focus, shape-focus))
        if all(((new_point[0] - x[0]) ** 2 + (new_point[1] - x[1]) ** 2 >= 300) for x in position_list):
            position_list.append(new_point)
            amp_list.append(random.randint(amp_min, amp_max))

    return position_list, amp_list


def generate_data(num_src, focus, amp_max, amp_min, dist=5, fwhm=5, shape=(250, 250), bg_level=100, bg_noise_level=10):

    positions, amplitudes = generate_sources(num_src, shape[0], focus, amp_max, amp_min)

    src_img = create_image(positions, amplitudes, fwhm, shape, bg_level, bg_noise_level)
    img_list = [src_img]
    homography_list = []
    while len(img_list) < 4:
        while True:
            homography = generate_homography()
            corners = np.array([[0, 0, 1], [shape[0], 0, 1], [0, shape[1], 1], [shape[0], shape[1], 1]])
            transformed_corners = np.dot(homography, corners.T).T
            if np.all(abs(transformed_corners[:, :2] - corners[:, :2]) < dist):
                break
        new_pos = new_centroids(positions, homography)
        new_img = create_image(new_pos, amplitudes, fwhm, shape, bg_level, bg_noise_level)
        print(homography)
        homography_list.append(homography)
        img_list.append(new_img)

    return img_list, homography_list
