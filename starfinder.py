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
mask_min = 50
mask_max = 200


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
        print(nname)

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

        # Create a .fits file containing only the telescope images of the stars
        hdu_list = fits.PrimaryHDU(tele_list)

        hdu_list.writeto('./stars/images/{}_{}.fits'
                         .format(nname, 'pt'))

        # Close .fits file
        gravity_file.close()


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
    threshold = np.mean(blured)*2
    max_value = np.max(blured)
    # Filter pixeis
    binary = cv2.threshold(img, threshold, max_value, thresh_type)[1]
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


def find_stars(img, std, img_type='normal', exp_fwhm=7.0, flux_min_normal=5, flux_min_stitch=25, peak_min_stitch=45):
    """
        Function used to identificate and qualify stars in an image 

        :param img: image to find stars
        :param std: standard deviation of the background
        :param img_type: 'normal' (Default) -> for raw image or stacking of telescopes
                         'stitched' -> for master image
        :param exp_fwhm: expected fwhm for the stars (optional)
        :param flux_min_normal : minimum flux for the 'normal' star_finder to consider a star (optional)
        :param flux_min_stitch: minimum flux for the 'stitched' star_finder to consider a star (optional)
        :param peak_min_stitch: minimum flux for the 'stitched' star_finder to consider a star (optional)
        :return: list containing all the stars that meet the criteria 
    """
    # Setup for Star Finder
    daofind = DAOStarFinder(fwhm=exp_fwhm, threshold=3. * std)

    # Create mask for region of interest
    mask = np.zeros(img.shape, dtype=bool)
    mask[:][:] = True
    mask[mask_min:mask_max, mask_min:mask_max] = False

    # To identify stars in the frames and stacked telescope images
    if img_type == 'normal':
        # Find sources in an image, cataloging them
        stars = daofind(img, mask=mask)
        # In case no stars are identified that fit our criteria
        if type(stars) == astropy.units.decorators.NoneType:
            return stars
        # Filter stars by flux
        stars = stars[stars['flux'] > flux_min_normal]

        return stars
    # To identify stars in the master image
    elif img_type == 'stitched':
        # Find sources in an image, cataloging them
        stars = daofind(img, mask=mask)
        # In case no stars are identified that fit our criteria
        if type(stars) == astropy.units.decorators.NoneType:
            return stars
        # Filter stars by flux and peak
        stars = stars[stars['flux'] > flux_min_stitch]
        stars = stars[stars['peak'] > peak_min_stitch]

        return stars
    # In case of unknown input, the stars are not filtered to fit extra criteria
    else:
        # Find sources in an image, cataloging them
        stars = daofind(img, mask=mask)
        # In case no stars are identified that fit our criteria

        return stars


def star_stats(img_o, src, std, name='Default Name', exp_fwhm=9, plt_gauss=True, save=False):
    """
    Function to calculate Full Width Half Maximum of a star

    :param img_o: image to acquire the star
    :param src: coordinates of the centroid of the star
    :param std: standard deviation used to acquire stars
    :param name: name of the file
    :param exp_fwhm: expected odd fwhm for the star (optional)
    :param plt_gauss: True -> plt brighest star and fitted gaussian |
                      False -> don't plt brighest star and fitted gaussian
    :param save: True -> Save the image in "brightest" folder |
                 False -> Don't save the image
    :return: full width half maximum of the star, the standard deviation of the position of the star and the maximum
    amplitude of the star
    """
    # Copy the image
    img = img_o.copy()
    # Isolate the star
    star_pixels = stars_box(img, src, exp_fwhm)
    # Mask star
    masked = mask_roi(star_pixels)
    # Acquire max value of the star
    amp_max = np.max(masked)

    # Estimate initial parameters for 2D Gaussian fit
    amp = amp_max - np.min(star_pixels)
    # Centroid of the box containing the star
    x_mean, y_mean = exp_fwhm, exp_fwhm
    # Initial guess for standard deviation
    x_stddev = y_stddev = std
    # Initial guess for rotation angle (radians)
    theta = 0.0
    init_params = {'amplitude': amp, 'x_mean': x_mean, 'y_mean': y_mean,
                   'x_stddev': x_stddev, 'y_stddev': y_stddev, 'theta': theta}
    gauss2d_init = Gaussian2D(**init_params)

    # Fit 2D Gaussian model to star pixels
    fit_p = LevMarLSQFitter()
    y, x = np.mgrid[:2*exp_fwhm, :2*exp_fwhm]
    gauss2d_fit = fit_p(gauss2d_init, x, y, masked)

    x_std = gauss2d_fit.x_stddev.value
    y_std = gauss2d_fit.y_stddev.value

    # Calculate FWHM from standard deviation of best-fit Gaussian
    fwhm = np.mean([2.3548 * x_std, 2.3548 * y_std])

    if plt_gauss is True:
        # Create a figure and subplots
        fig, axs = plt.subplots(nrows=1, ncols=2)

        # Plot each image in a subplot
        axs[0].imshow(star_pixels, origin='lower', cmap='inferno', vmin=0, vmax=amp)
        axs[1].imshow(gauss2d_fit(x, y), origin='lower', cmap='inferno', vmin=0, vmax=amp)

        # Idetify telescope in subplot
        axs[0].set_title('Original Star')
        axs[1].set_title('Fitted Gaussian')

        plt.tight_layout()
        if save is True:
            plt.savefig('./stars/brightest/{}_{}.png'.format(name, 'brightest_star'))
        plt.show()

    return fwhm, x_std, y_std, amp_max


def main_stars(src, num):
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


def stars_features(img, exp_fwhm=7, flux_min_normal=5):
    """
    Function that identifies the stars and acquires their coordinates for later matching

    :param img: image to acquire stars
    :param exp_fwhm: expected fwhm for the stars (optional)
    :param flux_min_normal : minimum flux for the 'normal' star_finder to consider a star (optional)
    :return: list containing the coordinates of the identified stars
    """
    # Acquire standard deviation
    _, _, _, std = background_noise(img)
    # Find sources in the image
    img_src = find_stars(img, std, 'normal', exp_fwhm, flux_min_normal)

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


def frame_quality(img_list, num, exp_fwhm=7, flux_min_normal=5):
    """
    Function that evaluates every frame in a file, creating a list containing every frame that detectes at least "num"
    of stars. It also removes the background from the frames.

    :param img_list: list of frames to be qualified
    :param num: number of stars required for the frame to be deemed usable
    :param exp_fwhm: expected fwhm for the stars (optional)
    :param flux_min_normal : minimum flux for the 'normal' star_finder to consider a star (optional)
    :return: return a list for the usable frames organized by telescope
    """
    frame_list = []
    total_weight = []
    # Iterate between all frames and all telescopes
    for tele in range(len(img_list[0])):
        tele_list = []
        weight_list = []
        for frame in range(len(img_list)):
            img = img_list[frame][tele]
            # Acquire background and standard deviation of the frame
            bg, _, _, std = background_noise(img)
            # Prepare frame for identification of stars
            img = prepare_stars(img, bg)
            # Acquire sources in the images
            stars = find_stars(img, std, 'normal', exp_fwhm, flux_min_normal)
            # In case no stars are identified that fit our criteria
            if type(stars) == astropy.units.decorators.NoneType:
                continue
            if len(stars) >= num:
                tele_list.append(img)
        frame_list.append(tele_list)

    return frame_list


def stitch_quality(stitched_img, num, exp_fwhm=7, flux_min_stitch=25, peak_min_stitch=45):
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
    stars = find_stars(stitched_img, std, 'stitched', exp_fwhm, flux_min_stitch, peak_min_stitch)
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


def final_stitcher(img_list):
    """
    Function to find homography and stitch all the telescope perspectives into a master image
    :param img_list: list of images of the four telescopes
    :return: master image
    """
    final_img = []
    # Define the telescope 1 as the reference for the stitching
    master_img = img_list[0]
    warped_list = [master_img]

    # Acquire coordinates from the reference image
    master_coords = stars_features(master_img)

    # Iteratively finde the homography and warp all the telescope images
    for tele in range(1, len(img_list)):

        tele_coords = stars_features(img_list[tele])

        # Match the stars between the two images
        matches, master_match, tele_match = stars_matcher(master_coords, tele_coords)

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


def show_stars(img, amp_max, title='Default Name', ext='', save=False):
    """
    Plot a single image of the stars

    :param img: image to be ploted
    :param amp_max: maximum amplitude of the brightest star
    :param title: title of the plot and name of the file
    :param ext: extention for the file
    :param save: If 'True' save file to relative path: ./stars/stitched/
    """

    # Setup Image Normalization for plot
    norm = ImageNormalize(stretch=SqrtStretch(), vmax=amp_max)

    plt.imshow(img, norm=norm, origin='lower', cmap='inferno',
               interpolation='nearest')
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    if save is True:
        plt.savefig('./stars/stitched/{}_{}.png'.format(title, ext))
    plt.show()


def show_apertures(img, src, amp_max, title='Default Name', save=False):
    """
    Plot apertures for a single image of the stars

    :param img: image to be ploted
    :param src: sources of the image to be identified
    :param amp_max: maximum amplitude of the brightest star
    :param title: title of the plot and name of the file
    :param save: If 'True' save file to relative path: ./stars/stitched/
    """
    # Apply circles in the positions of the sources
    positions = np.transpose((src['xcentroid'], src['ycentroid']))
    apertures = CircularAperture(positions, r=4.0)

    # Setup Image Normalization for plot
    norm = ImageNormalize(stretch=SqrtStretch(), vmax=amp_max)

    # Plot image
    plt.imshow(img, cmap='inferno', origin='lower', norm=norm,
               interpolation='nearest')
    apertures.plot(color='white', lw=1.5, alpha=0.5)
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    if save is True:
        plt.savefig('./stars/stitched/{}_{}.png'.format(title, "aprt"))
    plt.show()


def show_telescopes(img_list, amp_max, name='Default Name', save=False):
    """
    Plot all stacked images for the telescopes into a single image

    :param img_list: list of images of all four telescopes
    :param amp_max: maximum amplitude of the brightest star
    :param name: name of the file
    :param save: If 'True' save file to relative path: ./stars/stitched/
    """
    # Setup Image Normalization for plot
    norm = ImageNormalize(stretch=SqrtStretch(), vmax=amp_max)

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
        plt.savefig('./stars/stitched/{}_{}.png'.format(name, "telescopes"))
    plt.show()


def print_centroid(positions, stdev_est, names, title='Default Name', name='Default Name',
                   xlabel='file', ylabel='value', save=False):
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
        plt.savefig('./stars/graphs/{}.png'.format(name))

    plt.show()


def print_fwhm(positions, names, save=False):
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
        plt.savefig('./stars/graphs/{}.png'.format('fwhm_files'))

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
