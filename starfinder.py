import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
from pathlib import Path, PureWindowsPath
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import cv2
from photutils.aperture import CircularAperture
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder, IRAFStarFinder
from math import sqrt

# TODO organizar as bibliotecas e packages

# # Path to the original .fits files and conversion to the user OS
# # fits_files = PureWindowsPath("./stars/fits/*.fits")
# fits_files = PureWindowsPath("C:/Users/Sengo/Desktop/Dissertação/stars/fits/*.fits")
# fits_files = Path(fits_files)
# fits_files = glob.glob(str(fits_files))
#
# # Path to the the images containing the stars divides by telescope .fits files and conversion to the user OS
# # stars_files = PureWindowsPath("./stars/images/*.fits")
# stars_files = PureWindowsPath("C:/Users/Sengo/Desktop/Dissertação/stars/images/*.fits")
# stars_files = Path(stars_files)
# stars_files = glob.glob(str(stars_files))

# Width and Height for each telescope in original .fits file
min_h = 0
max_h = 250
t1_min = 0
t1_max = t2_min = 250
t2_max = t3_min = 500
t3_max = t4_min = 750
t4_max = 1000

# Number of caracters in the path and the '.fits'
path = 46
f_type = -5

# Setup for Background Estimator
bkg_estimator = MedianBackground()
sigma_clip = SigmaClip(sigma=3.0)

# Setup Image Normalization for plot
norm = ImageNormalize(stretch=SqrtStretch())


# Function that opens every .fits file in the fits folder and divides the images based on telescope and frame. When
# opening the new .fits, the data is organized in an array where the first position denotes the frame and the second
# position the telescope
def div_files(fits_files):
    # Iterates between every .fits file in the fits folder
    for fits_file in fits_files:
        # Name of the new file
        nname = fits_file[path:-f_type]

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

        # hdu_list.writeto('./stars/images/{}_{}.fits'
        #                 .format(nname, 'pt'))
        hdu_list.writeto('C:/Users/Sengo/Desktop/Dissertação/stars/images/{}_{}.fits'
                         .format(nname, 'pt'))

        # Close .fits file
        gravity_file.close()


# TODO: NÃO SEI AS DIMENSÕES QUE É SUPOSTO RECORTAR NA IMAGEM

# Background from image
# def backgroundNoise():
#     bg_file = fits.open('C:/Users/Sengo/Desktop/Dissertação/stars/outros/ACQ_dark07_20171229_DIT_mean.fits')
#
#     img_bg = bg_file[0].data
#     bg = img_bg[0:250, 0:250]
#     mean, median, std = sigma_clipped_stats(bg, sigma=3.0)
#
#     bg_file.close()
#
#     return bg


# Function to estimate background image and acquire mean, median and standard deviation of the original image
# with the background removed.
def backgroundNoise(img):
    # Aquire background estimation
    bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    # Acquire mean, median and standard deviation of original image without background
    mean, median, std = sigma_clipped_stats(img - bkg.background, sigma=3.0)

    return bkg.background, mean, median, std


# TODO: COMO RECORTAR OS DEAD PIXEIS DO FICHEIRO FITS PARA APLICAR AQUI (DIMENSÕES)
# Funtion to prepare the image for further analysis, removing the background
def prepareStars(img, noise):
    # Subtract the noise
    img = img - noise

    return img


# Function used to acquire the centroid, peak and flux of the sources in the image
def findStars(img, std):
    # Setup for Star Finder
    daofind = DAOStarFinder(fwhm=5.0, threshold=15. * std, brightest=3)
    # Find sources in an image, cataloging them
    sources = daofind(img)

    return sources


# Function to acquire position of the three brightest stars in the sources catalog of the image
def brightestStars(src):
    # Organize the catalog by flux and acquire the three brightest stars
    brightest = src['flux'].argsort()[::-1][:3]

    return brightest


# Function that decomposes the catalog into an list for easier access
def brightestKeypoints(src):
    # Acquire the positions in the catalog for the three brightest stars
    positions = brightestStars(src)

    star_data = []
    # For the three brightest stars, append relevant data into a list
    for i in range(0, len(positions)):
        star = [src['id'][positions[i]], src['xcentroid'][positions[i]], src['ycentroid'][positions[i]],
                src['peak'][positions[i]], src['flux'][positions[i]], src['mag'][positions[i]]]
        star_data.append(star)

    return star_data


# Function that puts the centroids in an list for the Star Matcher
def starsCoordinates(b_src):
    stars_coords = []
    # Append x and y coordinate respectively into a list
    for i in range(0, len(b_src)):
        coord = [b_src[i][1], b_src[i][2]]
        stars_coords.append(coord)

    return stars_coords


# Function that matches the three brightest of two telescopes through the coordinates
def starsMatcher(src_coords, dst_coords):
    matches_list = []
    match_1 = []
    match_2 = []

    # Match the coordinates of the same star between the two images
    for c1 in range(len(src_coords)):
        for c2 in range(len(dst_coords)):
            # Calculate distance between two stars in different images
            d = sqrt((src_coords[c1][0] - dst_coords[c2][0]) ** 2 + (src_coords[c1][1] - dst_coords[c2][1]) ** 2)
            # If the distance is lower than 10 px, the two stars match
            if d < 10:
                match = [src_coords[c1], dst_coords[c2]]
                matches_list.append(match)
                match_1.append(src_coords[c1])
                match_2.append(dst_coords[c2])

    # Append the same corner of the images as matches
    match_1.append([0, 0])
    match_2.append([0, 0])
    matches_list.append([[0, 0], [0, 0]])

    return matches_list, np.array(match_1), np.array(match_2)


# Function to print the catalog of the stars in an image
def printSources(src):
    # For consistent table output
    for col in src.colnames:
        src[col].info.format = '%.8g'

    print(src)


# Function that warps images given the estimated homography
def warpImage(src_img, H):
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


# Function that joins two images, given the estimated homography
def joinImages_homography(src_img, dst_img, H):
    # Define the corner points of the images (they share all four corners)
    dst_corners = np.array([[0, 0], [0, dst_img.shape[0]], [dst_img.shape[1], dst_img.shape[0]], [dst_img.shape[1], 0]],
                           dtype=np.float64)

    # Warp the source image to the destination image using the estimated homography
    warped_img = warpImage(src_img, H)

    # Apply the warped image as a mask on the destination image
    mask = np.zeros_like(dst_img, dtype=np.float64)
    mask = cv2.fillConvexPoly(mask, dst_corners.astype(np.int32), (1,))
    masked_dst_img = cv2.multiply(dst_img, mask)

    # Combine the warped image and the masked destination image
    result = cv2.add(warped_img, masked_dst_img, dtype=cv2.CV_64F) / 2

    return result


# Function that joins two images
def joinImages(img_1, img_2):
    # Create an image which is the average between the two
    #result = cv2.add(img_1, img_2, dtype=cv2.CV_64F) / 2
    result = (img_1 + img_2) / 2

    return result


# Calculate the new coordinates to estimate the homography
def newCoords(matches_list):
    new_coords = []
    # The new coordinates are the average of the two
    for coords in matches_list:
        c_x = (coords[0][0] + coords[1][0]) / 2
        c_y = (coords[0][1] + coords[1][1]) / 2

        # Append the new coordinates to an array
        new_c = [c_x, c_y]
        new_coords.append(new_c)

    return np.array(new_coords)


# Plot a single image of the stars
def showStars(img, title):
    plt.imshow(img, norm=norm, origin='lower', cmap='Greys_r',
               interpolation='nearest')
    plt.title(title, fontweight='bold')
    plt.show()

