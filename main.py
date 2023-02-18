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
import starfinder as sf

# Path to the original .fits files and conversion to the user OS
# fits_files = PureWindowsPath("./stars/fits/*.fits")
fits_files = PureWindowsPath("C:/Users/Sengo/Desktop/Dissertação/stars/fits/*.fits")
fits_files = Path(fits_files)
fits_files = glob.glob(str(fits_files))

# Path to the the images containing the stars divides by telescope .fits files and conversion to the user OS
# stars_files = PureWindowsPath("./stars/images/*.fits")
stars_files = PureWindowsPath("C:/Users/Sengo/Desktop/Dissertação/stars/images/*.fits")
stars_files = Path(stars_files)
stars_files = glob.glob(str(stars_files))

# Number of caracters in the path and the '.fits'
path = 46
f_type = -5

# Setup Image Normalization for plot
norm = ImageNormalize(stretch=SqrtStretch())

for fits_file in stars_files:

    stars_file = fits.open(fits_file)

    name = stars_file[path:-f_type]

    images = stars_file[0].data
# stars_file = fits.open('C:/Users/Sengo/Desktop/Dissertação/stars/images/'
#                        'GRAVI.2021-06-25T04_17_48.414_pt.fits')
# images = stars_file[0].data

    tele_list = []
    for tele in range(len(images[0])):

        res_tele = images[0][tele]
        for frame in range(len(images)):

            # Method of combining frames
            res_bg = sf.backgroundNoise(res_tele)
            res_tele = sf.prepareStars(res_tele, res_bg[0])

            frame_bg = sf.backgroundNoise(images[frame][tele])
            images[frame][tele] = sf.prepareStars(images[frame][tele], frame_bg[0])

            res_tele = sf.joinImages(res_tele, images[frame][tele])

        tele_list.append(res_tele)

        # Method of correcting the positions for 1 chosen frame
        #     res_bg = sf.backgroundNoise(res_tele)
        #     res_tele = sf.prepareStars(res_tele, res_bg[0])
        #     res_stars = sf.findStars(res_tele, res_bg[3])
        #     res_kp = sf.brightestKeypoints(res_stars)
        #     res_coords = sf.starsCoordinates(res_kp)
        #     # print(master_coords)
        #
        #     frame_bg = sf.backgroundNoise(images[frame][tele])
        #     images[frame][tele] = sf.prepareStars(images[frame][tele], frame_bg[0])
        #     frame_stars = sf.findStars(images[frame][tele], frame_bg[3])
        #     frame_kp = sf.brightestKeypoints(frame_stars)
        #     frame_coords = sf.starsCoordinates(frame_kp)
        #
        #
        #
        #     matches, frame_match, res_match = sf.starsMatcher(frame_coords, res_coords)
        #     H, status = cv2.findHomography(frame_match, res_match)
        #
        #     res = sf.joinImages_homography(images[frame][tele], res_tele, H)
        #     print(frame)
        #
        #
        # tele_list.append(res_tele)

    plt.subplot(141)
    plt.imshow(tele_list[0], norm=norm, origin='lower', cmap='Greys_r',
               interpolation='nearest')
    plt.title("Tele 1", fontweight='bold')

    plt.subplot(142)
    plt.imshow(tele_list[1], norm=norm, origin='lower', cmap='Greys_r',
               interpolation='nearest')
    plt.title("Tele 2", fontweight='bold')

    plt.subplot(143)
    plt.imshow(tele_list[2], norm=norm, origin='lower', cmap='Greys_r',
               interpolation='nearest')
    plt.title("Tele 3", fontweight='bold')

    plt.subplot(144)
    plt.imshow(tele_list[3], norm=norm, origin='lower', cmap='Greys_r',
               interpolation='nearest')
    plt.title("Tele 4", fontweight='bold')
    plt.show()

    master_img = tele_list[0]
    for tele in range(1, len(tele_list)):

        #
        # master_bg = sf.backgroundNoise(master_img)
        # master_img = sf.prepareStars(master_img, master_bg[0])
        # master_stars = sf.findStars(master_img, master_bg[3])
        # master_kp = sf.brightestKeypoints(master_stars)
        # master_coords = sf.starsCoordinates(master_kp)
        # print(master_coords)
        #
        # tele_bg = sf.backgroundNoise(tele_list[tele])
        # tele_list[tele] = sf.prepareStars(tele_list[tele], tele_bg[0])
        # tele_stars = sf.findStars(tele_list[tele], tele_bg[3])
        # tele_kp = sf.brightestKeypoints(tele_stars)
        # tele_coords = sf.starsCoordinates(tele_kp)
        #
        # matches, tele_match, master_match = sf.starsMatcher(tele_coords, master_coords)
        # H, status = cv2.findHomography(tele_match, master_match)
        #
        # res = sf.joinImages(tele_list[tele], master_img, H)

        master_bg = sf.backgroundNoise(master_img)
        master_img = sf.prepareStars(master_img, master_bg[0])
        master_stars = sf.findStars(master_img, master_bg[3])
        master_kp = sf.brightestKeypoints(master_stars)
        master_coords = sf.starsCoordinates(master_kp)

        tele_bg = sf.backgroundNoise(tele_list[tele])
        tele_list[tele] = sf.prepareStars(tele_list[tele], tele_bg[0])
        tele_stars = sf.findStars(tele_list[tele], tele_bg[3])
        tele_kp = sf.brightestKeypoints(tele_stars)
        tele_coords = sf.starsCoordinates(tele_kp)

        matches, master_match, tele_match = sf.starsMatcher(master_coords, tele_coords)
        new_positions = sf.newCoords(matches)

        master_homography, _ = cv2.findHomography(master_match, new_positions)
        tele_homography, _ = cv2.findHomography(tele_match, new_positions)

        warped_master = sf.warpImage(master_img, master_homography)
        warped_tele = sf.warpImage(tele_list[tele], tele_homography)

        master_img = sf.joinImages(warped_master, warped_tele)

    sf.showStars(master_img, "{}".format(name))

    stars_file.close()


# plt.imshow(tele_list[0], norm=norm, origin='lower', cmap='Greys_r',
#                interpolation='nearest')
# plt.show()


