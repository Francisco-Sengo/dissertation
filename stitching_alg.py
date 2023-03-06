from astropy.io import fits
import glob
from pathlib import Path, PureWindowsPath
import starfinder as sf
import os
import warnings

'''''''''
Example of usage for the starfinder.py library and image stitching alghorithm for images acquired by GRAVITY.
Ideally, for visualization of the ploted graphs, images where there are the same number of objects is sugested.

Before using the use of the function div_files from starfinder.py is required. There is an example of it´s usage in my
GitHub. This function will acess all the files in an specified folder, (relative to the code: stars/fits), 
converting all the fits files to a fits file where only the images remain. The files are then put in a specific folder,
(relative to the code: stars/images). Following this, the script can be ran and all the images in the "images" folder 
will be stitched.

The algorithm is as follows:
    1- All the frames from all the telescopes are evaluated and qualified for their quality (usable/not usable), and 
appended to a list with a different format for easier stacking
    2- The frames are then stacked together per telescope
    3- The first telescope image (Telescope #1) is then used has reference for the homography
    4- Then, and iterativelly, the estimated homography of each image is calculated in relation to the before mensioned 
    image and warped.
    5- Finally the warped images are stacked and evaluated for their quality
'''''''''

# Turn off warning in case the DAOStarFinder doesn´t identify any stars that fit our criteria
warnings.filterwarnings('ignore')

'''''''                    SETUP PARAMETERS                                                                      '''''''
# Path to the the images containing the stars divides by telescope .fits files and conversion to the user OS
stars_files = PureWindowsPath("./stars/images/*.fits")
stars_files = Path(stars_files)
stars_files = glob.glob(str(stars_files))

# Number of expecteded stars for image assessment
num_stars = 6

# Setup for plots
# True -> Yes | False -> No
# For aperture of master image
plt_aprt = True
# For simple of master image
plt_img = True
# For all stacked telescopes
plt_tele = True
# For the centroids of the brightest stars of all files
plt_centroids = True
# For the isolated brightest star and the fitted gaussian
plt_gauss = True
# For the fwhm of the brightest stars of all files
plt_fwhm = True

# Setup for save plots
# True -> Yes | False -> No
# For aperture of master image
save_aprt = False
# For simple of master image
save_img = False
# For all stacked telescopes
save_tele = False
# For the centroids of the brightest stars of all files
save_centroids = False
# For the isolated brightest star and the fitted gaussian
save_gauss = False
# For the fwhm of the brightest stars of all files
save_fwhm = True

'''''''                    OTHER SETUPS                                                                          '''''''
# Name list for plots
name_list = []

# Lists to store master images
master_list = []

# Lists to store centroid values of the brightest star of all the images
pos_x = []
pos_y = []

# List to store the standard deviation of the centroid of the brightest star of all the images
x_std_list = []
y_std_list = []

# List to store FWHM of the brightest star in each image
fwhm_list = []

file_counter = 0
'''''''-------------------STITCHING ALGORITHM--------------------------------------------------------------------'''''''
# Loop to access all files in fits folder
for fits_file in stars_files:

    # Open the fits file
    stars_file = fits.open(fits_file)

    # Get name to name files
    name = os.path.splitext(os.path.basename(fits_file))[0][:29]

    # Extract data from .fits file
    images = stars_file[0].data

    print(name, "(", file_counter + 1, "/", len(stars_files), ")")

    '''''''                 TELESCOPE STACKING                                                                   '''''''
    # Check quality of all frames of all telescopes
    frame_quality = sf.frame_quality(images, num_stars)

    # Stack all the images deemed usable by the frame_quality() function
    tele_list = sf.tele_stitcher(frame_quality)

    tele_bg, _, _, tele_std = sf.background_noise(tele_list[0])
    tele_stars = sf.find_stars(tele_list[0], tele_std, flux_min_normal=15)
    brightest_tele = sf.main_stars(tele_stars, 1)

    # Acquire maximum amplite of telescope one for the plot
    _, _, _, amp_max_tele = sf.star_stats(tele_list[0], brightest_tele[0], tele_std, plt_gauss=False)

    # Plot stacked images
    if plt_tele is True:
        sf.show_telescopes(tele_list, amp_max_tele, name, save_tele)

    '''''''                 IMAGE STITCHING                                                                      '''''''

    # Check if there are any usefull images from the previous step
    if len(tele_list) == 4:
        final_img = sf.final_stitcher(tele_list)

        if len(final_img) == 0:
            print("UNFIT FILE FOR IMAGE STITCHING")
            continue

        # Check the quality of the master image
        if sf.stitch_quality(final_img, 2) is True:
            master_bg, _, _, std = sf.background_noise(final_img)
            master_stars = sf.find_stars(final_img, std, 'stitched')

            # Append both the master image and the name of the file to a list
            master_list.append(final_img)
            name_list.append(name[14:22])

            # Append centroid positions of brighetest star to lists
            brightest = sf.main_stars(master_stars, 1)

            # Calculate the FWHM, standard deviation of the position of the centroid and maximum amplitude
            # of the brightest star
            fwhm, x_std, y_std, amp_max = sf.star_stats(final_img, brightest[0], std, name,
                                                        plt_gauss=plt_gauss, save=save_gauss)

            # Append stats to corresponding lists for plots
            x_std_list.append(x_std)
            y_std_list.append(y_std)
            fwhm_list.append(fwhm)
            pos_x.append(brightest[0][1])
            pos_y.append(brightest[0][2])

            # Plot apertures in the master image
            if plt_aprt is True:
                sf.show_apertures(final_img, master_stars, amp_max, name, save_aprt)

            # Plot the master image
            if plt_img is True:
                sf.show_stars(final_img, amp_max, name, 'master', save_img)

            # Print the sources of the master image
            sf.print_sources(master_stars)

    # Close the fits file
    stars_file.close()

    # Increase file counter
    file_counter = file_counter + 1
    print("_______________________________________________________________________________________________________")

'''''''                 DATA ANALYSIS                                                                            '''''''
#%%
# Plot the positions of the centroids of the brightest star
if plt_centroids is True:
    sf.print_centroid(pos_x, x_std_list, name_list,
                      'x of centroids', 'pos_x', 'files', 'x', save_centroids)
    sf.print_centroid(pos_y, y_std_list, name_list,
                      'y of centroids', 'pos_y', 'files', 'y', save_centroids)

if plt_fwhm is True:
    sf.print_fwhm(fwhm_list, name_list, save_fwhm)

print("FINISH")

