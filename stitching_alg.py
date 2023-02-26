from astropy.io import fits
import glob
from pathlib import Path, PureWindowsPath
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from tqdm import tqdm
import starfinder as sf
import os
import warnings

'''''''''
Example of usage for the starfinder.py library and image stitching alghorithm for images acquired by GRAVITY.
It's designed for SrgA object, but with further development it will be applicable to most cases.

Before using the use of the function div_files from starfinder.py is required. There is an example of it´s usage in my
GitHub. This function will acess all the files in an specified folder, (relative to the code: stars/fits), 
converting all the fits files to a fits file where only the images remain. The files are then put in a specific folder,
(relative to the code: stars/images). Following this, the script can be ran and all the images in the "images" folder 
will be stitched.

The algorithm is as follows:
    1- All the frames from all the telescopes are evaluated and qualified for their quality (usable/not usable), and 
appended to a list with a different format for easier stacking
    2- The frames are then stacked together per telescope
    3- The result stacked images are then evaluated to and qualified for their quality (usable/not usable)
    4- The first telescope image deemed usable (usually Telescope #1) is then used has reference for the homography
    5- Then, and iterativelly, the estimated homography of each image is calculated in relation to the before mensioned 
    image and warped.
    6- Finally the warped images are stacked and evaluated for their quality
'''''''''

# Turn off warning in case the DAOStarFinder doesn´t identify any stars that fit our criteria
warnings.filterwarnings('ignore')

'''''''                    SETUP PARAMETERS                                                                      '''''''
# Path to the the images containing the stars divides by telescope .fits files and conversion to the user OS
stars_files = PureWindowsPath("./stars/images/*.fits")
stars_files = Path(stars_files)
stars_files = glob.glob(str(stars_files))

# Number of expecteded main stars for image assessment
num_stars = 3

# Setup for plots
# True -> Yes | False -> No
plt_aprt = True
plt_img = True
plt_tele = True
plt_centroids = True

# Setup for save plots
# True -> Yes | False -> No
# For aperture of master image
save_aprt = True
# For simple of master image
save_img = True
# For all stacked telescopes
save_tele = True
# For the centroids in the files with three stars or all files
save_centroids = True

'''''''                    OTHER SETUPS                                                                          '''''''
# Name list for plots
name_list = []

# Setup Image Normalization for plot
norm = ImageNormalize(stretch=SqrtStretch())

# Lists to store master images
master_list = []

# Lists to store centroid values of the brightest star of all the images
pos_x = []
pos_y = []

'''''''---------------------STITCHING ALGORITHM------------------------------------------------------------------'''''''
# Loop to access all files in fits folder
for fits_file in tqdm(stars_files):

    print("NEW FILE")
    # Open the fits file
    stars_file = fits.open(fits_file)

    # Get name to name files
    name = os.path.splitext(os.path.basename(fits_file))[0][:29]

    # Extract data from .fits file
    images = stars_file[0].data

    '''''''                 TELESCOPE STACKING                                                                   '''''''
    # Check quality of all frames of all telescopes
    frame_quality = sf.frame_quality(images, num_stars)

    # Stack all the images deemed usable by the frame_quality() function
    tele_list = sf.tele_stitcher(frame_quality)

    # Plot stacked images
    if plt_tele is True:
        sf.show_telescopes(tele_list, name, save_tele)

    '''''''                 IMAGE STITCHING                                                                      '''''''
    # Check the quality of the final images of all telescopes to deem which are usable for the stitching process
    stitch_checked = sf.tele_quality(tele_list, num_stars)

    # Check if there are any usefull images from the previous step
    if len(stitch_checked) > 1:
        final_img = sf.final_stitcher(stitch_checked)

        # Check the quality of the master image
        if sf.stitch_quality(final_img, num_stars) is True:
            master_bg, _, _, std = sf.background_noise(final_img)
            master_stars = sf.find_stars(final_img, std, 'stitched')

            # Append both the master image and the name of the file to a list
            master_list.append(final_img)
            name_list.append(name[20:30])
            # Append centroid positions of brighetest star to lists
            brightest = sf.main_stars(master_stars, 1)
            pos_x.append(brightest[0][1])
            pos_y.append(brightest[0][2])

            # Plot apertures in the master image
            if plt_aprt is True:
                sf.show_apertures(final_img, master_stars, name, save_aprt)

            # Plot the master image
            if plt_img is True:
                sf.show_stars(final_img, name, 'master', save_img)

            # Print the sources of the master image
            sf.print_sources(master_stars)

    # Close the fits file
    stars_file.close()

'''''''                 DATA ANALYSIS                                                                            '''''''
#%%
# Plot the positions of the centroids of the brightest star
if plt_centroids is True:
    sf.print_centroid(pos_x, name_list, 'x of centroids', 'pos_x', 'all', 'files', 'x', save_centroids)
    sf.print_centroid(pos_y, name_list, 'y of centroids', 'pos_y', 'all', 'files', 'y', save_centroids)
print("FINISH")

