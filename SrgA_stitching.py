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
Example of usage for the starfinder.py library in the case of images acquired by GRAVITY of the SgrA object, 
for two case scenarios:
   - There are only two of the main stars visible -> in this case, due to the lack of points to estimate a proper 
   homography, the results don't prove to be as good as the next case. Besides this, in many of the frames, the supposed
   brightest star isn´t constant, so a special filter is made to correct that for data analysis.
   - Given that are three main stars in this scenario, an estimate of the homography is much more robust, having 
   perfomed better.

This file also contains an example of how to properly use all the functions in the library, but a simpler version of 
this algorithm is also available at GitHub.
'''''''''

# Turn off warning in case the DAOStarFinder doesn´t identify any stars that fit our criteria
warnings.filterwarnings('ignore')

'''''''                    SETUP PARAMETERS                                                                      '''''''
# Path to the the images containing the stars divides by telescope .fits files and conversion to the user OS
stars_files = PureWindowsPath("./stars/images/*.fits")
stars_files = Path(stars_files)
stars_files = glob.glob(str(stars_files))

# Number of expecteded main stars for image assessment
num_stars = 2

# Setup for plots
# True -> Yes | False -> No
plt_aprt = True
plt_img = True
plt_tele = True
plt_centroids_all = True
plt_centroids_three = True

# Setup for save plots
# True -> Yes | False -> No
# For aperture of master image
save_aprt = True
# For simple of master image
save_img = True
# For all stacked telescopes
save_tele = True
# For the centroids in the files with three stars or all files
save_centroids_three = True
save_centroids_all = True

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

# Lists to store centroid values of the brightest star of only the files with three stars
pos_x_b = []
pos_y_b = []

# List containing only the names of the files with three stars that the stitched images that passed the quality test
main_name = []

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
    frame_quality = sf.frame_quality(images, 2)

    # Stack all the images deemed usable by the frame_quality() function
    tele_list = sf.tele_stitcher(frame_quality)

    # Plot stacked images
    if plt_tele is True:
        sf.show_telescopes(tele_list, name, save_tele)

    '''''''                 IMAGE STITCHING                                                                      '''''''
    # Check the quality of the final images of all telescopes to deem which are usable for the stitching process
    stitch_checked = sf.tele_quality(tele_list, 2)

    # Check if there are any usefull images from the previous step
    if len(stitch_checked) > 1:
        final_img = sf.final_stitcher(stitch_checked)

        # Check the quality of the master image
        if sf.stitch_quality(final_img, 2) is True:
            master_bg, _, _, std = sf.background_noise(final_img)
            master_stars = sf.find_stars(final_img, std, 'stitched')

            # Append both the master image and the name of the file to a list
            master_list.append(final_img)
            name_list.append(name[20:30])

            # To filter out the images that only contain two stars
            if len(master_stars) < 3:
                # Append centroid positions
                _, matches, _ = sf.stars_matcher(sf.get_coordinates(master_stars), [[128, 158]])
                pos_x.append(matches[0][0])
                pos_y.append(matches[0][1])

            # For the images that contain the full three stars
            else:
                # Append centroid positions and file names to list
                brightest = sf.main_stars(master_stars, 1)
                pos_x.append(brightest[0][1])
                pos_y.append(brightest[0][2])
                main_name.append(name[20:30])
                pos_x_b.append(brightest[0][1])
                pos_y_b.append(brightest[0][2])

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
if plt_centroids_all is True:
    sf.print_centroid(pos_x, name_list, 'x of centroids', 'pos_x', 'all', 'files', 'x', save_centroids_all)
    sf.print_centroid(pos_y, name_list, 'y of centroids', 'pos_y', 'all', 'files', 'y', save_centroids_all)
if plt_centroids_three is True:
    sf.print_centroid(pos_y_b, main_name, 'y of centroids',  'centroid_y', 'three', 'files', 'y', save_centroids_three)
    sf.print_centroid(pos_x_b, main_name, 'x of centroids', 'centroid_x', 'three', 'files', 'x', save_centroids_three)
print("FINISH")

