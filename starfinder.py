from PIL import Image
import numpy as np
from astropy.io import fits
import matplotlib.image
import matplotlib.pyplot as plt
import glob
from pathlib import Path, PureWindowsPath

# Path to the .fits files and conversion to the user OS
fits_files = PureWindowsPath("C:/Users/Sengo/Desktop/Dissertação/stars/fits/*.fits")
fits_files = Path(fits_files)
fits_files = glob.glob(str(fits_files))

# Width and Height for each telescope in original .fits file
min_h = 0
max_h = 250
t1_min = 0
t1_max = t2_min = 250
t2_max = t3_min = 500
t3_max = t4_min = 750
t4_max = 1000


# Function that opens every .fits file in the fits folder and divides the images based on telescope and frame. When
# opening the new .fits, the data is organized in an array where the first position denotes the frame and the second
# position the telescope
def div_files():
    # Iterates between every .fits file in the fits folder
    for fits_file in fits_files:
        # name of the file with 46 being the number of caracters in the path and -5 the '.fits'
        nname = fits_file[46:-5]
        gravity_file = fits.open(fits_file)

        images = gravity_file[4].data

        tele_list = []

        # Iterates between every frame of a fits file
        for image in images[:][:]:

            # Divide every frame by telescope, storing while maintaining frame order
            t_list = [image[min_h:max_h, t1_min:t1_max], image[min_h:max_h, t2_min:t2_max],
                      image[min_h:max_h, t3_min:t3_max], image[min_h:max_h, t4_min:t4_max]]

            tele_list.append(t_list)

        # Create a .fits file containing only the telescope images of the stars
        hdu_list = fits.PrimaryHDU(tele_list)
        hdu_list.writeto('C:/Users/Sengo/Desktop/Dissertação/stars/images/{}_{}.fits'
                         .format(nname, 'pt'))

        gravity_file.close()
