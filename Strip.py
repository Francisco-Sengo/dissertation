from PIL import Image
import numpy as np
from astropy.io import fits
import matplotlib.image
import glob

fits_files = glob.glob("C:/Users/Sengo/Desktop/Dissertação/stars/fits/*.fits")


for fits_file in fits_files:

    nname = fits_file[13:-5]
    gravity_file = fits.open(fits_file)

    images = gravity_file[4].data

    i = 0
    for image in images[:][:]:

        my_roi = image[0:250, 0:1000]

        matplotlib.image.imsave('C:/Users/Sengo/Desktop/Dissertação/stars/images/{}_{}.png'.format(nname, i), my_roi)
        i = i + 1


    gravity_file.close()


