from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS
import numpy as np
import sys
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from drizzle import drizzle
# from drizzlepac import astrodrizzle
# import drizzlepac
import starfinder as sf
import cv2

# stars_file = fits.open('./stars/images/GRAVI.2020-03-07T07_46_09.234_pt.fits')
#
# exp_time = stars_file[0].header['EXPTIME']
#
# images = stars_file[0].data
# norm = ImageNormalize(stretch=SqrtStretch())
#
# #%%
# w = WCS(stars_file[4].header, naxis=2).sub(2)
# w.array_shape = (250, 250)
# print(w)

images, name, exp_time, w = sf.open_file('./stars/images/GRAVI.2020-03-07T07_46_09.234_pt.fits')

#%%
driz1 = drizzle.Drizzle(outwcs=w)
driz2 = drizzle.Drizzle(outwcs=w)
driz3 = drizzle.Drizzle(outwcs=w)
driz4 = drizzle.Drizzle(outwcs=w)
for image in images:
    driz1.add_image(image[0], w, expin=exp_time)
    driz2.add_image(image[1], w, expin=exp_time)
    driz3.add_image(image[2], w, expin=exp_time)
    driz4.add_image(image[3], w, expin=exp_time)


#%%
img = [driz1.outsci, driz2.outsci, driz3.outsci, driz4.outsci]



#%%
# plt.imshow(img[0], cmap='inferno', norm=norm)
# plt.colorbar()
# plt.show()
#
# plt.imshow(img[1], cmap='inferno', norm=norm)
# plt.colorbar()
# plt.show()
#
# plt.imshow(img[2], cmap='inferno', norm=norm)
# plt.colorbar()
# plt.show()
#
# plt.imshow(img[3], cmap='inferno', norm=norm)
# plt.colorbar()
# plt.show()
#%%


for image in img:
    # _, mean, _, _ = sf.background_noise(image)
    # image = image-mean
    _, _, _, std = sf.background_noise(image)
    stars = sf.find_stars(image, std, flux_min=0, exp_fwhm=6)
    brightest_tele = sf.main_stars(stars, 1)
    # print(len(tele_stars))
    # Acquire maximum amplite of telescope one for the plot
    _, _, _, _, amp_max_tele = sf.star_stats(image, brightest_tele[0], std, plt_gauss=False)
    sf.show_apertures(image, stars, amp_max_tele)
    sf.print_sources(stars)

# sys.exit(0)
#%%
final_img = []
# Define the telescope 1 as the reference for the stitching
master_img = img[0]
warped_list = [master_img]

# Acquire coordinates from the reference image
master_coords = sf.stars_features(master_img, flux_min=0, exp_fwhm=6)

# Iteratively finde the homography and warp all the telescope images
for tele in range(1, len(img)):

    tele_coords = sf.stars_features(img[tele], flux_min=0, exp_fwhm=6)
    # print(tele_coords)
    # Match the stars between the two images
    matches, master_match, tele_match = sf.stars_matcher(master_coords, tele_coords, distance=6)

    print(matches)

    if matches == 0:
        print("NOT ENOUGH MATCHES TO CALCULATE HOMOGRAPHY")

    else:
    # Estimate the homography using the cv2 library
        H, _ = cv2.findHomography(tele_match, master_match)

        # Warp image base on the estimated homography
        warped_tele = sf.warp_image(img[tele], H)
        # Append the warped image from the telescope into a list
        warped_list.append(warped_tele)

    # Stack images to form the master image
final_img = sf.join_images(warped_list)

#%%
_, mean, _, std = sf.background_noise(final_img)
stars = sf.find_stars(final_img, std, flux_min=0, exp_fwhm=5)
sf.show_apertures(final_img, stars, np.max(final_img))

#TODO ADICIONAR WCS QUANDO ABRIR O FITS FILE ORIGINAL
#TODO ALTERAR CODIGO PARA USAR ISTO PARA JUNTAR FRAMES PER TELE
#TODO TESTAR ISTO NO STITCHING
#TODO TESTAR MAIS "DEFINIÇÕES" NAS FUNCOES DO PHOTUTILS E ASTROPY
#TODO COMENTAR STARFINDER.PY
#TODO GUARDAR IMAGENS COMO FITS TAMBEM
