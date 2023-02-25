from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.background import Background2D, MedianBackground, MeanBackground, ModeEstimatorBackground, MMMBackground, \
    SExtractorBackground, BiweightLocationBackground
from photutils.datasets import load_star_image
hdu = load_star_image()
data = hdu.data[0:401, 0:401]

import numpy as np

stars_file = fits.open('C:/Users/Sengo/Desktop/Dissertação/stars/images/GRAVI.2021-06-25T04_17_48.414_pt.fits')
bg_file = fits.open('C:/Users/Sengo/Desktop/Dissertação/stars/outros/ACQ_dark07_20171229_DIT_mean.fits')


images = stars_file[0].data

img = images[20][0]

print("Without BG extraction")
mean, median, std = sigma_clipped_stats(img, sigma=3.0)
print((mean, median, std))

norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(img, norm=norm, origin='lower', cmap ='Spectral',
           interpolation='nearest')
plt.title("Without BG extraction", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

###########################################################################

print("BiweightLocationBackground")

sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = BiweightLocationBackground()
bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(img - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(img - bkg.background, norm=norm, origin='lower', cmap ='magma',
           interpolation='nearest')
plt.title("BiweightLocationBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("SExtractorBackground")

bkg_estimator = SExtractorBackground()
bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(img - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(img - bkg.background, norm=norm, origin='lower', cmap ='viridis',
           interpolation='nearest')
plt.title("SExtractorBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("MMMBackground")

bkg_estimator = MMMBackground()
bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(img - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(img - bkg.background, norm=norm, origin='lower', cmap ='viridis',
           interpolation='nearest')
plt.title("MMMBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("ModeEstimatorBackground")

bkg_estimator = ModeEstimatorBackground()
bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(img - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(img - bkg.background, norm=norm, origin='lower', cmap ='plasma',
           interpolation='nearest')
plt.title("ModeEstimatorBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("MeanBackground")

bkg_estimator = MeanBackground()
bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(img - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(img - bkg.background, norm=norm, origin='lower', cmap ='cividis',
           interpolation='nearest')
plt.title("MeanBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("MedianBackground")

bkg_estimator = MedianBackground()
bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(img - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(img - bkg.background, norm=norm, origin='lower', cmap ='inferno',
           interpolation='nearest')
plt.title("MedianBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

img_bg = bg_file[0].data

bg = img_bg[0:250, 0:250]

print("Actual Background")

mean, median, std = sigma_clipped_stats(img - bg, sigma=3.0)
print((mean, median, std))

plt.imshow(img - bg, norm=norm, origin='lower', cmap='inferno',
           interpolation='nearest')
plt.title("Actual Background", fontweight ='bold')
plt.show()

###########################################################################

print("Without BG extraction w/ sintetic data")
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
print((mean, median, std))

norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, norm=norm, origin='lower', cmap ='magma',
           interpolation='nearest')
plt.title("Without BG extraction", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

###########################################################################

print("BiweightLocationBackground w/ sintetic data")

sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = BiweightLocationBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(data - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(data - bkg.background, norm=norm, origin='lower', cmap ='magma',
           interpolation='nearest')
plt.title("BiweightLocationBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("SExtractorBackground w/ sintetic data")

bkg_estimator = SExtractorBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(data - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(data - bkg.background, norm=norm, origin='lower', cmap ='cividis',
           interpolation='nearest')
plt.title("SExtractorBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("MMMBackground w/ sintetic data")

bkg_estimator = MMMBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(data - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(data - bkg.background, norm=norm, origin='lower', cmap ='cividis',
           interpolation='nearest')
plt.title("MMMBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("ModeEstimatorBackground w/ sintetic data")

bkg_estimator = ModeEstimatorBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(data - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(data - bkg.background, norm=norm, origin='lower', cmap ='Greys_r',
           interpolation='nearest')
plt.title("ModeEstimatorBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("MeanBackground w/ sintetic data")

bkg_estimator = MeanBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(data - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(data - bkg.background, norm=norm, origin='lower', cmap ='Greys_r',
           interpolation='nearest')
plt.title("MeanBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")

#############################################################################

print("MedianBackground w/ sintetic data")

bkg_estimator = MedianBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
mean, median, std = sigma_clipped_stats(data - bkg.background, sigma=3.0)
print((mean, median, std))

plt.imshow(data - bkg.background, norm=norm, origin='lower', cmap ='Greys_r',
           interpolation='nearest')
plt.title("MedianBackground", fontweight ='bold')
plt.show()

print("_____________________________________________________________________")
