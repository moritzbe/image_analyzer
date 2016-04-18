

# The goal of this exercise is to get aqcainted with noise, and the related 
# concepts of noise power spectra and correlation.

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.signal as nss

##############################################################################
# Part A: noise and correlation
# create noise, and calculate its noise power spectrum and correlation
##############################################################################

# Create a 100x100 array of Gaussian noise with mean=0 and standard deviation 
# sigma = 1. Use the function numpy.rand.randn.
# Then use scipy.ndimage.gaussian_filter to create a low and high pass filtered 
# version of your noise. (Remember from lecture 2: a high pass can be modelled as 
# the original image minus the low pass image)
sigma = 1

white_noise = np.random.randn(100, 100)
low_pass    = nd.gaussian_filter(white_noise, sigma)
high_pass   = white_noise - low_pass

# Calculate and plot the noise power spectra of each noise signal.
nps_white = np.abs(np.fft.fft2(white_noise))**2
nps_low   = np.abs(np.fft.fft2(low_pass))**2
nps_high  = np.abs(np.fft.fft2(high_pass))**2

# Calculate and plot the auto-correlation of each noise signal using the 
# correlation theorem. Center the maximum cross-correlation in the middle of the
# image, as already shown in the lecture for white noise.
corr_white = nss.correlate2d(white_noise, white_noise, mode="same")
corr_low = nss.correlate2d(low_pass, low_pass, mode="same")
corr_high = nss.correlate2d(high_pass, high_pass, mode="same")

##############################################################################
# Part B: Image shift using cross-correlation
# Use cross-correlation for a simple image registration task 
##############################################################################

# Read in the two images worldA and worldB. Both images show the same object, but
# shifted by a small amount relative to each other. The task is to estimate the
# shift using cross-correlation.

im_shifted1 = plt.imread('worldA.jpg') / 255.
im_shifted2 = plt.imread('worldB.jpg') / 255.
im_shifted1 = im_shifted1.mean(2)
im_shifted2 = im_shifted2.mean(2)

# Calculate the cross-correlation between the two images
ccorr = nss.correlate2d(im_shifted1, im_shifted2, mode="same")

# Calculate the shift as a vector tuple. (you might want to use numpy.argmax and
# np.unravel_index, or numpy.where)
maxi = np.argmax(ccor)
shift_y, shift_x = np.unravel_index(corr.argmax(), corr.shape)

# Print to screen the shifts
print corr[shift_y, shift_x]
print shift_y
print shift_x


#The crosscorrelation will be highest if the shift from the correlation equals 
#the shift between the two images, so you have to search for the co-ordinate
#of the maximum cross-correlation relative to the origin.

##############################################################################
# Plot results 
# feel free to use the code below.
##############################################################################


#Part A
plt.figure(num=1, figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(white_noise, interpolation='nearest')
plt.gray()
plt.title('white noise spatial domain')
plt.subplot(1,3,2)
plt.imshow(low_pass, interpolation='nearest')
plt.gray()
plt.title('low pass spatial domain')
plt.subplot(1,3,3)
plt.imshow(high_pass, interpolation='nearest')
plt.gray()
plt.title('high pass spatial domain')

plt.figure(num=2, figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(nps_white, interpolation='nearest')
plt.gray()
plt.title('white noise power spectrum')
plt.subplot(1,3,2)
plt.imshow(nps_low, interpolation='nearest')
plt.gray()
plt.title('low pass power spectrum')
plt.subplot(1,3,3)
plt.imshow(nps_high, interpolation='nearest')
plt.gray()
plt.title('high pass power spectrum')

plt.figure(num=3, figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(corr_white, interpolation='nearest')
plt.gray()
plt.title('white noise autocorrelation')
plt.subplot(1,3,2)
plt.imshow(corr_low, interpolation='nearest')
plt.gray()
plt.title('low pass noise autocorrelation')
plt.subplot(1,3,3)
plt.imshow(corr_high, interpolation='nearest')
plt.gray()
plt.title('high pass noise autocorrelation')

# #Part B
plt.figure(num=4, figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(im_shifted1)
plt.title('image1')
plt.subplot(1,3,2)
plt.imshow(im_shifted2)
plt.title('image2')
plt.subplot(1,3,3)
plt.imshow(ccorr)
plt.title('crosscorrelation')


plt.show()
