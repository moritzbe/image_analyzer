# Count the number of stripes in an image:

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd


# load tree image
im = plt.imread('redtest.jpg') / 255.
# im = im.mean(2)
sh = im.shape

# select color channels
im_red = im[:,:,0]
im_green = im[:,:,1]
im_blue = im[:,:,2]

#prepare templates
red_temp = im_red
green_temp = im_green
blue_temp = im_blue

# get image thresholds from image
red_threshold = im_red[385:,555:]
green_threshold = im_green[385:,555:]
blue_threshold = im_blue[385:,555:]

red_threshold_average = np.average(red_threshold)
green_threshold_average = np.average(green_threshold)
blue_threshold_average = np.average(blue_threshold)

red_temp[red_temp > red_threshold_average] = 1
green_temp[green_temp > green_threshold_average] = 1
blue_temp[blue_temp > blue_threshold_average] = 1

red_temp = 1 - red_temp


im_2 = plt.imread('Drawing.png')
# Let's create a convolution kernel (PSF) and produce the convolved image. We 
# want the convolved image to suffer from motion blur in the direction of the 
# diagonal. The function np.diag creates an appropriate convolution kernel. the
# kernel should be 51x51 pixels with 1 on the diagonal, 0 otherwise, and then 
# normalized so that the sum of the diagonal is 1.

# M = 51
# psf = np.diag(np.ones(M)) / M
# im_conv = nd.convolve(im, psf, mode='wrap')

# # Add zero-mean Gaussian noise with a standard deviation sigma to the image
# # Hint: look at np.random.randn
# sigma = .01
# im_noisy = im + (sigma*np.random.randn(51, 51))

# # In order to use Fourier space deconvolution we need to zeropad our convolution
# # kernel to the same size as the original image
# psf_pad = np.zeros_like(im)
# psf_pad[sh[0]//2-M/2:sh[0]//2+M/2+1,sh[1]//2-M/2:sh[1]//2+M/2+1] = psf

# # Now we'll try out "naive" deconvolution by dividing the noisy, blurred image 
# # by the filter function in the Fourier domain
# im_deconv = ...

# # As soon as you add a little noise to the image, the naive deconvolution will go
# # wrong, since for white noise, the noise power will exceed the signal power
# # for high frequencies. Since the inverse filtering also enhances the high 
# # frequncies the result will be nonsense

# # Let's first define the Wiener deconvolution in a seperate function.
# def wiener_deconv(img, psf, nps):
#     """\
#     This function performs an image deconvolution using a Wiener filter.
#     Parameters:
#         img: convolved image
#         psf: the convolution kernel
#         nps: noise power spectrum of the image, you will have to choose an 
#             appropriate value
#     """
#     # Apart from the noise power spectrum (nps), which is passed as a
#     # parameter, you'll also need the frequency representation of your psf,
#     # the power spectrum of the filter and the signal power spectrum (sps).
#     # Calculate them.

#     # f_psf = ???
#     # sps_psf = ???
#     # sps = ???

#     # create the Wiener filter
#     # wiener_filter = ???
    
#     # Do a Fourier space convolution of the image with the wiener filter
#     deconv_img = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(img) * wiener_filter)))
    
#     return deconv_img

# # Try out Wiener deconvolution.
# # Assume white noise, i.e. a noise power spectrum that has a constant value 
# # for all frequencies. Try out a few values to get a good result.
# nps = 0.0

# im_deconv_W = wiener_deconv(im_noisy, psf_pad, nps)

# The Wiener filter is essentially the same as the naive filter, only with an
# additional weighting factor that depends on the SNR in the image in frequency
# domain. Frequencies where the noise power exceeds the signal power will be 
# damped.

plt.figure(1);plt.clf()
plt.subplot(2,2,1)
plt.imshow(red_temp, cmap='gray');plt.title('Red');
plt.subplot(2,2,2)
plt.imshow(green_temp, cmap='gray');plt.title('Green');
plt.subplot(2,2,3)
plt.imshow(blue_temp, cmap='gray');plt.title('Blue');
plt.subplot(2,2,4)
plt.imshow(im_2, cmap='gray');plt.title('');

plt.show()