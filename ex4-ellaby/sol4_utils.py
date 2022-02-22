from scipy.signal import convolve2d
import numpy as np
import imageio
from skimage.color import rgb2gray
from scipy import signal
from scipy import ndimage


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
    """ this function reads an image file and converts it into a given representation """
    img = imageio.imread(filename)
    if np.amax(img) > 1:
        img = img / (256 - 1)  # normalized if needed
    dim = len(img.shape)
    if representation == 1 and dim == 3:
        return rgb2gray(img)
    else:
        return img


def blur(im, filter):
    blurred_x = ndimage.filters.convolve(im, filter)
    blurred_x_y = ndimage.filters.convolve(blurred_x, filter.transpose())
    return blurred_x_y


def reduce(im, filter):
    blurred_x_y = blur(im, filter)
    sampled_rows = np.array(blurred_x_y[::2])
    samples_rows_cols = sampled_rows.transpose()[::2]
    return samples_rows_cols.transpose()


def create_filter(size):
    filter = np.array([1, 1])
    for i in range(size - 2):
        filter = signal.convolve(filter, [1, 1])
    return np.array([filter]) / filter.sum()


def build_gaussian_pyramid(im, max_levels, filter_size):
    filter = create_filter(filter_size)
    gaussian_pyramid = []
    curr_layer = im
    for i in range(max_levels):
        if len(curr_layer) < 16 or len(curr_layer[0]) < 16:
            break
        gaussian_pyramid.append(curr_layer)
        curr_layer = reduce(curr_layer, filter)
    return gaussian_pyramid, filter
