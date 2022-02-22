from scipy import ndimage
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray
import os
RANGE = 256
import scipy.misc





def read_image(filename, representation):
    """ this function reads an image file and converts it into a given representation """
    img = imageio.imread(filename)
    if np.amax(img) > 1:
        img = img / (RANGE - 1)  # normalized if needed
    dim = len(img.shape)
    if representation == 1 and dim == 3:
        return rgb2gray(img)
    else:
        return img


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


def create_filter(size):
    filter = np.array([1, 1])
    for i in range(size - 2):
        filter = signal.convolve(filter, [1, 1])
    return np.array([filter]) / filter.sum()


def reduce(im, filter):
    blurred_x_y = blur(im, filter)
    sampled_rows = np.array(blurred_x_y[::2])
    samples_rows_cols = sampled_rows.transpose()[::2]
    return samples_rows_cols.transpose()


def blur(im, filter):
    blurred_x = ndimage.filters.convolve(im, filter)
    blurred_x_y = ndimage.filters.convolve(blurred_x, filter.transpose())
    return blurred_x_y


def build_laplacian_pyramid(im, max_levels, filter_size):
    gaussian_pyramid, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        laplacian_pyramid.append(gaussian_pyramid[i] - expand(gaussian_pyramid[i + 1], filter))
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid, filter


def expand(im, filter):
    z_rows = np.zeros((im.shape[0], 2 * im.shape[1]), dtype=im.dtype)
    z_rows[:,::2] = im
    z_rows = z_rows.transpose()
    z_columns = np.zeros((z_rows.shape[0], 2 * z_rows.shape[1]), dtype=im.dtype)
    z_columns[:,::2] = z_rows
    z_columns = z_columns.transpose()
    expanded = blur(z_columns, 2 * filter)
    return expanded


def laplacian_to_image(lpyr, filter_vec, coeff):
    new_img = lpyr[-1] * coeff[-1]
    for i in (range(len(lpyr))[-2::-1]):
        new_img = lpyr[i] + coeff[i] * expand(new_img, filter_vec)
    return new_img


def render_pyramid(pyr, levels):
    pyr = np.array(pyr, dtype=object)
    result = normalize_to_0_1(pyr[0])
    length = pyr[0].shape[0]
    for i in range(1, levels):
        mini_img = normalize_to_0_1(pyr[i])
        zeros = np.zeros((length - mini_img.shape[0], mini_img.shape[1]))
        mini_img = np.concatenate([mini_img, zeros])
        result = np.concatenate([result, mini_img], axis=1)
    return result


def display_pyramid(pyr, levels):
    im = render_pyramid(pyr, levels)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()


def normalize_to_0_1(mat):
    max = np.max(mat)
    min = np.min(mat)
    normalized = (mat - min) / (max - min)
    return normalized


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    mask = mask.astype(dtype=np.float64)
    l1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gm, filter3 = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    l_out = []
    for k in range(len(l1)):
        layer_k = (gm[k] * l1[k]) + ((1 - gm[k]) * l2[k])
        l_out.append(layer_k)
    return laplacian_to_image(l_out, filter1, np.ones(len(l_out)))


def RGB_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    im1_RGB = im1.shape[-1] == 3
    im2_RGB = im2.shape[-1] == 3
    if im1_RGB and im2_RGB:
        im1_r = im1[:, :, 0]
        im1_g = im1[:, :, 1]
        im1_b = im1[:, :, 2]
        im2_r = im2[:, :, 0]
        im2_g = im2[:, :, 1]
        im2_b = im2[:, :, 2]
        red = pyramid_blending(im1_r, im2_r, mask, max_levels, filter_size_im, filter_size_mask)
        green = pyramid_blending(im1_g, im2_g, mask, max_levels, filter_size_im, filter_size_mask)
        blue = pyramid_blending(im1_b, im2_b, mask, max_levels, filter_size_im, filter_size_mask)
        return np.dstack((red, green, blue))
    if (not im1_RGB) and (not im2_RGB):
        return pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
    if im1_RGB and (not im2_RGB):
        im1_r = im1[:, :, 0]
        im1_g = im1[:, :, 1]
        im1_b = im1[:, :, 2]
        red = pyramid_blending(im1_r, im2, mask, max_levels, filter_size_im, filter_size_mask)
        green = pyramid_blending(im1_g, im2, mask, max_levels, filter_size_im, filter_size_mask)
        blue = pyramid_blending(im1_b, im2, mask, max_levels, filter_size_im, filter_size_mask)
        return np.dstack((red, green, blue))
    else:
        im2_r = im2[:, :, 0]
        im2_g = im2[:, :, 1]
        im2_b = im2[:, :, 2]
        red = pyramid_blending(im1, im2_r, mask, max_levels, filter_size_im, filter_size_mask)
        green = pyramid_blending(im1, im2_g, mask, max_levels, filter_size_im, filter_size_mask)
        blue = pyramid_blending(im1, im2_b, mask, max_levels, filter_size_im, filter_size_mask)
        return np.dstack((red, green, blue))


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def show_example(middle_img, backgroung_img , mask_img, middle_title, background_title):
    im1 = read_image(relpath(middle_img), 2)
    im2 = read_image(relpath(backgroung_img), 2)
    mask = read_image(relpath(mask_img), 1)

    mask = np.around(mask)
    bin_mask = mask.astype(np.bool)

    im_blend = RGB_blending(im1, im2, bin_mask, 4, 3, 3)

    fig = plt.figure(figsize=(8, 8))
    rows, cols = 2, 2
    first_im = fig.add_subplot(rows, cols, 1)
    plt.imshow(im1, vmin=0, vmax=1)
    first_im.title.set_text(middle_title)
    second_im = fig.add_subplot(rows, cols, 2)
    plt.imshow(im2, vmin=0, vmax=1)
    second_im.title.set_text(background_title)
    third_im = fig.add_subplot(rows, cols, 3)
    plt.imshow(bin_mask, cmap=plt.cm.gray, vmin=0, vmax=1)
    third_im.title.set_text('mask')
    fourth_im = fig.add_subplot(rows, cols, 4)
    plt.imshow(im_blend, vmin=0, vmax=1)
    fourth_im.title.set_text('blend')
    plt.show()
    return im1, im2, bin_mask, im_blend


def blending_example1():
    return show_example('externals/Ben Gurion.jpg', 'externals/Charlie Small Face.jpg',
                 'externals/Charlie Ben Gurion Mask.jpg',
                 'Ben gurion speech', 'Charlie sitting')


def blending_example2():
    return show_example('externals/Bridge.jpg', 'externals/Charlie Bridge.jpg', 'externals/Charlie Bridge Mask.jpg',
                 'lunch on a skyscraper', 'Charlie chilling')


# if __name__ == '__main__':
#     blending_example1()
#     blending_example2()