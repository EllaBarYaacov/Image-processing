import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray
import sys

RANGE = 256
RGB_TO_YIQ = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
YIQ_TO_RGB = np.linalg.inv(RGB_TO_YIQ)


def read_image(filename, representation):
    """
    this function reads an image file and converts it into a given representation
    :param filename: image to process
    :param representation: 1 for gray scale, 2 for rgb
    :return: image from filename represented in the given representation
    """
    img = imageio.imread(filename)
    if np.amax(img) > 1:
        img = img / (RANGE - 1)  # normalized if needed
    dim = len(img.shape)
    if representation == 1 and dim == 3:
        return rgb2gray(img)
    else:
        return img


def imdisplay(filename, representation):
    """
    this function opens a new figure and displays the loaded image in the converted representation
    :param filename:  picture to show
    :param representation: 1 for gray scale, 2 for rgb
    :return: None
    """
    converted = read_image(filename, representation)
    plt.imshow(converted, cmap=plt.cm.gray)
    plt.show()


def rgb2yiq(imRGB):
    """
    this function converts rgb to yiq
    :param imRGB: photo to convert
    :return: converted photo
    """
    return imRGB.dot(RGB_TO_YIQ.T)


def yiq2rgb(imYIQ):
    """
    this function converts yiq to rgb
    :param imYIQ: photo to convert
    :return: converted photo
    """
    return imYIQ.dot(YIQ_TO_RGB.T)

def split_RGB_to_Y_I_Q(img):
    """
    split RGB to three seperate channels: Y I Q
    """
    yiq = rgb2yiq(img)
    Y = yiq[:, :, 0]
    I = yiq[:, :, 1]
    Q = yiq[:, :, 2]
    return [Y, I, Q]

def histogram_equalize(im_orig):
    """
    this function equalizes the histogram of the pjoto
    :param im_orig: image to equalize
    :return: equalized inage
    """
    img = im_orig
    rgb_case = len(im_orig.shape) == 3
    # if rgb than use only the Y channel
    if rgb_case:
        img, I, Q = split_RGB_to_Y_I_Q(im_orig)

    img = (img * (RANGE - 1)).round().astype(np.uint8)  # multiply image values by 255
    hist, bins = np.histogram(img, bins=np.arange(RANGE + 1))
    cum_hist = np.cumsum(hist)
    min = np.amin(img)
    LUT = np.round(((cum_hist - cum_hist[min])/(cum_hist[RANGE - 1] - cum_hist[min])) * (RANGE - 1))
    new_img = LUT[img]
    new_hist, new_bins = np.histogram(new_img, bins=np.arange(RANGE +1))
    new_img /= (RANGE - 1)
    if rgb_case:
        new_img = yiq2rgb(np.dstack((new_img, I, Q)))  # pile up the channels of y, i, q
    return[new_img, hist, new_hist]


def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig: ing to proccess
    :param n_quant: number of gray shades
    :param n_iter: number of iterations
    :return: quantized img and error array
    """
    img = im_orig
    rgb_case = len(im_orig.shape) == 3
    if rgb_case:
        img, I, Q = split_RGB_to_Y_I_Q(im_orig)
    img = (img * (RANGE - 1)).round().astype(np.uint8)
    Zs = init_Zs(img, n_quant)
    Qs = find_Qs(Zs, img)
    errors = []
    old_Zs = []
    for i in range(n_iter):
        Zs = find_Zs(Qs)
        if Zs == old_Zs:
            break
        else:
            old_Zs = Zs
        Qs = find_Qs(Zs, img)
        error = calculate_error(Zs, Qs, img)
        errors.append(error)
    quant_img = quantize_by_ZnQ(Zs, Qs, img)

    quant_img = quant_img/(RANGE-1)
    if rgb_case:
        quant_img = yiq2rgb(np.dstack((quant_img, I, Q)))  # pile up the channels of y, i, q
    return [quant_img, errors]


def init_Zs(im_orig, n_quants):
    """
    initials Zs that divide all the pixels evenly between all bins
    :param im_orig: img to proccess
    :param n_quants: number of bins wanted
    :return: Zs array
    """
    all_Zs = []
    all_Zs.append(0.0)
    interval = float(1.0/n_quants)
    for i in range(1, n_quants):
        z = np.quantile(im_orig, float(interval * (i)))
        all_Zs.append(z)
    all_Zs.append(255.0)
    return all_Zs


def find_Qs(Zs, im_orig):
    """ finds Qs givvan Zs by the formula """
    hist, bins = np.histogram(im_orig, bins=np.arange(RANGE + 1))
    gray_values = np.arange(256)
    Qs = []
    for i in range(len(Zs) - 1):
        z_start = int(Zs[i])
        if(Zs[i] != 0 and Zs[i] != 255):
            z_start += 1
        z_end = int(Zs[i+1]) + 1
        denominator = np.sum(hist[z_start:z_end])  # ￿￿￿MECHANE
        numerator = np.dot(gray_values[z_start:z_end],  hist[z_start: z_end])  # MONE
        Qs.append(numerator/denominator)
    return Qs


def find_Zs(Qs):
    """ find Zs by Qs in the formula"""
    pass
    Zs = []
    Zs.append(0)
    for i in range(len(Qs) - 1):
        Zs.append(int((Qs[i] + Qs[i+1])/2))
    Zs.append(RANGE - 1)
    return Zs


def calculate_error(Zs, Qs, im_orig):
    """ calculates error """
    LUT = make_LUT(Zs, Qs)
    all_colors = np.arange(RANGE)
    diff = all_colors - LUT
    square_diff = diff * diff
    hist, bins = np.histogram(im_orig, bins=np.arange(RANGE + 1))
    error = np.dot(square_diff, hist)
    return error

def make_LUT(Zs, Qs):
    """ calculated look up table """
    maps = []
    for i in range(len(Qs)):
        length = int(Zs[i + 1]) - int(Zs[i])
        if (i == len(Qs) - 1):
            length += 1
        map = [Qs[i]] * length
        maps.append(map)
    LUT = np.concatenate(maps)
    return LUT


def quantize_by_ZnQ(Zs, Qs, im_orig):
    """ given Zs and Qs and original img return quantized img """
    LUT = make_LUT(Zs, Qs)
    new_img = LUT[im_orig]
    return new_img





if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "/Users/ellaby/Downloads/ex1_presubmit/presubmit_externals/monkey.jpg"


    photo = read_image(file, 2)
    plt.imshow(photo)
    plt.show()

    yiq = rgb2yiq(photo)
    plt.imshow(yiq)
    plt.show()

    rgb = yiq2rgb(yiq)
    plt.imshow(rgb)
    plt.show()

    eq = histogram_equalize(photo)[0]
    plt.imshow(eq)
    plt.show()

    quantized = quantize(photo, 2, 2)[0]
    plt.imshow(quantized)
    plt.show()