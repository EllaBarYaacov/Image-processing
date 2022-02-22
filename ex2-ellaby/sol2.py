import numpy as np
import scipy.io.wavfile as wv
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import imageio
from skimage.color import rgb2gray
# import matplotlib.pyplot as plt
# import pandas as pd




def init_DFT_natrix(N):
    """ initiate DFT matrix of size N x N"""
    u = x = np.arange(N)
    # u = x = np.arange(-np.floor(N / 2), np.ceil(N / 2))
    ux = np.outer(x, u)
    i = np.complex(0, 1)
    DFT_mat = np.cos((2 * np.pi * ux) / N) - i * np.sin((2 * np.pi * ux) / N)
    return DFT_mat


def init_IDFT_natrix(N):
    """ initiate DFT matrix of size N x N"""
    u = x = np.arange(N)
    # u = x = np.arange(-np.floor(N / 2), np.ceil(N / 2))
    ux = np.outer(x, u)
    i = np.complex(0, 1)
    IDFT_mat = np.cos((2 * np.pi * ux) / N) + i * np.sin((2 * np.pi * ux) / N)
    return IDFT_mat / N


def DFT(signal):
    """ this function transforms a 1D discrete signal to its 1D Fourier representation """
    DFT_mat = init_DFT_natrix(signal.shape[0])
    return np.dot(DFT_mat, signal)


def IDFT(fourier_signal):
    """ this function transforms a 1D Fourier representation to its 1D discrete signal """
    IDFT_mat = init_IDFT_natrix(fourier_signal.shape[0])
    return np.dot(IDFT_mat, fourier_signal)


def DFT2(image):
    """ this function transforms a 2D discrete signal to its 2D Fourier representation """
    extra_dim = len(image.shape) == 3
    DFT_mat_1 = init_DFT_natrix(image.shape[0])
    DFT_mat_2 = init_DFT_natrix(image.shape[1])
    if extra_dim:
        image = image[:, :, 0]
    first_DFT = np.dot(DFT_mat_1, image)
    second_DFT = np.dot(DFT_mat_2, first_DFT.transpose())
    result = second_DFT.transpose()
    if extra_dim:
        result = result[..., np.newaxis]
    return result


def IDFT2(fourier_image):
    """ this function transforms a 2D Fourier representation to its 2D discrete signal """
    extra_dim = len(fourier_image.shape) == 3
    DFT_mat_1 = init_IDFT_natrix(fourier_image.shape[0])
    DFT_mat_2 = init_IDFT_natrix(fourier_image.shape[1])
    if extra_dim:
        fourier_image = fourier_image[:, :, 0]
    first_DFT = np.dot(DFT_mat_1, fourier_image)
    second_DFT = np.dot(DFT_mat_2, first_DFT.transpose())
    result = second_DFT.transpose()
    if extra_dim:
        result = result[..., np.newaxis]
    return result


def change_rate(filename, ratio):
    """ this function creates new wav file with duration time of original_duration/ratio """
    orig_wav = wv.read(filename)
    wv.write("change_rate.wav", int(orig_wav[0] * ratio), orig_wav[1])


def change_samples(filename, ratio):
    """ this function creates new wav file with same rate but different samples """
    rate, samples = wv.read(filename)
    new_samples = resize(samples, ratio).real
    wv.write("change_samples.wav", rate, new_samples)
    return new_samples


def resize(data, ratio):
    """ this function returns resized data by ratio"""
    fourier = DFT(data)
    shifted_f = np.fft.fftshift(fourier)
    # shifted_f = fourier
    if ratio > 1:
        resized = clip_high_freq(shifted_f, ratio)
    elif ratio < 1:
        resized = pad_with_zeros(shifted_f, ratio)
    else:
        return data.astype('float64')
    shifted_back = np.fft.ifftshift(resized)
    # shifted_back = resized
    return IDFT(shifted_back)


def clip_high_freq(data, ratio):
    """ this function clips high frequencies from data according to ratio """
    N = data.shape[0]
    new_N = int(N / ratio)
    diff = N - new_N
    left = int(np.ceil(diff / 2))
    right = N - int(np.floor(diff / 2))
    return data[left: right]

def pad_with_zeros(data, ratio):
    """ this function pads data with zeros according to ratio """
    N = data.shape[0]
    new_N = int(N/ratio)
    diff = new_N - N
    left = int(np.floor(diff/2))
    right = int(np.ceil(diff/2))
    return np.concatenate([np.zeros(left), data, np.zeros(right)])


def resize_spectrogram(data, ratio):
    """this function resizes data spectogram"""
    spectrogram = stft(data)
    new_spectrogram = np.apply_along_axis(resize, 1, spectrogram, ratio)
    return istft(new_spectrogram)


def resize_vocoder(data, ratio):
    spectrogram = stft(data)
    return istft(phase_vocoder(spectrogram, ratio))


def conv_der(im):
    x_der_conv = np.array([[0.5, 0, -0.5]])
    y_der_conv = x_der_conv.transpose()
    im_x_derived = signal.convolve2d(im, x_der_conv, mode='same')
    im_y_derived = signal.convolve2d(im, y_der_conv, mode='same')
    magnitude = np.sqrt(np.abs(im_x_derived) ** 2 + np.abs(im_y_derived) ** 2)
    return magnitude


def fourier_der(im):
    im_x_derived = fourier_der_by_axis(im, 0)
    im_y_derived = fourier_der_by_axis(im, 1)
    magnitude = np.sqrt(np.abs(im_x_derived) ** 2 + np.abs(im_y_derived) ** 2)
    return magnitude


def fourier_der_by_axis(im, axis):
    fourier = DFT2(im)
    shifted_fourier = np.fft.fftshift(fourier)
    # shifted_fourier = fourier
    if axis == 0:
        multiplied = multiply_rows(shifted_fourier)
        multiplied *= (2 * np.pi)/im.shape[0]
    else:
        multiplied = multiply_columns(shifted_fourier)
        multiplied *= (2 * np.pi) / im.shape[1]
    shifted_back = np.fft.ifftshift(multiplied)
    # shifted_back = multiplied
    return IDFT2(shifted_back)


def multiply_rows(mat):
    N = mat.shape[0]
    vec = np.arange(-np.floor(N/2), np.ceil(N/2)).transpose()
    vec = vec[..., np.newaxis]
    return mat * vec


def multiply_columns(mat):
    N = mat.shape[1]
    vec = np.arange(-np.floor(N/2), np.ceil(N/2)).transpose()
    vec = vec[..., np.newaxis]
    return (mat.transpose() * vec).transpose()


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def read_image(filename, representation):
    """
    this function reads an image file and converts it into a given representation
    :param filename: image to process
    :param representation: 1 for gray scale, 2 for rgb
    :return: image from filename represented in the given representation
    """
    img = imageio.imread(filename)
    if np.amax(img) > 1:
        img = img / (256 - 1)  # normalized if needed
    dim = len(img.shape)
    if representation == 1 and dim == 3:
        return rgb2gray(img)
    else:
        return img


def pad_filter_with_zeros(filter, shape):
    """ this function pads a filter to the size of shape - for my tests"""
    zeros = np.zeros((shape[0], shape[1]))
    x = int((shape[0]-filter.shape[0])/2)
    y = int((shape[1]-filter.shape[1])/2)
    zeros[x :x + filter.shape[0],y :y+filter.shape[1]] = filter
    return zeros


def create_big_filter(little_filter, size):
    """ this function returns filter that is like little_filter only 'size' times bigger - for my tests"""
    new_filter = np.zeros((little_filter.shape[0] * size, little_filter.shape[1] * size))
    for i in range(little_filter.shape[0]):
        for j in range(little_filter.shape[1]):
            cube = np.ones((size, size)) * little_filter[i][j]
            new_filter[i*size:(i+1)*size, j*size:(j+1)*size] = cube
    return new_filter

# MY TESTS
# if __name__ == '__main__':
#     # show image
#     filename = '/Users/ellaby/Documents/year D/image_processing/exes/ex2-ellaby/external/city.jpg'
#     im = read_image(filename, 1)
#     plt.imshow(im, cmap=plt.cm.gray)
#     plt.show()
#
#     # fourier image
#     fourier = DFT2(im)
#     plt.imshow(fourier.real, cmap=plt.cm.gray)
#     plt.show()
#
#     # double fourier image
#     fourier2 = DFT2(fourier)
#     plt.imshow(fourier2.real, cmap=plt.cm.gray)
#     plt.show()
#
#     # possible filters
#     filt_ones = np.ones((1, 1))
#     filt_der_x = np.array([[-1, 0, 1]])
#     filt_der_y = np.array([[-1], [0], [1]])
#     filt_sum_der_x_der_y = filt_der_x + filt_der_y
#
#     # convolution with filter
#     filt = create_big_filter(filt_ones, 20)
#     convolved = signal.convolve2d(im, filt, boundary='symm', mode='same')
#     plt.imshow(convolved, cmap=plt.cm.gray)
#     plt.show()
#
#     # show filter padded with seros
#     padded_filt = pad_filter_with_zeros(filt, im.shape)
#     plt.imshow(padded_filt, cmap=plt.cm.gray)
#     plt.show()
#
#     # DFT to filter
#     F_filt = DFT2(padded_filt.real)
#     plt.imshow(np.abs(F_filt), cmap=plt.cm.gray)
#     plt.show()
#
#     F_im = DFT2(im)
#     mult = F_filt * F_im
#     new_im = IDFT2(mult)
#     # need do shift.....
#     new_im = np.fft.fftshift(new_im)
#     plt.imshow(new_im.real, cmap=plt.cm.gray)
#     plt.show()

