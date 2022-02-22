import numpy as np
import matplotlib.pyplot as plt
import sol2


def pad_with_zeros2(filter, shape):
    zeros = np.zeros((shape[0], shape[1]))
    x = int((shape[0]-filter.shape[0])/2)
    y = int((shape[1]-filter.shape[1])/2)
    zeros[x :x + filter.shape[0],y :y+filter.shape[1]] = filter
    return zeros


def create_big_filter(little_filter, size):
    new_filter = np.zeros((little_filter.shape[0] * size, little_filter.shape[1] * size))
    for i in range(little_filter.shape[0]):
        for j in range(little_filter.shape[1]):
            cube = np.ones((size, size)) * little_filter[i][j]
            new_filter[i*size:(i+1)*size, j*size:(j+1)*size] = cube
    return new_filter


if __name__ == '__main__':
    filt = create_big_filter(np.array([[1]]), 10)
    padded_filt = pad_with_zeros2(filt, (1024, 1024))
    plt.imshow(padded_filt, cmap=plt.cm.gray)
    plt.show()

    F_filt = sol2.DFT2(padded_filt.real)
    plt.imshow(np.abs(F_filt), cmap=plt.cm.gray)
    plt.show()

