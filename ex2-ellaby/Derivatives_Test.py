from skimage import data
from skimage import color as sk
from matplotlib import pyplot as plt
import sol2
import numpy as np

def read_image(image, representation):
    image = image.astype('float64')
    image = image/255
    if(representation == 1 and len(image.shape) == 3):
        if(image.shape[2] == 3):
            image = sk.rgb2gray(image)
        else:
            # I downloaded some RGBA images for testing so I added this:
            image = sk.rgb2gray(sk.rgba2rgb(image))
    return image

def get_images():
    images = []
    images.append(data.astronaut())
    images.append(data.logo())
    images.append(data.rocket())
    images.append(data.clock())
    images.append(data.camera())


    bw_images = []
    i = 0
    for image in images:
        i += 1
        bw_image = read_image(image, 1)
        bw_images.append(bw_image)
    return bw_images

def show_three_images(one, two, three, title=""):
    fig, axs = plt.subplots(1,3)
    fig.suptitle(title)
    axs[0].imshow(one, cmap="gray")
    axs[1].imshow(two, cmap="gray")
    axs[2].imshow(three, cmap="gray")

    axs[0].set_title("original")
    axs[1].set_title("convolution")
    axs[2].set_title("fourier")
    plt.show()
    return None

def save_three_images(one, two, three, title):
    fig, axs = plt.subplots(1,3)
    fig.suptitle(title)
    axs[0].imshow(one, cmap="gray")
    axs[1].imshow(two, cmap="gray")
    axs[2].imshow(three, cmap="gray")

    axs[0].set_title("original")
    axs[1].set_title("convolution")
    axs[2].set_title("fourier")
    plt.savefig(title)
    return None

def compare_der():
    images = get_images()
    i = 0
    for image in images:
        der1 = sol2.conv_der(image)
        der2 = sol2.fourier_der(image)
        title = f'{i}.jpg'
        show_three_images(image, der1, der2)
        save_three_images(image, der1, der2, title)
        i += 1

compare_der()

