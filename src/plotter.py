import numpy as np
import matplotlib.pyplot as plt


# Interactive plot mode
# import matplotlib
# matplotlib.use('TkAgg')

def show_image(image: np.ndarray, grayscale=True):
    plt.figure()
    if grayscale:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def compare_images(image1, image2, grayscale=True):
    plt.figure()

    if grayscale:
        plt.subplot(1, 2, 1)  # last param is the subplot number
        plt.imshow(image1, cmap='gray', vmin=0, vmax=255)
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap='gray', vmin=0, vmax=255)
    else:
        plt.subplot(1, 2, 1)  # last param is the subplot number
        plt.imshow(image1, vmin=0, vmax=255)
        plt.subplot(1, 2, 2)
        plt.imshow(image2, vmin=0, vmax=255)

    plt.show()


if __name__ == '__main__':
    from src.loader import ImageLoader, PathLoader, PuzzleType

    path = PathLoader(1, PuzzleType.AFFINE)
    image1 = ImageLoader(path.get_image(1))
    image2 = ImageLoader(path.get_image(2))
    compare_images(image1.grayscale_image, image2.grayscale_image)
    compare_images(image1.color_image, image2.color_image, grayscale=False)
    show_image(image1.grayscale_image)
