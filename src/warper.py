import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib
import matplotlib.pyplot as plt

from loader import ImageLoader, TranformLoader


# Interactive plot mode
# matplotlib.use('TkAgg')

class Warper:   # todo: once we have sift, create transformation hierarchy(tree form)
    def __init__(self, image: ImageLoader, tranform: TranformLoader):
        """
        Warps <image> with <transform> and pastes it to <base_image>, if <base_image> does not exist, one is created.
        """
        self.image_loader = image
        self.transform_loader = tranform
        self.warped_images = []

    def warp_first(self, grayscale=True) -> np.ndarray:
        # if self.transform.type == TranformLoader.Type.AFFINE: # todo: check why our transforms have values in the last row.
        #     transform_func = cv2.warpAffine
        # else:  # Homography
        #     transform_func = cv2.warpPerspective
        image = self.image_loader.grayscale_image if grayscale else self.image_loader.color_image
        transform = self.transform_loader.transform
        self.warped_images.append(self.warp(image, transform))
        return self.warped_images[0]

    def warp(self, image: np.ndarray, transform: np.ndarray):
        return cv2.warpPerspective(image, inv(transform),
                                   (self.transform_loader.width, self.transform_loader.height),
                                   self.blank_full_image, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    @property
    def blank_full_image(self):
        return np.zeros(shape=(self.transform_loader.width, self.transform_loader.height))


if __name__ == '__main__':
    transform = TranformLoader(r'assignment_1\puzzles\puzzle_affine_1\warp_mat_1__H_521__W_760_.txt')
    # transform = TranformLoader(r'assignment_1\puzzles\puzzle_affine_2\warp_mat_1__H_537__W_735_.txt')
    transform.load()

    image = ImageLoader(r'assignment_1\puzzles\puzzle_affine_1\pieces\piece_1.jpg')
    # image = ImageLoader(r'assignment_1\puzzles\puzzle_affine_2\pieces\piece_1.jpg')
    image.load()

    new_image = np.zeros(shape=(transform.height, transform.width))

    warper = Warper(image, transform)
    new_image = warper.warp_first()

    plt.figure()
    plt.subplot(1, 2, 1)  # last param is the subplot number
    plt.imshow(image.grayscale_image, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(new_image, cmap='gray', vmin=0, vmax=255)
    plt.show()
