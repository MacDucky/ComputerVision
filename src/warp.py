import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib
import matplotlib.pyplot as plt

from loader import ImageLoader, TranformLoader


# Interactive plot mode
# matplotlib.use('TkAgg')

class Warper:
    def __init__(self, image: ImageLoader, tranform: TranformLoader, base_image: np.ndarray = None):
        """
        Warps <image> with <transform> and pastes it to <base_image>, if <base_image> does not exist, one is created.
        """
        self.image_loader = image
        self.transform_loader = tranform
        self.base_image = base_image
        if base_image is None:
            self.base_image = np.zeros(shape=(tranform.width, tranform.height))

    def warp(self, grayscale=True) -> np.ndarray:
        # if self.transform.type == TranformLoader.Type.AFFINE: # todo: check why our transforms have values in the last row.
        #     transform_func = cv2.warpAffine
        # else:  # Homography
        #     transform_func = cv2.warpPerspective
        image = self.image_loader.grayscale_image if grayscale else self.image_loader.color_image
        transform = self.transform_loader.transform()
        self.base_image = cv2.warpPerspective(image, inv(transform),
                                              (self.transform_loader.width, self.transform_loader.height),
                                              self.base_image, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return self.base_image


if __name__ == '__main__':
    # todo add file manager: given puzzle directory, return path to tranform and to image dir

    transform = TranformLoader(r'assignment_1\puzzles\puzzle_affine_1\warp_mat_1__H_521__W_760_.txt')
    # transform = TranformLoader(r'assignment_1\puzzles\puzzle_affine_2\warp_mat_1__H_537__W_735_.txt')
    transform.load()

    image = ImageLoader(r'assignment_1\puzzles\puzzle_affine_1\pieces\piece_1.jpg')
    # image = ImageLoader(r'assignment_1\puzzles\puzzle_affine_2\pieces\piece_1.jpg')
    image.load()

    new_image = np.zeros(shape=(transform.height, transform.width))

    warper = Warper(image, transform, base_image=new_image)
    new_image = warper.warp()

    plt.figure()
    plt.subplot(1, 2, 1)  # last param is the subplot number
    plt.imshow(image.grayscale_image, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(new_image, cmap='gray', vmin=0, vmax=255)
    plt.show()
