import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib
import matplotlib.pyplot as plt

from loader import ImageLoader, TransformLoader, PuzzleType
from transform import Transform


# Interactive plot mode
# matplotlib.use('TkAgg')

class Warper:  # todo: once we have sift, create transformation hierarchy(tree form)
    def __init__(self, image: ImageLoader, transform: TransformLoader):
        """
        Warps <image> with <transform> and pastes it to <base_image>, if <base_image> does not exist, one is created.
        """
        self.image_loader = image
        self.transform_loader = transform
        self.warped_images = []

    def warp_first(self, grayscale=True) -> np.ndarray:
        """ Warps base image, with the provided transformation. """
        # if self.transform.type == TranformLoader.Type.AFFINE: # todo: check why our transforms have values in the last row.
        #     transform_func = cv2.warpAffine
        # else:  # Homography
        #     transform_func = cv2.warpPerspective
        image = self.image_loader.grayscale_image if grayscale else self.image_loader.color_image
        transform = self.transform_loader.transform
        return self.warp(image, Transform.from_transform(transform=transform, type_=self.transform_loader.type))

    def warp(self, image: np.ndarray, transform: Transform):
        if transform.type == PuzzleType.AFFINE and (transform.transform[2, :] == np.array([0, 0, 1])).all():
            # todo: refactor after Simon answers the question.
            self.warped_images.append(
                cv2.warpAffine(image, transform.itransform[:2, :],
                               (self.transform_loader.width, self.transform_loader.height), dst=None,
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
        else:
            self.warped_images.append(cv2.warpPerspective(image, transform.itransform,
                                                          (self.transform_loader.width, self.transform_loader.height),
                                                          dst=None, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
        return self.warped_images[-1]

    # @property
    # def blank_full_image(self):
    #     return np.zeros(shape=(self.transform_loader.width, self.transform_loader.height))


if __name__ == '__main__':
    transform = TransformLoader(r'assignment_1\puzzles\puzzle_affine_1\warp_mat_1__H_521__W_760_.txt')
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
