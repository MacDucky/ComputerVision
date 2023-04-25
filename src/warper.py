from functools import reduce

import numpy as np
import cv2

from loader import ImageLoader, TransformLoader, PuzzleType
from transform import Transform


class Warper:  # todo: once we have sift, create transformation hierarchy(tree form)
    def __init__(self, image: ImageLoader, transform: TransformLoader):
        """
        Warps <image> with <transform> and pastes it to <base_image>, if <base_image> does not exist, one is created.
        """
        self.image_loader = image
        self.transform_loader = transform
        self.warped_images = []
        self.is_grayscale: bool = True

    def warp_first(self, grayscale=True) -> np.ndarray:
        """ Warps base image, with the provided transformation. """
        if self.is_grayscale != grayscale:  # if mode is different, clear warped to recompute
            self.warped_images.clear()
            self.is_grayscale = grayscale
        if self.warped_images:
            return self.warped_images[0]
        image = self.image_loader.grayscale_img if self.is_grayscale else self.image_loader.color_img
        transform = self.transform_loader.transform
        return self.warp(image, Transform.from_transform(transform=transform, type_=self.transform_loader.type))

    def warp(self, image: np.ndarray, transform: Transform):
        if transform.type == PuzzleType.AFFINE and (transform.transform[2, :] == np.array([0, 0, 1])).all():
            self.warped_images.append(
                cv2.warpAffine(image, transform.itransform[:2, :],
                               (self.transform_loader.width, self.transform_loader.height), dst=None,
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
        else:
            self.warped_images.append(cv2.warpPerspective(image, transform.itransform,
                                                          (self.transform_loader.width, self.transform_loader.height),
                                                          dst=None, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
        return self.warped_images[-1]

    @property
    def merged_image(self) -> np.ndarray:
        return reduce(np.maximum, [np.array(image) for image in self.warped_images])

    # @property
    # def coverage_image(self) -> np.ndarray[np.int8]:
    #     return reduce(np.count_nonzero)
