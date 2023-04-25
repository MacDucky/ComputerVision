from functools import reduce

import numpy as np
import cv2

from loader import ImageLoader, TransformLoader, PuzzleType
from transform import Transform


class Warper:  # todo: once we have sift, create transformation hierarchy(tree form)
    def __init__(self, image: ImageLoader, transform: TransformLoader, pieces_amount: int):
        """
        Warps <image> with <transform> and pastes it to <base_image>, if <base_image> does not exist, one is created.
        """
        self.image_loader = image
        self.transform_loader = transform
        self.puzzle_pieces = pieces_amount
        self.warped_images: list[None | np.ndarray] = [None] * pieces_amount
        self.is_grayscale: bool = True

    def warp_first(self, grayscale=True) -> np.ndarray:
        """ Warps base image, with the provided transformation. """
        if self.is_grayscale != grayscale:  # if mode is different, clear warped to recompute
            self.warped_images = [None] * self.puzzle_pieces
            self.is_grayscale = grayscale
        if self.warped_images[0]:
            return self.warped_images[0]
        image = self.image_loader.grayscale_img if self.is_grayscale else self.image_loader.color_img
        transform = self.transform_loader.transform
        return self.warp(image, Transform.from_transform(transform=transform, type_=self.transform_loader.type), 0)

    def warp(self, image: np.ndarray, transform: Transform, place_at: int) -> np.ndarray:
        if transform.type == PuzzleType.AFFINE and (transform.transform[2, :] == np.array([0, 0, 1])).all():
            self.warped_images[place_at] = cv2.warpAffine(image, transform.itransform[:2, :],
                                                          (self.transform_loader.width, self.transform_loader.height),
                                                          dst=None,
                                                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        else:
            self.warped_images[place_at] = cv2.warpPerspective(image, transform.itransform,
                                                               (self.transform_loader.width,
                                                                self.transform_loader.height),
                                                               dst=None, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return self.warped_images[place_at]

    def merged_image(self, transform_array, *unselected_indices) -> np.ndarray:
        unselected_set = set(unselected_indices)
        has_transform_indices = set(i for i, t in enumerate(transform_array, start=1) if t)
        solved_to_unselect = unselected_set.intersection(has_transform_indices)
        if has_transform_indices <= solved_to_unselect:
            return np.zeros(shape=(self.transform_loader.height, self.transform_loader.width))
        return reduce(np.maximum,
                      [np.array(image) for i, image in enumerate(self.warped_images, start=1) if
                       i not in solved_to_unselect])
