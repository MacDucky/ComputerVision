from typing import Self

import cv2
import numpy as np
from abc import ABC, abstractmethod
from numpy.linalg import inv

from loader import *
from sift import SiftMatcher, SiftData


class Transform:
    def __init__(self, keypoint_matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint]]):
        self.source_kp = [t[0] for t in keypoint_matches]
        self.dest_kp = [t[1] for t in keypoint_matches]
        self.source_points = cv2.KeyPoint_convert(self.source_kp)
        self.dest_points = cv2.KeyPoint_convert(self.dest_kp)
        self._transform = []
        self._itransform = []

    @property
    def transform(self):
        return self._transform.copy()

    @property
    def itransform(self):
        return self._itransform.copy()

    @property
    @abstractmethod
    def type(self) -> PuzzleType:
        pass

    @classmethod
    def from_transform(cls, transform: np.ndarray, type_: PuzzleType) -> Self:
        instance = AffineTransform([]) if type_ == PuzzleType.AFFINE else HomographyTransform([])
        instance._transform = transform
        instance._itransform = inv(transform)
        return instance


class AffineTransform(Transform):
    def __init__(self, keypoint_matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint]]):
        super().__init__(keypoint_matches)
        if keypoint_matches:
            self._transform: np.ndarray = np.vstack(
                [cv2.getAffineTransform(self.source_points, self.dest_points), [0, 0, 1]])
            self._itransform: np.ndarray = None
            eigen_ratio, self._itransform = cv2.invert(self._transform,
                                                       flags=cv2.DECOMP_SVD)  # inv(self._transform)
            i = 0

    @property
    def type(self):
        return PuzzleType.AFFINE


class HomographyTransform(Transform):
    def __init__(self, keypoint_matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint]]):
        super().__init__(keypoint_matches)
        if keypoint_matches:
            self._transform: np.ndarray = cv2.getPerspectiveTransform(self.source_points, self.dest_points,
                                                                      solveMethod=cv2.DECOMP_SVD)
            self._itransform: np.ndarray = inv(self._transform)

    @property
    def type(self) -> PuzzleType:
        return PuzzleType.HOMOGRAPHY


if __name__ == '__main__':
    path = PathLoader(1, PuzzleType.AFFINE)
    image3 = ImageLoader(path.get_image_path(3))
    image4 = ImageLoader(path.get_image_path(4))

    matcher = SiftMatcher(SiftData(image3), SiftData(image4))

    matches = matcher.get_n_random_matches(3)
    af = AffineTransform(matches)
    t = af.transform
    it = af.itransform
    i = 0
