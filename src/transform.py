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
        if keypoint_matches:
            self._transform = self._create_transform()
            T_to_invert = self._transform if self._transform.shape == (3, 3) else np.vstack(
                [self._transform, [0, 0, 1]])
            _, self._itransform = cv2.invert(T_to_invert, flags=cv2.DECOMP_SVD)

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
        eigen_ratio, instance._itransform = cv2.invert(transform, flags=cv2.DECOMP_SVD)
        return instance

    @abstractmethod
    def _create_transform(self):
        pass


class AffineTransform(Transform):
    def __init__(self, keypoint_matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint]]):
        super().__init__(keypoint_matches)

    @property
    def type(self):
        return PuzzleType.AFFINE

    def _create_transform(self):
        mat_vecs = []
        dest_vec = []
        for p, d in zip(self.source_points, self.dest_points):
            mat_vecs.append(np.array([*p, 1, 0, 0, 0]))
            mat_vecs.append(np.array([0, 0, 0, *p, 1]))
            dest_vec.append(np.array(d))
        T = np.vstack(mat_vecs)
        dest_vec = np.hstack(dest_vec)
        _, T_inv = cv2.invert(T, flags=cv2.DECOMP_SVD)
        return np.matmul(T_inv, dest_vec).reshape((2, 3))


class HomographyTransform(Transform):
    def __init__(self, keypoint_matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint]]):
        super().__init__(keypoint_matches)

    @property
    def type(self) -> PuzzleType:
        return PuzzleType.HOMOGRAPHY

    def _create_transform(self):
        mat_vecs = []
        for s, d in zip(self.source_points, self.dest_points):
            mat_vecs.append(np.array([*(-s), -1, 0, 0, 0, s[0] * d[0], s[1] * d[0], d[0]]))
            mat_vecs.append(np.array([0, 0, 0, *(-s), -1, s[0] * d[1], s[0] * d[0], d[1]]))
        T = np.vstack(mat_vecs)
        _, T_inv = cv2.invert(T, flags=cv2.DECOMP_SVD)
        return np.matmul(T_inv, np.zeros(9)).reshape((3, 3))


if __name__ == '__main__':
    path = PathLoader(1, PuzzleType.AFFINE)
    image3 = ImageLoader(path.get_image_path(1))
    image4 = ImageLoader(path.get_image_path(2))

    matcher = SiftMatcher(SiftData(image3), SiftData(image4))

    matches = matcher.get_n_random_matches(3)
    af = AffineTransform(matches)
    t = af.transform
    it = af.itransform
    i = 0
