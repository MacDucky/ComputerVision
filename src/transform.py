import cv2
import numpy as np
from numpy.linalg import inv

from src.loader import *
from src.sift import SiftMatcher, SiftData


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


class AffineTransform(Transform):
    KP_NUM = 3

    def __init__(self, keypoint_matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint]]):
        super().__init__(keypoint_matches)
        if len(keypoint_matches) != self.KP_NUM:
            raise ValueError(f'Affine transformations require {self.KP_NUM} matches')
        self._transform: np.ndarray = np.vstack(
            [cv2.getAffineTransform(self.source_points, self.dest_points), [0, 0, 1]])
        self._itransform: np.ndarray = inv(self._transform)


class HomographyTransform(Transform):
    KP_NUM = 4

    def __init__(self, keypoint_matches: list[tuple[cv2.KeyPoint, cv2.KeyPoint]]):
        super().__init__(keypoint_matches)
        if len(keypoint_matches) == self.KP_NUM:
            raise ValueError(f'Homography transformations require {self.KP_NUM} matches')
        self._transform: np.ndarray = cv2.getPerspectiveTransform(self.source_points, self.dest_points,
                                                                  solveMethod=cv2.DECOMP_SVD)
        self._itransform: np.ndarray = inv(self._transform)


if __name__ == '__main__':
    path = PathLoader(1, PuzzleType.AFFINE)
    image3 = ImageLoader(path.get_image(3))
    image4 = ImageLoader(path.get_image(4))

    matcher = SiftMatcher(SiftData(image3), SiftData(image4))

    matches = matcher.get_n_random_matches(3)
    af = AffineTransform(matches)
    t = af.transform
    it = af.itransform

    i = 0
