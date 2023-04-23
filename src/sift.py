import random

import numpy as np
import cv2
from cv2 import DMatch
from scipy.spatial.distance import cdist

from src.loader import *
from src.plotter import *


class SiftData:
    """ Creates Sift key points and Sift descriptors to the image --> [image, key points, descriptors]"""

    def __init__(self, image: ImageLoader):
        sift = cv2.SIFT_create()
        self.image: ImageLoader = image
        self.keypoints, self.descriptors = sift.detectAndCompute(image.grayscale_image, None)

    def __getitem__(self, index: int):
        return self.keypoints[index], self.descriptors[index]


class SiftMatcher:

    def __init__(self, source_data: SiftData, dest_data: SiftData, ratio_threshold=0.8):
        self.source_data = source_data
        self.dest_data = dest_data
        self.source_index: int = source_data.image.image_index
        self.dest_index: int = dest_data.image.image_index

        # entry (i,j) == distance from descriptor i in source and descriptor j in dest.
        # NxM Matrix, where N is size of source descriptor, and M the size of dest descriptor.
        self.__diffmatrix = cdist(source_data.descriptors, dest_data.descriptors)
        min2_indices = np.argpartition(self.__diffmatrix, 2)[:, :2]
        min2_values = np.ones((len(self.source_data.keypoints), 2))

        for row, indices in enumerate(min2_indices):  # fill values
            min2_values[row] = self.__diffmatrix[row, indices]

        self.matches: dict[int, [int]] = {}
        for row, (min1, min2) in enumerate(min2_values[:, :]):
            if min2 == 0 or 0 <= min1 / min2 <= ratio_threshold:
                self.matches[row] = min2_indices[row][0]

    def get_matched(self):
        """ Generates match map between keypoints """
        for i, j in self.matches.items():
            yield DMatch(_distance=self.__diffmatrix[i, j], _queryIdx=i, _trainIdx=j,
                         _imgIdx=0)  # fixme, imgIdx==dest image index in comparison

    def filter_by_dmatches(self, dmatches):
        return [(self.source_data.keypoints[s.queryIdx], self.dest_data.keypoints[s.trainIdx]) for s in dmatches]

    def get_n_random_matches(self, n: int):
        assert n < len(self.matches), 'Number of samples must be smaller than amount of matches'
        samples: list[DMatch] = random.sample(list(self.get_matched()), n)
        return self.filter_by_dmatches(samples)

    def get_number_of_matches(self) -> int:
        return len(self.source_data.keypoints)
