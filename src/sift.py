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


class Matcher:

    def __init__(self, source_data: SiftData, dest_data: SiftData):
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
            if min2 == 0 or 0 <= min1 / min2 <= 0.8:
                self.matches[row] = min2_indices[row][0]

    def get_matched(self):
        """ Generates matched tuples of (keypoint, descriptor) """
        for i, j in self.matches.items():
            yield DMatch(_distance=self.__diffmatrix[i, j], _queryIdx=i, _trainIdx=j,
                         _imgIdx=0)  # fixme, imgIdx==dest image index in comparison


if __name__ == '__main__':
    # path = PathLoader(1, PuzzleType.AFFINE)
    # image1 = ImageLoader(path.get_image(1))
    # image2 = ImageLoader(path.get_image(2))
    # image1 = image1.grayscale_image
    # image2 = image2.grayscale_image
    # sift_data1 = SiftData(image1)
    # sift_data2 = SiftData(image2)
    #
    # difmatrix = cdist(sift_data1.descriptors, sift_data2.descriptors)
    #
    # b = np.ones((len(sift_data1.keypoints), 2))
    # a = np.argpartition(difmatrix, 2)[:, :2]
    # for row, indices in enumerate(a):
    #     b[row] = difmatrix[row, indices]
    #
    # matches = {}
    # for row, (min1, min2) in enumerate(b[:, :]):
    #     if min2 == 0 or 0 <= min1 / min2 <= 0.8:
    #         matches[row] = a[row][0]

    path = PathLoader(1, PuzzleType.AFFINE)
    image1 = ImageLoader(path.get_image(3))
    image2 = ImageLoader(path.get_image(4))
    final_image = draw_matches(image1.grayscale_image, image2.grayscale_image,
                               Matcher(SiftData(image1), SiftData(image2)))
    show_image(final_image)
