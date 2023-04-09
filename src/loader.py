from numpy import ndarray
import numpy as np
from copy import deepcopy
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
from enum import Enum


# Interactive plot mode
# matplotlib.use('TkAgg')


class Loader:
    def __init__(self, path: str):
        self.path = path

    def load(self):
        pass


class ImageLoader(Loader):

    def __init__(self, path: str):
        super().__init__(path)
        self.color_image: ndarray = None
        self.grayscale_image: ndarray = None

    def load(self):
        # with open(self.path, 'rb') as fp:
        self.color_image = cv2.imread(self.path)
        self.grayscale_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

    @property
    def color_img(self):
        return self.color_image.copy()

    def grayscale_img(self):
        return self.grayscale_image.copy()


class TranformLoader(Loader):
    class Type(Enum):
        AFFINE = 'affine'
        HOMOGRAPHY = 'homography'

    def __init__(self, path: str):
        super().__init__(path)
        self._transform: ndarray = None
        self.__parse_filename()

    def __parse_filename(self):
        filename: str = os.path.basename(self.path).split('.')[0].rstrip('_')
        self._type = TranformLoader.Type.AFFINE if TranformLoader.Type.AFFINE.value in os.path.dirname(
            self.path) else TranformLoader.Type.HOMOGRAPHY
        suffix = filename[filename.rfind('H'):]
        suffix = suffix.split('__')
        self.height = int(suffix[0].split('_')[-1])
        self.width = int(suffix[1].split('_')[-1])

    def load(self):
        self._transform = np.loadtxt(self.path)

    def transform(self) -> ndarray:
        return self._transform.copy()

    @property  # This is like a getter
    def type(self) -> Type:
        return self._type


# if __name__ == '__main__':
#     # a = ImageLoader(r'assignment_1\puzzles\puzzle_affine_1\pieces\piece_1.jpg')
#     # a.load()
#
#
#
#     # a =
#     # a.load()
#     # print(a.transform())
#     transform = TranformLoader(r'assignment_1\puzzles\puzzle_affine_1\warp_mat_1__H_521__W_760_.txt')
#     transform.load()
#
#     image = ImageLoader(r'assignment_1\puzzles\puzzle_affine_1\pieces\piece_1.jpg')
#     image.load()
#
#     new_image = np.zeros(shape=(transform.height, transform.width))

