from numpy import ndarray
import numpy as np
import cv2
import os
from enum import Enum
from abc import ABC, abstractmethod


class PuzzleType(Enum):
    AFFINE = 'affine'
    HOMOGRAPHY = 'homography'


class Loader(ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def load(self):
        pass


class ImageLoader(Loader):
    """ Loads an image in a given path. --> [Grayscale Image, Color Image]"""

    def __init__(self, path: str):
        super().__init__(path)
        self.color_image: ndarray = None
        self.grayscale_image: ndarray = None
        self.load()

    def load(self):
        # with open(self.path, 'rb') as fp:
        self.color_image = cv2.imread(self.path)
        self.grayscale_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

    @property
    def color_img(self) -> np.ndarray:
        return self.color_image.copy()

    def grayscale_img(self) -> np.ndarray:
        return self.grayscale_image.copy()

    @property
    def image_index(self) -> int:
        return int(self.path.split('.', maxsplit=1)[0].rsplit('_', maxsplit=1)[-1])


class TranformLoader(Loader):
    """ Loads transformation file. --> [transform, transform_type, height, width]"""

    def __init__(self, path: str):
        super().__init__(path)
        self._transform: ndarray = None
        self.__parse_filename()
        self.load()

    def __parse_filename(self):
        filename: str = os.path.basename(self.path).split('.')[0].rstrip('_')
        self.type = PuzzleType.AFFINE if PuzzleType.AFFINE.value in os.path.dirname(self.path) \
            else PuzzleType.HOMOGRAPHY
        suffix = filename[filename.rfind('H'):]
        suffix = suffix.split('__')
        self.height = int(suffix[0].split('_')[-1])
        self.width = int(suffix[1].split('_')[-1])

    def load(self):
        self._transform = np.loadtxt(self.path)

    @property
    def transform(self) -> ndarray:
        return self._transform.copy()


class PathLoader:
    """ Loads paths to relevant image directory / transformation file."""
    BASE_PUZZLE_PATH = 'assignment_1/puzzles'

    def __init__(self, puzzle_num, puzzle_type: PuzzleType):
        self.images_path = os.path.join(os.path.abspath(self.BASE_PUZZLE_PATH),
                                        f'puzzle_{puzzle_type.value}_{puzzle_num}', 'pieces')
        transform_path = os.path.join(os.path.dirname(self.images_path))
        warp_file_full_name = list(filter(lambda x: x.startswith('warp'), os.listdir(transform_path)))[0]
        self.transform_path = os.path.join(transform_path, warp_file_full_name)

    def get_image(self, image_num: int) -> str | None:
        files = os.listdir(self.images_path)
        for file in files:
            num = file.split('.')[0].split('_')[-1]
            if int(num) == image_num:
                break
        else:
            return None
        return os.path.join(self.images_path, file)
