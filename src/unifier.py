from copy import deepcopy
import numpy as np

from loader import ImageLoader, TransformLoader, PuzzleType, PathLoader
from src.plotter import show_image, show_coverage_image, compare_images
from warper import Warper
from ransac import Ransac
from sift import SiftData
from transform import Transform


class ImageUnifier:
    MIN_MATCHES_NUM = 4
    SUCCESS_RATE = 0.4  # inliers / matches
    RADIUS_THRESHOLD = 1
    RATIO_TEST = 0.6
    RANSAC_STOP_PARAM = 0.999  # see Ransac.StopCriteria for info
    RANSAC_STOP_CRITERIA = Ransac.StopCriteria.DYNAMIC

    def __init__(self, puzzle_index: int, puzzle_type: PuzzleType):
        # collecting the images and the initial transform
        self.paths = PathLoader(puzzle_index, puzzle_type)
        self.images: list[ImageLoader] = [ImageLoader(p) for p in self.paths.all_images]
        self.sift_datas: list[SiftData] = [SiftData(image) for image in self.images]
        init_image = self.images[0]
        base_transform = TransformLoader(self.paths.transform_path)
        self.puzzle_type = puzzle_type

        # Creating the warper and the transform array
        self.warper = Warper(init_image, base_transform, pieces_amount=len(self.images))
        self.transform_array: list[Transform | None] = [None] * len(self.images)
        self.transform_array[0] = Transform.from_transform(base_transform.transform, self.puzzle_type)

    def add_node(self, idx_parent: int, idx_son: int) -> bool:
        ransac = Ransac(self.sift_datas[idx_parent - 1], self.sift_datas[idx_son - 1], self.puzzle_type,
                        stop_param=self.RANSAC_STOP_PARAM, stop_criteria=self.RANSAC_STOP_CRITERIA,
                        ratio_threshold=self.RATIO_TEST)
        if len(ransac.matcher.matches) >= self.MIN_MATCHES_NUM:
            ransac.fit_transforms(self.RADIUS_THRESHOLD)
            if ransac.best_inlier_rate > self.SUCCESS_RATE:
                itransform = ransac.best_transform.itransform
                self.transform_array[idx_son - 1] = Transform.from_transform(
                    np.matmul(self.transform_array[idx_parent - 1].transform, itransform), self.puzzle_type)
                # self.warper.warp(self.images[idx_son - 1].grayscale_image, self.transArray[idx_son - 1])
                return True
        return False

    def warp_images(self, grayscale=True):
        self.warper.warp_first(grayscale)  # at least 1 image is warped by this point.

        # if more than 1 image is warped, this means it was warped previously.
        if len([im for im in self.warper.warped_images if im is not None]) > 1:
            return

        for index, image, transform in zip(range(1, len(self.images[1:]) + 1),
                                           self.images[1:],
                                           self.transform_array[1:]):
            if transform:
                self.warper.warp(image.grayscale_img if grayscale else image.color_img, transform, index)

    def build_data(self) -> list[int]:
        """"Build the array and the warper, and return number of unsuitable pieces"""
        num_pieces = len(self.images)
        left_idx_pieces = list(range(2, num_pieces + 1))
        last_visited_idx_pieces = [1]
        # Continue if we didn't go over all the pieces, and we visit any new piece in the loop
        # break out of the while loop if:
        # 1. a transformation was not found for any of the images that already have a transformation
        # 2. we found all transformations
        while len(left_idx_pieces) > 0 and len(last_visited_idx_pieces) > 0:
            visited_idx_pieces = []
            for i in last_visited_idx_pieces:
                for j in left_idx_pieces:
                    if self.add_node(i, j):
                        visited_idx_pieces.append(j)
                        left_idx_pieces.remove(j)
            last_visited_idx_pieces = visited_idx_pieces
        return left_idx_pieces  # pieces without a transformation

    def merged_image(self, grayscale: bool = True, *unselect_images) -> np.ndarray:
        if all(x is None for x in self.warper.warped_images) or self.warper.is_grayscale != grayscale:
            self.warp_images(grayscale)
        return self.warper.merged_image(self.transform_array, *unselect_images)

    @classmethod
    def special_parameters(cls):
        return [attr for attr in dir(cls) if
                not callable(getattr(cls, attr)) and not attr.startswith("__") and attr.isupper()]


if __name__ == '__main__':
    unifier = ImageUnifier(5, PuzzleType.AFFINE)
    unifier.build_data()
    unifier.warp_images()
    hide_images = []
    show_coverage_image(deepcopy(unifier.warper.warped_images), *hide_images, show_image_idx=True)
    merged_image = unifier.merged_image(False, *hide_images)
    show_image(merged_image)

    merged_image = unifier.merged_image(False)
    show_image(merged_image)
