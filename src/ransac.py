import typing
import cv2
import numpy as np

from loader import PathLoader, ImageLoader, TransformLoader, PuzzleType
from warper import Warper
from transform import Transform, AffineTransform, HomographyTransform
from sift import SiftData, SiftMatcher
from itertools import combinations


class Ransac:

    def __init__(self, puzzle_index: int, puzzle_type: PuzzleType):
        """ Finds optimal warps for given puzzle """
        self.paths = PathLoader(puzzle_index, puzzle_type)
        self.images: list[ImageLoader] = [ImageLoader(p) for p in self.paths.all_images]
        self.base_transform_loader = TransformLoader(self.paths.transform_path)
        self.base_transform = Transform.from_transform(self.base_transform_loader.transform,
                                                       self.base_transform_loader.type)
        self.sift_datas: list[SiftData] = [SiftData(image) for image in self.images]
        self.best_transforms: dict[tuple[int, int], [tuple[float, Transform]]] = {}
        self.n_trials = 2500

    def fit_transforms(self, radius_threshold=50):
        for dest_image, to_fit_image in combinations(self.images, r=2):
            i = dest_image.image_index
            j = to_fit_image.image_index
            image1 = dest_image.grayscale_image
            image2 = to_fit_image.grayscale_image
            sift_data1 = self.sift_datas[i - 1]
            sift_data2 = self.sift_datas[j - 1]
            matcher = SiftMatcher(sift_data1, sift_data2)
            for _ in range(self.n_trials):
                best_fit_pts, _ = self.best_transforms.get((i, j), (0, None))
                if self.base_transform_loader.type == PuzzleType.AFFINE:
                    r_matches = matcher.get_n_random_matches(3)
                    t = AffineTransform(r_matches)
                else:
                    r_matches = matcher.get_n_random_matches(4)
                    t = HomographyTransform(r_matches)
                itransform = t.itransform
                homogeneous_pts = np.hstack((pts := cv2.KeyPoint_convert(matcher.dest_data.keypoints),
                                             np.ones((len(pts), 1))))
                mapped_vectors = [((res := np.matmul(itransform, v)) / res[2])[:2] for v in homogeneous_pts]
                under_threshold = 0
                # fit_pts = [] # todo add fitting with fit_pts after each attempted transform
                for s, d in zip(cv2.KeyPoint_convert(matcher.source_data.keypoints), mapped_vectors):
                    distance = np.linalg.norm(d - s)
                    if distance < radius_threshold:
                        under_threshold += 1
                        # fit_pts.append(d)
                if under_threshold > best_fit_pts:
                    self.best_transforms[(i, j)] = under_threshold, t

            i = 0


if __name__ == '__main__':
    # ransac = Ransac(1, PuzzleType.AFFINE)
    # ransac.fit_transforms()
    # n_close, t = ransac.best_transforms.get((1, 2))
    # print(f'RANSAC rt: 50, RANSAC_SUCCESSES: {n_close}')
    # t.transform.dump('best_t.txt')

    paths = PathLoader(1, PuzzleType.AFFINE)
    images: list[ImageLoader] = [ImageLoader(p) for p in paths.all_images]
    base_transform_loader = TransformLoader(paths.transform_path)
    base_transform = Transform.from_transform(base_transform_loader.transform, base_transform_loader.type)
    pkled_t = np.load('best_t.txt', allow_pickle=True)
    warper = Warper(images[0], transform=base_transform_loader)
    from plotter import compare_images, draw_matches

    pkled_mul = np.matmul(base_transform_loader.transform, pkled_t)
    compare_images(images[0].grayscale_image, images[1].grayscale_image)
    compare_images(warper.warp_first(), warper.warp(images[-1].grayscale_image,
                                                    Transform.from_transform(pkled_mul, PuzzleType.AFFINE)))

    i = 0
