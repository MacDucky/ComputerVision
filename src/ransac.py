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
        self.best_transforms: dict[tuple[int, int], [tuple[int, Transform]]] = {}
        self.n_trials = 2500
        self.radius_threshold = None

    def fit_transforms(self, ratio_threshold=0.8, radius_threshold=50):
        self.radius_threshold = radius_threshold
        for dest_image, to_fit_image in combinations(self.images, r=2):
            # setup for two images
            i = dest_image.image_index
            j = to_fit_image.image_index
            sift_data1 = self.sift_datas[i - 1]
            sift_data2 = self.sift_datas[j - 1]
            matcher = SiftMatcher(sift_data1, sift_data2, ratio_threshold=ratio_threshold)

            for trial in range(self.n_trials):
                # get number of best fit points (and transform) given indices of src and dst images
                best_fit_pts, _ = self.best_transforms.get((i, j), (0, None))
                # handling creation of transforms between 2 images
                puzzle_type = self.base_transform_loader.type
                if puzzle_type == PuzzleType.AFFINE:
                    r_matches = matcher.get_n_random_matches(3)
                    t = AffineTransform(r_matches)
                else:  # homography
                    r_matches = matcher.get_n_random_matches(4)
                    t = HomographyTransform(r_matches)

                transform = t.transform
                homogeneous_pts = np.hstack((pts := cv2.KeyPoint_convert(matcher.source_data.keypoints),
                                             np.ones((len(pts), 1))))
                # todo: warp affine transforms without the division
                mapped_vectors = np.matmul(transform, homogeneous_pts.T)  # warp points (unnormalized homogenous!!)
                under_threshold = 0
                inlier_indices = []
                for index, warped_point in enumerate(mapped_vectors.T):
                    if index not in matcher.matches:
                        continue
                    normalized_point = warped_point / warped_point[2]
                    dest_point = np.array(matcher.dest_data[matcher.matches[index]][0].pt)
                    distance = np.linalg.norm((normalized_point[:2]) - dest_point)
                    if distance < radius_threshold:
                        under_threshold += 1
                        inlier_indices.append(index)

                under_threshold_refit, t_refit = self.__refit_by_adjacent_points(matcher, puzzle_type, inlier_indices)
                if under_threshold_refit > under_threshold:
                    under_threshold = under_threshold_refit
                    t = t_refit
                if under_threshold > best_fit_pts:
                    self.best_transforms[(i, j)] = under_threshold, t

            i = 0

    # def __refit_by_adjacent_points(self, matcher: SiftMatcher, puzzle_type: PuzzleType, inlier_indices=None):
    def __refit_by_adjacent_points(self, matcher: SiftMatcher, puzzle_type: PuzzleType, inlier_indices):
        # if not inlier_indices:
        #     inlier_indices = range(matcher.get_number_of_matches())
        matched_src_kp_under_threshold: list[cv2.KeyPoint] = []
        matched_dest_kp_under_threshold: list[cv2.KeyPoint] = []
        for match in matcher.get_matched():
            src_index, dest_index = match.queryIdx, match.trainIdx
            if src_index in inlier_indices:
                matched_src_kp_under_threshold.append(matcher.source_data[src_index][0])
                matched_dest_kp_under_threshold.append(matcher.dest_data[matcher.matches[src_index]][0])
        keypoint_matches = [tuple(s, d) for s, d in
                            zip(matched_src_kp_under_threshold, matched_dest_kp_under_threshold)]
        if puzzle_type == PuzzleType.AFFINE:
            t = AffineTransform(keypoint_matches)
        else:
            t = HomographyTransform(keypoint_matches)

        transform = t.transform
        homogeneous_pts = np.hstack((pts := cv2.KeyPoint_convert(matcher.source_data.keypoints),
                                     np.ones((len(pts), 1))))
        # todo: warp affine transforms without the division
        mapped_vectors = np.matmul(transform, homogeneous_pts.T)  # warp points (unnormalized homogenous!!)
        under_threshold = 0
        for source_point, warped_point in zip(cv2.KeyPoint_convert(matcher.source_data.keypoints), mapped_vectors.T):
            normalized_point = warped_point / warped_point[2]
            distance = np.linalg.norm((normalized_point[:2]) - source_point)
            if distance < self.radius_threshold:
                under_threshold += 1
        return under_threshold, t

    # @staticmethod
    # def __create_transform(kp_matches, puzzle_type, random_from_matches=False):
    #     if puzzle_type == PuzzleType.AFFINE:
    #         r_matches = matcher.get_n_random_matches(3)
    #         t = AffineTransform(r_matches)
    #     else:  # homography
    #         r_matches = matcher.get_n_random_matches(4)
    #         t = HomographyTransform(r_matches)


if __name__ == '__main__':
    ransac = Ransac(1, PuzzleType.AFFINE)
    radius_thresh = 70
    ransac.fit_transforms(ratio_threshold=0.65, radius_threshold=radius_thresh)
    n_close, t = ransac.best_transforms.get((1, 2))
    print(f'RANSAC rt: {radius_thresh}, RANSAC_SUCCESSES: {n_close}')
    t.transform.dump('best_t.txt')
    #
    paths = PathLoader(1, PuzzleType.AFFINE)
    images: list[ImageLoader] = [ImageLoader(p) for p in paths.all_images]
    base_transform_loader = TransformLoader(paths.transform_path)
    base_transform = Transform.from_transform(base_transform_loader.transform, base_transform_loader.type)
    pkled_t = np.load('best_t.txt', allow_pickle=True)
    warper = Warper(images[0], transform=base_transform_loader)
    from plotter import compare_images, draw_matches

    pkled_mul = np.matmul(base_transform_loader.transform, pkled_t)
    # pkled_mul = np.matmul(base_transform_loader.transform, t.transform)
    final_t = Transform.from_transform(pkled_mul, PuzzleType.AFFINE)
    final_t = Transform.from_transform(final_t.itransform, PuzzleType.AFFINE)
    compare_images(images[0].grayscale_image, images[1].grayscale_image)
    compare_images(warper.warp_first(), warper.warp(images[-1].grayscale_image, final_t))

    i = 0
