import cv2
import numpy as np

from loader import PuzzleType
from transform import AffineTransform, HomographyTransform
from sift import SiftData, SiftMatcher


class Ransac:

    def __init__(self, sift_data_parent: SiftData, sift_data_son: SiftData, puzzle_type: PuzzleType,
                 ratio_threshold=0.8):
        """ Finds optimal warps for given puzzle """
        self.sift_data_parent = sift_data_parent
        self.sift_data_son = sift_data_son
        self.puzzle_type = puzzle_type
        self.matcher = SiftMatcher(sift_data_parent, sift_data_son, ratio_threshold=ratio_threshold)
        self.n_trials = 200
        self.radius_threshold = None
        self.best_transform = None
        self.best_inliers = 0

    def fit_transforms(self, radius_threshold: float = 1.0):
        self.radius_threshold = radius_threshold
        for trial in range(self.n_trials):
            # check which puzzle type it is and building the transform
            if self.puzzle_type == PuzzleType.AFFINE:
                r_matches = self.matcher.get_n_random_matches(3)
                t = AffineTransform(r_matches)
            else:  # homography
                r_matches = self.matcher.get_n_random_matches(4)
                t = HomographyTransform(r_matches)

            # creating mapped vectors
            transform = t.transform
            homogeneous_pts = np.hstack((pts := cv2.KeyPoint_convert(self.matcher.source_data.keypoints),
                                         np.ones((len(pts), 1))))
            mapped_vectors = np.matmul(transform, homogeneous_pts.T)  # warp points (unnormalized homogenous!!)
            under_threshold = 0
            inlier_indices = []

            #
            for index, warped_point in enumerate(mapped_vectors.T):
                if index not in self.matcher.matches:
                    continue
                if self.puzzle_type == PuzzleType.HOMOGRAPHY:
                    normalized_point = warped_point / warped_point[2]
                else:
                    normalized_point = warped_point
                dest_point = np.array(self.matcher.dest_data[self.matcher.matches[index]][0].pt)
                distance = np.linalg.norm((normalized_point[:2]) - dest_point)
                if distance < radius_threshold:
                    under_threshold += 1
                    inlier_indices.append(index)

            under_threshold_refit, t_refit = self.__refit_by_adjacent_points(inlier_indices)
            if under_threshold_refit > under_threshold:
                under_threshold = under_threshold_refit
                t = t_refit
            if under_threshold > self.best_inliers:
                # self.best_transforms[(i, j)] = under_threshold, t
                self.best_transform = t
                self.best_inliers = under_threshold

    def __refit_by_adjacent_points(self, inlier_indices):
        # if not inlier_indices:
        #     inlier_indices = range(matcher.get_number_of_matches())
        matched_src_kp_under_threshold: list[cv2.KeyPoint] = []
        matched_dest_kp_under_threshold: list[cv2.KeyPoint] = []
        for match in self.matcher.get_matched():
            src_index, dest_index = match.queryIdx, match.trainIdx
            if src_index in inlier_indices:
                matched_src_kp_under_threshold.append(self.matcher.source_data[src_index][0])
                matched_dest_kp_under_threshold.append(self.matcher.dest_data[self.matcher.matches[src_index]][0])
        keypoint_matches = [(s, d) for s, d in
                            zip(matched_src_kp_under_threshold, matched_dest_kp_under_threshold)]
        if self.puzzle_type == PuzzleType.AFFINE:
            if len(keypoint_matches) < 3:
                return 0, None
            t = AffineTransform(keypoint_matches)
        else:
            if len(keypoint_matches) < 4:
                return 0, None
            t = HomographyTransform(keypoint_matches)

        transform = t.transform
        homogeneous_pts = np.hstack((pts := cv2.KeyPoint_convert(self.matcher.source_data.keypoints),
                                     np.ones((len(pts), 1))))
        mapped_vectors = np.matmul(transform, homogeneous_pts.T)  # warp points (unnormalized homogenous!!)
        under_threshold = 0
        for source_point, warped_point in zip(cv2.KeyPoint_convert(self.matcher.source_data.keypoints),
                                              mapped_vectors.T):
            if self.puzzle_type == PuzzleType.HOMOGRAPHY:
                normalized_point = warped_point / warped_point[2]
            else:
                normalized_point = warped_point
            distance = np.linalg.norm((normalized_point[:2]) - source_point)
            if distance < self.radius_threshold:
                under_threshold += 1
        return under_threshold, t


if __name__ == '__main__':
    pass
    # ransac = Ransac(1, PuzzleType.AFFINE)
    # radius_thresh = 0.1
    # ransac.fit_transforms(ratio_threshold=0.8, radius_threshold=radius_thresh)
    # n_close, t = ransac.best_transforms.get((1, 2))
    # print(f'RANSAC rt: {radius_thresh}, RANSAC_SUCCESSES: {n_close}')
    # t.transform.dump('best_t.txt')

    # paths = PathLoader(1, PuzzleType.AFFINE)
    # images: list[ImageLoader] = [ImageLoader(p) for p in paths.all_images]
    # base_transform_loader = TransformLoader(paths.transform_path)
    # base_transform = Transform.from_transform(base_transform_loader.transform, base_transform_loader.type)
    # pkled_t = np.load('best_t.txt', allow_pickle=True)
    # picture1_to2 = Transform.from_transform(pkled_t, PuzzleType.AFFINE)
    # warper = Warper(images[0], transform=base_transform_loader)
    #
    # pkled_mul = np.matmul(base_transform_loader.transform, picture1_to2.itransform)
    # compare_images(images[0].color_image, images[1].color_image, False)
    # im1 = warper.warp_first(False)
    # im2 = warper.warp(images[-1].color_image, Transform.from_transform(pkled_mul, PuzzleType.AFFINE))
    # compare_images(im1, im2)
    # show_image(warper.merged_image)
