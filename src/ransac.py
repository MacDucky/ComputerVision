from enum import Enum
import cv2
import numpy as np
from math import log2, ceil

from loader import PuzzleType
from transform import AffineTransform, HomographyTransform
from sift import SiftData, SiftMatcher


class Ransac:
    class StopCriteria(str,Enum):
        # Satisfy amount of stop_param = [n] inliers
        INLIER_SAT = 'inliers'

        # After stop_param = [n] trials
        N_TRIALS = 'n_trials'

        # after log(1-P)/log(1-p**k) where stop_param = [P := Overall Success rate] and p := inlier rate
        DYNAMIC = 'dynamic'

    def __init__(self, sift_data_parent: SiftData, sift_data_son: SiftData, puzzle_type: PuzzleType,
                 stop_param: int | float, stop_criteria: StopCriteria,
                 ratio_threshold=0.8):
        """ Finds optimal warps for given puzzle """
        self.sift_data_parent = sift_data_parent
        self.sift_data_son = sift_data_son
        self.puzzle_type = puzzle_type
        self.matcher = SiftMatcher(sift_data_parent, sift_data_son, ratio_threshold=ratio_threshold)
        self.radius_threshold = None
        self.best_transform = None
        self.best_inliers = 0
        self.best_inlier_rate = 0
        self.stop_condition_param = stop_param
        self.stop_criteria = stop_criteria
        self.current_trial = 0

    def is_under_n_trials(self) -> bool:
        res = self.current_trial < self.stop_condition_param
        self.current_trial += 1
        return res

    def dynamic_criteria(self) -> bool:
        if self.best_inlier_rate == 0:
            return True
        try:
            min_trials = log2(1 - self.stop_condition_param) / \
                         log2(1 - self.best_inlier_rate ** (3 if self.puzzle_type == PuzzleType.AFFINE else 4))
        except ValueError:
            return False
        temp = self.current_trial
        self.current_trial += 1
        return temp < min_trials

    def get_trial_stop_condition(self):
        if self.stop_criteria == Ransac.StopCriteria.INLIER_SAT:
            return lambda: self.best_inliers < self.stop_condition_param
        elif self.stop_criteria == Ransac.StopCriteria.N_TRIALS:
            return self.is_under_n_trials
        else:  # DYNAMIC
            return self.dynamic_criteria

    def fit_transforms(self, radius_threshold: float = 1.0):
        self.radius_threshold = radius_threshold
        run_condition = self.get_trial_stop_condition()
        while run_condition():
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
                self.best_transform = t
                self.best_inliers = under_threshold
                self.best_inlier_rate = self.best_inliers / len(self.matcher.matches)

    def __refit_by_adjacent_points(self, inlier_indices):
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