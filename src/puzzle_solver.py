import os
from copy import deepcopy
import json

import cv2

from loader import PuzzleType
from unifier import ImageUnifier
from plotter import show_coverage_image, show_image
from ransac import Ransac


class PuzzleSolver:
    SOLUTION_FILENAME = 'solution_{solved}_{total}.jpeg'
    COVERAGE_FILENAME = 'coverage.jpeg'
    PARTIAL_SOL_FILENAME = 'piece_{i}_relative.jpeg'

    def __init__(self, puzzle_num: int, puzzle_type: PuzzleType):
        self.puzzle_num: int = puzzle_num
        self.puzzle_type: PuzzleType = puzzle_type
        self.image_unifier = ImageUnifier(puzzle_num, puzzle_type)
        self.paths = self.image_unifier.paths
        self.base_solution_dir = os.path.join(self.paths.puzzle_base_dir, 'solution')

    def negate_indices_to_keep(self, keep_indices_: list[int]) -> list[int]:
        keep_indices = set(keep_indices_)
        all_indices = []
        for file in os.listdir(self.base_solution_dir):
            if file.startswith('piece'):
                all_indices.append(int(file.split('_')[1]))
        return sorted(list(set(all_indices).difference(keep_indices)))

    def __set_unifier_special_params(self, **unifier_params):
        sp = self.image_unifier.special_parameters()
        dct = {}
        for param in sp:
            if unifier_params.get(param):
                dct[param] = unifier_params.get(param)
                setattr(self.image_unifier, param, unifier_params.get(param))
        default_keys = set(sp).difference(set(dct))
        for key in default_keys:
            dct[key] = getattr(self.image_unifier, key)
        return dct

    def create_solution(self, *hide_images, show_image_idx: bool = False, interactive: bool = True, **unifier_args):
        special_param_dct = self.__set_unifier_special_params(**unifier_args)
        unsolved_pieces: list[int] = self.image_unifier.build_data()
        total = len(self.image_unifier.images)
        solved = len(self.image_unifier.images) - len(unsolved_pieces)
        self.image_unifier.warp_images()
        cov_image = show_coverage_image(deepcopy(self.image_unifier.warper.warped_images), *hide_images,
                                        show_image_idx=show_image_idx, no_gui=not interactive)
        sol_image = self.image_unifier.merged_image(False, *hide_images)
        if interactive:
            show_image(sol_image, grayscale=False)
            positive = {'Y', 'y', 'yes', 'Yes'}
            negative = {'No', 'no', 'N', 'n'}
            while (answer := input('Save solution? [Y/n] ')) not in positive.union(negative):
                print('Incorrect choice, Try again.\n')
            if answer in negative:
                return

        # base_solution_dir = os.path.join(self.paths.puzzle_base_dir, 'solution')
        os.makedirs(self.base_solution_dir, exist_ok=True)
        full_sol_filename = os.path.join(self.base_solution_dir,
                                         self.SOLUTION_FILENAME.format(total=total, solved=solved))
        full_cov_filename = os.path.join(self.base_solution_dir, self.COVERAGE_FILENAME)
        cv2.imwrite(full_sol_filename, cv2.cvtColor(sol_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(full_cov_filename, cv2.cvtColor(cov_image, cv2.COLOR_RGB2BGR))

        for index, image_rel in enumerate(self.image_unifier.warper.warped_images, start=1):
            try:
                if not image_rel:
                    continue
            except:
                pass
            rel_piece_filename = os.path.join(self.base_solution_dir, self.PARTIAL_SOL_FILENAME.format(i=index))
            cv2.imwrite(rel_piece_filename, cv2.cvtColor(image_rel, cv2.COLOR_RGB2BGR))
        with open(os.path.join(self.base_solution_dir, 'params.json'), 'w') as fp:
            json.dump(special_param_dct, fp)
        print('Saved')


if __name__ == '__main__':
    # for i in range(6, 11):
    #     solver = PuzzleSolver(i, PuzzleType.HOMOGRAPHY)
    #     unifier_args = {'MIN_MATCHES_NUM': 4, 'SUCCESS_RATE': 0.265, 'RADIUS_THRESHOLD': 5, 'RATIO_TEST': 0.7,
    #                     #  'RANSAC_STOP_PARAM': 200, 'RANSAC_STOP_CRITERIA': Ransac.StopCriteria.N_TRIALS}
    #                     'RANSAC_STOP_PARAM': 0.999, 'RANSAC_STOP_CRITERIA': Ransac.StopCriteria.DYNAMIC}
    #     hide_images: list[int] = []  # 8A - [14, 17, 21, 24]
    #     solver.create_solution(*hide_images, show_image_idx=True, interactive=False, **unifier_args)
    solver = PuzzleSolver(10, PuzzleType.HOMOGRAPHY)
    unifier_args = {'MIN_MATCHES_NUM': 4, 'SUCCESS_RATE': 0.1, 'RADIUS_THRESHOLD': 10, 'RATIO_TEST': 0.45,
                     'RANSAC_STOP_PARAM': 20, 'RANSAC_STOP_CRITERIA': Ransac.StopCriteria.N_TRIALS}
                    # 'RANSAC_STOP_PARAM': 0.8, 'RANSAC_STOP_CRITERIA': Ransac.StopCriteria.DYNAMIC}

    keep_images: list[int] = [1]
    hide_images = solver.negate_indices_to_keep(keep_images)
    # hide_images = []
    # 9A[2, 3, 8, 9, 10, 13, 19, 20, 22, 29, 30, 31, 34, 35, 38, 39, 14, 18, 19, 52,56,49]  # 8A - [14, 17, 21, 24]
    solver.create_solution(*hide_images, show_image_idx=True, interactive=True, **unifier_args)
