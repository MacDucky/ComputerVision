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

        base_solution_dir = os.path.join(self.paths.puzzle_base_dir, 'solution')
        os.makedirs(base_solution_dir, exist_ok=True)
        full_sol_filename = os.path.join(base_solution_dir, self.SOLUTION_FILENAME.format(total=total, solved=solved))
        full_cov_filename = os.path.join(base_solution_dir, self.COVERAGE_FILENAME)
        cv2.imwrite(full_cov_filename, cv2.cvtColor(sol_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(full_cov_filename, cv2.cvtColor(cov_image, cv2.COLOR_RGB2BGR))

        for index, image_rel in enumerate(self.image_unifier.warper.warped_images, start=1):
            rel_piece_filename = os.path.join(base_solution_dir, self.PARTIAL_SOL_FILENAME.format(i=index))
            cv2.imwrite(rel_piece_filename, image_rel)
        with open(os.path.join(base_solution_dir, 'params.json'), 'w') as fp:
            json.dump(special_param_dct, fp)
        print('Saved')


if __name__ == '__main__':
    solver = PuzzleSolver(2, PuzzleType.AFFINE)
    unifier_args = {'MIN_MATCHES_NUM': 4, 'SUCCESS_RATE': 0.4, 'RADIUS_THRESHOLD': 0.4, 'RATIO_TEST': 0.6,
                    'RANSAC_STOP_PARAM': 0.999, 'RANSAC_STOP_CRITERIA': Ransac.StopCriteria.DYNAMIC}
    hide_images: list[int] = []
    solver.create_solution(*hide_images, show_image_idx=True, interactive=True, **unifier_args)
