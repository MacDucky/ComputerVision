import os
import cv2
import numpy as np
from loader import ImageLoader
from camera import Camera
from numpy import ndarray
from timeit import default_timer as perf_counter


class Stereo:
    DISPARITY_WIN_SIZE = (9, 9)
    RHO_THRESHOLD = 1
    GAUSSIAN_SIGMA = 5

    def __init__(self, image_left: ImageLoader, image_right: ImageLoader, camera_left: Camera, camera_right: Camera,
                 max_disparity: int):
        self.image_left = image_left
        self.image_right = image_right

        # Camera maps
        self.camera_left = camera_left
        self.camera_right = camera_right
        # Disparities Maps
        self.disparity_left: None | ndarray = None
        self.disparity_right: None | ndarray = None
        self.max_disparity = max_disparity
        # Depth Maps
        self.depth_left: None | ndarray = None
        self.depth_right: None | ndarray = None
        # Back Projection Matrix from camera 1 (all the pixels represented in camera axis 1)
        self.__back_projection: None | ndarray = None

    @staticmethod
    def __create_census(sliding_window: ndarray, rho: int) -> ndarray:
        pad_h, pad_w = (sliding_window.shape[2] - 1) // 2, (sliding_window.shape[3] - 1) // 2

        # create a sliding window comparison of center values per window
        center_value = sliding_window[:, :, pad_h, pad_w]
        center_value = np.full_like(sliding_window, center_value[:, :, None, None])

        # create a sliding window output values
        zero_one = np.full(sliding_window.shape, 1)
        one_zero = np.full(sliding_window.shape, 2)
        zero_zero = np.full(sliding_window.shape, 0)

        # apply census mapping on sliding windows
        census_values = np.where(sliding_window > (center_value + rho), one_zero,
                                 np.where(sliding_window < (center_value - rho), zero_one,
                                          zero_zero))

        # unpack decimals to bits
        census_bits = np.unpackbits(census_values.astype(np.uint8), axis=-1)
        census_bits = census_bits.flatten().reshape((*sliding_window.shape, 8))
        # reduce to 2 bit representation
        final_census = np.array(census_bits)[:, :, :, :, -2:]

        # remove center value at each window
        remove_center_mask = np.ones_like(final_census, dtype=bool)
        remove_center_mask[:, :, pad_h, pad_w, :] = False
        final_census = final_census[remove_center_mask]

        return final_census.reshape(
            (*sliding_window.shape[:2], 2 * (sliding_window.shape[2] * sliding_window.shape[3] - 1)))

    @staticmethod
    def __create_census_map(image: ndarray, window_size: tuple, rho: int):
        """
        Returns a census map from a given image

        Parameters:
            image(ndarray): a position of the camera origin in world coordinates.
            window_size(tuple): the size of the window
            rho: threshold for reducing the noisy effect
        Returns:
            A census map of the image
        """
        padded_height, padded_width = (window_size[0] - 1) // 2, (window_size[1] - 1) // 2
        padded_image = np.pad(image.astype(int), (padded_height, padded_width), constant_values=0)
        sliding_window = np.lib.stride_tricks.sliding_window_view(padded_image, window_size)
        census_map = Stereo.__create_census(sliding_window, rho)
        return census_map

    @staticmethod
    def __winner_takes_all(census_image_src: ndarray, census_image_candidate: ndarray, is_left_im: bool,
                           arm_length: int, max_cost: int):
        """
        Returns the cost image of the source image given the census source image and the other census image.

        Parameters:
            census_image_src: the census source image.
            census_image_candidate: the other census image.
            is_left_im(bool): True if the source image is the left image.
            arm_length(int): the size of the cost array for each pixel.
            max_cost: maximum cost of a census
        Returns:
            return the disparity
        """
        height, width, _ = census_image_src.shape
        disparity = np.full((height, width, 4), max_cost)
        for d in range(1, arm_length + 1):
            x = np.arange(width)

            x_right = x[:] - d if is_left_im else x[:] + d
            invalid_indices = (x_right < 0) | (x_right >= width)
            x_right = np.where(invalid_indices, 0, x_right)

            cost_volume = np.where(invalid_indices, np.inf, np.sum(
                np.bitwise_xor(census_image_src[:, :, :], census_image_candidate[:, x_right, :]), axis=2))

            cost_volume = cv2.GaussianBlur(cost_volume, (15, 15), Stereo.GAUSSIAN_SIGMA)

            # mask for: new minimum is found
            min_mask = cost_volume < disparity[..., 0]

            # mask for: another minimum is found
            equal_mask = np.logical_and(cost_volume == disparity[..., 0], np.logical_not(min_mask))

            # mask for: another minimum is found and it is the third
            triple_mask = disparity[..., 2] == 1

            # mask for: another minimum is found and we already have three or more
            more_than_triple_mask = disparity[..., 2] >= 2

            disparity[min_mask, 0] = cost_volume[min_mask]
            disparity[min_mask, 1] = d
            disparity[min_mask, 2] = 0
            disparity[min_mask, 3] = d

            disparity[equal_mask & triple_mask, 1] = disparity[equal_mask & triple_mask, 3]

            disparity[equal_mask & more_than_triple_mask, 1] = 0

            disparity[equal_mask, 2] += 1
            disparity[equal_mask, 3] = d

        disparity = disparity[..., 1].astype(int)
        return disparity

    @staticmethod
    def __consistency_test(src_disparity: ndarray, dest_disparity: ndarray, is_left_img: int):
        """
        Returns an improved disparity

        Parameters:
            src_disparity: source disparity, the disparity we want to improve
            dest_disparity: destination disparity, the disparity we use to improve the source disparity
            is_left_img: is the source image is the left image?
        Returns:
             source disparity
        """
        height, width = src_disparity.shape
        consistency_map = np.zeros((height, width), dtype=np.float32)
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        r_x_coords = x_coords - (2 * is_left_img - 1) * src_disparity
        valid_coords = np.logical_and(r_x_coords >= 0, r_x_coords < width)

        valid_left_disparity = src_disparity[valid_coords]
        valid_right_disparity = dest_disparity[y_coords[valid_coords], r_x_coords[valid_coords].astype(int)]

        consistency_mask = valid_left_disparity == valid_right_disparity
        consistency_map[y_coords[valid_coords][consistency_mask], x_coords[valid_coords][consistency_mask]] = \
            valid_left_disparity[consistency_mask]

        return consistency_map

    @staticmethod
    def __bonus(disparity):
        height, width = disparity.shape
        window_size = 25
        window_half = window_size // 2

        for y in range(height):
            for x in range(width):
                if disparity[y, x] == 0:
                    ymin = max(0, y - window_half)
                    ymax = min(height, y + window_half + 1)
                    xmin = max(0, x - window_half)
                    xmax = min(width, x + window_half + 1)

                    neighbors = disparity[ymin:ymax, xmin:xmax]
                    unique, counts = np.unique(neighbors, return_counts=True)
                    most_common_value = unique[np.argmax(counts)]

                    disparity[y, x] = most_common_value
        return disparity

    def calc_disparity_maps(self, window_size: tuple[int, int], rho: int):
        """
        Create disparity maps to both left & right images, then apply a winner takes it all + postprocessing.
        """
        start_time_disparity = perf_counter()
        # Census map creation

        start_time_census_left = perf_counter()
        census_map_left = self.__create_census_map(self.image_left.grayscale_img, window_size, rho)
        total_time_census_left = perf_counter() - start_time_census_left
        print(f'Total time taken during left census map creation: {total_time_census_left}')

        start_time_census_right = perf_counter()
        census_map_right = self.__create_census_map(self.image_right.grayscale_img, window_size, rho)
        total_time_census_left = perf_counter() - start_time_census_right
        print(f'Total time taken during right census map creation: {total_time_census_left}')

        max_cost = 2 * (window_size[0] * window_size[1] - 1)

        # the cost aggregations for every pixel in the left image
        start_time_census_right = perf_counter()
        left_disparity = self.__winner_takes_all(census_map_left, census_map_right, True, self.max_disparity,
                                                 max_cost=max_cost)
        total_time_census_left = perf_counter() - start_time_census_right
        print(f'Total time taken during winner takes all technique for the left disparity: {total_time_census_left}')

        start_time_wta_right = perf_counter()
        right_disparity = self.__winner_takes_all(census_map_right, census_map_left, False, self.max_disparity,
                                                  max_cost=max_cost)
        total_time_wta_right = perf_counter() - start_time_wta_right
        print(f'Total time taken during winner takes all technique for the right disparity: {total_time_wta_right}')

        start_time_consistency_left = perf_counter()
        self.disparity_left = Stereo.__consistency_test(left_disparity, right_disparity, True)
        total_time_consistency_left = perf_counter() - start_time_consistency_left
        print(
            f'Total time taken during consistency test technique for the left disparity: {total_time_consistency_left}')

        start_time_consistency_right = perf_counter()
        self.disparity_right = Stereo.__consistency_test(right_disparity, left_disparity, False)
        total_time_consistency_right = perf_counter() - start_time_consistency_right
        print(
            f'Total time taken during consistency test technique for the right disparity: {total_time_consistency_right}')

        start_time_bonus_left = perf_counter()
        self.disparity_left = Stereo.__bonus(self.disparity_left)
        total_time_bonus_left = perf_counter() - start_time_bonus_left
        print(
            f'Total time taken during consistency test technique for the left disparity: {total_time_bonus_left}')

        start_time_bonus_right = perf_counter()
        self.disparity_left = Stereo.__bonus(self.disparity_right)
        total_time_bonus_right = perf_counter() - start_time_bonus_right
        print(
            f'Total time taken during consistency test technique for the left disparity: {total_time_bonus_right}')

        total_time_disparity = perf_counter() - start_time_disparity
        print(f'Total time taken during disparity creation: {total_time_disparity}')

        # disparity_left_im = cv2.normalize(self.disparity_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
        #                                   dtype=cv2.CV_8U)
        # disparity_right_im = cv2.normalize(self.disparity_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
        #                                    dtype=cv2.CV_8U)
        # compare_images(disparity_left_im, disparity_right_im)

    def calc_depth_maps(self):
        """
        Calculates depth maps for the cameras.
        """
        left_cam_origin = self.camera_left.origin[:-1]
        right_cam_origin = self.camera_right.origin[:-1]
        baseline_len = np.linalg.norm(left_cam_origin - right_cam_origin)
        focal_len = self.camera_left.focal_len  # assumes same intrinsics for both cameras
        similarity_ratio = baseline_len * focal_len
        if self.depth_left is None:
            self.depth_left = np.divide(similarity_ratio, self.disparity_left, where=self.disparity_left != 0)
            self.depth_left = np.nan_to_num(self.depth_left)

        if self.depth_right is None:
            self.depth_right = np.divide(similarity_ratio, self.disparity_right, where=self.disparity_right != 0)
            self.depth_right = np.nan_to_num(self.depth_right)

    def calc_back_projection(self):
        """ Calculates the back projection for the left camera in the setup. """
        i_intrinsics = np.linalg.pinv(self.camera_left.intrinsic_transform)
        height, width = self.depth_left.shape

        # Creating a pixel coordinate matrix
        x = np.linspace(0, width - 1, width).astype(int)
        y = np.linspace(0, height - 1, height).astype(int)
        [x, y] = np.meshgrid(x, y)
        pixel_coordinates = np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

        # Scale each point on the line from camera center and the pixel coordinate on the image plane(as camera coords),
        # to its original z-value
        inverse_points = i_intrinsics @ pixel_coordinates
        inverse_points = inverse_points / inverse_points[2]
        self.__back_projection = inverse_points * self.depth_left.flatten()

    @property
    def back_projection(self):
        if self.disparity_left is None or self.disparity_right is None:
            self.calc_disparity_maps(window_size=self.DISPARITY_WIN_SIZE, rho=self.RHO_THRESHOLD)
        if self.depth_left is None or self.depth_right is None:
            self.calc_depth_maps()
        if self.__back_projection is None:  # generalize to right camera as well
            self.calc_back_projection()
        return self.__back_projection.copy()

    @classmethod
    def set_disparity_params(cls, window_size: tuple[int, int], rho_thresh: int, gaussian_sigma: int):
        cls.DISPARITY_WIN_SIZE = window_size
        cls.RHO_THRESHOLD = rho_thresh
        cls.GAUSSIAN_SIGMA = gaussian_sigma
