import os.path

import cv2
import numpy as np
from loader import ImageLoader
from camera import Camera
from numpy import ndarray
from timeit import default_timer as perf_counter


class Stereo:
    def __init__(self, image_left: ImageLoader, image_right: ImageLoader, camera_left: Camera, camera_right: Camera,
                 max_disparity: int):
        self.image_left = image_left
        self.image_right = image_right
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
        neg_neg = np.full(sliding_window.shape, 3)

        # apply census mapping on sliding windows
        census_values = np.where(sliding_window == -1, neg_neg,
                                 np.where(sliding_window > (center_value + rho), one_zero,
                                          np.where(sliding_window < (center_value - rho), zero_one,
                                                   zero_zero)))

        # unpack decimals to bits
        census_bits = np.unpackbits(census_values.astype(np.uint8), axis=-1)
        census_bits = census_bits.flatten().reshape((*sliding_window.shape, 8))
        # reduce to 2 bit representation
        census_bits = np.array(census_bits)[:, :, :, :, -2:]

        # map [1,1] to [-1,-1]
        final_census = np.where((census_bits == [1, 1]).all(axis=-1, keepdims=True), [-1, -1], census_bits)

        # remove center value at each window
        remove_center_mask = np.ones_like(final_census, dtype=bool)
        remove_center_mask[:, :, pad_h, pad_w, :] = False
        final_census = final_census[remove_center_mask]

        return final_census.reshape(
            (*sliding_window.shape[:2], 2 * (sliding_window.shape[2] * sliding_window.shape[3] - 1)))

    @staticmethod
    def create_census_map(image: ndarray, window_size: tuple, rho: int):
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
        padded_image = np.pad(image.astype(int), (padded_height, padded_width), constant_values=-1)
        sliding_window = np.lib.stride_tricks.sliding_window_view(padded_image, window_size)
        census_map = Stereo.__create_census(sliding_window, rho)
        return census_map

    @staticmethod
    def compare_census(census1: list, census2: list):
        res = sum((np.clip(c1 + c2, a_min=0, a_max=2) % 2 for c1, c2 in zip(census1, census2)))
        return res

    @staticmethod
    def winner_takes_all(census_image_src: ndarray, census_image_candidates: ndarray, is_left_im: bool,
                         arm_length: int = 30):
        """
        Returns the cost image of the source image given the census source image and the other census image.

        Parameters:
            census_image_src: the census source image.
            census_image_candidates: the other census image.
            is_left_im(bool): True if the source image is the left image.
            arm_length(int): the size of the cost array for each pixel.
        Returns:
            return the cost image
        """
        height, width, _ = census_image_src.shape
        disparity = np.zeros((height, width))
        delta = -1
        if not is_left_im:
            delta = 1
        for i in range(height):
            for j in range(width):
                cost = (
                    Stereo.compare_census(census_image_src[i, j], census_image_candidates[i, j + delta * (k + 1)]) for k
                    in range(arm_length))
                cost = np.array(list(cost))
                # Get the indices that would sort the array
                sorted_indices = np.argsort(cost)
                # Get the indices of the least 4 values
                least_indices = sorted_indices[:4]

                # Sort the least indices to maintain stability
                min_indices = np.sort(least_indices)

                values = cost[min_indices]
                if values.min() != values.max():
                    values = values[:-1]
                    if values.min() == values.max():
                        disparity[i, j] = min_indices[1]
                    else:
                        disparity[i, j] = min_indices[0]
        return disparity

        # min_cost = min(cost)
        # min_indices = [index for index, value in enumerate(cost) if value == min_cost]
        # len_cost_min = len(min_indices)
        # if len_cost_min > 0:
        #     if len_cost_min == 1:
        #         disparity[i, j] = min_indices[0] + 1
        #     elif len_cost_min == 2 and min_indices[1] - min_indices[0] == 1:
        #         disparity[i, j] = min_indices[0] + 1
        #     elif len_cost_min == 3 and min_indices[2] - min_indices[0] == 2:
        #         disparity[i, j] = min_indices[1] + 1
        # print(i*height+j)

    # @staticmethod
    # def winner_takes_all(cost_image: ndarray):
    #     """
    #     Returns a disparity map given the cost image.
    #
    #     Parameters:
    #         cost_image(ndarray): image of all the candidates cost of a specific pixel.
    #     Returns:
    #          A disparity map (not final).
    #     """
    #     height, width = cost_image.shape
    #     disparity = np.zeros(shape=(height, width))
    #     for i in range(height):
    #         for j in range(width):
    #             min_cost = min(cost_image[i, j])
    #             min_indices = [index for index, value in enumerate(cost_image[i, j]) if value == min_cost]
    #             len_cost_min = len(min_indices)
    #             if len_cost_min > 0:
    #                 if len_cost_min == 1:
    #                     disparity[i, j] = min_indices[0] + 1
    #                 elif len_cost_min == 2 and min_indices[1] - min_indices[0] == 1:
    #                     disparity[i, j] = min_indices[0] + 1
    #                 elif len_cost_min == 3 and min_indices[2] - min_indices[0] == 2:
    #                     disparity[i, j] = min_indices[1] + 1
    #     return disparity

    def create_disparities(self, window_size: tuple[int, int], rho: int):
        """
        Create disparity maps to both left & right images, then apply a winner takes it all + postprocessing.
        """
        # Census map creation
        from multiprocessing import Pool
        start_time = perf_counter()
        with Pool(processes=2) as pool:
            left_res = pool.apply_async(self.create_census_map, (self.image_left.grayscale_img, window_size, rho))
            right_res = pool.apply_async(self.create_census_map, (self.image_right.grayscale_img, window_size, rho))
            census_map_left = left_res.get()
            census_map_right = right_res.get()
        total_time = perf_counter() - start_time
        print(f'Total time taken during census map creation: {total_time}')

        # the cost aggregations for every pixel in the left image
        left_disparity = self.winner_takes_all(census_map_left, census_map_right, True)
        right_disparity = self.winner_takes_all(census_map_right, census_map_left, False)

        # Threshold the disparity maps
        # todo: make a threshold using max disparity

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
        _, i_intrinsics = cv2.invert(self.camera_left.intrinsic_transform, flags=cv2.DECOMP_SVD)
        height, width = self.depth_left.shape

        # Creating a pixel coordinate matrix
        x = np.linspace(0, width - 1, width).astype(int)
        y = np.linspace(0, height - 1, height).astype(int)
        [x, y] = np.meshgrid(x, y)
        pixel_coordinates = np.vstack(
            (x.flatten(), y.flatten(), np.ones_like(x.flatten()) * self.camera_left.focal_len))

        # Scale each point on the line from camera center and the pixel coordinate on the image plane(as camera coords),
        # to its original z-value
        self.__back_projection = i_intrinsics @ pixel_coordinates * self.depth_left.flatten()

    @property
    def back_projection(self):
        if self.depth_left is None:
            self.calc_depth_maps()
        if self.__back_projection is None:
            self.calc_back_projection()
        return self.__back_projection.copy()


if __name__ == '__main__':
    left_camera = Camera.basic_camera_at_position(position=ndarray([0, 0, 0]))
    right_camera = Camera.basic_camera_at_position(position=ndarray([1, 0, 0]))
    left_im_path = os.path.abspath(r'\assignment_2\example\im_left.jpg')
    right_im_path = os.path.abspath(r'\assignment_2\example\im_right.jpg')
    left_image = ImageLoader(left_im_path)
    right_image = ImageLoader(right_im_path)
    stereo = Stereo(left_image, right_image, left_camera, right_camera, 255)
    stereo.create_disparities(window_size=tuple((5, 5)), rho=2)
