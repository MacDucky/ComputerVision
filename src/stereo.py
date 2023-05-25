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

        self.depth_left = self.depth_right = np.loadtxt(
            r'C:\Users\User\PycharmProjects\pythonProject\ComputerVision\assignment_2\example\depth_left.txt',
            delimiter=',')

    @staticmethod
    def create_census(image: ndarray, window_size: tuple[int, int], rho: int, i: int, j: int) -> ndarray:
        pad_h, pad_w = int((window_size[0] - 1) / 2), int((window_size[1] - 1) / 2)
        og_image = image[pad_h:-pad_h, pad_w:-pad_w]

        i_range = np.arange(i, i + window_size[0])
        j_range = np.arange(j, j + window_size[1])

        xx, yy = np.meshgrid(i_range, j_range)

        zero_one = np.full(window_size, 1)
        one_zero = np.full(window_size, 2)
        zero_zero = np.full(window_size, 0)
        neg_neg = np.full(window_size, 3)

        census_transform_values = np.where(image[xx, yy] == -1, neg_neg,
                                           np.where(image[xx, yy] > (og_image[i, j] + rho), one_zero,
                                                    np.where(image[xx, yy] < (og_image[i, j] - rho), zero_one,
                                                             zero_zero)))
        census_bits = np.unpackbits(census_transform_values.astype(np.uint8), axis=1)
        reshaped_array = census_bits.reshape(census_transform_values.shape[0], census_transform_values.shape[1] * 8)

        # Convert each row of reshaped_array into a separate NumPy array
        bit_array = np.hsplit(reshaped_array, census_transform_values.shape[1])
        bit_array = np.array(bit_array)[:, :, -2:]
        final_census = np.where(bit_array == np.full((window_size[0], window_size[1], 2), (1, 1)),
                                np.full((window_size[0], window_size[1], 2), (-1, -1)), bit_array)
        flattened = final_census.flatten()
        return np.hstack([flattened[:int(flattened.shape[0] / 2) - 1], flattened[int(flattened.shape[0] / 2) + 1:]])
        # bit_array = np.where(np.all(bit_array)
        # i_range = np.arange(padded_height, height + padded_height)
        # j_range = np.arange(padded_width, width + padded_width)

        # census_transform = -np.ones((window_size[0] * window_size[1], 2))
        #
        # i_min = max(i - int((window_size[0] - 1) / 2), 0)
        # i_max = min(i + int((window_size[0] - 1) / 2) + 1, height)
        # j_min = max(j - int((window_size[1] - 1) / 2), 0)
        # j_max = min(j + int((window_size[1] - 1) / 2) + 1, width)
        #
        # i_range = np.arange(i_min, i_max)
        # j_range = np.arange(j_min, j_max)
        #
        # i_indices = i_range[:, np.newaxis]
        # j_indices = j_range
        #
        # value_mat = np.array([image[i, j]] * (len(i_indices) * len(j_indices))).reshape(
        #     (len(i_indices), len(j_indices)))
        #
        # census_transform_indices = np.where(True, np.arange(census_transform.shape[0]), 0)
        # census_transform_values = np.where(image[i_indices, j_indices] > value_mat[:, :] + rho, [0, 1,-2],
        #                                    np.where(image[i_indices, j_indices] < value_mat[:, :] - rho, [1, 0,-2],
        #                                             [0, 0,-2]))
        #
        # census_transform[census_transform_indices] = census_transform_values
        #
        # return census_transform.flatten()

    # @staticmethod
    # def create_census(image: ndarray, window_size: tuple[int, int], rho: int, i: int, j: int) -> ndarray:
    #     height, width = image.shape
    #     census_transform = -np.ones((window_size[0] * window_size[1], 2))
    #
    #     i_range = np.arange(i - int((window_size[0] - 1) / 2), i + int((window_size[0] - 1) / 2) + 1)
    #     j_range = np.arange(j - int((window_size[1] - 1) / 2), j + int((window_size[0] - 1) / 2) + 1)
    #
    #     valid_i = (i_range >= 0) & (i_range < height)
    #     valid_j = (j_range >= 0) & (j_range < width)
    #
    #     # i_indices = np.where(valid_i, i_range, 0)
    #     # j_indices = np.where(valid_j, j_range, 0)
    #
    #     i_indices = np.where(valid_i)[0]
    #     j_indices = np.where(valid_j)[0]
    #
    #     census_transform_indices = np.where(valid_i & valid_j,
    #                                         np.arange(census_transform.shape[0]).reshape((window_size[0], -1)), 0)
    #     census_transform_values = np.where(image[i_indices, j_indices] > image[i, j] + rho, [0, 1],
    #                                        np.where(image[i_indices, j_indices] < image[i, j] - rho, [1, 0], [0, 0]))
    #
    #     census_transform[census_transform_indices] = census_transform_values
    #
    #     return census_transform.flatten()

    # @staticmethod
    # def create_census(image: ndarray, window_size: tuple[int, int], rho: int, i: int, j: int) -> ndarray:
    #     height, width = image.shape
    #     census_transform = []
    #     # todo meshgrid for window of size window size
    #     for i_new in range(i - int((window_size[0] - 1) / 2), i + int((window_size[0] - 1) / 2) + 1):
    #         for j_new in range(j - int((window_size[1] - 1) / 2), j + int((window_size[0] - 1) / 2) + 1):
    #             if i_new < 0 or j_new < 0 or i_new >= height or j_new >= width:
    #                 census_transform.extend([-1, -1])
    #             elif i == i_new and j == j_new:
    #                 pass
    #             elif image[i_new, j_new] > image[i, j] + rho:
    #                 census_transform.extend([0, 1])
    #             elif image[i_new, j_new] < image[i, j] - rho:
    #                 census_transform.extend([1, 0])
    #             else:
    #                 census_transform.extend([0, 0])
    #     return np.array(census_transform)

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
        print('Running...')
        height, width = image.shape
        census_map = np.empty((height, width, 2 * window_size[0] * window_size[1] - 2))
        padded_height, padded_width = int((window_size[0] - 1) / 2), int((window_size[1] - 1) / 2)
        new_image = np.pad(image.astype(int), (padded_height, padded_width), constant_values=-1)
        for i in range(height):
            for j in range(width):
                start_time = perf_counter()
                census_map[i, j] = Stereo.create_census(new_image, window_size, rho, i, j)
                total_time = perf_counter() - start_time
                print(f'[{i},{j}] time: {total_time}')
        print('Done')
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
        Create disparity maps to both left & right images
        """
        """creating census maps"""
        from multiprocessing import Pool
        start_time = perf_counter()
        with Pool(processes=2) as pool:
            left_res = pool.apply_async(self.create_census_map, (self.image_left.grayscale_img, window_size, rho))
            right_res = pool.apply_async(self.create_census_map, (self.image_right.grayscale_img, window_size, rho))
            census_map_left = left_res.get()
            census_map_right = right_res.get()
        total_time = perf_counter() - start_time
        print(f'Total time taken during create_census_map: {total_time}')
        # census_map_left = self.create_census_map(self.image_left.grayscale_img, window_size, rho=rho)
        # total_time = perf_counter() - start_time
        # print(f'Time taken in create_census_map: {total_time}')
        # start_time = perf_counter()
        # census_map_right = self.create_census_map(self.image_right.grayscale_img, window_size, rho=rho)
        # total_time = perf_counter() - start_time
        # print(f'Time taken in create_census_map: {total_time}')

        """the cost aggregations for every pixel in the left image"""
        left_disparity = self.winner_takes_all(census_map_left, census_map_right, True)
        right_disparity = self.winner_takes_all(census_map_right, census_map_left, False)
        # right_costs_image = self.cost_aggregation(census_map_right, census_map_left, False)

        """Winner takes all method to creat two disparities maps"""
        # self.disparity_left = self.winner_takes_all(left_costs_image)
        # self.disparity_right = self.winner_takes_all(right_costs_image)

        """Threshold the disparity maps"""
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
    left_image = ImageLoader(
        r'C:\Users\User\PycharmProjects\pythonProject\ComputerVision\assignment_2\example\im_left.jpg')
    right_image = ImageLoader(
        r'C:\Users\User\PycharmProjects\pythonProject\ComputerVision\assignment_2\example\im_right.jpg')
    stereo = Stereo(left_image, right_image, left_camera, right_camera, 255)
    stereo.create_disparities(window_size=tuple((5, 5)), rho=2)
