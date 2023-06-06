import cv2
import numpy as np
from loader import ImageLoader
from camera import Camera
from numpy import ndarray


class Stereo:
    def __init__(self, image_left: ImageLoader, image_right: ImageLoader, camera_left: Camera, camera_right: Camera):
        self.image_left = image_left
        self.image_right = image_right

        # Camera maps
        self.camera_left = camera_left
        self.camera_right = camera_right
        # Disparities Maps
        self.disparity_left: None | ndarray = None
        self.disparity_right: None | ndarray = None
        # Depth Maps
        self.depth_left: None | ndarray = None
        self.depth_right: None | ndarray = None
        # Back Projection Matrix from camera 1 (all the pixels represented in camera axis 1)
        self.__back_projection: None | ndarray = None

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
            self.calc_disparity_maps()
        if self.depth_left is None or self.depth_right is None:
            self.calc_depth_maps()
        if self.__back_projection is None:  # generalize to right camera as well
            self.calc_back_projection()
        return self.__back_projection.copy()
