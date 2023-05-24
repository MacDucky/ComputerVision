import cv2
import numpy as np
from numpy import ndarray
from src.stereo import Stereo
from src.camera import Camera
from src.plotter import show_image
from src.loader import ImageLoader


# class NdArrayWithColor(ndarray):
#
#     def __init__(self, shape, color: ndarray, dtype=None, buffer=None, offset=0, strides=None, order=None):
#         super().__init__(shape, dtype, buffer, offset, strides, order)
#         self.color = color


class Synthesizer:
    def __init__(self, stereo: Stereo, start_x_pos: float, end_x_pos: float, num_synth_pictures: int):
        """
        Synthesizes images from translated camera along x-axis.

        Note: camera intrinsics are trivial in this implementation.

        Parameters:
            stereo(Stereo): Stereo object containing original 2-camera setup
            start_x_pos(float): Start position of the synthesized images along x-axis
            end_x_pos(float): End position of the synthesized images along x-axis
            num_synth_pictures(int): Number of synthesized images
        """
        self.__stereo = stereo
        self.__start_pos = start_x_pos
        self.__end_pos = end_x_pos
        self.__num_images = num_synth_pictures
        self.cameras: None | list[Camera] = None
        self.__create_cameras_at_xpositions()

    def __create_cameras_at_xpositions(self):
        step = (self.__end_pos - self.__start_pos) / (self.__num_images - 1)  # Calculate the step size
        x_positions = (self.__start_pos + i * step for i in range(self.__num_images))  # Generate the iterable
        if not self.cameras:
            self.cameras = [self.__stereo.camera_left.duplicate_camera_at_position(position=np.array([x, 0, 0])) for x
                            in x_positions]

    def synthesize(self) -> list[ndarray]:
        back_projected = self.__stereo.back_projection
        # back_projected = back_projected.T[back_projected[2] != 0].T
        back_projected = np.vstack([back_projected, np.ones(back_projected[0].size)])
        reprojected_images = []
        for camera in self.cameras:
            reprojected = camera.full_transform @ back_projected
            reprojected = reprojected.T[reprojected.T[:, 2] != 0].T
            reprojected = reprojected / reprojected[2]
            reprojected = reprojected[:2]

            if camera == self.__stereo.camera_left:
                orig_coords = reprojected
            else:
                orig_coords = self.__calculate_original_points(camera, self.__stereo.depth_left, reprojected)

            # orig_nan_mask = np.isnan(orig_coords.T).any(axis=1)
            # reprojected_nan_mask = np.isnan(reprojected.T).any(axis=1)
            # orig_coords = orig_coords.T[~orig_nan_mask]
            # reprojected = reprojected.T[~reprojected_nan_mask]
            image = self.__stereo.image_left.color_img
            reconstructed_image = self.__interpolate(image, orig_coords, reprojected)
            reprojected_images.append(reconstructed_image)
        return reprojected_images

    def __calculate_original_points(self, camera: Camera, depth_map: ndarray, reprojected_coords: ndarray):
        # todo: make this an inverse (mapping) reprojection
        # (ft)/z + x_r
        baseline_len = np.linalg.norm(self.__stereo.camera_left.origin - camera.origin)
        focal_len = self.__stereo.camera_left.focal_len

        # _, i_intrinsic = cv2.invert(camera.intrinsic_transform, flags=cv2.DECOMP_SVD)

        original_pixel_coords = ((baseline_len * focal_len) / depth_map[depth_map != 0].flatten()) + reprojected_coords
        return original_pixel_coords

    @staticmethod
    def __interpolate(image, orig_coords, dest_coords):
        # todo: convert from nearest neighbour to bilinear interpolation
        dest_coords = np.round(dest_coords)
        im = np.zeros_like(image)
        os = orig_coords.astype(int).T
        ds = dest_coords.astype(int).T
        # im[ds[:, 1], ds[:, 0]] = image[os[:, 1], os[:, 0]]

        # Get image dimensions
        image_height, image_width = image.shape[:2]

        # Filter out-of-bounds coordinates
        valid_indices = (os[:, 1] < image_height) & (os[:, 0] < image_width)

        # Apply valid indices to os and ds arrays
        valid_os = os[valid_indices]
        valid_ds = ds[valid_indices]

        # Assign values to im array
        im[valid_ds[:, 1], valid_ds[:, 0]] = image[valid_os[:, 1], valid_os[:, 0]]
        return im
        # return interpolated_image


if __name__ == '__main__':
    im_l = ImageLoader(r'..\assignment_2\example\im_left.jpg')
    left_cam = Camera(o=np.array([0, 0, 0]), phi=np.array([0, 0, 0]))
    left_cam.load_intrinsics_from_file(r'C:\Users\Dany\PycharmProjects\ComputerVision\assignment_2\example\K.txt')
    right_cam = left_cam.duplicate_camera_at_position(position=np.array([0.1, 0, 0]))
    s = Synthesizer(Stereo(im_l, None, left_cam, right_cam), 0, 0.1, 11)
    for c in s.cameras:
        print(c, end='\n\n')
    for im in s.synthesize():
        show_image(im)
