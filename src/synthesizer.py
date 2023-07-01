import os.path

import cv2
import numpy as np
from numpy import ndarray
from src.stereo import Stereo
from src.camera import Camera
from src.plotter import show_image, compare_images
from src.loader import ImageLoader


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
        back_projected = np.vstack([back_projected, np.ones(back_projected[0].size)])
        reprojected_images = []
        for camera in self.cameras:
            # reproject points to virtual camera
            reprojected = camera.full_transform @ back_projected
            reprojected = np.divide(reprojected, reprojected[2], where=reprojected[2] != 0)
            reprojected = reprojected[:2]

            image = self.__stereo.image_left.color_img
            reproj_pts = reprojected.T.reshape((*image.shape[:2], -1))
            reprojected_image = self.__interpolate(image, reproj_pts)
            reprojected_images.append(reprojected_image)
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
    def __interpolate(image, coords):
        rounded_coords = np.round(coords).astype(int)

        # Height and width constraints
        height_constraint = (0, image.shape[0] - 1)
        width_constraint = (0, image.shape[1] - 1)

        # Extract the first and second coordinates into separate arrays
        coords_x = rounded_coords[:, :, 0]  # First coordinate
        coords_y = rounded_coords[:, :, 1]  # Second coordinate

        # Create masks for height and width constraints
        width_mask = (coords_x >= width_constraint[0]) & (coords_x <= width_constraint[1])
        height_mask = (coords_y >= height_constraint[0]) & (coords_y <= height_constraint[1])

        # Combine the masks
        in_boundary = height_mask & width_mask
        in_boundary = in_boundary.repeat(2).reshape(*in_boundary.shape, -1)

        # filter out bad indices
        reprojected_idxs = np.where(in_boundary, rounded_coords, np.zeros_like(rounded_coords))

        og_xx, og_yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        interpolated_image = np.zeros_like(image)
        interpolated_image[reprojected_idxs[:, :, 1], reprojected_idxs[:, :, 0]] = image[og_yy, og_xx, :]

        return interpolated_image


if __name__ == '__main__':
    for i in range(5):
        base_path = os.path.abspath(f'../assignment_2/set_{i+1}/')
        im_l = ImageLoader(os.path.join(base_path, 'im_left.jpg'))
        im_r = ImageLoader(os.path.join(base_path, 'im_right.jpg'))
        cam_intrinsics = os.path.join(base_path, 'K.txt')
        max_disp_path = os.path.join(base_path, 'max_disp.txt')
        with open(max_disp_path, 'r') as fp:
            max_disp = int(fp.readline())
        left_cam = Camera(o=np.array([0, 0, 0]), phi=np.array([0, 0, 0]))
        left_cam.load_intrinsics_from_file(cam_intrinsics)
        right_cam = left_cam.duplicate_camera_at_position(position=np.array([0.1, 0, 0]))
        s = Synthesizer(Stereo(im_l, im_r, left_cam, right_cam, max_disp), 0, 0.1, 11)
        for c in s.cameras:
            print(c, end='\n\n')

        disp_l = s._Synthesizer__stereo.disparity_left = np.loadtxt(os.path.join(base_path, 'disp_left.txt'), delimiter=',')
        disp_r = s._Synthesizer__stereo.disparity_right = np.loadtxt(os.path.join(base_path, 'disp_right.txt'),
                                                                     delimiter=',')

        synthesize_images = s.synthesize()

        disp_l = s._Synthesizer__stereo.disparity_left
        disp_l_txt_path = os.path.join(base_path, 'disp_left.txt')
        disp_r = s._Synthesizer__stereo.disparity_right
        disp_r_txt_path = os.path.join(base_path, 'disp_right.txt')
        np.savetxt(disp_l_txt_path, disp_l, delimiter=',')
        np.savetxt(disp_r_txt_path, disp_r, delimiter=',')

        corrected_disp_l = show_image(disp_l, normalize=True, equalize=False)
        disp_l_img_path = disp_l_txt_path.replace('.txt', '.jpg')
        corrected_disp_r = show_image(disp_r, normalize=True, equalize=False)
        disp_r_img_path = disp_r_txt_path.replace('.txt', '.jpg')
        cv2.imwrite(disp_l_img_path, corrected_disp_l)
        cv2.imwrite(disp_r_img_path, corrected_disp_r)

        depth_l = s._Synthesizer__stereo.depth_left
        depth_l_txt_path = os.path.join(base_path, 'depth_left.txt')
        depth_r = s._Synthesizer__stereo.depth_right
        depth_r_txt_path = os.path.join(base_path, 'depth_right.txt')
        np.savetxt(depth_l_txt_path, depth_l, delimiter=',')
        np.savetxt(depth_r_txt_path, depth_r, delimiter=',')

        corrected_depth_l = show_image(depth_l, normalize=True, equalize=True)
        depth_l_img_path = depth_l_txt_path.replace('.txt', '.jpg')
        corrected_depth_r = show_image(depth_r, normalize=True, equalize=True)
        depth_r_img_path = depth_r_txt_path.replace('.txt', '.jpg')
        cv2.imwrite(depth_l_img_path, corrected_depth_l)
        cv2.imwrite(depth_r_img_path, corrected_depth_r)

        for i, im in enumerate(synthesize_images, start=1):
            s = str(i).zfill(2)
            synth_path = os.path.join(base_path, f'synth_{s}.jpg')
            image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cv2.imwrite(synth_path, image_rgb)
