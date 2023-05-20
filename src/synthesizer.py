import cv2
import numpy as np
from numpy import ndarray
from src.stereo import Stereo
from src.camera import Camera
from src.plotter import show_image
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
            self.cameras = [Camera.basic_camera_at_position(position=np.array([x, 0, 0])) for x in x_positions]

    def synthesize(self) -> list[ndarray]:
        back_projected = self.__stereo.back_projection
        # back_projected = back_projected.T[back_projected[2] != 0].T
        back_projected = np.vstack([back_projected, np.ones(back_projected[0].size)])
        reprojected_images = []
        for camera in self.cameras:
            reprojected = camera.full_transform @ back_projected
            reprojected += 1 ** -10
            reprojected = reprojected / reprojected[2]
            reprojected = reprojected[:2]

            if camera == self.__stereo.camera_left:
                orig_coords = reprojected
            else:
                orig_coords = self.__calculate_original_points(camera, self.__stereo.depth_left, reprojected)

            orig_nan_mask = np.isnan(orig_coords.T).any(axis=1)
            reprojected_nan_mask = np.isnan(reprojected.T).any(axis=1)
            orig_coords = orig_coords.T[~orig_nan_mask]
            reprojected = reprojected.T[~reprojected_nan_mask]
            image = self.__stereo.image_left.color_img
            # reconstructed_image = np.zeros_like(image)
            hw, coord_size = self.__stereo.image_left.grayscale_img.shape, -1
            shape = hw[0], hw[1], coord_size
            reprojected = reprojected.reshape(shape)
            orig_coords = orig_coords.reshape(shape)
            reconstructed_image = self.__bilinear_interpolation(image, orig_coords)
            show_image(reconstructed_image)

        i = 0
        # return reprojected_images

    def __calculate_original_points(self, camera: Camera, depth_map: ndarray, reprojected_coords: ndarray):
        # z/(ft) + x_r
        baseline_len = np.linalg.norm(self.__stereo.camera_left.origin - camera.origin)
        focal_len = self.__stereo.camera_left.focal_len
        original_pixel_coords = (depth_map.flatten() / (baseline_len * focal_len)) + reprojected_coords
        return original_pixel_coords

    @staticmethod
    def __bilinear_interpolation(imgs, pix_coords):
        """
        Construct a new image by bilinear sampling from the input image.
        Args:
            imgs:                   [H, W, C]
            pix_coords:             [h, w, 2]
        :return:
            sampled image           [h, w, c]
        """
        img_h, img_w, img_c = imgs.shape
        pix_h, pix_w, pix_c = pix_coords.shape
        out_shape = (pix_h, pix_w, img_c)

        pix_x, pix_y = np.split(pix_coords, [1], axis=-1)  # [pix_h, pix_w, 1]
        pix_x = pix_x.astype(np.float32)
        pix_y = pix_y.astype(np.float32)

        # Rounding
        pix_x0 = np.floor(pix_x)
        pix_x1 = pix_x0 + 1
        pix_y0 = np.floor(pix_y)
        pix_y1 = pix_y0 + 1

        # Clip within image boundary
        y_max = (img_h - 1)
        x_max = (img_w - 1)
        zero = np.zeros([1])

        pix_x0 = np.clip(pix_x0, zero, x_max)
        pix_y0 = np.clip(pix_y0, zero, y_max)
        pix_x1 = np.clip(pix_x1, zero, x_max)
        pix_y1 = np.clip(pix_y1, zero, y_max)

        # Weights [pix_h, pix_w, 1]
        wt_x0 = pix_x1 - pix_x
        wt_x1 = pix_x - pix_x0
        wt_y0 = pix_y1 - pix_y
        wt_y1 = pix_y - pix_y0

        # indices in the image to sample from
        dim = img_w

        # Apply the lower and upper bound pix coord
        base_y0 = pix_y0 * dim
        base_y1 = pix_y1 * dim

        # 4 corner vertices
        idx00 = (pix_x0 + base_y0).flatten().astype(int)
        idx01 = (pix_x0 + base_y1).astype(int)
        idx10 = (pix_x1 + base_y0).astype(int)
        idx11 = (pix_x1 + base_y1).astype(int)

        # Gather pixels from image using vertices
        imgs_flat = imgs.reshape([-1, img_c]).astype(np.float32)
        im00 = imgs_flat[idx00].reshape(out_shape)
        im01 = imgs_flat[idx01].reshape(out_shape)
        im10 = imgs_flat[idx10].reshape(out_shape)
        im11 = imgs_flat[idx11].reshape(out_shape)

        # Apply weights [pix_h, pix_w, 1]
        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1
        output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
        return output


if __name__ == '__main__':
    im_l = ImageLoader(r'C:\Users\Dany\PycharmProjects\ComputerVision\assignment_2\example\im_left.jpg')
    s = Synthesizer(Stereo(im_l, None, Camera.basic_camera_at_position(position=np.array([0, 0, 0])),
                           Camera.basic_camera_at_position(position=(np.array([0.1, 0, 0])))), 0, 0.1, 11)
    for c in s.cameras:
        print(c, end='\n\n')
    for im in s.synthesize():
        show_image(im)
