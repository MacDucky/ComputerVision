import numpy as np
from loader import ImageLoader
from camera import Camera
class Stereo:
    def __init__(self, image_left : ImageLoader, image_right : ImageLoader, camera_left : Camera, camera_right: Camera):
        self.image_left = image_left
        self.image_right = image_right
        self.camera_left = camera_left
        self.camera_right = camera_right
        # Disparities Maps
        self.disparity_left = None
        self.disparity_right = None
        # Depth Maps
        self.depth_left = None
        self.depth_right = None
        # Back Projection Matrix from camera 1 (all the pixels represented in camera axis 1)
        self.back_projection = None

    def creat_disps(self)

