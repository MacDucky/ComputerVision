import numpy as np
from loader import ImageLoader
from camera import Camera
class Stereo:
    def __init__(self, iml : ImageLoader, imr : ImageLoader, caml : Camera, camr: Camera):
        self.iml = iml
        self.imr = imr
        self.caml = caml
        self.camr = camr
        # Disparities Maps
        self.displ = None
        self.dispr = None
        # Depth Maps
        self.depl = None
        self.depr = None
        # Reprojection Matrix
        self.reprojection = None


