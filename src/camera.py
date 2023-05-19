import numpy as np


class Camera:
    """
    A pinhole inspired camera model.

    See for details:
    https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec
    https://ksimek.github.io/2012/08/22/extrinsic/

    The model is the following parameters:
        -   Intrisic transformation (K) consists of:

            -   *f* **-** focal length, distance from focal point (camera origin) and image plane.
            -   *roh_x,roh_y* **-** meters/pixel, these parameters define how 'wide' the pixel is.
            -   *theta* **-** skew angle between x and y axes.
            -   *p_x,p_y* **-** translation correction between image and pixel planes.

        -   Extrinsic transformation ([R|-RC]) consists of:

            -   *c_x,c_y,c_z* **-** camera origin coordinates with respect to ***world*** axis.
            -   *phi_x,phi_y,phi_z* **-** angles with respect to camera origin.
    Final transformation from world coordinates to pixel coordinates:

    """

    def __init__(self):
        pass

    def intrinsic_transform(self):
        pass

    def extrinsic_transform(self):
        pass
