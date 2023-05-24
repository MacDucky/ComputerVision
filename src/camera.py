import numpy as np
from numpy import ndarray
from math import cos, sin, tan, pi
from copy import deepcopy


class Camera:
    """
    A pinhole inspired camera model.

    See for details:
    https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec
    https://ksimek.github.io/2012/08/22/extrinsic/

    The model is constructed from the following parameters:
        -   Intrisic transformation (K) consists of:

            -   *f* **-** focal length, distance from focal point (camera origin) and image plane.
            -   *roh_u,roh_v =: roh* **-** meters/pixel, these parameters define how 'wide' the pixel is.
            -   *skew_theta* **-** skew angle between x and y axes. No skew = pi/2.
            -   *t_x,t_y =: t* **-** translation between image and pixel planes.

        -   Extrinsic transformation ([R|-RC]) consists of:

            -   *o_x,o_y,o_z =: o* **-** camera origin coordinates with respect to ***world*** axis.
            -   *phi_x,phi_y,phi_z =: phi* **-** angles with respect to camera origin.
    Final transformation from world coordinates to pixel coordinates:
    *P[x,y,w] = K \* [I|0] \* [R|-Rc] \* P_w[x,y,z,w]*
    """

    def __init__(self, o: ndarray, phi: ndarray, focal_length: float = None, roh: ndarray = None,
                 skew_theta: float = None, t: ndarray = None):
        self.focal_len = focal_length
        self.roh = roh
        self.skew_theta = skew_theta
        self.t = t
        self.o = o
        self.phi = phi
        self._intrinsic_t: None | ndarray = None
        self._extrinsic_t: None | ndarray = None
        self._full_t: None | ndarray = None

    def load_intrinsics_from_file(self, path: str):
        self._intrinsic_t = np.loadtxt(path)
        self.focal_len = self._intrinsic_t[0, 0]
        assert self._intrinsic_t[0, 0] == self._intrinsic_t[1, 1], 'Camera intrinsics have non uniform pixels!'

    @property
    def intrinsic_transform(self) -> ndarray:
        if self._intrinsic_t is None:
            self._intrinsic_t = self.__calc_intrinsic()
        return self._intrinsic_t.copy()

    @property
    def extrinsic_transform(self) -> ndarray:
        if self._extrinsic_t is None:
            self._extrinsic_t = self.__calc_extrinsic()
        return self._extrinsic_t.copy()

    @property
    def full_transform(self) -> ndarray:
        if self._full_t is None:
            self._full_t = self.__calc_world_to_pixel()
        return self._full_t.copy()

    @property
    def origin(self):
        return self.o.copy()

    def duplicate_camera_at_position(self, position: ndarray, phi: ndarray = None):
        """
        Returns a camera with same simple (normalized) intrinsic calibration at specified extrinsic calibration.

        Parameters:
            position(ndarray): a position of the camera origin in world coordinates.
            phi(ndarray): (Optional) rotation angles (phi_x, phi_y,phi_z) of the camera
        Returns:
            A camera at the specified position
        """
        cam_copy = deepcopy(self)
        cam_copy.o = position
        cam_copy.phi = np.array([0, 0, 0]) if phi is None else phi
        cam_copy._extrinsic_t = None  # In order to recompute extrinsic.
        return cam_copy

    @staticmethod
    def __rotation(phi: ndarray):
        """ Camera rotation matrix (R) (3x3) """
        phi_x, phi_y, phi_z = phi
        rot_x = np.array([[1, 0, 0], [0, cos(phi_x), -sin(phi_x)], [0, sin(phi_x), cos(phi_x)]])
        rot_y = np.array([[cos(phi_y), 0, sin(phi_y)], [0, 1, 0], [-sin(phi_y), 0, cos(phi_y)]])
        rot_z = np.array([[cos(phi_z), -sin(phi_z), 0], [sin(phi_z), cos(phi_z), 0], [0, 0, 1]])
        return rot_x @ rot_y @ rot_z

    @staticmethod
    def __basic_projection():
        """ returns [I|0] (3x4) """
        return np.hstack([np.identity(3), np.zeros((3, 1))])

    def __calc_intrinsic(self):
        """ Returns K (3x3) """
        alpha = self.focal_len / self.roh[0]
        beta = self.focal_len / self.roh[1]
        intrinsics = np.array([[alpha, alpha / tan(self.skew_theta), self.t[0]], [0, beta, self.t[1]], [0, 0, 1]])
        return intrinsics

    def __calc_extrinsic(self):
        """ Returns [R|-Rc] (4x4)"""
        rot_mat = self.__rotation(self.phi)
        rot_with_zeros = np.vstack((rot_mat, np.zeros(3)))
        translation_vec = np.hstack((-rot_mat @ self.o, 1)) if self.o.size == 3 else self.o
        return np.hstack((rot_with_zeros, translation_vec.reshape(4, 1)))

    def __calc_world_to_pixel(self):
        return self.intrinsic_transform @ self.__basic_projection() @ self.extrinsic_transform

    def __str__(self):
        s = '-----------intrinsics-----------\n'
        s += str(self.intrinsic_transform)
        s += '\n-----------extrinsics-----------\n'
        s += str(self.extrinsic_transform)
        s += '\n-----------full transform-----------\n'
        s += str(self.full_transform)
        return s

    def __eq__(self, other):
        return \
                self.focal_len == other.focal_len and \
                np.array_equal(self.roh, other.roh) and \
                self.skew_theta == other.skew_theta and \
                np.array_equal(self.t, other.t) and \
                np.array_equal(self.o, other.o) and \
                np.array_equal(self.phi, other.phi)
