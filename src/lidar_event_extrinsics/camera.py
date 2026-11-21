"""Pinhole camera with radial-tangential lens distortion.

A point in the camera frame (z forward, x right, y down) is divided by its depth to
get normalized coordinates (x, y). The lens bends (x, y) into (x_d, y_d), and the
intrinsics scale and shift that into pixels (u, v). The distortion model is the one
ROS calls plumb_bob and OpenCV uses by default, with coefficients k1, k2, p1, p2, k3.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray  # k1, k2, p1, p2, k3

    def distort(self, x, y):
        k1, k2, p1, p2, k3 = self.distortion
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return x_distorted, y_distorted

    def project(self, points_camera, min_depth=0.3, max_normalized_radius=1.5):
        """Project (N, 3) camera-frame points to pixels.

        Returns uv (N, 2), depth (N,), and a boolean mask of points that are in front
        of the camera and land inside the image. Points far off-axis are dropped too,
        because the distortion polynomial folds back on itself there and could place
        them inside the image by mistake.
        """
        points_camera = np.asarray(points_camera, dtype=np.float64)
        depth = points_camera[:, 2]

        # Divide only where depth is positive, so points behind the camera give no warnings.
        in_front = depth > min_depth
        safe_depth = np.where(in_front, depth, 1.0)
        x = points_camera[:, 0] / safe_depth
        y = points_camera[:, 1] / safe_depth

        within_lens = np.sqrt(x * x + y * y) < max_normalized_radius
        x_distorted, y_distorted = self.distort(x, y)

        u = self.fx * x_distorted + self.cx
        v = self.fy * y_distorted + self.cy
        uv = np.stack([u, v], axis=1)

        # Pixel (i, j) covers [i - 0.5, i + 0.5), so a point counts if it rounds to a valid pixel.
        column = np.round(u)
        row = np.round(v)
        in_image = (column >= 0) & (column < self.width) & (row >= 0) & (row < self.height)

        visible = in_front & within_lens & in_image
        return uv, depth, visible
