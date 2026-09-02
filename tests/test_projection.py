import math
import unittest

import numpy as np

from adas_gaze_realtime import GazeAwareADAS


def make_geometry_only_adas():
    adas = GazeAwareADAS.__new__(GazeAwareADAS)
    adas.frame_width = 640
    adas.frame_height = 360
    adas.K_lr = np.array([[465.0, 0.0, 320.0], [0.0, 465.0, 180.0], [0.0, 0.0, 1.0]])
    adas.R_lr_from_pro = np.eye(3)
    adas.T_lr_from_pro_m = np.zeros(3)
    adas.R_pro_from_model = np.eye(3)
    adas.last_projection_status = "not-evaluated"
    return adas


class ProjectionTests(unittest.TestCase):
    def test_identity_ray_projects_to_principal_point(self):
        adas = make_geometry_only_adas()
        point, direction = adas.project_gaze_to_lr((0.0, 0.0), origin_pro=np.zeros(3))
        self.assertEqual(point, (320, 180))
        np.testing.assert_allclose(direction, [0, 0, 1])
        self.assertEqual(adas.last_projection_status, "ok")

    def test_missing_model_rotation_is_invalid_not_center(self):
        adas = make_geometry_only_adas()
        adas.R_pro_from_model = None
        point, _ = adas.project_gaze_to_lr((0.0, 0.0), origin_pro=np.zeros(3))
        self.assertIsNone(point)
        self.assertEqual(adas.last_projection_status, "missing-model-to-pro-calibration")

    def test_ray_behind_camera_is_invalid(self):
        adas = make_geometry_only_adas()
        adas.R_pro_from_model = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
        )
        point, _ = adas.project_gaze_to_lr((0.0, 0.0), origin_pro=np.zeros(3))
        self.assertIsNone(point)
        self.assertEqual(adas.last_projection_status, "ray-behind-lr-camera")

    def test_out_of_fov_ray_is_not_clipped_to_border(self):
        adas = make_geometry_only_adas()
        point, _ = adas.project_gaze_to_lr((math.radians(60.0), 0.0), origin_pro=np.zeros(3))
        self.assertIsNone(point)
        self.assertEqual(adas.last_projection_status, "ray-outside-lr-fov")


if __name__ == "__main__":
    unittest.main()
