import math
import unittest

import cv2
import numpy as np

from gaze_geometry import (
    FACE_ALIGNMENT_TARGET,
    build_face_alignment_transform,
    model_angles_to_direction,
    validate_proper_rotation,
)


class FaceAlignmentTests(unittest.TestCase):
    def test_canonical_face_does_not_reflect(self):
        transform = build_face_alignment_transform(*FACE_ALIGNMENT_TARGET)
        np.testing.assert_allclose(
            transform, np.float64([[1, 0, 0], [0, 1, 0]]), atol=1e-6
        )
        self.assertGreater(np.linalg.det(transform[:, :2]), 0.0)

    def test_reversed_eye_mapping_is_a_reflection(self):
        reversed_target = np.float32([[152, 86], [72, 86], [112, 158]])
        transform = cv2.getAffineTransform(FACE_ALIGNMENT_TARGET, reversed_target)
        self.assertLess(np.linalg.det(transform[:, :2]), 0.0)

    def test_rotation_and_scale_preserve_handedness(self):
        center = np.float32([112.0, 112.0])
        for angle_degrees, scale in ((-25.0, 0.8), (0.0, 1.0), (17.0, 1.4)):
            angle = math.radians(angle_degrees)
            rotation = np.float32(
                [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            )
            source = (FACE_ALIGNMENT_TARGET - center) @ (scale * rotation).T + center
            transform = build_face_alignment_transform(*source)
            homogeneous = np.column_stack([source, np.ones(3, dtype=np.float32)])
            mapped = homogeneous @ transform.T
            np.testing.assert_allclose(mapped, FACE_ALIGNMENT_TARGET, atol=1e-3)
            self.assertGreater(np.linalg.det(transform[:, :2]), 0.0)

    def test_mirrored_input_is_rejected(self):
        mirrored = FACE_ALIGNMENT_TARGET.copy()
        mirrored[:, 0] = 224.0 - mirrored[:, 0]
        with self.assertRaisesRegex(ValueError, "mirrored"):
            build_face_alignment_transform(*mirrored)

    def test_degenerate_landmarks_are_rejected(self):
        with self.assertRaises(ValueError):
            build_face_alignment_transform(
                np.float32([100, 80]), np.float32([104, 80]), np.float32([102, 150])
            )
        with self.assertRaises(ValueError):
            build_face_alignment_transform(
                np.float32([152, 86]), np.float32([72, 86]), np.float32([112, 86])
            )


class RayConventionTests(unittest.TestCase):
    def test_model_angle_axes_and_norm(self):
        np.testing.assert_allclose(model_angles_to_direction(0.0, 0.0), [0, 0, 1])
        self.assertGreater(model_angles_to_direction(0.2, 0.0)[0], 0.0)
        self.assertLess(model_angles_to_direction(0.0, -0.2)[1], 0.0)
        self.assertAlmostEqual(np.linalg.norm(model_angles_to_direction(0.3, -0.4)), 1.0)

    def test_rotation_validation_rejects_reflection_and_scale(self):
        np.testing.assert_allclose(validate_proper_rotation(np.eye(3)), np.eye(3))
        with self.assertRaises(ValueError):
            validate_proper_rotation(np.diag([-1.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            validate_proper_rotation(2.0 * np.eye(3))



if __name__ == "__main__":
    unittest.main()
