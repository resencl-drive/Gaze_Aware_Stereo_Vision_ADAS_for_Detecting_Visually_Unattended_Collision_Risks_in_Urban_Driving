"""Pure, testable geometry helpers for gaze processing."""

from __future__ import annotations

import math
import cv2
import numpy as np


# Preserve the landmark order used by the already validated model pipeline.
FACE_ALIGNMENT_TARGET = np.float32(
    [
        [72.0, 86.0],
        [152.0, 86.0],
        [112.0, 158.0],
    ]
)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def build_face_alignment_transform(
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    mouth: np.ndarray,
    *,
    min_eye_distance: float = 8.0,
) -> np.ndarray:
    """Return a proper (non-reflecting) affine face transform.

    The source order is kept identical to the established OpenVINO integration.
    Degenerate inputs and transforms that would change that handedness are
    rejected.
    """

    source = np.float32([left_eye, right_eye, mouth])
    if source.shape != (3, 2) or not np.isfinite(source).all():
        raise ValueError("face landmarks must be finite 2D points")
    if float(np.linalg.norm(source[0] - source[1])) < min_eye_distance:
        raise ValueError("inter-eye distance is too small")

    source_area = _cross2(source[1] - source[0], source[2] - source[0])
    target_area = _cross2(
        FACE_ALIGNMENT_TARGET[1] - FACE_ALIGNMENT_TARGET[0],
        FACE_ALIGNMENT_TARGET[2] - FACE_ALIGNMENT_TARGET[0],
    )
    if abs(source_area) < 1e-3:
        raise ValueError("face landmarks are nearly collinear")
    if source_area * target_area <= 0.0:
        raise ValueError("face landmarks are mirrored")

    transform = cv2.getAffineTransform(source, FACE_ALIGNMENT_TARGET)
    linear_det = float(np.linalg.det(transform[:, :2]))
    if not np.isfinite(transform).all() or linear_det <= 1e-9:
        raise ValueError("face alignment would reflect or collapse the image")
    return transform


def validate_proper_rotation(rotation: np.ndarray, *, atol: float = 1e-5) -> np.ndarray:
    """Validate and return a finite 3D proper rotation matrix."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=atol):
        raise ValueError("rotation must have determinant +1")
    return rotation


def model_angles_to_direction(yaw: float, pitch: float) -> np.ndarray:
    """Convert internal ``(yaw, pitch)`` to a unit model-frame direction.

    This function defines only the model frame. A separately calibrated proper
    rotation is required before treating the result as a PRO-camera ray.
    """

    direction = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or norm < 1e-12:
        raise ValueError("gaze angles do not define a finite direction")
    return direction / norm
