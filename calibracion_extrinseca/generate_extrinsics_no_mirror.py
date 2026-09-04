# SPDX-License-Identifier: AGPL-3.0-only
"""Generate the OAK-D Pro-to-LR extrinsic calibration without a mirror.

The checkerboard is observed at matching physical locations by each camera.
Each observation is converted to a checkerboard pose with solvePnP. The two
sets of board centers are then aligned with rigid ICP to estimate

    p_lr = R @ p_pro + T

The JSON output retains the observations so the estimate can be audited or
recomputed without reconnecting the cameras.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import depthai as dai
import numpy as np


CHECKERBOARD_INNER_CORNERS = (7, 10)
CHECKERBOARD_SQUARE_SIZE_M = 0.020
CORNER_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


def checkerboard_points() -> np.ndarray:
    points = np.zeros(
        (CHECKERBOARD_INNER_CORNERS[0] * CHECKERBOARD_INNER_CORNERS[1], 3),
        dtype=np.float32,
    )
    points[:, :2] = (
        np.mgrid[
            0 : CHECKERBOARD_INNER_CORNERS[0],
            0 : CHECKERBOARD_INNER_CORNERS[1],
        ]
        .T.reshape(-1, 2)
        * CHECKERBOARD_SQUARE_SIZE_M
    )
    return points


OBJECT_POINTS = checkerboard_points()


def detect_checkerboard(frame: np.ndarray) -> tuple[np.ndarray | None, bool]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD_INNER_CORNERS, flags
    )
    if not found:
        return None, False
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
    return refined, True


def solve_checkerboard_pose(
    corners: np.ndarray, intrinsics: np.ndarray, distortion: np.ndarray
) -> dict[str, np.ndarray] | None:
    solved, rotation_vector, translation = cv2.solvePnP(
        OBJECT_POINTS, corners, intrinsics, distortion
    )
    if not solved:
        return None
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return {"R": rotation, "t": translation.reshape(3, 1)}


def nearest_neighbor_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    squared_distances = np.sum(
        (source[:, None, :] - target[None, :, :]) ** 2, axis=2
    )
    return np.argmin(squared_distances, axis=1)


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the proper rigid transform that maps source points to target."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1, :] *= -1
        rotation = right_transpose.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Align unmatched board-center observations using rigid ICP."""
    translation = target.mean(axis=0) - source.mean(axis=0)
    rotation = np.eye(3)
    for _ in range(max_iterations):
        transformed = (rotation @ source.T).T + translation
        matches = nearest_neighbor_indices(transformed, target)
        next_rotation, next_translation = rigid_alignment(source, target[matches])
        converged = (
            np.linalg.norm(next_rotation - rotation) < tolerance
            and np.linalg.norm(next_translation - translation) < tolerance
        )
        if converged:
            break
        rotation, translation = next_rotation, next_translation
    return rotation, translation


def deserialize_poses(items: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
    return [
        {
            "R": np.asarray(item["R"], dtype=np.float64),
            "t": np.asarray(item["t"], dtype=np.float64).reshape(3, 1),
        }
        for item in items
    ]


def serialize_poses(poses: list[dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    return [
        {"R": pose["R"].tolist(), "t": pose["t"].tolist()} for pose in poses
    ]


def estimate_extrinsics(
    pro_poses: list[dict[str, np.ndarray]],
    lr_poses: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, float]:
    pair_count = min(len(pro_poses), len(lr_poses))
    if pair_count < 3:
        raise ValueError("At least three PRO/LR pose pairs are required")

    pro_poses = pro_poses[:pair_count]
    lr_poses = lr_poses[:pair_count]
    pro_centers = np.asarray([pose["t"].ravel() for pose in pro_poses])
    lr_centers = np.asarray([pose["t"].ravel() for pose in lr_poses])
    rotation, translation = icp(pro_centers, lr_centers)

    pro_points = np.vstack(
        [(pose["R"] @ OBJECT_POINTS.T + pose["t"]).T for pose in pro_poses]
    )
    lr_points = np.vstack(
        [(pose["R"] @ OBJECT_POINTS.T + pose["t"]).T for pose in lr_poses]
    )
    transformed = (rotation @ pro_points.T).T + translation
    matches = nearest_neighbor_indices(transformed, lr_points)
    errors = np.linalg.norm(transformed - lr_points[matches], axis=1)
    rms_mm = float(np.sqrt(np.mean(errors**2)) * 1000.0)
    return rotation, translation, rms_mm


def calibration_payload(
    pro_poses: list[dict[str, np.ndarray]],
    lr_poses: list[dict[str, np.ndarray]],
    pro_intrinsics: np.ndarray,
    pro_distortion: np.ndarray,
    lr_intrinsics: np.ndarray,
    lr_distortion: np.ndarray,
) -> dict[str, Any]:
    pair_count = min(len(pro_poses), len(lr_poses))
    rotation, translation, rms_mm = estimate_extrinsics(pro_poses, lr_poses)
    return {
        "R": rotation.tolist(),
        "T": translation.tolist(),
        "translation_unit": "m",
        "rms_mm": rms_mm,
        "n_pro": pair_count,
        "n_lr": pair_count,
        "K_pro": pro_intrinsics.tolist(),
        "D_pro": pro_distortion.tolist(),
        "K_lr": lr_intrinsics.tolist(),
        "D_lr": lr_distortion.tolist(),
        "poses_pro": serialize_poses(pro_poses[:pair_count]),
        "poses_lr": serialize_poses(lr_poses[:pair_count]),
    }


def validate_payload(data: dict[str, Any]) -> None:
    required = {
        "R",
        "T",
        "translation_unit",
        "rms_mm",
        "n_pro",
        "n_lr",
        "K_pro",
        "D_pro",
        "K_lr",
        "D_lr",
        "poses_pro",
        "poses_lr",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Calibration JSON is missing: {sorted(missing)}")
    if data["translation_unit"] != "m":
        raise ValueError("translation_unit must be 'm'")
    rotation = np.asarray(data["R"], dtype=np.float64)
    translation = np.asarray(data["T"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape not in {(3,), (3, 1)}:
        raise ValueError("R must be 3x3 and T must contain three values")


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    validate_payload(data)
    return data


def save_payload(path: Path, data: dict[str, Any]) -> None:
    validate_payload(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    temporary_path.replace(path)


def recompute_payload(data: dict[str, Any]) -> dict[str, Any]:
    return calibration_payload(
        deserialize_poses(data["poses_pro"]),
        deserialize_poses(data["poses_lr"]),
        np.asarray(data["K_pro"], dtype=np.float64),
        np.asarray(data["D_pro"], dtype=np.float64),
        np.asarray(data["K_lr"], dtype=np.float64),
        np.asarray(data["D_lr"], dtype=np.float64),
    )


def create_lr_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()
    camera = pipeline.create(dai.node.ColorCamera)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    camera.setPreviewSize(896, 512)
    camera.setInterleaved(False)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camera.setFps(20)
    output = pipeline.create(dai.node.XLinkOut)
    output.setStreamName("preview")
    camera.preview.link(output.input)
    return pipeline


def create_pro_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()
    camera = pipeline.create(dai.node.ColorCamera)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setPreviewSize(300, 300)
    camera.setInterleaved(False)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    output = pipeline.create(dai.node.XLinkOut)
    output.setStreamName("preview")
    camera.preview.link(output.input)
    return pipeline


def camera_socket() -> dai.CameraBoardSocket:
    return getattr(dai.CameraBoardSocket, "CAM_A", dai.CameraBoardSocket.RGB)


def scaled_intrinsics(
    calibration: Any,
    native_size: tuple[int, int],
    preview_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    native_width, native_height = native_size
    preview_width, preview_height = preview_size
    socket = camera_socket()
    intrinsics = np.asarray(
        calibration.getCameraIntrinsics(socket, native_width, native_height),
        dtype=np.float64,
    )
    intrinsics[0] *= preview_width / native_width
    intrinsics[1] *= preview_height / native_height
    distortion = np.asarray(
        calibration.getDistortionCoefficients(socket), dtype=np.float64
    )
    return intrinsics, distortion


def get_device_info(mxid: str) -> dai.DeviceInfo:
    devices = {device.getMxId(): device for device in dai.Device.getAllAvailableDevices()}
    if mxid not in devices:
        available = ", ".join(sorted(devices)) or "none"
        raise RuntimeError(f"Device {mxid!r} was not found; available devices: {available}")
    return devices[mxid]


def load_existing_observations(
    path: Path,
) -> tuple[
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
]:
    if not path.exists():
        return [], [], None
    data = load_payload(path)
    intrinsics = (
        np.asarray(data["K_pro"], dtype=np.float64),
        np.asarray(data["D_pro"], dtype=np.float64),
        np.asarray(data["K_lr"], dtype=np.float64),
        np.asarray(data["D_lr"], dtype=np.float64),
    )
    return (
        deserialize_poses(data["poses_pro"]),
        deserialize_poses(data["poses_lr"]),
        intrinsics,
    )


def collect_observations(
    lr_device_id: str,
    pro_device_id: str,
    output_path: Path,
    resume: bool,
) -> dict[str, Any]:
    if lr_device_id == pro_device_id:
        raise ValueError("The LR and Pro device IDs must be different")
    lr_info = get_device_info(lr_device_id)
    pro_info = get_device_info(pro_device_id)

    with dai.Device(
        create_lr_pipeline(), lr_info, maxUsbSpeed=dai.UsbSpeed.HIGH
    ) as lr_device, dai.Device(
        create_pro_pipeline(), pro_info, maxUsbSpeed=dai.UsbSpeed.HIGH
    ) as pro_device:
        lr_intrinsics, lr_distortion = scaled_intrinsics(
            lr_device.readCalibration(), (1920, 1200), (896, 512)
        )
        pro_intrinsics, pro_distortion = scaled_intrinsics(
            pro_device.readCalibration(), (1920, 1080), (300, 300)
        )

        if resume:
            pro_poses, lr_poses, saved_intrinsics = load_existing_observations(
                output_path
            )
            if saved_intrinsics is not None:
                (
                    pro_intrinsics,
                    pro_distortion,
                    lr_intrinsics,
                    lr_distortion,
                ) = saved_intrinsics
        else:
            pro_poses, lr_poses = [], []

        lr_queue = lr_device.getOutputQueue("preview", maxSize=4, blocking=False)
        pro_queue = pro_device.getOutputQueue("preview", maxSize=4, blocking=False)
        clicked = False

        def click_callback(event: int, _x: int, _y: int, _flags: int, _data: Any) -> None:
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked = True

        cv2.namedWindow("OAK-D Pro", cv2.WINDOW_NORMAL)
        cv2.namedWindow("OAK-D LR", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("OAK-D Pro", click_callback)
        cv2.setMouseCallback("OAK-D LR", click_callback)
        print("Click either window to capture; press Esc to calculate and save.")
        print("Capture the board with PRO, then place it at the same physical")
        print("location for the LR capture. Repeat at varied positions and angles.")

        mode = "PRO"
        pending_pro_pose: dict[str, np.ndarray] | None = None
        lr_frame = None
        pro_frame = None
        while True:
            lr_packet = lr_queue.tryGet()
            pro_packet = pro_queue.tryGet()
            if lr_packet is not None:
                lr_frame = lr_packet.getCvFrame()
            if pro_packet is not None:
                pro_frame = pro_packet.getCvFrame()
            if lr_frame is None or pro_frame is None:
                cv2.waitKey(10)
                continue

            pro_corners, pro_found = detect_checkerboard(pro_frame)
            lr_corners, lr_found = detect_checkerboard(lr_frame)
            pro_display = pro_frame.copy()
            lr_display = lr_frame.copy()
            if pro_found:
                cv2.drawChessboardCorners(
                    pro_display,
                    CHECKERBOARD_INNER_CORNERS,
                    pro_corners,
                    True,
                )
            if lr_found:
                cv2.drawChessboardCorners(
                    lr_display, CHECKERBOARD_INNER_CORNERS, lr_corners, True
                )
            cv2.putText(
                pro_display,
                f"PRO pairs={len(pro_poses)} mode={mode}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                lr_display,
                f"LR pairs={len(lr_poses)} mode={mode}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("OAK-D Pro", pro_display)
            cv2.imshow("OAK-D LR", lr_display)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            if not clicked:
                continue
            clicked = False
            if mode == "PRO" and pro_found:
                pending_pro_pose = solve_checkerboard_pose(
                    pro_corners, pro_intrinsics, pro_distortion
                )
                if pending_pro_pose is not None:
                    mode = "LR"
                    print("PRO pose captured; capture LR at the same location.")
            elif mode == "LR" and lr_found and pending_pro_pose is not None:
                lr_pose = solve_checkerboard_pose(
                    lr_corners, lr_intrinsics, lr_distortion
                )
                if lr_pose is not None:
                    pro_poses.append(pending_pro_pose)
                    lr_poses.append(lr_pose)
                    pending_pro_pose = None
                    mode = "PRO"
                    print(f"Stored pair {len(pro_poses)}.")

        cv2.destroyAllWindows()
        return calibration_payload(
            pro_poses,
            lr_poses,
            pro_intrinsics,
            pro_distortion,
            lr_intrinsics,
            lr_distortion,
        )


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).with_name("extrinsics_pro_to_lr_no_mirror.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-device-id", help="OAK-D LR MXID")
    parser.add_argument("--pro-device-id", help="OAK-D Pro MXID")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore observations in an existing output file.",
    )
    parser.add_argument(
        "--recompute",
        type=Path,
        help="Recompute a calibration from the observations in an existing JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recompute:
        payload = recompute_payload(load_payload(args.recompute))
    else:
        if not args.lr_device_id or not args.pro_device_id:
            raise SystemExit(
                "--lr-device-id and --pro-device-id are required for capture"
            )
        payload = collect_observations(
            args.lr_device_id,
            args.pro_device_id,
            args.output,
            resume=not args.fresh,
        )

    save_payload(args.output, payload)
    print(f"Saved calibration to {args.output}")
    print(f"Pose pairs: {payload['n_pro']}")
    print(f"RMS: {payload['rms_mm']:.6f} mm")


if __name__ == "__main__":
    main()
