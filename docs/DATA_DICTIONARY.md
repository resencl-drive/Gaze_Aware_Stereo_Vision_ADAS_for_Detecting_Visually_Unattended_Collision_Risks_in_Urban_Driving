# Data dictionary

This document describes the data consumed and produced by the released code.
Model binaries and datasets are not included in the repository.

## Conventions

| Convention | Definition |
|---|---|
| Image origin | Upper-left pixel |
| Image x-axis | Positive to the right |
| Image y-axis | Positive downward |
| 3D transform | `p_lr = R_lr_from_pro @ p_pro + T_lr_from_pro_m` |
| 3D distance | Metres unless a field ends in `_mm` |
| Angles | Radians unless a field ends in `_deg` |
| Wall-clock sample time | Unix seconds in `timestamp` fields |
| Report creation time | ISO 8601 string |
| Missing stereo depth in internal processing | Sentinel `999.0` m; not a physical measurement |

## ETH-XGaze training input

The trainer accepts one or more HDF5 files selected with `--train-glob` and
`--val-glob`. Dataset acquisition and preprocessing are external to this
repository.

| Dataset key | Type and shape | Unit/range | Meaning |
|---|---|---|---|
| `face_patch` | `uint8` or numeric array, `N x H x W x 3` or `N x 3 x H x W` | Pixel intensity | Normalized face crop. Historical files use BGR; `--input-color rgb` supports RGB. |
| `face_gaze` | Floating array, `N x 2` | Radians | Ground-truth `(pitch, yaw)` by default; configurable with `--gaze-order`. |

The training pipeline resizes images to 224 x 224, converts them to RGB when
required, applies the documented augmentation, and performs ImageNet
normalization.

## Training-run outputs

Every run under `runs/<run-name>/` contains:

| File/field | Type | Meaning |
|---|---|---|
| `config.json` | JSON object | Resolved arguments, model settings, device, and input-file counts. |
| `metrics.csv` | CSV | One row per epoch. |
| `metrics.csv.epoch` | Integer | One-based epoch number. |
| `metrics.csv.lr` | Float | Learning rate after the epoch scheduler step. |
| `metrics.csv.train_loss` | Float | Mean combined training loss. |
| `metrics.csv.train_angular_error_deg` | Float, degrees | Mean training angular error. |
| `metrics.csv.validation_loss` | Float | Mean combined validation loss. |
| `metrics.csv.validation_angular_error_deg` | Float, degrees | Mean validation angular error. |
| `metrics.csv.epoch_time_seconds` | Float, seconds | Epoch wall-clock duration. |
| `best.pt` | PyTorch state dictionary | Parameters with the lowest validation angular error. |
| `last.pt` | PyTorch checkpoint | Latest model, optimizer, scheduler, epoch, configuration, and best error. |

The combined loss is `0.5 * SmoothL1 + 0.5 * angular_loss` under the paper
configuration.

## Extrinsic-calibration JSON

The active file is
`calibracion_extrinseca/extrinsics_pro_to_lr_no_mirror.json`.

| Field | Type/shape | Unit | Meaning |
|---|---|---:|---|
| `R` | Float matrix, `3 x 3` | Dimensionless | Proper rotation from the OAK-D Pro frame to the OAK-D LR frame. |
| `T` | Float vector, length 3 | Defined by `translation_unit` | Translation from Pro to LR. |
| `translation_unit` | String | — | Must be `m` or `mm`; the published file uses `m`. |
| `rms_mm` | Float | mm | Nearest-neighbour checkerboard-point RMS after rigid alignment. |
| `n_pro` | Integer | Poses | Number of retained Pro observations. |
| `n_lr` | Integer | Poses | Number of retained LR observations. |
| `K_pro` | Float matrix, `3 x 3` | Pixels | Pro RGB intrinsic matrix at calibration resolution. |
| `D_pro` | Float array | Model-specific | Pro lens-distortion coefficients from device calibration. |
| `K_lr` | Float matrix, `3 x 3` | Pixels | LR RGB intrinsic matrix at calibration resolution. |
| `D_lr` | Float array | Model-specific | LR lens-distortion coefficients from device calibration. |
| `poses_pro[]` | Array of pose objects | — | Retained checkerboard poses in the Pro frame. |
| `poses_lr[]` | Array of pose objects | — | Retained checkerboard poses in the LR frame. |
| `poses_*.R` | Float matrix, `3 x 3` | Dimensionless | Checkerboard rotation returned by `solvePnP`. |
| `poses_*.t` | Float vector, `3 x 1` | m | Checkerboard translation returned by `solvePnP`. |

The published JSON contains five pose pairs. The included generator can
recompute `R`, `T`, and `rms_mm` exactly from these observations.

## Runtime report JSON

Reports are written only when `--save-reports` is enabled. The filename is
`outputs/adas_gaze_report_<YYYYMMDD_HHMMSS>.json`.

### Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `timestamp` | ISO 8601 string | Report creation time. |
| `total_samples` | Integer | Number of manual gaze-validation clicks. |
| `models` | Object keyed by model name | Aggregate gaze-validation metrics. |
| `raw_samples` | Array | Individual manual validation samples. |
| `method_configuration` | Object | Calibration, model, geometry, threshold, and valid-frame settings. |
| `alarm_metrics` | Object | Alarm counts, rates, and optional latency summaries. |
| `alarm_events` | Array | One record per positive alarm frame. |
| `adas_metrics` | Object | Motion, detection, TTC, and visual-attention counts. |
| `fps_metrics` | Object | Recent and whole-session processing rates. |

### `raw_samples[]`

| Field | Type | Unit | Meaning |
|---|---|---:|---|
| `timestamp` | Float | Unix seconds | Sample time. |
| `model` | String | — | Gaze-model identifier. |
| `gaze_point` | Integer pair | Pixels | Projected gaze coordinate in the LR image. |
| `click_point` | Integer pair | Pixels | User-selected reference coordinate. |
| `error_px` | Float | Pixels | Euclidean point error. |
| `angular_error_deg` | Float | Degrees | Angular error derived from camera intrinsics. |
| `nss` | Float | Standard deviations | Normalized Scanpath Saliency score. |

### `models.<model_name>`

| Field group | Type/unit | Meaning |
|---|---|---|
| `samples` | Integer | Number of validation samples. |
| `mean_error_px`, `std_error_px`, `min_error_px`, `max_error_px`, `median_error_px`, `p95_error_px`, `p99_error_px` | Float, pixels | Point-error summary. |
| `mean_angular_error_deg`, `std_angular_error_deg`, `min_angular_error_deg`, `max_angular_error_deg`, `median_angular_error_deg`, `p95_angular_error_deg`, `p99_angular_error_deg` | Float, degrees | Angular-error summary. |
| `mean_nss`, `std_nss` | Float | NSS summary. |
| `pct_good_nss` | Float, percent | Samples with NSS greater than 1.0. |

### `method_configuration`

| Field | Type/unit | Meaning |
|---|---|---|
| `extrinsics_file` | String | Configured calibration path. |
| `extrinsic_rmse_mm` | Float or null, mm | RMS read from the calibration file. |
| `R_lr_from_pro` | `3 x 3` float matrix | Active Pro-to-LR rotation. |
| `T_lr_from_pro_m` | Length-3 float vector, m | Active Pro-to-LR translation. |
| `K_lr_active`, `K_pro_active` | `3 x 3` float matrices, pixels | Active camera intrinsics. |
| `yolo_model`, `facial_landmark_model` | Strings | Configured model paths. |
| `facial_landmarks_count` | Integer | Number of predicted landmarks; 35. |
| `face_alignment_size` | Integer pair, pixels | Gaze-model crop size; `[224, 224]`. |
| `driver_rgbd_depth_used_for_gaze_origin` | Boolean | Whether RGB-D face depth defines the ray origin. |
| `face_candidate_frames` | Integer, frames | Frames containing a face candidate. |
| `landmark_valid_frames` | Integer, frames | Frames with valid facial landmarks. |
| `rgbd_face_depth_valid_frames` | Integer, frames | Frames with valid face-origin depth. |
| `yolo_confidence` | Float | Detection threshold; 0.5. |
| `max_depth_m` | Float, m | Evaluated stereo range; 30.0. |
| `lane_width_m` | Float, m | Assumed lane width; 3.5. |
| `motion_threshold_px` | Float, pixels | Optical-flow motion threshold; 1.5. |
| `shi_tomasi_max_corners` | Integer | Maximum tracked features; 400. |
| `lucas_kanade_window` | Integer pair, pixels | Optical-flow window; `[11, 11]`. |
| `track_centroid_threshold_px` | Float, pixels | Frame-to-frame association threshold; 60. |
| `depth_smoothing_window` | Integer, frames | Per-track depth-history length; 5. |
| `minimum_approach_speed_mps` | Float, m/s | Minimum closing speed used for finite TTC; 0.1. |
| `ttc_threshold_s` | Float, seconds | Critical TTC threshold; 2.0. |

### Alarm and ADAS metrics

| Field | Type/unit | Meaning |
|---|---|---|
| `alarm_metrics.total_frames_processed` | Integer, frames | Frames entering the evaluated ADAS branch. |
| `alarm_metrics.risk_condition_frames` | Integer, frames | Frames with an evaluated object within 30 m. |
| `alarm_metrics.alarm_positive_frames` | Integer, frames | Frames satisfying the complete warning condition. |
| `alarm_metrics.audible_warning_count` | Integer | Beeps emitted after cooldown filtering. |
| `alarm_metrics.risk_condition_rate_pct` | Float, percent | Risk-condition frames divided by processed frames. |
| `alarm_metrics.operational_activation_ratio_pct` | Float, percent | Positive-alarm frames divided by risk-condition frames. |
| `alarm_metrics.latency_ms_mean/min/max/p95` | Float, ms | Optional end-to-end latency summaries. |
| `adas_metrics.moving_frames_count` | Integer, frames | Frames classified as moving. |
| `adas_metrics.total_detections_while_moving` | Integer | Evaluated detections in moving frames. |
| `adas_metrics.ttc_positive_records` | Integer | Detection records with critical TTC. |
| `adas_metrics.unattended_ttc_positive_records` | Integer | Critical-TTC records not intersected by gaze. |
| `adas_metrics.ttc_positive_frames` | Integer, frames | Frames containing at least one critical-TTC record. |
| `adas_metrics.ttc_positive_frame_rate_pct` | Float, percent | Critical-TTC frames divided by moving frames. |

Each `alarm_events[]` entry contains Unix `timestamp`, `latency_total_ms`,
`latency_gaze_ms`, `object_missed`, `driver_looking_at`, and Boolean
`beep_emitted`.

### `fps_metrics`

| Field | Type/unit | Meaning |
|---|---|---|
| `program_fps_recent_avg` | Float, FPS | Mean program throughput over the recent 30-frame window. |
| `camera_lr_fps_recent_avg`, `camera_pro_fps_recent_avg` | Float, FPS | Recent inter-arrival-rate means. |
| `camera_lr_fps_total_session`, `camera_pro_fps_total_session` | Float, FPS | Frames received divided by session duration. |
| `program_frames_processed_for_fps` | Integer, frames | Frames used by the program FPS counter. |
| `camera_lr_frames_received`, `camera_pro_frames_received` | Integer, frames | Received RGB-frame counts. |

## Optional media outputs

`--record-video`, `--save-captures`, and `--save-dataset-frames` create media
under `outputs/`. Driver-facing images may contain personally identifiable
biometric information. These options are disabled by default and should be
used only with appropriate consent, access control, and retention procedures.
