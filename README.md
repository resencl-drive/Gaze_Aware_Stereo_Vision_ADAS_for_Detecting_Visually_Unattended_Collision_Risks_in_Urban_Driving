
# Real-Time ADAS Gaze and Planar-Distance Estimation

> **Repository scope:** This folder contains source code and numerical calibration files only. It does not include the manuscript, model weights, images, videos, or participant data. 

This repository contains the implementation associated with the paper, including driver gaze estimation, ADAS object detection, depth estimation, time-to-collision (TTC) computation, and performance metrics using two OAK-D cameras.

## Privacy

By default, the application does not write videos, images, dataset frames, or reports to disk. Interface metrics are maintained in memory, and the physical identifiers of the cameras are not included in the source code.

Optional recording features may store images of the driver. These features should only be used under appropriate informed-consent procedures and with a suitable data-retention policy.

## Installation

The project targets Python 3.11, and its dependencies are pinned in `requirements.txt`.

Full system execution requires two OAK-D cameras and the model weights described in the associated paper. These weights are not distributed with this repository.

Model weights (`.pt`) and DepthAI model files (`.blob`) are intentionally excluded from version control.

Refer to `models/README.md` for the expected model locations. Each required file must be obtained from an authorized source and placed in the corresponding directory, or its location must be provided through the available command-line arguments.

## Running the System

To start the real-time application:

```bash
python adas_gaze_realtime.py --lr-device-id MXID_1 --pro-device-id MXID_2
```

The driver-facing processing branch uses RGB-D data from the OAK-D Pro. It detects the driver’s face, estimates 35 facial landmarks, aligns the face crop, and computes the 3D origin of the gaze ray using stereo-depth information.

Frames without valid facial landmarks or valid facial-depth measurements are not considered successfully processed for the driver gaze estimation branch.

By default, the application uses a ResNet-101 gaze-estimation backbone together with the extrinsic calibration stored in:

`calibracion_extrinseca/extrinsics_pro_to_lr_no_mirror.json`

The existing validated processing convention interprets the model direction
directly in the OAK-D Pro frame and then applies the published PRO-to-LR
extrinsic. That behavior remains unchanged.

The validated extrinsic file remains the active calibration and is not
modified at runtime.

Camera roles must be assigned explicitly because USB enumeration order is not
stable. This keeps hardware identifiers out of the repository while preventing
an accidental LR/Pro inversion:

```bash
python adas_gaze_realtime.py --lr-device-id MXID_1 --pro-device-id MXID_2
```

Optional outputs can be enabled with:

```bash
python adas_gaze_realtime.py \
  --lr-device-id MXID_1 \
  --pro-device-id MXID_2 \
  --save-reports \
  --record-video \
  --save-captures \
  --save-dataset-frames
```

These optional outputs are written to `outputs/` by default, but only when the corresponding flag is enabled. In particular:

- `--save-reports`: saves the JSON validation report
- `--record-video`: saves the processed video stream
- `--save-captures`: saves captured images (which may include faces)
- `--save-dataset-frames`: saves LR frames for YOLO dataset preparation

## Model Weights

This repository does not redistribute model weights through Git, GitHub Releases, Git LFS, or any other distribution mechanism.

Only the expected model paths are defined in the repository, and the application does not silently download model weights at runtime.

Users are responsible for obtaining all required models from authorized sources and for complying with the applicable licenses and usage restrictions.

## License and Citation

The source code is released under the GNU AGPL-3.0 license to maintain compatibility with the Ultralytics dependency.

See `LICENSE` and `THIRD_PARTY_NOTICES.md` for additional licensing information.

Although this project is primarily intended for academic research, the source code is distributed under the terms of the GNU AGPL-3.0 license.

In particular, when the system is used with a checkpoint derived from ETH-XGaze, the corresponding model remains subject to its original licensing terms, including restrictions to non-commercial academic research where applicable. Such checkpoints must not be redistributed through this repository.


