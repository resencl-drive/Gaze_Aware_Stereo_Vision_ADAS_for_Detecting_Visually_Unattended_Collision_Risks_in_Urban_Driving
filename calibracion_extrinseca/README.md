# Extrinsic calibration

`generate_extrinsics_no_mirror.py` estimates the rigid transformation from the
OAK-D Pro coordinate frame to the OAK-D LR coordinate frame:

```text
p_lr = R @ p_pro + T
```

The calibration uses a checkerboard with 7 x 10 inner corners and 20 mm
squares. It alternates captures from the two cameras without reflecting either
image. At least three pairs are required; a larger, spatially varied set is
recommended.

## Capture a new calibration

Camera identifiers are supplied at runtime and are never stored in source:

```bash
python calibracion_extrinseca/generate_extrinsics_no_mirror.py \
  --lr-device-id MXID_1 \
  --pro-device-id MXID_2 \
  --fresh
```

Click either preview window to capture the current camera. Capture one PRO
pose, move the checkerboard to the corresponding physical location for the LR
camera, and capture the LR pose. Repeat at different positions and angles.
Press `Esc` to calculate and save the calibration.

Without `--fresh`, observations from an existing output JSON are loaded so
additional pairs can be collected.

## Audit an existing calibration

The published JSON retains all checkerboard poses. Its transformation and RMS
can therefore be recomputed without cameras:

```bash
python calibracion_extrinseca/generate_extrinsics_no_mirror.py \
  --recompute calibracion_extrinseca/extrinsics_pro_to_lr_no_mirror.json \
  --output outputs/recomputed_extrinsics.json
```

The output path should be different from the validated calibration unless the
replacement has been independently checked. Generated JSON files under
`outputs/` are ignored by Git.
