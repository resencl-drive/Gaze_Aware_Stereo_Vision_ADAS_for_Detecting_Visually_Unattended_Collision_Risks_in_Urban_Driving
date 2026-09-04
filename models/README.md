# External models

This directory intentionally contains no model weights. The application does
not download weights automatically and fails early when a required file is
missing.

The default configuration expects:

- `resnet101_v2.pt`
- `yolov8n-seg.pt`
- `face-detection-retail-0004.blob`
- `facial-landmarks-35-adas-0002.blob`

See [`manifest.json`](manifest.json) for the model versions, interfaces,
verified file sizes, SHA-256 fingerprints, upstream sources, and distribution
status. A fingerprint identifies the artifact used for the reported system; it
does not grant permission to redistribute that artifact.

Paths can be overridden with `--gaze-model`, `--yolo-model`,
`--face-detector-model`, and `--landmark-model`. Replacements must preserve the
interfaces documented in the manifest. In particular, a detection-only YOLO
model is not a drop-in replacement for the required instance-segmentation
model.

Users are responsible for obtaining each model from an authorized source and
complying with its license and dataset terms. Do not commit model files to this
repository.
