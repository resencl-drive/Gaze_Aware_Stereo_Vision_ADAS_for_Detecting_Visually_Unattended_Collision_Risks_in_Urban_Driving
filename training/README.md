# Gaze-model training

This directory contains the publication-ready training implementation for the
gaze-model family evaluated in the associated paper. It supports ResNet-50,
ResNet-101, EfficientNetV2-S, ConvNeXt Tiny, and MobileNetV3 Large.

## Data

The dataset is not distributed with this repository. Obtain ETH-XGaze from its
official maintainers and comply with its license and data-use conditions.

The trainer reads one or more HDF5 files. By default, each file must contain:

- `face_patch`: face images as `H x W x 3` or `3 x H x W` arrays
- `face_gaze`: gaze labels in radians, ordered as `(pitch, yaw)`

The historical preprocessed files stored images in BGR order. Use
`--input-color rgb` if your HDF5 images are already RGB. The image and label
keys can be changed with `--image-key` and `--label-key`.

## Environment

Install the main project dependencies and the training-specific HDF5 reader:

```bash
python -m pip install -r requirements.txt
python -m pip install -r training/requirements.txt
```

## Reproducing the ResNet-101 configuration

Run the command from the repository root and quote wildcard paths so the
trainer can expand them consistently:

```bash
python training/train_gaze_models.py \
  --model resnet101 \
  --train-glob "/path/to/eth-xgaze/train/**/*.h5" \
  --val-glob "/path/to/eth-xgaze/validation/**/*.h5" \
  --run-name resnet101_v2_clean
```

The default ResNet-101 settings implement the configuration in Table 2 of the
paper: ImageNet
initialization, batch size 24, learning rate `5e-5`, 40 epochs, AdamW, cosine
learning-rate decay, and early-stopping patience 10. The loss is

```text
0.5 * SmoothL1(prediction, target) + 0.5 * angular_loss(prediction, target)
```

The training pipeline resizes crops to `224 x 224`, applies ImageNet
normalization, color jitter, Gaussian blur, and random erasing, and does not use
horizontal flipping.

Model-specific defaults are selected automatically. Explicit command-line
arguments such as `--batch-size`, `--lr`, `--pretrained`, or `--no-pretrained`
override them.

## Outputs

Each run produces the following under `runs/<run-name>/`:

- `config.json`: resolved command-line configuration
- `metrics.csv`: per-epoch loss, angular error, learning rate, and runtime
- `last.pt`: resumable training checkpoint
- `best.pt`: model state with the lowest validation angular error

The `runs/` directory, HDF5 data, and PyTorch checkpoints are ignored by Git.
Do not commit datasets, participant images, or trained weights.
