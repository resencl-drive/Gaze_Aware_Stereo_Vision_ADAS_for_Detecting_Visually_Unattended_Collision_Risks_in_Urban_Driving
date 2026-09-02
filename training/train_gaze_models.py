# SPDX-License-Identifier: AGPL-3.0-only
"""Train the gaze-regression models evaluated in the associated paper.

The input HDF5 files are expected to contain face crops and two gaze angles in
radians. ETH-XGaze uses the ``(pitch, yaw)`` ordering by default.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


# Hyperparameters reported in Table 2 of the paper.
PAPER_CONFIGS: dict[str, dict[str, Any]] = {
    "resnet50": {"batch_size": 64, "lr": 1e-4, "pretrained": True},
    "resnet101": {"batch_size": 24, "lr": 5e-5, "pretrained": True},
    "efficientnet_v2_s": {"batch_size": 32, "lr": 5e-5, "pretrained": False},
    "convnext_tiny": {"batch_size": 64, "lr": 5e-5, "pretrained": False},
    "mobilenet_v3_large": {"batch_size": 128, "lr": 1e-4, "pretrained": True},
}


class EarlyStopping:
    """Stop after the monitored validation metric ceases to improve."""

    def __init__(self, patience: int = 10, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_value = math.inf

    def update(self, value: float) -> bool:
        if value < self.best_value - self.delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def append_csv_row(path: Path, row: dict[str, Any], overwrite: bool = False) -> None:
    mode = "w" if overwrite or not path.exists() else "a"
    with path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)


def angles_to_unit_vector(angles: torch.Tensor, order: str) -> torch.Tensor:
    """Convert batches of pitch/yaw angles in radians to 3D unit vectors."""
    if order == "pitch_yaw":
        pitch, yaw = angles[:, 0], angles[:, 1]
    elif order == "yaw_pitch":
        yaw, pitch = angles[:, 0], angles[:, 1]
    else:
        raise ValueError("order must be 'pitch_yaw' or 'yaw_pitch'")

    vector = torch.stack(
        (
            torch.cos(pitch) * torch.sin(yaw),
            torch.sin(pitch),
            torch.cos(pitch) * torch.cos(yaw),
        ),
        dim=1,
    )
    return nn.functional.normalize(vector, dim=1)


def angular_error_deg(
    predicted: torch.Tensor, target: torch.Tensor, order: str
) -> torch.Tensor:
    predicted_vector = angles_to_unit_vector(predicted, order)
    target_vector = angles_to_unit_vector(target, order)
    cosine = (predicted_vector * target_vector).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


class AngularLoss(nn.Module):
    """Mean angular separation between predicted and target gaze vectors."""

    def __init__(self, gaze_order: str = "pitch_yaw") -> None:
        super().__init__()
        self.gaze_order = gaze_order

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_vector = angles_to_unit_vector(predicted, self.gaze_order)
        target_vector = angles_to_unit_vector(target, self.gaze_order)
        cosine = (predicted_vector * target_vector).sum(dim=1)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cosine).mean()


class CombinedGazeLoss(nn.Module):
    """Weighted Smooth L1 and angular loss used in the paper."""

    def __init__(self, alpha: float = 0.5, gaze_order: str = "pitch_yaw") -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss()
        self.angular = AngularLoss(gaze_order)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth_l1 = self.smooth_l1(predicted, target)
        angular = self.angular(predicted, target)
        return self.alpha * smooth_l1 + (1.0 - self.alpha) * angular


class XGazeH5Dataset(Dataset):
    """Read face crops and gaze labels lazily from one or more HDF5 files."""

    def __init__(
        self,
        h5_files: list[str],
        image_key: str = "face_patch",
        label_key: str = "face_gaze",
        input_color: str = "bgr",
        transform: Any = None,
    ) -> None:
        if not h5_files:
            raise ValueError("No HDF5 files were provided")
        self.h5_files = sorted(h5_files)
        self.image_key = image_key
        self.label_key = label_key
        self.input_color = input_color
        self.transform = transform
        self.cumulative_lengths: list[int] = []
        self._file_handles: dict[str, h5py.File] = {}

        total = 0
        for filename in self.h5_files:
            with h5py.File(filename, "r") as h5_file:
                if image_key not in h5_file or label_key not in h5_file:
                    raise KeyError(
                        f"{filename} must contain '{image_key}' and '{label_key}'"
                    )
                if len(h5_file[image_key]) != len(h5_file[label_key]):
                    raise ValueError(f"Image/label count mismatch in {filename}")
                total += len(h5_file[image_key])
                self.cumulative_lengths.append(total)
        self.total_length = total

    def __len__(self) -> int:
        return self.total_length

    def _location(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.total_length
        if index < 0 or index >= self.total_length:
            raise IndexError(index)
        file_index = bisect_right(self.cumulative_lengths, index)
        start = 0 if file_index == 0 else self.cumulative_lengths[file_index - 1]
        return file_index, index - start

    def _handle(self, file_index: int) -> h5py.File:
        filename = self.h5_files[file_index]
        if filename not in self._file_handles:
            self._file_handles[filename] = h5py.File(filename, "r")
        return self._file_handles[filename]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_index, local_index = self._location(index)
        h5_file = self._handle(file_index)
        image = np.asarray(h5_file[self.image_key][local_index])
        gaze = np.asarray(h5_file[self.label_key][local_index])

        if image.ndim != 3:
            raise ValueError(f"Expected a 3D image array, received shape {image.shape}")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] != 3:
            raise ValueError(f"Expected three image channels, received shape {image.shape}")
        if self.input_color == "bgr":
            image = image[:, :, ::-1]

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = np.clip(image, 0.0, 1.0) * 255.0
            else:
                image = np.clip(image, 0.0, 255.0)
            image = image.astype(np.uint8)
        pil_image = Image.fromarray(np.ascontiguousarray(image))
        tensor = self.transform(pil_image) if self.transform else transforms.ToTensor()(pil_image)
        return tensor, torch.as_tensor(gaze, dtype=torch.float32)


def make_regression_head(in_features: int, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LayerNorm(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2.0),
        nn.Linear(256, 2),
    )


def build_model(model_name: str, pretrained: bool, dropout: float) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    constructor = getattr(models, model_name)
    model = constructor(weights=weights)

    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = make_regression_head(in_features, dropout)
    else:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = make_regression_head(in_features, dropout)
    return model


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
            normalization,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ]
    )
    validation_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalization]
    )
    return train_transform, validation_transform


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    gaze_order: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_angular_error = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            angular = angular_error_deg(
                predictions.detach(), targets, gaze_order
            ).mean()
            total_loss += loss.item() * batch_size
            total_angular_error += angular.item() * batch_size
            total_samples += batch_size
            if training and batch_index % 500 == 0:
                print(
                    f"  batch={batch_index} loss={loss.item():.4f} "
                    f"angular_error={angular.item():.2f} deg"
                )

    return total_loss / total_samples, total_angular_error / total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train gaze regressors on ETH-XGaze-style HDF5 files."
    )
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--val-glob", required=True)
    parser.add_argument("--image-key", default="face_patch")
    parser.add_argument("--label-key", default="face_gaze")
    parser.add_argument("--input-color", choices=("bgr", "rgb"), default="bgr")
    parser.add_argument(
        "--gaze-order", choices=("pitch_yaw", "yaw_pitch"), default="pitch_yaw"
    )
    parser.add_argument("--model", choices=tuple(PAPER_CONFIGS), default="resnet101")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the paper configuration for ImageNet initialization.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--loss-alpha", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.data_fraction <= 1.0:
        raise ValueError("--data-fraction must be in the interval (0, 1]")

    paper_config = PAPER_CONFIGS[args.model]
    batch_size = args.batch_size or paper_config["batch_size"]
    learning_rate = args.lr or paper_config["lr"]
    pretrained = (
        paper_config["pretrained"] if args.pretrained is None else args.pretrained
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files = glob.glob(args.train_glob, recursive=True)
    validation_files = glob.glob(args.val_glob, recursive=True)
    if not train_files or not validation_files:
        raise FileNotFoundError("The training or validation glob matched no files")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model}_{timestamp}"
    run_directory = args.output_root / run_name
    run_directory.mkdir(parents=True, exist_ok=True)

    train_transform, validation_transform = make_transforms()
    dataset_options = {
        "image_key": args.image_key,
        "label_key": args.label_key,
        "input_color": args.input_color,
    }
    train_dataset: Dataset = XGazeH5Dataset(
        train_files, transform=train_transform, **dataset_options
    )
    validation_dataset = XGazeH5Dataset(
        validation_files, transform=validation_transform, **dataset_options
    )
    if args.data_fraction < 1.0:
        sample_count = max(1, int(len(train_dataset) * args.data_fraction))
        train_dataset = Subset(train_dataset, range(sample_count))

    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, **loader_options
    )

    model = build_model(args.model, pretrained, args.dropout).to(device)
    criterion = CombinedGazeLoss(args.loss_alpha, args.gaze_order)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    start_epoch = 1
    best_validation_error = math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_error = float(
            checkpoint.get("best_validation_angular_error_deg", math.inf)
        )

    configuration = vars(args).copy()
    configuration.update(
        {
            "batch_size": batch_size,
            "lr": learning_rate,
            "pretrained": pretrained,
            "device": str(device),
            "train_file_count": len(train_files),
            "validation_file_count": len(validation_files),
            "output_root": str(args.output_root),
            "resume": str(args.resume) if args.resume else None,
        }
    )
    save_json(run_directory / "config.json", configuration)

    print(f"Run: {run_name}")
    print(
        f"Model: {args.model}; device: {device}; batch: {batch_size}; "
        f"learning rate: {learning_rate:g}; ImageNet initialization: {pretrained}"
    )
    early_stopping = EarlyStopping(args.patience)
    early_stopping.best_value = best_validation_error
    metrics_path = run_directory / "metrics.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_loss, train_error = run_epoch(
            model, train_loader, criterion, device, args.gaze_order, optimizer
        )
        validation_loss, validation_error = run_epoch(
            model, validation_loader, criterion, device, args.gaze_order
        )
        scheduler.step()

        improved = validation_error < best_validation_error
        if improved:
            best_validation_error = validation_error
            torch.save(model.state_dict(), run_directory / "best.pt")

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_angular_error_deg": best_validation_error,
            "config": configuration,
        }
        torch.save(checkpoint, run_directory / "last.pt")

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_loss, 6),
            "train_angular_error_deg": round(train_error, 6),
            "validation_loss": round(validation_loss, 6),
            "validation_angular_error_deg": round(validation_error, 6),
            "epoch_time_seconds": round(time.time() - started, 2),
        }
        append_csv_row(
            metrics_path, row, overwrite=epoch == 1 and args.resume is None
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs}: val_loss={validation_loss:.4f}, "
            f"val_angular_error={validation_error:.4f} deg"
        )
        if early_stopping.update(validation_error):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Training complete. Best validation error: {best_validation_error:.4f} deg")


if __name__ == "__main__":
    main()
