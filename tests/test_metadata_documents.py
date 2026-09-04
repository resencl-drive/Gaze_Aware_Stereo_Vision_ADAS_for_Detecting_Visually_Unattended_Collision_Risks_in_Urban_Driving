import json
from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ModelManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest_path = REPOSITORY_ROOT / "models" / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            cls.manifest = json.load(stream)

    def test_required_model_set_is_complete(self):
        identifiers = {model["id"] for model in self.manifest["models"]}
        self.assertEqual(
            identifiers,
            {
                "gaze-resnet101-v2",
                "yolov8n-seg-coco",
                "face-detection-retail-0004",
                "facial-landmarks-35-adas-0002",
            },
        )
        self.assertTrue(all(model["required"] for model in self.manifest["models"]))

    def test_fingerprints_and_sizes_are_well_formed(self):
        for model in self.manifest["models"]:
            with self.subTest(model=model["id"]):
                self.assertRegex(model["sha256"], re.compile(r"^[0-9a-f]{64}$"))
                self.assertGreater(model["artifact_size_bytes"], 0)

    def test_default_paths_are_ignored_by_git_policy(self):
        ignored_suffixes = {".pt", ".pth", ".blob", ".onnx", ".engine"}
        for model in self.manifest["models"]:
            with self.subTest(model=model["id"]):
                path = Path(model["default_path"])
                self.assertEqual(path.parts[0], "models")
                self.assertIn(path.suffix, ignored_suffixes)

    def test_gaze_training_architecture_is_explicit(self):
        gaze_model = next(
            model
            for model in self.manifest["models"]
            if model["id"] == "gaze-resnet101-v2"
        )
        self.assertEqual(
            gaze_model["training_architecture"],
            {
                "dropout_1": 0.30,
                "dropout_2": 0.15,
                "activation": "ReLU",
                "normalization": "LayerNorm",
                "output_dimensions": 2,
                "output_order": ["pitch_rad", "yaw_rad"],
            },
        )


if __name__ == "__main__":
    unittest.main()
