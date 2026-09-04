import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIRECTORY = REPOSITORY_ROOT / "calibracion_extrinseca"
GENERATOR_PATH = CALIBRATION_DIRECTORY / "generate_extrinsics_no_mirror.py"
CALIBRATION_PATH = CALIBRATION_DIRECTORY / "extrinsics_pro_to_lr_no_mirror.json"

module_spec = importlib.util.spec_from_file_location(
    "generate_extrinsics_no_mirror", GENERATOR_PATH
)
generator = importlib.util.module_from_spec(module_spec)
assert module_spec.loader is not None
module_spec.loader.exec_module(generator)


class CalibrationRecomputationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CALIBRATION_PATH.open("r", encoding="utf-8") as stream:
            cls.original = json.load(stream)
        cls.recomputed = generator.recompute_payload(cls.original)

    def test_stored_observations_reproduce_the_published_transform(self):
        np.testing.assert_allclose(
            self.recomputed["R"], self.original["R"], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            self.recomputed["T"], self.original["T"], rtol=0.0, atol=1e-12
        )
        self.assertAlmostEqual(
            self.recomputed["rms_mm"], self.original["rms_mm"], delta=1e-9
        )

    def test_rotation_is_proper_and_orthonormal(self):
        rotation = np.asarray(self.recomputed["R"])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)

    def test_observation_counts_are_preserved(self):
        self.assertEqual(self.recomputed["n_pro"], 5)
        self.assertEqual(self.recomputed["n_lr"], 5)


if __name__ == "__main__":
    unittest.main()
