import unittest

import torch.nn as nn

from gaze_models import get_gaze_model


class GazeModelArchitectureTests(unittest.TestCase):
    def test_dropout_matches_training_architecture(self):
        model = get_gaze_model()
        probabilities = [
            layer.p for layer in model.fc if isinstance(layer, nn.Dropout)
        ]
        self.assertEqual(probabilities, [0.30, 0.15])


if __name__ == "__main__":
    unittest.main()
