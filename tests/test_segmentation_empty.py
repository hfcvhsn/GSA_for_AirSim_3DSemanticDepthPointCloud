import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.gsa import GroundedMobileSAM


class SegmentationEmptyTests(unittest.TestCase):
    def test_no_detections_produces_unknown_mask(self):
        pipeline = GroundedMobileSAM.__new__(GroundedMobileSAM)
        pipeline.official = SimpleNamespace(load_image=lambda path: (Image.new("RGB", (8, 6)), torch.zeros(3, 6, 8)))
        pipeline.device, pipeline.caption = "cpu", "tree ."
        pipeline.token_groups = [[1]]
        pipeline.settings = {"box_threshold": .25, "text_threshold": .25, "nms_threshold": .5}
        pipeline.model = lambda images, captions: {"pred_logits": torch.full((1, 2, 4), -20.),
                                                   "pred_boxes": torch.ones(1, 2, 4) * .5}
        rgb, semantic, instance, confidence, records = pipeline.segment("unused")
        self.assertEqual(rgb.shape, (6, 8, 3))
        self.assertEqual(semantic.dtype, np.uint16)
        self.assertFalse(semantic.any())
        self.assertFalse(instance.any())
        self.assertFalse(confidence.any())
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
