# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare released FP32 models on CPU and Metal using actual walking poses.

Run on Apple Silicon after downloading both checkpoints:
    MOTIONBRICKS_MPS_PARITY_TESTS=1 .venv/bin/python -m unittest tests.motionbricks.test_real_mps_parity
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from docs.examples._rig import load_rig, sample_window
from motionbricks.model.config import MotionBricksInferenceConfig


_CASES = (
    ("early walk", 0.1, 1.3, False),
    ("walk", 0.9, 1.8, False),
    ("later walk", 2.6, 3.7, False),
    ("placed target", 0.9, 1.8, True),
)
_TOKEN_COUNTS = (6, 11, 16)


@unittest.skipUnless(os.environ.get("MOTIONBRICKS_MPS_PARITY_TESTS") == "1", "set MOTIONBRICKS_MPS_PARITY_TESTS=1")
class RealMetalParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def generate(self, rig_name, provider):
        with patch.dict(os.environ, {"MOTIONBRICKS_ONNX_PROVIDER": provider}):
            rig = load_rig(rig_name)
            with rig.engine as engine:
                skeleton = (rig.skeleton,) if rig.skeleton is not None else ()
                context, target, output = engine.create_input(*skeleton), engine.create_input(*skeleton), engine.create_output(*skeleton)
                context_poses, target_poses = wp.empty_like(context.poses), wp.empty_like(target.poses)
                results = {}
                for label, start, end, placed in _CASES:
                    offset = (
                        wp.transform(wp.vec3(0.5, -0.2, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35))
                        if placed else None
                    )
                    sample_window(rig, start, context_poses)
                    sample_window(rig, end, target_poses, root_offset=offset)
                    context.update_from_poses(context_poses, stream=engine.stream)
                    target.update_from_poses(target_poses, stream=engine.stream)
                    for count in _TOKEN_COUNTS:
                        engine.generate_animation(context, target, output, config=MotionBricksInferenceConfig(allowed_token_counts=(count,)))
                        results[(label, count)] = (
                            output.poses.numpy()[:output.num_frames].copy(),
                            output.foot_contacts.numpy()[:output.num_frames].copy(),
                        )
                return results

    def check_rig(self, rig_name):
        reference = self.generate(rig_name, "cpu")
        metal = self.generate(rig_name, "mps")
        self.assertEqual(reference.keys(), metal.keys())
        for (label, count), expected in reference.items():
            with self.subTest(rig=rig_name, case=label, tokens=count):
                actual = metal[(label, count)]
                np.testing.assert_allclose(actual[0], expected[0], atol=2e-3, rtol=1e-4)
                np.testing.assert_allclose(actual[1], expected[1], atol=2e-3, rtol=1e-4)

    def test_g1(self):
        self.check_rig("unitree_g1")

    def test_soma(self):
        self.check_rig("soma")
