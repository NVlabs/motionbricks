# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the app's background model loader and main-thread Metal inference."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from app.utils.model_loading import create_model_engine
from motionbricks.io.csv import load_csv
from motionbricks.model.config import MotionBricksInferenceConfig
from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP32_ONNX_PATH
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine


class _Viewer:
    def __init__(self):
        self.show_ui = True
        self.loading_text = []

    def show_loading_splash(self, text):
        self.loading_text.append(text)

    def hide_loading_splash(self):
        pass

    def begin_frame(self, time):
        pass

    def end_frame(self):
        pass

    def is_running(self):
        return True


@unittest.skipUnless(os.environ.get("MOTIONBRICKS_MPS_APP_TESTS") == "1", "set MOTIONBRICKS_MPS_APP_TESTS=1")
class RealMetalAppLoaderTests(unittest.TestCase):
    def check_worker_load_then_main_thread_generation(self, provider):
        wp.init()
        viewer = _Viewer()
        clip_path = (Path(__file__).resolve().parents[2] / "assets/motions/unitree_g1/locomotion/"
                     "bs_1d_linear/Neutral_walk_forward_002__A057.csv")
        with wp.ScopedDevice("cpu"), patch.dict(os.environ, {"MOTIONBRICKS_ONNX_PROVIDER": provider}):
            clip = load_csv(clip_path, device="cpu")
            with create_model_engine(viewer, MotionBricksUnitreeG1Engine, UNITREE_G1_N1_FP32_ONNX_PATH) as engine:
                self.assertTrue(viewer.show_ui)
                self.assertTrue(viewer.loading_text)
                self.assertIsNone(engine.stream)
                self.assertIsNotNone(engine._mps_model)
                context, target, output = engine.create_input(), engine.create_input(), engine.create_output()
                a, b = wp.empty_like(context.poses), wp.empty_like(target.poses)
                for i in range(4):
                    clip.sample(0.9 + i / 30.0, a[i])
                    clip.sample(1.8 + i / 30.0, b[i])
                context.update_from_poses(a)
                target.update_from_poses(b)
                engine.generate_animation(context, target, output,
                                          config=MotionBricksInferenceConfig(allowed_token_counts=(6,)))
                self.assertEqual(output.num_frames, 18)
                self.assertTrue(np.isfinite(output.poses.numpy()[:output.num_frames]).all())

    def test_explicit_metal(self):
        self.check_worker_load_then_main_thread_generation("mps")

    def test_automatic_metal(self):
        self.check_worker_load_then_main_thread_generation("auto")
