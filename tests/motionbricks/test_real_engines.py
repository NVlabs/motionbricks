# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real TensorRT engines generate sane motion from the bundled models.

Each generation's length must come from the token counts its config allows.

These tests load the SOMA and G1 engines in FP32 and FP16. They need an NVIDIA GPU.
The first run builds the engines into .trt-cache, which takes several minutes.
Set the MOTIONBRICKS_ENGINE_TESTS environment variable to 1 to run them.
Run them after changing the ONNX models, the decode kernels or TensorRT.
"""

import os
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from motionbricks.inference.options import TensorRTBuildOptions
from motionbricks.io.bvh import load_bvh
from motionbricks.io.csv import load_csv
from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.model.config import MotionBricksInferenceConfig
from motionbricks.model.contract import valid_frame_count
from motionbricks.model.soma.contract import SOMA_N1_CONTRACT, SOMA_N1_FP16_ONNX_PATH, SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP16_ONNX_PATH, UNITREE_G1_N1_FP32_ONNX_PATH
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter


_REPO = Path(__file__).resolve().parents[2]
_WALK = "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057"
# The first context ends 1 s into the walk. Each generation targets the walk poses at the next time.
_CONTEXT_END = 1.0
_TARGET_TIMES = (1.6, 3.2)


def _clip_window(clip, poses, start, to_runtime):
    """Sample four clip poses at 30 Hz from start, into an array shaped like poses."""
    window = wp.empty((4, poses.shape[1]), dtype=poses.dtype, device=poses.device)
    for index in range(4):
        clip.sample(start + index / 30.0, window[index], root_offset=to_runtime)
    return window


def _degrees_between(a, b):
    return np.degrees((Rotation.from_quat(a).inv() * Rotation.from_quat(b)).magnitude())


@unittest.skipUnless(os.environ.get("MOTIONBRICKS_ENGINE_TESTS") == "1", "set MOTIONBRICKS_ENGINE_TESTS=1 to run")
class RealEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cuda:0" if wp.is_cuda_available() else "cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def check_walk(self, engine, clip, to_runtime, context, target, output, poses, errors):
        """Generate along the walk without endpoint correction.

        The first generation starts from clip poses. Each later one starts from the last four
        generated poses, the way the demos chain them. poses is the output's pose array.
        Each foot's contacts must reach high and low values along the walk.
        """
        config = MotionBricksInferenceConfig(correct_generated_diff=False)
        contacts = []
        with wp.ScopedStream(engine.stream):
            context_poses = _clip_window(clip, poses, _CONTEXT_END - 3 / 30.0, to_runtime)
            for target_time in _TARGET_TIMES:
                target_poses = _clip_window(clip, poses, target_time, to_runtime)
                context.update_from_poses(context_poses, stream=engine.stream)
                target.update_from_poses(target_poses, stream=engine.stream)
                engine.generate_animation(context, target, output, config=config)
                raw = poses.numpy()[:output.num_frames]
                self.assert_walk(raw, context_poses.numpy()[-1], target_poses.numpy()[0], errors)
                contacts.append(output.foot_contacts.numpy()[:output.num_frames])
                context_poses = wp.clone(poses[output.num_frames - 4:output.num_frames])
        contacts = np.concatenate(contacts)
        self.assertTrue(((contacts >= 0.0) & (contacts <= 1.0)).all())
        for side in (contacts[:, :2], contacts[:, 2:]):
            self.assertGreater(side.max(), 0.8)
            self.assertLess(side.min(), 0.2)

    def assert_walk(self, raw, last_context, first_target, errors):
        """errors(a, b) gives the root distance in meters and each joint's turn in degrees between two poses."""
        self.assertTrue(np.isfinite(raw).all())
        # A wrong decode would leave the raw motion far from the poses it was generated from.
        for pose, end in ((raw[0], last_context), (raw[-1], first_target)):
            distance, turns = errors(pose, end)
            self.assertLess(distance, 0.05)
            self.assertLess(turns.mean(), 10.0)
        for previous, pose in zip(raw[:-1], raw[1:]):
            distance, turns = errors(previous, pose)
            self.assertLess(distance, 0.15)
            self.assertLess(turns.max(), 45.0)

    def check_token_counts(self, engine, clip, to_runtime, context, target, output, poses):
        """Each generation's length must come from the allowed token counts in its config.

        One allowed count forces the length. Several let the model choose among them.
        """
        with wp.ScopedStream(engine.stream):
            context_poses = _clip_window(clip, poses, _CONTEXT_END - 3 / 30.0, to_runtime)
            context.update_from_poses(context_poses, stream=engine.stream)
            target.update_from_poses(_clip_window(clip, poses, _TARGET_TIMES[0], to_runtime), stream=engine.stream)
            for allowed in ((6,), (11,), (16,), (12, 13), (7, 15)):
                with self.subTest(allowed=allowed):
                    config = MotionBricksInferenceConfig(allowed_token_counts=allowed)
                    engine.generate_animation(context, target, output, config=config)
                    lengths = [valid_frame_count(count) for count in allowed]
                    self.assertIn(output.num_frames, lengths)

    def soma_engine(self, fp16):
        """Return the SOMA engine, followed by the other arguments of check_walk."""
        skeleton = load_skeleton_from_usd(
            get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH,
        )
        clip = load_bvh(str(_REPO / f"assets/motions/soma/{_WALK}.bvh"), skeleton)
        to_runtime = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
        # The skeleton also has finger joints, which the model does not predict.
        joints = [skeleton.joint_names.index(name) for name in SOMA_N1_CONTRACT.joint_names]

        def errors(a, b):
            return np.linalg.norm(a[0, :3] - b[0, :3]), _degrees_between(a[joints, 3:], b[joints, 3:])

        engine = MotionBricksSOMAEngine(
            SOMA_N1_FP16_ONNX_PATH if fp16 else SOMA_N1_FP32_ONNX_PATH, model_to_runtime=to_runtime,
            build_options=TensorRTBuildOptions(enable_fp16=fp16), cache_dir=_REPO / ".trt-cache",
        )
        self.addCleanup(engine.close)
        context, target = engine.create_input(skeleton), engine.create_input(skeleton)
        output = engine.create_output(skeleton)
        return engine, clip, to_runtime, context, target, output, output.poses, errors

    def g1_engine(self, fp16):
        """Return the G1 engine, followed by the other arguments of check_walk."""
        clip = load_csv(_REPO / f"assets/motions/unitree_g1/{_WALK}.csv", device="cuda:0")
        to_runtime = wp.transform_identity()

        def errors(a, b):
            # A G1 pose holds the root position and XYZW rotation, then one hinge angle per joint.
            turns = np.append(_degrees_between(a[3:7], b[3:7]), np.degrees(np.abs(a[7:] - b[7:])))
            return np.linalg.norm(a[:3] - b[:3]), turns

        engine = MotionBricksUnitreeG1Engine(
            UNITREE_G1_N1_FP16_ONNX_PATH if fp16 else UNITREE_G1_N1_FP32_ONNX_PATH,
            build_options=TensorRTBuildOptions(enable_fp16=fp16), cache_dir=_REPO / ".trt-cache",
        )
        self.addCleanup(engine.close)
        context, target, output = engine.create_input(), engine.create_input(), engine.create_output()
        return engine, clip, to_runtime, context, target, output, output.poses, errors

    def test_soma_chains_walks_between_clip_poses(self):
        for fp16 in (False, True):
            with self.subTest(fp16=fp16):
                self.check_walk(*self.soma_engine(fp16))

    def test_g1_chains_walks_between_clip_poses(self):
        for fp16 in (False, True):
            with self.subTest(fp16=fp16):
                self.check_walk(*self.g1_engine(fp16))

    def test_soma_generates_only_allowed_token_counts(self):
        for fp16 in (False, True):
            with self.subTest(fp16=fp16):
                self.check_token_counts(*self.soma_engine(fp16)[:-1])

    def test_g1_generates_only_allowed_token_counts(self):
        for fp16 in (False, True):
            with self.subTest(fp16=fp16):
                self.check_token_counts(*self.g1_engine(fp16)[:-1])


if __name__ == "__main__":
    unittest.main()
