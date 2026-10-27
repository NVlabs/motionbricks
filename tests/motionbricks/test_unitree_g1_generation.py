# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 generation behavior with synthetic poses and controlled predictor output."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from motionbricks.inference.options import TensorRTBuildOptions
from motionbricks.inference.errors import TensorContractError
from motionbricks.model.config import MotionBricksInferenceConfig
from motionbricks.model.kernels import write_motionbricks_conditions_kernel
from motionbricks.model.unitree_g1 import kernels
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer


def _poses(count):
    frames = np.arange(count, dtype=np.float32)
    poses = np.zeros((count, 36), dtype=np.float32)
    poses[:, :3] = np.column_stack((10 + frames * .02, -7 + frames * .01, .8 + frames * .002))
    angles = np.column_stack((frames * .01 - .1, frames * .005 + .2, frames * .03 + 1.4))
    poses[:, 3:7] = Rotation.from_euler("xyz", angles).as_quat()
    poses[:, 7:] = frames[:, None] * .013 + np.arange(29)[None] * .025
    return poses


def _assert_poses(test, actual, expected, atol=3e-6):
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=atol, rtol=0)
    np.testing.assert_allclose(actual[:, 7:], expected[:, 7:], atol=atol, rtol=0)
    delta = Rotation.from_quat(actual[:, 3:7]).inv() * Rotation.from_quat(expected[:, 3:7])
    test.assertLess(float(delta.magnitude().max()), atol)


class G1GenerationKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.previous_device = wp.get_device()
        wp.set_device("cpu")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def test_alignment_preserves_height_tilt_and_hinges(self):
        source = _poses(4)
        # Quaternion sign and length must not change the represented heading.
        source[:, 3:7] *= np.array([-1.03, 1.03, -1.03, -1.03])[:, None]
        source_device = wp.array(source, dtype=wp.float32)
        alignment = wp.empty(1, dtype=wp.transform)
        encoded = wp.empty((1, 4, 36), dtype=wp.float32)
        root_matrix = Rotation.from_quat(source[-1, 3:7]).as_matrix()
        heading = np.arctan2(root_matrix[1, 0], root_matrix[0, 0])
        origin = source[-1, :3] * [1, 1, 0]

        for desired in ((0., 1.), (1., 0.), (-3., 2.)):
            with self.subTest(desired=desired):
                wp.launch(kernels.compute_guide_alignment_kernel, dim=1,
                          inputs=[wp.vec2(*desired), wp.transform_identity(), source_device], outputs=[alignment])
                wp.launch(kernels.encode_motionbricks_inputs_kernel, dim=4,
                          inputs=[wp.transform_identity(), source_device, alignment], outputs=[encoded[0]])
                actual = encoded.numpy()[0]
                rotation = Rotation.from_euler("z", np.arctan2(desired[1], desired[0]) - heading)
                expected = source.copy()
                expected[:, :3] = rotation.apply(source[:, :3] - origin)
                expected[:, 3:7] = (rotation * Rotation.from_quat(source[:, 3:7])).as_quat()
                _assert_poses(self, actual, expected)
                np.testing.assert_array_equal(actual[:, 2], source[:, 2])
                np.testing.assert_array_equal(actual[:, 7:], source[:, 7:])
                np.testing.assert_allclose(actual[-1, :2], 0, atol=1e-7)
                np.testing.assert_allclose(np.linalg.norm(actual[:, 3:7], axis=1), 1, atol=1e-7)
        np.testing.assert_array_equal(source_device.numpy(), source)

    def test_world_restoration_trims_only_valid_frames(self):
        poses = _poses(58)
        world_rotation = Rotation.from_euler("z", -2.1)
        origin = np.array([103., -71., 0.], dtype=np.float32)
        align = wp.transform(wp.vec3(*origin), wp.quat(*world_rotation.as_quat()))
        alignment = wp.array([align], dtype=wp.transform)
        local = poses.copy()
        local[:, :3] = world_rotation.inv().apply(poses[:, :3] - origin)
        local[:, 3:7] = (world_rotation.inv() * Rotation.from_quat(poses[:, 3:7])).as_quat()
        output = wp.full((61, 36), 999., dtype=wp.float32)
        output_count = wp.empty(1, dtype=wp.int32)
        pointer = output.ptr
        unfiltered = wp.array([1.0], dtype=wp.float32)
        raw_contacts, output_contacts = wp.zeros((1, 64, 4), dtype=wp.float32), wp.empty((61, 4), dtype=wp.float32)

        for tokens in (16, 6, 8):
            count = tokens * 4 - 6
            raw = np.full((1, 64, 36), np.nan, dtype=np.float32)
            raw[0, 3:3 + count] = local[:count]
            before = output.numpy()
            wp.launch(kernels.decode_motionbricks_output_kernel, dim=output.shape[0], inputs=[
                alignment, wp.transform_identity(), wp.array(raw[0], dtype=wp.float32), raw_contacts[0],
                wp.array([tokens], dtype=wp.int32), unfiltered,
            ], outputs=[output, output_contacts, output_count])
            actual = output.numpy()
            self.assertEqual(output_count.numpy()[0], count)
            _assert_poses(self, actual[:count], poses[:count], atol=3e-5)
            np.testing.assert_array_equal(actual[count:], before[count:])
            self.assertEqual(output.ptr, pointer)

    def test_root_filter_keeps_steady_motion_and_removes_frame_jitter(self):
        poses = _poses(64)
        world_rotation = Rotation.from_euler("z", 0.7)
        origin = np.array([-4., 9., 0.], dtype=np.float32)
        alignment = wp.array([wp.transform(wp.vec3(*origin), wp.quat(*world_rotation.as_quat()))], dtype=wp.transform)
        local = poses.copy()
        local[:, :3] = world_rotation.inv().apply(poses[:, :3] - origin)
        local[:, 3:7] = (world_rotation.inv() * Rotation.from_quat(poses[:, 3:7])).as_quat()
        # One centimetre of alternating jitter on every axis, the fastest change a 30 Hz clip can hold.
        local[:, :3] += 0.01 * (-1.0) ** np.arange(64)[:, None]
        root_filter = wp.array(kernels.ROOT_POSITION_FILTER, dtype=wp.float32)
        output = wp.empty((58, 36), dtype=wp.float32)
        output_count = wp.empty(1, dtype=wp.int32)
        raw_contacts, output_contacts = wp.zeros((1, 64, 4), dtype=wp.float32), wp.empty((58, 4), dtype=wp.float32)

        for tokens in (16, 6):
            raw = np.full((1, 64, 36), np.nan, dtype=np.float32)
            raw[0, :tokens * 4] = local[:tokens * 4]
            wp.launch(kernels.decode_motionbricks_output_kernel, dim=output.shape[0], inputs=[
                alignment, wp.transform_identity(), wp.array(raw[0], dtype=wp.float32), raw_contacts[0],
                wp.array([tokens], dtype=wp.int32), root_filter,
            ], outputs=[output, output_contacts, output_count])
            count = tokens * 4 - 6
            actual = output.numpy()[:count]
            expected = poses[3:3 + count]
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=1e-3, rtol=0)
            np.testing.assert_array_equal(actual[:, 7:], expected[:, 7:])
            delta = Rotation.from_quat(actual[:, 3:7]).inv() * Rotation.from_quat(expected[:, 3:7])
            self.assertLess(float(delta.magnitude().max()), 3e-6)

    def test_masks_clear_stale_targets_and_unknown_derivatives(self):
        masks = [wp.full((1, 8), -1, dtype=wp.int32) for _ in range(3)]
        tokens = wp.full((1, 11), -1, dtype=wp.int32)
        for target_count, allowed in ((4, (6, 9, 16)), (1, (8,)), (3, (7, 14)), (2, (6, 16))):
            config = MotionBricksInferenceConfig(allowed_token_counts=allowed)
            wp.launch(write_motionbricks_conditions_kernel, dim=11,
                      inputs=[target_count, config.allowed_token_bits()], outputs=[*masks, tokens])
            expected = np.array([1] * (4 + target_count) + [0] * (4 - target_count))
            np.testing.assert_array_equal(masks[0].numpy()[0], expected)
            np.testing.assert_array_equal(masks[2].numpy()[0], expected)
            expected[[3, 4 + target_count - 1]] = 0
            np.testing.assert_array_equal(masks[1].numpy()[0], expected)
            np.testing.assert_array_equal(tokens.numpy()[0], [int(n in allowed) for n in range(6, 17)])


class _ControlledG1Engine(MotionBricksUnitreeG1Engine):
    """Exercise the real generation path, replacing only the TensorRT backend."""

    def _initialize(self, build_options, *, cache_dir, force_rebuild, progress_monitor):
        self.build_options = build_options
        self._tensor_specs = {spec.name: spec for spec in self.contract.tensor_specs}
        self._buffers = {
            spec.name: wp.empty(spec.shape, dtype=wp.float32 if spec.dtype == np.dtype(np.float32) else wp.int32,
                                device=self.device)
            for spec in self.contract.tensor_specs
        }
        self.prediction = wp.array(_poses(64)[None], dtype=wp.float32, device=self.device)
        self.contacts = wp.zeros((1, 64, 4), dtype=wp.float32, device=self.device)
        self.tokens = 8

    def _enqueue_tensorrt(self):
        with wp.ScopedStream(self.stream):
            wp.copy(self.device_tensor("pred_qpos"), self.prediction)
            wp.copy(self.device_tensor("pred_foot_contacts"), self.contacts)
            self.device_tensor("pred_num_tokens").assign([self.tokens])


class G1GenerationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is unavailable")
        cls.previous_device = wp.get_device()
        wp.set_device("cuda:0")

    @classmethod
    def tearDownClass(cls):
        wp.set_device(cls.previous_device)

    def setUp(self):
        self.engine = _ControlledG1Engine("controlled.onnx", bake_cuda_graph=False)
        self.addCleanup(self.engine.close)
        self.context = self.engine.create_input()
        self.target = self.engine.create_input()
        self.output = self.engine.create_output()
        with wp.ScopedStream(self.engine.stream):
            self.source = wp.array(_poses(4), dtype=wp.float32)
            self.context.update_from_poses(self.source, stream=self.engine.stream)
            self.target.update_from_poses(self.source, stream=self.engine.stream)

    def test_tf32_is_off_by_default_and_explicit_options_win(self):
        self.assertIs(self.engine.build_options.enable_tf32, False)
        for options, expected in (
            (TensorRTBuildOptions(enable_fp16=True), False),
            (TensorRTBuildOptions(enable_tf32=True), True),
        ):
            with self.subTest(options=options):
                engine = _ControlledG1Engine("controlled.onnx", build_options=options, bake_cuda_graph=False)
                self.addCleanup(engine.close)
                self.assertIs(engine.build_options.enable_fp16, options.enable_fp16)
                self.assertIs(engine.build_options.enable_tf32, expected)

    def test_default_and_explicit_alignment_set_the_encoded_heading(self):
        for config, heading in ((None, 0.0), (MotionBricksInferenceConfig(heading_alignment=(0., 1.)), np.pi / 2)):
            self.engine.generate_animation(self.context, self.target, self.output, config=config)
            last_context = self.engine.device_tensor("context_qpos").numpy()[0, -1]
            forward = Rotation.from_quat(last_context[3:7]).as_matrix()[:, 0]
            self.assertAlmostEqual(np.arctan2(forward[1], forward[0]), heading, places=6)

    def test_model_to_runtime_moves_the_same_motion_into_world_space(self):
        # A Y up world, turned and moved away from the origin.
        rotation = Rotation.from_euler("xz", (90, 40), degrees=True)
        to_world = wp.transform(wp.vec3(3.0, -2.0, 0.5), wp.quat(*rotation.as_quat()))
        engine = _ControlledG1Engine("controlled.onnx", model_to_runtime=to_world, bake_cuda_graph=False)
        self.addCleanup(engine.close)
        context, target, output = engine.create_input(), engine.create_input(), engine.create_output()
        world = _poses(4)
        for frame in world:
            frame[:7] = np.asarray(wp.mul(to_world, wp.transform(wp.vec3(*frame[:3]), wp.quat(*frame[3:7]))))
        with wp.ScopedStream(engine.stream):
            world_device = wp.array(world, dtype=wp.float32)
            context.update_from_poses(world_device, stream=engine.stream)
            target.update_from_poses(world_device, stream=engine.stream)
        config = MotionBricksInferenceConfig(allowed_token_counts=(8,), correct_generated_diff=False)
        self.engine.generate_animation(self.context, self.target, self.output, config=config)
        engine.generate_animation(context, target, output, config=config)

        # Both engines see the same model space input.
        for name in ("context_qpos", "target_qpos"):
            _assert_poses(
                self, engine.device_tensor(name).numpy()[0], self.engine.device_tensor(name).numpy()[0], atol=1e-5,
            )
        expected = self.output.poses.numpy()[:self.output.num_frames].copy()
        for frame in expected:
            frame[:7] = np.asarray(wp.mul(to_world, wp.transform(wp.vec3(*frame[:3]), wp.quat(*frame[3:7]))))
        self.assertEqual(output.num_frames, self.output.num_frames)
        _assert_poses(self, output.poses.numpy()[:output.num_frames], expected, atol=1e-5)

    def test_repeated_generation_trims_padding_into_the_same_output(self):
        engine = self.engine
        pointer = self.output.poses.ptr
        for count in (16, 6, 8, 14):
            engine.tokens = count
            result = engine.generate_animation(self.context, self.target, self.output,
                config=MotionBricksInferenceConfig(allowed_token_counts=(count,), correct_generated_diff=False))
            self.assertIs(result, self.output)
            self.assertEqual(result.poses.ptr, pointer)
            self.assertEqual(result.num_frames, count * 4 - 6)
            self.assertAlmostEqual(result.duration, (count * 4 - 7) / 30)

    def test_generation_copies_each_output_frames_foot_contacts_clamped(self):
        contacts = np.arange(64 * 4, dtype=np.float32).reshape(1, 64, 4) / 128.0 - 0.5
        self.engine.contacts = wp.array(contacts, dtype=wp.float32, device=self.engine.device)
        self.engine.generate_animation(self.context, self.target, self.output,
            config=MotionBricksInferenceConfig(allowed_token_counts=(8,), correct_generated_diff=False))
        frames = self.output.num_frames
        # Output frame 0 is the raw frame after the three padding frames.
        np.testing.assert_array_equal(
            self.output.foot_contacts.numpy()[:frames], np.clip(contacts[0, 3:3 + frames], 0, 1),
        )

    def test_invalid_model_lengths_do_not_publish_output(self):
        for correction in (False, True):
            config = MotionBricksInferenceConfig(allowed_token_counts=(8,), correct_generated_diff=correction)
            for tokens in (5, 17, 7):
                with self.subTest(tokens=tokens, correction=correction):
                    self.output.num_frames = 26
                    self.engine.tokens = tokens
                    with self.assertRaises(TensorContractError):
                        self.engine.generate_animation(self.context, self.target, self.output, config=config)
                    self.assertEqual(self.output.num_frames, 0)

    def test_uninitialized_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Target token"):
            self.engine.generate_animation(self.context, self.engine.create_input(), self.output)

    def test_output_without_foot_contacts_is_rejected(self):
        output = DOFAnimationBuffer(self.output.joint_names, 30.0, wp.empty_like(self.output.poses), num_frames=0)
        with self.assertRaisesRegex(ValueError, "foot_contacts"):
            self.engine.generate_animation(self.context, self.target, output)

    def test_correction_matches_endpoints_and_can_be_disabled_on_the_next_call(self):
        engine = self.engine
        for count, target_count in ((16, 1), (6, 4), (8, 2)):
            engine.tokens = count
            self.target.update_from_poses(self.source[:target_count], stream=engine.stream)
            engine.generate_animation(self.context, self.target, self.output,
                config=MotionBricksInferenceConfig(allowed_token_counts=(count,), correct_generated_diff=False))
            raw = self.output.poses.numpy()[:self.output.num_frames].copy()
            engine.generate_animation(self.context, self.target, self.output,
                config=MotionBricksInferenceConfig(allowed_token_counts=(count,), correct_generated_diff=True))
            actual = self.output.poses.numpy()[:self.output.num_frames].copy()
            expected = np.stack((self.context.poses.numpy()[-1], self.target.poses.numpy()[0]))
            _assert_poses(self, actual[[0, -1]], expected, atol=1e-5)
            self.assertTrue(np.isfinite(actual).all())
            engine.generate_animation(self.context, self.target, self.output,
                config=MotionBricksInferenceConfig(allowed_token_counts=(count,), correct_generated_diff=False))
            np.testing.assert_array_equal(self.output.poses.numpy()[:self.output.num_frames], raw)
            self.assertEqual(engine.timer.milliseconds["post_process"], 0)


if __name__ == "__main__":
    unittest.main()
