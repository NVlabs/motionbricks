# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Metal backend preserves the MotionBricks tensor contract and ownership."""

import unittest
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import warp as wp

try:
    from motionbricks.inference.onnx_engine import InferenceEngine
except ImportError:
    InferenceEngine = None

from motionbricks.inference.errors import EngineClosedError, EngineExecutionError, TensorContractError
from motionbricks.inference.options import TensorRTBuildOptions
from motionbricks.inference.types import TensorMode, TensorSpec


@unittest.skipIf(InferenceEngine is None, "ONNX Runtime is not installed")
class MetalBackendTests(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = wp.ScopedDevice("cpu")
        self.device.__enter__()
        self.addCleanup(self.device.__exit__, None, None, None)
        self.specs = (
            TensorSpec("input", TensorMode.INPUT, (1, 2), np.dtype("float32")),
            TensorSpec("out_b", TensorMode.OUTPUT, (1, 2), np.dtype("float32")),
            TensorSpec("out_a", TensorMode.OUTPUT, (1, 2), np.dtype("float32")),
        )

    def test_named_outputs_and_cpu_buffers_are_copied_and_close(self):
        def execute(model, inputs, outputs, arrays):
            np.testing.assert_array_equal(arrays["input"], [[1, 2]])
            return {"out_a": arrays["input"] + 10, "out_b": arrays["input"] + 20}

        with patch.dict("os.environ", {"MOTIONBRICKS_ONNX_PROVIDER": "mps"}), \
             patch("motionbricks.inference.mps.load_model", return_value=(object(), ("input",), ("out_a", "out_b"))), \
             patch("motionbricks.inference.mps.run_model", side_effect=execute):
            engine = InferenceEngine("bundled.onnx", expected_tensor_specs=self.specs)
            wp.copy(engine.device_tensor("input"), wp.array([[1, 2]], dtype=wp.float32, device="cpu"))
            engine.enqueue_async()
            np.testing.assert_array_equal(engine.device_tensor("out_a").numpy(), [[11, 12]])
            np.testing.assert_array_equal(engine.device_tensor("out_b").numpy(), [[21, 22]])
            engine.close()
            engine.close()
            with self.assertRaises(EngineClosedError):
                engine.enqueue_async()

    def test_rejects_mismatched_contract_and_nonunit_batch(self):
        with patch.dict("os.environ", {"MOTIONBRICKS_ONNX_PROVIDER": "mps"}), \
             patch("motionbricks.inference.mps.load_model", return_value=(object(), ("other",), ("out_a", "out_b"))):
            with self.assertRaisesRegex(TensorContractError, "tensor contract"):
                InferenceEngine("bundled.onnx", expected_tensor_specs=self.specs)
            with self.assertRaisesRegex(ValueError, "batch size of one"):
                InferenceEngine("bundled.onnx", expected_tensor_specs=self.specs,
                                build_options=TensorRTBuildOptions(batch_size=2))

    def test_execution_error_identifies_metal_backend(self):
        with patch.dict("os.environ", {"MOTIONBRICKS_ONNX_PROVIDER": "mps"}), \
             patch("motionbricks.inference.mps.load_model", return_value=(object(), ("input",), ("out_a", "out_b"))), \
             patch("motionbricks.inference.mps.run_model", side_effect=RuntimeError("device failed")):
            with InferenceEngine("bundled.onnx", expected_tensor_specs=self.specs) as engine:
                with self.assertRaisesRegex(EngineExecutionError, "Metal inference failed: device failed"):
                    engine.enqueue_async()

    def test_auto_selection_uses_metal_when_ready(self):
        with patch.dict("os.environ", {"MOTIONBRICKS_ONNX_PROVIDER": "auto"}), \
             patch("motionbricks.inference.mps.is_available_for_model", return_value=True), \
             patch("motionbricks.inference.mps.load_model", return_value=(object(), ("input",), ("out_a", "out_b"))) as load:
            with InferenceEngine("bundled.onnx", expected_tensor_specs=self.specs) as engine:
                self.assertIsNotNone(engine._mps_model)
                self.assertEqual(engine.provider, "mps")
            load.assert_called_once_with("bundled.onnx")

    def test_auto_selection_uses_cpu_when_metal_is_not_ready(self):
        with patch.dict("os.environ", {"MOTIONBRICKS_ONNX_PROVIDER": "auto"}), \
             patch("motionbricks.inference.mps.is_available_for_model", return_value=False), \
             patch("motionbricks.inference.onnx_engine.ort.InferenceSession") as session:
            session.return_value.get_inputs.return_value = []
            session.return_value.get_outputs.return_value = []
            with InferenceEngine("bundled.onnx", expected_tensor_specs=self.specs) as engine:
                self.assertIsNone(engine._mps_model)
                self.assertEqual(engine.provider, "cpu")
            self.assertEqual(session.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                         "Metal dependencies are not installed")
    def test_auto_selection_requires_complete_checkpoint(self):
        from motionbricks.inference.mps import is_available_for_model
        from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP32_ONNX_PATH
        from tools.onnx.checksums import CHECKPOINTS

        with TemporaryDirectory() as directory, \
             patch.dict("os.environ", {"MOTIONBRICKS_CHECKPOINT_DIR": directory}), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertFalse(is_available_for_model(UNITREE_G1_N1_FP32_ONNX_PATH))
            release = CHECKPOINTS["g1"]
            root = Path(directory) / "g1" / release["root_directory"]
            for entry in release["files"]:
                path = root / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertTrue(is_available_for_model(UNITREE_G1_N1_FP32_ONNX_PATH))
            (root / release["files"][-1]["path"]).unlink()
            self.assertFalse(is_available_for_model(UNITREE_G1_N1_FP32_ONNX_PATH))

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                         "Metal dependencies are not installed")
    def test_replaced_bundled_onnx_is_not_silently_ignored(self):
        from motionbricks.inference.mps import load_model
        from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP32_ONNX_PATH

        with patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.backends.mps.get_name", return_value="test GPU"), \
             patch("tools.onnx.checkpoints.sha256", return_value="0" * 64), \
             patch("tools.onnx.checkpoints.load_sources") as load_sources:
            with self.assertRaisesRegex(ValueError, "does not match the release"):
                load_model(UNITREE_G1_N1_FP32_ONNX_PATH)
            load_sources.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                         "Metal dependencies are not installed")
    def test_fp16_model_is_rejected_instead_of_silently_running_fp32(self):
        from motionbricks.inference.mps import load_model
        from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP16_ONNX_PATH

        with patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.backends.mps.get_name", return_value="test GPU"):
            with self.assertRaisesRegex(ValueError, "bundled FP32"):
                load_model(UNITREE_G1_N1_FP16_ONNX_PATH)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                         "Metal dependencies are not installed")
    def test_explicit_metal_reports_missing_checkpoints(self):
        from motionbricks.inference.mps import load_model
        from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP32_ONNX_PATH

        digest = "79258d3613988b74ab84a530352ea600bbe78a1048584f059e231e44641cea16"
        with TemporaryDirectory() as directory, \
             patch.dict("os.environ", {"MOTIONBRICKS_CHECKPOINT_DIR": directory}), \
             patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.backends.mps.get_name", return_value="test GPU"), \
             patch("tools.onnx.checkpoints.sha256", return_value=digest):
            with self.assertRaisesRegex(ValueError, "Missing checkpoint file.*MOTIONBRICKS_CHECKPOINT_DIR"):
                load_model(UNITREE_G1_N1_FP32_ONNX_PATH)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                         "Metal dependencies are not installed")
    def test_invalid_compilation_setting_fails_before_loading(self):
        from motionbricks.inference.mps import load_model

        with patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.backends.mps.get_name", return_value="test GPU"), \
             patch.dict("os.environ", {"MOTIONBRICKS_MPS_COMPILE": "yes"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                load_model("unused.onnx")
