# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import unittest
import numpy as np
import warp as wp

from motionbricks.inference.builder import TensorRTBuildOptions
from motionbricks.inference.engine import (
    InferenceEngine,
    _resolve_ort_dtype,
    _resolve_ort_shape,
)
from motionbricks.inference.errors import EngineBuildError
from motionbricks.model.contract import valid_frame_count
from motionbricks.model.soma.contract import SOMA_N1_CONTRACT, SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_CONTRACT, UNITREE_G1_N1_FP32_ONNX_PATH


def _populate_soma_inputs(engine: InferenceEngine) -> None:
    rng = np.random.RandomState(42)
    # Token count 6 is at index 0 (counts range from 6 to 16)
    engine.device_tensor("allowed_pred_num_tokens").assign(
        np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    )
    engine.device_tensor("has_global_root_values").assign(np.ones((1, 8), dtype=np.int32))
    engine.device_tensor("has_local_poses").assign(np.ones((1, 8), dtype=np.int32))
    engine.device_tensor("local_poses").assign(rng.randn(1, 8, 267).astype(np.float32) * 0.1)
    engine.device_tensor("global_root_values").assign(rng.randn(1, 8, 5).astype(np.float32) * 0.1)
    engine.device_tensor("local_root_values").assign(rng.randn(1, 8, 4).astype(np.float32) * 0.1)


def _populate_g1_inputs(engine: InferenceEngine) -> None:
    rng = np.random.RandomState(42)
    # Token count 6 is at index 0 (counts range from 6 to 16)
    engine.device_tensor("allowed_pred_num_tokens").assign(
        np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    )
    engine.device_tensor("has_global_root_values").assign(np.ones((1, 8), dtype=np.int32))
    engine.device_tensor("has_local_poses").assign(np.ones((1, 8), dtype=np.int32))
    engine.device_tensor("context_qpos").assign(rng.randn(1, 4, 36).astype(np.float32) * 0.1)
    engine.device_tensor("target_qpos").assign(rng.randn(1, 4, 36).astype(np.float32) * 0.1)


class ONNXRuntimeEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_soma_onnxruntime_inference(self) -> None:
        engine = InferenceEngine(SOMA_N1_FP32_ONNX_PATH, backend="cpu")
        self.addCleanup(engine.close)

        SOMA_N1_CONTRACT.validate_tensor_specs(engine.tensor_specs)

        _populate_soma_inputs(engine)
        engine.enqueue_async()
        engine.synchronize()

        tokens = int(engine.device_tensor("pred_num_tokens").numpy()[0])
        self.assertEqual(tokens, 6)

        pred_motions = engine.device_tensor("pred_global_motions").numpy()
        self.assertEqual(pred_motions.shape, (1, 64, 366))
        self.assertTrue(np.all(np.isfinite(pred_motions)))
        self.assertTrue(np.any(pred_motions != 0.0))
        self.assertEqual(valid_frame_count(tokens), 18)

    def test_unitree_g1_onnxruntime_inference(self) -> None:
        engine = InferenceEngine(UNITREE_G1_N1_FP32_ONNX_PATH, backend="cpu")
        self.addCleanup(engine.close)

        UNITREE_G1_N1_CONTRACT.validate_tensor_specs(engine.tensor_specs)

        _populate_g1_inputs(engine)
        engine.enqueue_async()
        engine.synchronize()

        tokens = int(engine.device_tensor("pred_num_tokens").numpy()[0])
        self.assertEqual(tokens, 6)

        pred_qpos = engine.device_tensor("pred_qpos").numpy()
        self.assertEqual(pred_qpos.shape, (1, 64, 36))
        self.assertTrue(np.all(np.isfinite(pred_qpos)))
        self.assertTrue(np.any(pred_qpos != 0.0))

        pred_contacts = engine.device_tensor("pred_foot_contacts").numpy()
        self.assertEqual(pred_contacts.shape, (1, 64, 4))
        self.assertTrue(np.all(np.isfinite(pred_contacts)))

    def test_soma_directml_inference_with_fallback(self) -> None:
        import onnxruntime as ort

        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise unittest.SkipTest("DirectML execution provider is not available")

        engine = InferenceEngine(SOMA_N1_FP32_ONNX_PATH, backend="directml")
        self.addCleanup(engine.close)

        SOMA_N1_CONTRACT.validate_tensor_specs(engine.tensor_specs)
        _populate_soma_inputs(engine)
        engine.enqueue_async()
        engine.synchronize()

        tokens = int(engine.device_tensor("pred_num_tokens").numpy()[0])
        self.assertEqual(tokens, 6)

        pred_motions = engine.device_tensor("pred_global_motions").numpy()
        self.assertEqual(pred_motions.shape, (1, 64, 366))
        self.assertTrue(np.all(np.isfinite(pred_motions)))
        self.assertTrue(np.any(pred_motions != 0.0))
        self.assertEqual(valid_frame_count(tokens), 18)

    def test_unitree_g1_directml_inference_with_fallback(self) -> None:
        import onnxruntime as ort

        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise unittest.SkipTest("DirectML execution provider is not available")

        engine = InferenceEngine(UNITREE_G1_N1_FP32_ONNX_PATH, backend="directml")
        self.addCleanup(engine.close)

        UNITREE_G1_N1_CONTRACT.validate_tensor_specs(engine.tensor_specs)
        _populate_g1_inputs(engine)
        engine.enqueue_async()
        engine.synchronize()

        tokens = int(engine.device_tensor("pred_num_tokens").numpy()[0])
        self.assertEqual(tokens, 6)

        pred_qpos = engine.device_tensor("pred_qpos").numpy()
        self.assertEqual(pred_qpos.shape, (1, 64, 36))
        self.assertTrue(np.all(np.isfinite(pred_qpos)))
        self.assertTrue(np.any(pred_qpos != 0.0))

        pred_contacts = engine.device_tensor("pred_foot_contacts").numpy()
        self.assertEqual(pred_contacts.shape, (1, 64, 4))
        self.assertTrue(np.all(np.isfinite(pred_contacts)))

    def test_invalid_backend_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InferenceEngine(SOMA_N1_FP32_ONNX_PATH, backend="unsupported_backend")

    def test_non_unit_batch_size_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InferenceEngine(
                SOMA_N1_FP32_ONNX_PATH,
                backend="cpu",
                build_options=TensorRTBuildOptions(batch_size=2),
            )

    def test_dynamic_inner_dimension_rejected(self) -> None:
        node_valid = SimpleNamespace(name="valid", shape=["batch_size", 8, 267])
        self.assertEqual(_resolve_ort_shape(node_valid, 1), (1, 8, 267))

        node_dynamic_inner = SimpleNamespace(name="invalid", shape=[1, "dynamic_seq", 267])
        with self.assertRaises(EngineBuildError):
            _resolve_ort_shape(node_dynamic_inner, 1)

    def test_ort_type_mapping(self) -> None:
        self.assertEqual(_resolve_ort_dtype("tensor(float)"), np.dtype(np.float32))
        self.assertEqual(_resolve_ort_dtype("tensor(float16)"), np.dtype(np.float16))
        self.assertEqual(_resolve_ort_dtype("tensor(int32)"), np.dtype(np.int32))
        self.assertEqual(_resolve_ort_dtype("tensor(int64)"), np.dtype(np.int64))
        self.assertEqual(_resolve_ort_dtype("tensor(bool)"), np.dtype(np.bool_))

        with self.assertRaises(EngineBuildError):
            _resolve_ort_dtype("tensor(complex64)")

    def test_load_or_build_serialized_engine_without_tensorrt(self) -> None:
        import motionbricks.inference.builder as builder
        from motionbricks.inference.builder import load_or_build_serialized_engine

        orig_trt = builder.trt
        try:
            builder.trt = None
            with self.assertRaises(RuntimeError):
                load_or_build_serialized_engine(
                    "dummy.onnx",
                    options=TensorRTBuildOptions(),
                    device_fingerprint={},
                    cache_dir=".trt-cache",
                    force_rebuild=False,
                    logger=None,
                )
        finally:
            builder.trt = orig_trt

    def test_auto_backend_retries_cpu_when_directml_init_fails(self) -> None:
        import onnxruntime as ort
        from unittest.mock import patch

        orig_session = ort.InferenceSession
        call_count = 0

        def mock_session(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            providers = kwargs.get("providers", [])
            if any("DmlExecutionProvider" in str(p) for p in providers):
                raise RuntimeError("DirectML device creation failed simulated error")
            return orig_session(*args, **kwargs)

        with patch("onnxruntime.InferenceSession", side_effect=mock_session):
            engine = InferenceEngine(SOMA_N1_FP32_ONNX_PATH, backend="auto")
            self.addCleanup(engine.close)
            self.assertFalse(engine._session_is_dml)
            if "DmlExecutionProvider" in ort.get_available_providers():
                self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
