# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import unittest
from types import SimpleNamespace

import numpy as np
try:
    import tensorrt as trt
except ImportError:
    trt = None

from motionbricks.inference.types import TensorMode, TensorSpec
from motionbricks.inference.builder import TensorRTBuildOptions, _specialize_batch_dimension, create_logger
from motionbricks.inference.errors import EngineBuildError
from motionbricks.inference.progress import TensorRTBuildProgressMonitor


class InferenceEngineTests(unittest.TestCase):
    def test_static_tensor_metadata(self) -> None:
        spec = TensorSpec("input", TensorMode.INPUT, (1, 8, 5), np.dtype(np.float32))

        self.assertEqual(spec.element_count, 40)
        self.assertEqual(spec.nbytes, 160)

    @unittest.skipIf(trt is None, "TensorRT is not installed")
    def test_tensorrt_messages_go_to_python_logging(self) -> None:
        warnings_and_up = create_logger("warning")
        with self.assertLogs("motionbricks.inference.tensorrt", level="DEBUG") as logs:
            warnings_and_up.log(trt.Logger.ERROR, "failed")
            warnings_and_up.log(trt.Logger.WARNING, "careful")
            warnings_and_up.log(trt.Logger.INFO, "below the minimum severity")
            create_logger("INFO").log(trt.Logger.INFO, "detail")
        self.assertEqual(
            [(record.levelno, record.getMessage()) for record in logs.records],
            [(logging.ERROR, "failed"), (logging.WARNING, "careful"), (logging.DEBUG, "detail")],
        )
        with self.assertRaisesRegex(ValueError, "Unknown TensorRT log severity"):
            create_logger("loud")

    def test_cache_payload_omits_default_precision_and_keeps_batch_size(self) -> None:
        defaults = TensorRTBuildOptions()
        self.assertNotIn("enable_fp16", defaults.cache_payload())
        self.assertTrue(
            TensorRTBuildOptions(enable_fp16=True).cache_payload()["enable_fp16"]
        )
        self.assertNotIn("enable_tf32", defaults.cache_payload())
        self.assertNotIn("enable_tf32", TensorRTBuildOptions(enable_tf32=True).cache_payload())
        self.assertIs(TensorRTBuildOptions(enable_tf32=False).cache_payload()["enable_tf32"], False)
        self.assertEqual(TensorRTBuildOptions(batch_size=3).cache_payload()["batch_size"], 3)

    def test_batch_specialization_keeps_nonbatch_dimensions(self) -> None:
        inputs = [SimpleNamespace(name="poses", shape=(-1, 8, 267)),
                  SimpleNamespace(name="mask", shape=(-1, 8)),
                  SimpleNamespace(name="constant", shape=(11,))]
        network = SimpleNamespace(num_inputs=len(inputs), get_input=inputs.__getitem__)
        _specialize_batch_dimension(network, 3)
        self.assertEqual([tensor.shape for tensor in inputs], [(3, 8, 267), (3, 8), (11,)])
        inputs[0].shape = (-1, -1, 267)
        with self.assertRaisesRegex(EngineBuildError, "Only the leading batch"):
            _specialize_batch_dimension(network, 1)

    def test_build_progress_monitor_tracks_phases_and_cancellation(self) -> None:
        monitor = TensorRTBuildProgressMonitor()

        monitor.phase_start("parent", None, 4)
        self.assertEqual(monitor.snapshot().phase_name, "parent")
        self.assertTrue(monitor.step_complete("parent", 0))
        self.assertEqual(monitor.snapshot().fraction, 0.25)

        monitor.phase_start("child", "parent", 2)
        self.assertTrue(monitor.step_complete("child", 0))
        self.assertEqual(monitor.snapshot().phase_name, "parent")
        self.assertEqual(monitor.snapshot().fraction, 0.25)
        monitor.phase_finish("child")
        self.assertEqual(monitor.snapshot().phase_name, "parent")

        monitor.request_cancel()
        self.assertFalse(monitor.step_complete("parent", 1))


if __name__ == "__main__":
    unittest.main()
