# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
import warp as wp

from motionbricks.inference.engine import InferenceEngine
from motionbricks.model.contract import valid_frame_count
from motionbricks.model.soma.contract import SOMA_N1_CONTRACT, SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_CONTRACT, UNITREE_G1_N1_FP32_ONNX_PATH


class ONNXRuntimeEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_soma_onnxruntime_inference(self) -> None:
        engine = InferenceEngine(SOMA_N1_FP32_ONNX_PATH, backend="cpu")
        self.addCleanup(engine.close)

        SOMA_N1_CONTRACT.validate_tensor_specs(engine.tensor_specs)

        # Allow 6 tokens
        engine.device_tensor("allowed_pred_num_tokens").assign(
            np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.int32)
        )
        engine.enqueue_async()
        engine.synchronize()

        tokens = int(engine.device_tensor("pred_num_tokens").numpy()[0])
        self.assertEqual(tokens, 6)

        pred_motions = engine.device_tensor("pred_global_motions").numpy()
        self.assertEqual(pred_motions.shape, (1, 64, 366))
        self.assertEqual(valid_frame_count(tokens), 18)

    def test_unitree_g1_onnxruntime_inference(self) -> None:
        engine = InferenceEngine(UNITREE_G1_N1_FP32_ONNX_PATH, backend="cpu")
        self.addCleanup(engine.close)

        UNITREE_G1_N1_CONTRACT.validate_tensor_specs(engine.tensor_specs)

        engine.device_tensor("allowed_pred_num_tokens").assign(
            np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.int32)
        )
        engine.enqueue_async()
        engine.synchronize()

        tokens = int(engine.device_tensor("pred_num_tokens").numpy()[0])
        self.assertEqual(tokens, 6)

        pred_qpos = engine.device_tensor("pred_qpos").numpy()
        self.assertEqual(pred_qpos.shape, (1, 64, 36))


if __name__ == "__main__":
    unittest.main()
