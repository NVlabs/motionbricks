"""The ONNX backend keeps the model contract and Warp buffers in sync."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

try:
    from motionbricks.inference import onnx_engine
except ImportError:
    onnx_engine = None

from motionbricks.inference.errors import EngineClosedError, TensorContractError
from motionbricks.inference.types import TensorMode, TensorSpec


@unittest.skipIf(onnx_engine is None, 'onnxruntime is not installed')
class OnnxEngineTests(unittest.TestCase):
    def test_dynamic_shapes_buffers_and_close(self):
        wp.init()
        specs = (
            TensorSpec('input', TensorMode.INPUT, (1, 2), np.dtype('float32')),
            TensorSpec('output', TensorMode.OUTPUT, (1, 2), np.dtype('float32')),
        )
        node = lambda name: SimpleNamespace(name=name, shape=['batch', 'width'], type='tensor(float)')
        session = SimpleNamespace(
            get_inputs=lambda: [node('input')], get_outputs=lambda: [node('output')],
            get_providers=lambda: ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
            run=lambda names, inputs: [inputs['input'] + 2],
        )
        with wp.ScopedDevice('cpu'), patch.dict('os.environ', {'MOTIONBRICKS_ONNX_PROVIDER': 'coreml'}), patch.object(onnx_engine.ort, 'get_available_providers', return_value=['CoreMLExecutionProvider']), patch.object(onnx_engine.ort, 'InferenceSession', return_value=session):
            engine = onnx_engine.InferenceEngine('unused.onnx', expected_tensor_specs=specs)
            wp.copy(engine.device_tensor('input'), wp.array([[3., 4.]], dtype=wp.float32, device='cpu'))
            engine.enqueue_async()
            np.testing.assert_array_equal(engine.device_tensor('output').numpy(), [[5., 6.]])
            session.run = lambda names, inputs: [np.zeros((1, 3), dtype=np.float32)]
            with self.assertRaises(TensorContractError):
                engine.enqueue_async()
            engine.close()
            engine.close()
            with self.assertRaises(EngineClosedError):
                engine.device_tensor('input')

    def test_missing_coreml_reports_error(self):
        wp.init()
        with wp.ScopedDevice('cpu'), patch.dict('os.environ', {'MOTIONBRICKS_ONNX_PROVIDER': 'coreml'}), patch.object(onnx_engine.ort, 'get_available_providers', return_value=['CPUExecutionProvider']):
            with self.assertRaisesRegex(RuntimeError, 'without the Core ML'):
                onnx_engine.InferenceEngine('unused.onnx')

    def test_cpu_thread_limit_and_validation(self):
        wp.init()
        with wp.ScopedDevice('cpu'), patch.dict('os.environ', {
            'MOTIONBRICKS_ONNX_PROVIDER': 'cpu', 'MOTIONBRICKS_ONNX_THREADS': '0',
        }):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                onnx_engine.InferenceEngine('unused.onnx')

        with wp.ScopedDevice('cpu'), patch.dict('os.environ', {
            'MOTIONBRICKS_ONNX_PROVIDER': 'cpu', 'MOTIONBRICKS_ONNX_THREADS': '3',
            'MOTIONBRICKS_COREML_COMPUTE_UNITS': 'invalid',
        }), patch.object(onnx_engine.ort, 'InferenceSession') as factory:
            factory.return_value.get_inputs.return_value = []
            factory.return_value.get_outputs.return_value = []
            engine = onnx_engine.InferenceEngine('unused.onnx')
            self.assertEqual(factory.call_args.kwargs['sess_options'].intra_op_num_threads, 3)
            engine.close()
