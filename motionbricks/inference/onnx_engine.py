# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apple Silicon inference with ONNX Runtime or PyTorch Metal."""

from pathlib import Path
from types import MappingProxyType
import logging
import os

import numpy as np
import onnxruntime as ort
import warp as wp

from .errors import EngineClosedError, EngineExecutionError, TensorContractError
from .types import TensorMode, TensorSpec

logger = logging.getLogger(__name__)
_TYPES = {
    'tensor(float)': np.float32,
    'tensor(float16)': np.float16,
    'tensor(int32)': np.int32,
    'tensor(int64)': np.int64,
    'tensor(bool)': np.bool_,
}


class InferenceEngine:
    """Run a model using CPU Warp buffers and the selected Apple inference backend."""

    default_enable_tf32 = True

    def __init__(self, onnx_path, *, build_options=None, stream=None, expected_tensor_specs=(), **_kwargs):
        wp.init()
        self._device = wp.get_device()
        if not self._device.is_cpu or stream is not None:
            raise ValueError('Apple inference requires Warp CPU buffers and no CUDA stream.')
        self._closed = False
        self._buffers = {}
        self._session = None
        self._mps_model = None
        self._mps_input_names = ()
        self._mps_output_names = ()
        provider = os.environ.get('MOTIONBRICKS_ONNX_PROVIDER', 'auto').lower()
        if provider not in {'auto', 'coreml', 'cpu', 'mps'}:
            raise ValueError(f'Unsupported ONNX provider: {provider!r}')
        if provider == 'auto':
            from .mps import is_available_for_model

            provider = 'mps' if expected_tensor_specs and is_available_for_model(onnx_path) else 'cpu'
        self._provider = provider
        if provider == 'mps':
            self._initialize_mps(onnx_path, build_options, expected_tensor_specs)
            return
        providers = ['CPUExecutionProvider']
        if provider == 'coreml':
            units = os.environ.get('MOTIONBRICKS_COREML_COMPUTE_UNITS', 'CPUAndGPU')
            if units not in {'CPUOnly', 'CPUAndGPU', 'CPUAndNeuralEngine', 'ALL'}:
                raise ValueError(f'Unsupported Core ML compute units: {units!r}')
            if 'CoreMLExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('ONNX Runtime was installed without the Core ML execution provider.')
            providers.insert(0, ('CoreMLExecutionProvider', {
                'ModelFormat': 'MLProgram', 'MLComputeUnits': units, 'RequireStaticInputShapes': '1',
            }))
            logger.info('Loading Core ML with compute units %s; unsupported operations run on CPU.', units)
        else:
            logger.info('Loading ONNX Runtime CPU execution provider.')
        options = ort.SessionOptions()
        # The default ONNX Runtime pool can oversubscribe Apple Silicon badly
        # when the viewer and other processes are active. Keep the default
        # modest while allowing applications to tune it for their machine.
        thread_count = os.environ.get('MOTIONBRICKS_ONNX_THREADS')
        if thread_count is None:
            options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        else:
            try:
                options.intra_op_num_threads = int(thread_count)
                if options.intra_op_num_threads < 1:
                    raise ValueError
            except ValueError as exc:
                raise ValueError('MOTIONBRICKS_ONNX_THREADS must be a positive integer.') from exc
        options.add_free_dimension_override_by_name('batch_size', build_options.batch_size if build_options else 1)
        self._session = ort.InferenceSession(str(Path(onnx_path)), sess_options=options, providers=providers)
        if provider == 'coreml' and 'CoreMLExecutionProvider' not in self._session.get_providers():
            raise RuntimeError('ONNX Runtime could not initialize the Core ML execution provider.')
        batch_size = build_options.batch_size if build_options is not None else 1
        specs = {}
        expected = {spec.name: spec for spec in expected_tensor_specs}
        for nodes, mode in ((self._session.get_inputs(), TensorMode.INPUT),
                            (self._session.get_outputs(), TensorMode.OUTPUT)):
            for node in nodes:
                try:
                    dtype = np.dtype(_TYPES[node.type])
                    contract = expected.get(node.name)
                    shape = tuple(
                        d if isinstance(d, int) else batch_size if i == 0 else contract.shape[i]
                        for i, d in enumerate(node.shape)
                    )
                    spec = TensorSpec(node.name, mode, shape, dtype)
                except (KeyError, TypeError, ValueError) as exc:
                    raise TensorContractError(f'Unsupported ONNX tensor {node.name}: {node.shape}, {node.type}') from exc
                specs[node.name] = spec
                self._buffers[node.name] = wp.zeros(shape, dtype=wp.dtype_from_numpy(dtype), device=self.device)
        self._tensor_specs = MappingProxyType(specs)

    def _initialize_mps(self, onnx_path, build_options, expected_tensor_specs):
        from .mps import load_model

        if build_options is not None and build_options.batch_size != 1:
            raise ValueError('Metal inference currently requires a batch size of one.')
        if not expected_tensor_specs:
            raise ValueError('Metal inference requires the model tensor contract.')
        self._mps_model, self._mps_input_names, self._mps_output_names = load_model(onnx_path)
        inputs = {spec.name for spec in expected_tensor_specs if spec.mode == TensorMode.INPUT}
        outputs = {spec.name for spec in expected_tensor_specs if spec.mode == TensorMode.OUTPUT}
        if inputs != set(self._mps_input_names) or outputs != set(self._mps_output_names):
            self._mps_model = None
            raise TensorContractError('Metal model input and output names do not match the tensor contract.')
        self._tensor_specs = MappingProxyType({spec.name: spec for spec in expected_tensor_specs})
        self._buffers = {
            spec.name: wp.zeros(spec.shape, dtype=wp.dtype_from_numpy(spec.dtype), device=self.device)
            for spec in expected_tensor_specs
        }

    @property
    def provider(self):
        """Selected provider: ``cpu``, ``coreml``, or ``mps``."""
        return self._provider

    @property
    def device(self):
        return self._device

    @property
    def stream(self):
        return None

    @property
    def tensor_specs(self):
        return self._tensor_specs

    @property
    def cuda_graph_baked(self):
        return False

    def device_tensor(self, name):
        self._ensure_open()
        return self._buffers[name]

    def enqueue_async(self):
        self._ensure_open()
        inputs = {name: self._buffers[name].numpy() for name, spec in self._tensor_specs.items()
                  if spec.mode == TensorMode.INPUT}
        outputs = [name for name, spec in self._tensor_specs.items() if spec.mode == TensorMode.OUTPUT]
        try:
            if self._mps_model is not None:
                from .mps import run_model

                results = run_model(self._mps_model, self._mps_input_names, self._mps_output_names, inputs)
                values = [results[name] for name in outputs]
            else:
                values = self._session.run(outputs, inputs)
            for name, value in zip(outputs, values, strict=True):
                spec = self._tensor_specs[name]
                if value.shape != spec.shape or value.dtype != spec.dtype:
                    raise TensorContractError(f'Unexpected output {name}: {value.shape}, {value.dtype}')
                wp.copy(self._buffers[name], wp.from_numpy(
                    value, dtype=wp.dtype_from_numpy(spec.dtype), device=self.device,
                ))
        except TensorContractError:
            raise
        except Exception as exc:
            backend = 'Metal' if self._provider == 'mps' else 'ONNX Runtime'
            raise EngineExecutionError(f'{backend} inference failed: {exc}') from exc

    def synchronize(self):
        self._ensure_open()
        wp.synchronize_device(self.device)

    def close(self):
        if not self._closed:
            self._buffers.clear()
            self._session = None
            self._mps_model = None
            self._closed = True

    def _ensure_open(self):
        if self._closed:
            raise EngineClosedError('Cannot use a closed inference engine.')

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
