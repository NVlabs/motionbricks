# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An inference engine that runs model plans with reusable device buffers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

try:
    import tensorrt as trt
except ImportError:
    trt = None

import warp as wp

from .builder import (
    SerializedEnginePlan,
    TensorRTBuildOptions,
    create_logger,
    load_or_build_serialized_engine,
    require_supported_tensorrt,
)
from .cache import TensorRTEngineCache
from .errors import (
    EngineBuildError,
    EngineClosedError,
    EngineDeserializationError,
    EngineExecutionError,
    TensorContractError,
)
from .types import TensorMode, TensorSpec, inspect_engine_tensors

logger = logging.getLogger(__name__)


class CPUStream:
    """A stream placeholder for CPU execution in Warp."""

    def __init__(self, device: wp.context.Device | None = None) -> None:
        self.device = device or wp.get_device("cpu")
        self.cuda_stream = None

    def __repr__(self) -> str:
        return f"<CPUStream on {self.device}>"


# Ensure wp.ScopedStream and wp.synchronize_stream handle CPU devices gracefully.
_orig_scoped_stream_enter = wp.ScopedStream.__enter__
_orig_scoped_stream_exit = wp.ScopedStream.__exit__


def _safe_scoped_stream_enter(self: Any) -> Any:
    if self.stream is not None and not getattr(getattr(self, "device", None), "is_cuda", False):
        if hasattr(self, "device_scope"):
            self.device_scope.__enter__()
        return self.stream
    return _orig_scoped_stream_enter(self)


def _safe_scoped_stream_exit(self: Any, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
    if self.stream is not None and not getattr(getattr(self, "device", None), "is_cuda", False):
        if hasattr(self, "device_scope"):
            self.device_scope.__exit__(exc_type, exc_value, traceback)
        return
    return _orig_scoped_stream_exit(self, exc_type, exc_value, traceback)


wp.ScopedStream.__enter__ = _safe_scoped_stream_enter
wp.ScopedStream.__exit__ = _safe_scoped_stream_exit

_orig_sync_stream = wp.synchronize_stream


def _safe_synchronize_stream(stream: Any = None) -> Any:
    if stream is not None and not getattr(getattr(stream, "device", None), "is_cuda", True):
        return
    return _orig_sync_stream(stream)


wp.synchronize_stream = _safe_synchronize_stream

_SUPPORTED_BACKENDS = ("auto", "tensorrt", "directml", "cpu")

_ORT_TYPE_TO_NP_DTYPE: dict[str, np.dtype] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(bool)": np.dtype(np.bool_),
}


def _resolve_ort_shape(node: Any, batch_size: int) -> tuple[int, ...]:
    shape = list(node.shape)
    if len(shape) > 0 and (shape[0] == "batch_size" or not isinstance(shape[0], int) or shape[0] <= 0):
        shape[0] = batch_size
    for idx, d in enumerate(shape):
        if not isinstance(d, int) or d <= 0:
            raise EngineBuildError(
                f"Tensor {node.name!r} has dynamic dimensions other than batch size: {node.shape}."
                " Only the leading batch dimension may be dynamic."
            )
    return tuple(shape)


def _resolve_ort_dtype(node_type: str) -> np.dtype:
    if node_type in _ORT_TYPE_TO_NP_DTYPE:
        return _ORT_TYPE_TO_NP_DTYPE[node_type]
    raise EngineBuildError(f"Unsupported ONNX Runtime tensor type {node_type!r}.")


class InferenceEngine:
    """Run an ONNX model with reusable device buffers.

    Supports TensorRT on CUDA devices and ONNX Runtime (DirectML / CPU) across
    all hardware platforms.
    """

    # Build options that leave enable_tf32 unset use this value.
    default_enable_tf32 = True

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        backend: str = "auto",
        build_options: TensorRTBuildOptions | None = None,
        stream: wp.Stream | None = None,
        cache_dir: str | Path = ".trt-cache",
        force_rebuild: bool = False,
        logger_severity: str = "INFO",
        build_progress_monitor: Any | None = None,
    ) -> None:
        """Load or build the inference engine and allocate the device buffers.

        Args:
            onnx_path: ONNX model to run.
            backend: Inference backend: "auto", "tensorrt", "directml", or "cpu".
            build_options: Build options for the plan. None uses the defaults. When
                enable_tf32 is None, default_enable_tf32 decides.
            stream: Stream for inference. None creates a new stream.
            cache_dir: Directory that holds the cached plans (TensorRT only).
            force_rebuild: Build a new plan even when a cached one matches.
            logger_severity: Least severe TensorRT message to forward:
                INTERNAL_ERROR, ERROR, WARNING, INFO, or VERBOSE.
            build_progress_monitor: Progress monitor for the build. None reports no progress.
        """
        wp.init()

        self._closed = False
        self._onnx_path = Path(onnx_path)
        backend_lower = backend.lower()
        if backend_lower not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend {backend!r}. Expected one of {_SUPPORTED_BACKENDS}."
            )
        self._requested_backend = backend_lower
        self._backend = backend_lower
        self._cuda_graph: Any | None = None
        self._buffers: dict[str, Any] = {}
        self._tensor_specs: Mapping[str, TensorSpec] = MappingProxyType({})
        self._session: Any | None = None
        self._session_is_dml = False
        self._input_names: list[str] = []
        self._output_names: list[str] = []

        device = wp.get_device()

        options = build_options or TensorRTBuildOptions()
        if options.enable_tf32 is None:
            options = replace(options, enable_tf32=self.default_enable_tf32)

        if type(self)._initialize is not InferenceEngine._initialize:
            self._stream = stream or (wp.Stream() if device.is_cuda else CPUStream(device))
            self._logger = create_logger(logger_severity) if (device.is_cuda and trt is not None) else None
            self._runtime = None
            self._engine = None
            self._context = None
            self._initialize(
                options,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
                progress_monitor=build_progress_monitor,
            )
            self._backend = "tensorrt"
            return

        use_tensorrt = False
        if backend_lower == "auto":
            if device.is_cuda and trt is not None:
                use_tensorrt = True
        elif backend_lower == "tensorrt":
            use_tensorrt = True

        if use_tensorrt:
            try:
                require_supported_tensorrt()
                if not device.is_cuda:
                    raise ValueError(f"TensorRT requires a CUDA device, got {device.alias!r}.")
                self._stream = stream or wp.Stream()
                if self._stream.device != device:
                    raise ValueError(
                        f"The stream uses {self._stream.device.alias}, but the current device is "
                        f"{device.alias}."
                    )
                self._logger = create_logger(logger_severity)
                self._runtime = None
                self._engine = None
                self._context = None
                self._initialize(
                    options,
                    cache_dir=cache_dir,
                    force_rebuild=force_rebuild,
                    progress_monitor=build_progress_monitor,
                )
                self._backend = "tensorrt"
                return
            except Exception as exc:
                if backend_lower == "tensorrt":
                    self._discard_resources(synchronize=False)
                    self._closed = True
                    raise
                logger.info(
                    "TensorRT unavailable or failed to initialize (%s). Falling back to ONNX Runtime.",
                    exc,
                )
                self._discard_resources(synchronize=False)

        # Initialize ONNX Runtime backend (DirectML / CPU)
        if not device.is_cuda:
            device = wp.get_device("cpu")
        self._stream = stream or (wp.Stream() if device.is_cuda else CPUStream(device))
        self._logger = None
        self._runtime = None
        self._engine = None
        self._context = None
        self._initialize_onnxruntime(options)
        self._backend = "onnxruntime"

    @property
    def device(self) -> wp.context.Device:
        """Warp device that runs inference."""
        return self._stream.device

    @property
    def stream(self) -> wp.Stream | CPUStream:
        """Stream that runs inference, including the baked CUDA graph."""
        return self._stream

    @property
    def tensor_specs(self) -> Mapping[str, TensorSpec]:
        """Metadata of every engine tensor, keyed by tensor name."""
        return self._tensor_specs

    @property
    def cuda_graph_baked(self) -> bool:
        """Whether bake_cuda_graph() has captured inference."""
        return self._cuda_graph is not None

    def device_tensor(self, name: str) -> Any:
        """Return the persistent Warp array bound to an inference tensor.

        Args:
            name: Tensor name, as a key of tensor_specs. An unknown name raises
                KeyError.
        """
        self._ensure_open()
        try:
            return self._buffers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown engine tensor {name!r}.") from exc

    def enqueue_async(self) -> None:
        """Enqueue one inference, or launch its baked CUDA graph."""
        self._ensure_open()
        if self._backend == "tensorrt":
            if self._cuda_graph is None:
                self._enqueue_tensorrt()
                return
            try:
                wp.capture_launch(self._cuda_graph, stream=self._stream)
            except Exception as exc:
                raise EngineExecutionError(f"CUDA graph launch failed: {exc}") from exc
            return

        # ONNX Runtime execution
        inputs = {name: self._buffers[name].numpy() for name in self._input_names}
        try:
            outputs = self._session.run(self._output_names, inputs)
        except Exception as exc:
            if self._session_is_dml:
                import onnxruntime as ort
                logger.info(
                    "DirectML execution encountered an unsupported op (%s); falling back to CPUExecutionProvider.",
                    exc,
                )
                try:
                    self._session = ort.InferenceSession(str(self._onnx_path), providers=["CPUExecutionProvider"])
                    self._session_is_dml = False
                    outputs = self._session.run(self._output_names, inputs)
                except Exception as cpu_exc:
                    raise EngineExecutionError(f"Fallback to CPUExecutionProvider failed: {cpu_exc}") from cpu_exc
            else:
                raise EngineExecutionError(f"ONNX Runtime inference failed: {exc}") from exc

        for name, out_val in zip(self._output_names, outputs):
            self._buffers[name].assign(out_val)

    def synchronize(self) -> None:
        """Wait for all work on the engine's stream to finish."""
        self._ensure_open()
        wp.synchronize_stream(self._stream)

    def bake_cuda_graph(self) -> None:
        """Warm up and capture only the TensorRT enqueue into a CUDA graph."""
        self._ensure_open()
        if self._cuda_graph is not None:
            return

        if self._backend != "tensorrt":
            self._cuda_graph = True
            return

        # TensorRT must finish its first-enqueue setup before capture.
        self._enqueue_tensorrt()
        wp.synchronize_stream(self._stream)

        capture_started = False
        try:
            wp.capture_begin(stream=self._stream)
            capture_started = True
            self._enqueue_tensorrt()
            graph = wp.capture_end(stream=self._stream)
            capture_started = False
        except Exception as exc:
            if capture_started:
                try:
                    wp.capture_end(stream=self._stream)
                except Exception:
                    pass
            raise EngineExecutionError(f"CUDA graph launch failed: {exc}") from exc

        if graph is None:
            raise EngineExecutionError("CUDA graph capture returned no graph.")
        self._cuda_graph = graph

    def close(self) -> None:
        """Wait for outstanding work and release owned resources."""
        if self._closed:
            return
        self._discard_resources(synchronize=True)
        self._closed = True

    def _initialize_onnxruntime(self, options: TensorRTBuildOptions) -> None:
        import onnxruntime as ort
        from motionbricks.model.contract import load_model_manifest

        if options.batch_size != 1:
            raise ValueError(
                f"ONNX Runtime backend requires batch_size=1, got {options.batch_size}."
            )

        available = ort.get_available_providers()
        providers: list[Any] = []
        self._session_is_dml = False
        if self._requested_backend == "directml":
            if "DmlExecutionProvider" not in available:
                raise RuntimeError(
                    "DirectML execution provider ('DmlExecutionProvider') is not available in ONNX Runtime."
                )
            providers.append(("DmlExecutionProvider", {"device_id": 0}))
            self._session_is_dml = True
        elif self._requested_backend == "auto":
            if "DmlExecutionProvider" in available:
                providers.append(("DmlExecutionProvider", {"device_id": 0}))
                self._session_is_dml = True
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self._session = ort.InferenceSession(str(self._onnx_path), sess_options=sess_opts, providers=providers)
        except Exception as exc:
            if self._session_is_dml and self._requested_backend == "auto":
                logger.info(
                    "DirectML initialization failed (%s); falling back to CPUExecutionProvider.",
                    exc,
                )
                self._session_is_dml = False
                self._session = ort.InferenceSession(
                    str(self._onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
                )
            else:
                raise

        specs: list[TensorSpec] = []
        manifest_found = False
        for json_file in self._onnx_path.parent.glob("*.json"):
            try:
                contract, fp32_path, fp16_path = load_model_manifest(json_file)
                if fp32_path.name == self._onnx_path.name or fp16_path.name == self._onnx_path.name:
                    specs = list(contract.tensor_specs)
                    manifest_found = True
                    break
            except Exception:
                continue

        if not manifest_found:
            for inp in self._session.get_inputs():
                shape = _resolve_ort_shape(inp, options.batch_size)
                dtype = _resolve_ort_dtype(inp.type)
                specs.append(TensorSpec(inp.name, TensorMode.INPUT, shape, dtype))
            for out in self._session.get_outputs():
                shape = _resolve_ort_shape(out, options.batch_size)
                dtype = _resolve_ort_dtype(out.type)
                specs.append(TensorSpec(out.name, TensorMode.OUTPUT, shape, dtype))

        self._tensor_specs = MappingProxyType({spec.name: spec for spec in specs})
        self._input_names = [spec.name for spec in specs if spec.mode == TensorMode.INPUT]
        self._output_names = [spec.name for spec in specs if spec.mode == TensorMode.OUTPUT]

        buffers: dict[str, Any] = {}
        for spec in self._tensor_specs.values():
            warp_dtype = wp.dtype_from_numpy(spec.dtype)
            buffers[spec.name] = wp.zeros(spec.shape, dtype=warp_dtype, device=self.device)
        self._buffers = buffers

    def _initialize(
        self,
        build_options: TensorRTBuildOptions,
        *,
        cache_dir: str | Path,
        force_rebuild: bool,
        progress_monitor: Any | None,
    ) -> None:
        if trt is None:
            raise EngineBuildError("TensorRT is not installed.")
        trt.init_libnvinfer_plugins(self._logger, "")

        def load_or_build(rebuild: bool) -> SerializedEnginePlan:
            return load_or_build_serialized_engine(
                self._onnx_path,
                options=build_options,
                device_fingerprint={"name": self.device.name, "architecture": self.device.arch},
                cache_dir=cache_dir,
                force_rebuild=rebuild,
                logger=self._logger,
                progress_monitor=progress_monitor,
            )

        plan = load_or_build(force_rebuild)
        try:
            self._deserialize(plan.data)
        except EngineDeserializationError as exc:
            TensorRTEngineCache(cache_dir).invalidate(plan.onnx_path.stem, plan.cache_key)
            if not plan.loaded_from_cache:
                raise EngineBuildError(
                    "TensorRT built a plan that the current runtime could not deserialize."
                ) from exc
            plan = load_or_build(True)
            try:
                self._deserialize(plan.data)
            except EngineDeserializationError as rebuild_exc:
                TensorRTEngineCache(cache_dir).invalidate(plan.onnx_path.stem, plan.cache_key)
                raise EngineBuildError(
                    "TensorRT rebuilt a plan that the current runtime could not deserialize."
                ) from rebuild_exc

        self._allocate_and_bind_buffers()

    _initialize_tensorrt = _initialize

    def _deserialize(self, serialized_engine: bytes) -> None:
        if not serialized_engine:
            raise EngineDeserializationError("Cannot deserialize an empty TensorRT plan.")

        self._context = None
        self._engine = None
        self._runtime = None
        try:
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise EngineDeserializationError("TensorRT rejected the serialized plan.")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise EngineDeserializationError(
                    "TensorRT could not create an execution context."
                )
            specs = inspect_engine_tensors(self._engine)
        except EngineDeserializationError:
            raise
        except Exception as exc:
            raise EngineDeserializationError(
                f"TensorRT could not load the serialized plan: {exc}"
            ) from exc

        self._tensor_specs = MappingProxyType({spec.name: spec for spec in specs})

    def _allocate_and_bind_buffers(self) -> None:
        buffers: dict[str, Any] = {}
        for spec in self._tensor_specs.values():
            warp_dtype = wp.dtype_from_numpy(spec.dtype)
            buffer = wp.zeros(spec.shape, dtype=warp_dtype)
            accepted = self._context.set_tensor_address(spec.name, int(buffer.ptr))
            if accepted is False:
                raise TensorContractError(
                    f"TensorRT rejected the buffer address for {spec.name!r}."
                )
            buffers[spec.name] = buffer
        self._buffers = buffers

    def _enqueue_tensorrt(self) -> None:
        try:
            submitted = self._context.execute_async_v3(self._stream.cuda_stream)
        except Exception as exc:
            raise EngineExecutionError(f"TensorRT inference submission failed: {exc}") from exc
        if submitted is False:
            raise EngineExecutionError("TensorRT failed to enqueue inference.")

    def _ensure_open(self) -> None:
        if self._closed:
            raise EngineClosedError("Cannot use a closed inference engine.")

    def _discard_resources(self, *, synchronize: bool) -> None:
        if synchronize and hasattr(self, "_stream"):
            try:
                wp.synchronize_stream(self._stream)
            except Exception:
                pass
        self._cuda_graph = None
        self._buffers.clear()
        self._session = None
        self._context = None
        self._engine = None
        self._runtime = None
        self._logger = None

    def __enter__(self) -> "InferenceEngine":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
