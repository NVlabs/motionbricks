# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An inference engine that runs a TensorRT plan with reusable device buffers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import tensorrt as trt
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
from .types import TensorSpec, inspect_engine_tensors


class InferenceEngine:
    """Run an ONNX model asynchronously in TensorRT with a fixed batch size.

    TensorRT runs on the current device, which must be a CUDA device. The engine
    reuses the same device buffers for each inference.

    TensorRT messages go to the ``motionbricks.inference.tensorrt`` logger. Info and
    verbose messages log at debug level.
    """

    # Build options that leave enable_tf32 unset use this value.
    default_enable_tf32 = True

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        build_options: TensorRTBuildOptions | None = None,
        stream: wp.Stream | None = None,
        cache_dir: str | Path = ".trt-cache",
        force_rebuild: bool = False,
        logger_severity: str = "INFO",
        build_progress_monitor: Any | None = None,
    ) -> None:
        """Load the cached plan or build a new one, then allocate the device buffers.

        Args:
            onnx_path: ONNX model to run.
            build_options: Build options for the plan. None uses the defaults. When
                enable_tf32 is None, default_enable_tf32 decides.
            stream: Stream for inference. None creates a new stream.
            cache_dir: Directory that holds the cached plans. A relative path starts
                at the working directory.
            force_rebuild: Build a new plan even when a cached one matches.
            logger_severity: Least severe TensorRT message to forward:
                INTERNAL_ERROR, ERROR, WARNING, INFO, or VERBOSE.
            build_progress_monitor: TensorRT progress monitor for the build. None
                reports no progress.
        """
        require_supported_tensorrt()
        wp.init()

        self._closed = False
        device = wp.get_device()
        if not device.is_cuda:
            raise ValueError(f"TensorRT requires a CUDA device, got {device.alias!r}.")
        self._stream = stream or wp.Stream()
        if self._stream.device != device:
            raise ValueError(
                f"The stream uses {self._stream.device.alias}, but the current device is "
                f"{device.alias}."
            )

        self._logger = create_logger(logger_severity)
        self._runtime: Any | None = None
        self._engine: Any | None = None
        self._context: Any | None = None
        self._cuda_graph: Any | None = None
        self._buffers: dict[str, Any] = {}
        self._tensor_specs: Mapping[str, TensorSpec] = MappingProxyType({})

        self._onnx_path = Path(onnx_path)

        options = build_options or TensorRTBuildOptions()
        if options.enable_tf32 is None:
            options = replace(options, enable_tf32=self.default_enable_tf32)
        try:
            self._initialize(
                options,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
                progress_monitor=build_progress_monitor,
            )
        except Exception:
            self._discard_resources(synchronize=False)
            self._closed = True
            raise

    @property
    def device(self) -> wp.context.Device:
        """Warp CUDA device that runs inference."""
        return self._stream.device

    @property
    def stream(self) -> wp.Stream:
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
        """Return the persistent Warp CUDA array bound to a TensorRT tensor.

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
        """Enqueue one TensorRT inference, or launch its baked CUDA graph."""

        self._ensure_open()
        if self._cuda_graph is None:
            self._enqueue_tensorrt()
            return
        try:
            wp.capture_launch(self._cuda_graph, stream=self._stream)
        except Exception as exc:
            raise EngineExecutionError(f"CUDA graph launch failed: {exc}") from exc

    def synchronize(self) -> None:
        """Wait for all work on the engine's stream to finish."""
        self._ensure_open()
        wp.synchronize_stream(self._stream)

    def bake_cuda_graph(self) -> None:
        """Warm up and capture only the TensorRT enqueue into a CUDA graph.

        Other CUDA work for the model can run before and after inference on the
        same stream.
        """

        self._ensure_open()
        if self._cuda_graph is not None:
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
            raise EngineExecutionError(f"CUDA graph capture failed: {exc}") from exc

        if graph is None:
            raise EngineExecutionError("CUDA graph capture returned no graph.")
        self._cuda_graph = graph

    def close(self) -> None:
        """Wait for outstanding work and release owned resources."""

        if self._closed:
            return
        self._discard_resources(synchronize=True)
        self._closed = True

    def _initialize(
        self,
        build_options: TensorRTBuildOptions,
        *,
        cache_dir: str | Path,
        force_rebuild: bool,
        progress_monitor: Any | None,
    ) -> None:
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
            # A cached plan that fails to load, such as one from another TensorRT build, is rebuilt once.
            plan = load_or_build(True)
            try:
                self._deserialize(plan.data)
            except EngineDeserializationError as rebuild_exc:
                TensorRTEngineCache(cache_dir).invalidate(plan.onnx_path.stem, plan.cache_key)
                raise EngineBuildError(
                    "TensorRT rebuilt a plan that the current runtime could not deserialize."
                ) from rebuild_exc

        self._allocate_and_bind_buffers()

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
        if synchronize:
            try:
                wp.synchronize_stream(self._stream)
            except Exception:
                pass
        self._cuda_graph = None
        self._buffers.clear()
        self._context = None
        self._engine = None
        self._runtime = None
        self._logger = None

    def __enter__(self) -> "InferenceEngine":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
