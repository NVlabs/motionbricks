# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and cache TensorRT plans for a chosen batch size."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any

try:
    import tensorrt as trt
except ImportError:
    trt = None

from .cache import (
    TensorRTEngineCache,
    cache_key_for,
    make_cache_identity,
    sha256_file,
)
from .errors import EngineBuildError


@dataclass(frozen=True, slots=True)
class TensorRTBuildOptions:
    """Settings that affect the generated TensorRT plan.

    Invalid values raise ValueError.

    Attributes:
        workspace_size_bytes: Memory TensorRT may use while building, in bytes. None
            keeps TensorRT's default limit.
        builder_optimization_level: Optimization level from 0 to 5. Higher levels spend
            more build time searching for faster kernels.
        enable_fp16: Allow TensorRT to use FP16 kernels.
        enable_tf32: Allow TensorRT to use TF32 kernels. None uses the engine's default
            for its model, or TensorRT's default when building without an engine.
            TensorRT's default permits TF32.
        batch_size: Batch size used to build the plan. It replaces a dynamic leading
            batch dimension in the ONNX model.
    """

    workspace_size_bytes: int | None = None
    builder_optimization_level: int = 3
    enable_fp16: bool = False
    enable_tf32: bool | None = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.workspace_size_bytes is not None and self.workspace_size_bytes <= 0:
            raise ValueError("workspace_size_bytes must be positive when specified.")
        if not 0 <= self.builder_optimization_level <= 5:
            raise ValueError("builder_optimization_level must be 0 to 5.")

    def cache_payload(self) -> dict[str, Any]:
        """Return the build options that go into the plan's cache key."""
        payload = asdict(self)
        # Omit default precision settings from the cache key. Include the batch
        # size so each size gets its own cached plan.
        if not self.enable_fp16:
            payload.pop("enable_fp16")
        if self.enable_tf32 is not False:
            payload.pop("enable_tf32")
        return payload


@dataclass(frozen=True, slots=True)
class SerializedEnginePlan:
    """TensorRT plan bytes with the source model and the key of its cache entry."""

    data: bytes = field(repr=False)
    onnx_path: Path
    cache_key: str
    loaded_from_cache: bool


def load_or_build_serialized_engine(
    onnx_path: str | Path,
    *,
    options: TensorRTBuildOptions,
    device_fingerprint: Mapping[str, Any],
    cache_dir: str | Path,
    force_rebuild: bool,
    logger: Any,
    progress_monitor: Any | None = None,
) -> SerializedEnginePlan:
    """Load a compatible cached plan, or build one from ONNX and cache it.

    make_cache_identity() lists the values a cached plan must match.

    Args:
        onnx_path: ONNX model to build from.
        options: Build options for the plan.
        device_fingerprint: Values that identify the GPU, such as its name and
            architecture.
        cache_dir: Directory that holds the cached plans.
        force_rebuild: Build a new plan even when a cached one matches.
        logger: TensorRT logger for the build, as create_logger() returns.
        progress_monitor: TensorRT progress monitor for the build. None reports no
            progress.
    """

    model_path = Path(onnx_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Could not find ONNX file: {model_path}.")

    onnx_sha256 = sha256_file(model_path)
    cache_identity = make_cache_identity(
        onnx_sha256=onnx_sha256,
        tensorrt_version=str(trt.__version__),
        build_options=options.cache_payload(),
        device_fingerprint=device_fingerprint,
    )
    cache_key = cache_key_for(cache_identity)
    cache_metadata = dict(cache_identity)
    cache_metadata["onnx_filename"] = model_path.name
    cache = TensorRTEngineCache(cache_dir)

    if not force_rebuild:
        data = cache.load(model_path.stem, cache_key)
        if data is not None:
            return SerializedEnginePlan(
                data=data,
                onnx_path=model_path,
                cache_key=cache_key,
                loaded_from_cache=True,
            )

    data = build_serialized_engine(
        model_path,
        options,
        logger,
        progress_monitor=progress_monitor,
    )
    cache.store(
        model_path.stem,
        cache_key,
        data,
        metadata=cache_metadata,
    )
    return SerializedEnginePlan(
        data=data,
        onnx_path=model_path,
        cache_key=cache_key,
        loaded_from_cache=False,
    )


def build_serialized_engine(
    onnx_path: Path,
    options: TensorRTBuildOptions,
    logger: Any,
    *,
    progress_monitor: Any | None = None,
) -> bytes:
    """Build a TensorRT plan with a fixed batch size from an ONNX model.

    The plan is not cached. Every failure raises EngineBuildError.

    Args:
        onnx_path: ONNX model to build from.
        options: Build options for the plan.
        logger: TensorRT logger for the build, as create_logger() returns.
        progress_monitor: TensorRT progress monitor for the build. None reports no
            progress.
    """

    if trt is None:
        raise EngineBuildError("TensorRT is not installed.")

    try:
        builder = trt.Builder(logger)

        # TensorRT 10 and newer always use explicit batch.
        network = builder.create_network(0)
        if network is None:
            raise EngineBuildError("TensorRT could not create a network definition.")
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            diagnostics = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            detail = "\n".join(diagnostics) if diagnostics else "No parser diagnostics were reported."
            raise EngineBuildError(f"TensorRT failed to parse {onnx_path}:\n{detail}")

        _specialize_batch_dimension(network, options.batch_size)
        _require_static_network(network)
        config = builder.create_builder_config()
        if config is None:
            raise EngineBuildError("TensorRT could not create a builder configuration.")
        if options.workspace_size_bytes is not None:
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                options.workspace_size_bytes,
            )
        config.builder_optimization_level = options.builder_optimization_level
        if options.enable_tf32 is False:
            config.clear_flag(trt.BuilderFlag.TF32)
        if options.enable_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if progress_monitor is not None:
            config.progress_monitor = progress_monitor

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise EngineBuildError(
                "TensorRT failed to build the plan. Retry with VERBOSE logging for "
                "detailed tactic diagnostics."
            )
        plan = bytes(serialized)
        if not plan:
            raise EngineBuildError("TensorRT returned an empty plan.")
        return plan
    except EngineBuildError:
        raise
    except Exception as exc:
        raise EngineBuildError(f"TensorRT failed while building {onnx_path}: {exc}") from exc


if trt is not None:
    _TENSORRT_SEVERITIES = {
        "INTERNAL_ERROR": trt.Logger.INTERNAL_ERROR,
        "ERROR": trt.Logger.ERROR,
        "WARNING": trt.Logger.WARNING,
        "INFO": trt.Logger.INFO,
        "VERBOSE": trt.Logger.VERBOSE,
    }
    _TENSORRT_LOG_LEVELS = {
        trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
        trt.Logger.ERROR: logging.ERROR,
        trt.Logger.WARNING: logging.WARNING,
        trt.Logger.INFO: logging.DEBUG,
        trt.Logger.VERBOSE: logging.DEBUG,
    }
    _LoggerBase = trt.ILogger
else:
    _TENSORRT_SEVERITIES = {}
    _TENSORRT_LOG_LEVELS = {}
    _LoggerBase = object

_TENSORRT_LOG = logging.getLogger("motionbricks.inference.tensorrt")


class _TensorRTLogger(_LoggerBase):
    """Forward TensorRT messages down to a minimum severity to Python logging."""

    def __init__(self, min_severity: Any) -> None:
        if trt is not None:
            trt.ILogger.__init__(self)
        self._levels = {
            severity: level
            for severity, level in _TENSORRT_LOG_LEVELS.items()
            if int(severity) <= int(min_severity)
        }

    def log(self, severity: Any, msg: str) -> None:
        level = self._levels.get(severity)
        if level is not None:
            _TENSORRT_LOG.log(level, msg)


def create_logger(severity: str) -> Any:
    """Create a TensorRT logger that forwards messages to Python logging.

    Args:
        severity: Least severe TensorRT message to forward: INTERNAL_ERROR, ERROR,
            WARNING, INFO, or VERBOSE. Case does not matter.
    """

    if trt is None:
        return None

    min_severity = _TENSORRT_SEVERITIES.get(severity.upper())
    if min_severity is None:
        raise ValueError(
            f"Unknown TensorRT log severity {severity!r}. Expected one of {tuple(_TENSORRT_SEVERITIES)}."
        )
    return _TensorRTLogger(min_severity)


def require_supported_tensorrt() -> None:
    """Check that TensorRT 10 or newer is installed.

    InferenceEngine uses its named tensor API.
    """

    if trt is None:
        raise RuntimeError("TensorRT is not installed. MotionBricks requires TensorRT 10 or newer.")

    version = str(trt.__version__)
    try:
        major_version = int(version.split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise RuntimeError(f"Cannot interpret TensorRT version {version!r}.") from exc
    if major_version < 10:
        raise RuntimeError(
            f"TensorRT {version} is not supported. MotionBricks requires TensorRT 10 or newer."
        )


def _specialize_batch_dimension(network: Any, batch_size: int) -> None:
    """Set the batch size in TensorRT before building the plan.

    A fixed batch size lets inference reuse its buffers and CUDA graph.
    The ONNX file stays unchanged. It can be used to build plans for other
    batch sizes.
    """
    for index in range(int(network.num_inputs)):
        tensor = network.get_input(index)
        shape = tuple(int(dimension) for dimension in tensor.shape)
        if not any(dimension < 0 for dimension in shape):
            continue
        if not shape or shape[0] != -1 or any(dimension <= 0 for dimension in shape[1:]):
            raise EngineBuildError(
                f"Only the leading batch dimension may be dynamic: {tensor.name!r}: {shape}."
            )
        tensor.shape = (batch_size, *shape[1:])


def _require_static_network(network: Any) -> None:
    dynamic_tensors: list[str] = []
    for tensor_kind, count, getter in (
        ("input", network.num_inputs, network.get_input),
        ("output", network.num_outputs, network.get_output),
    ):
        for index in range(int(count)):
            tensor = getter(index)
            shape = tuple(int(dimension) for dimension in tensor.shape)
            if any(dimension < 0 for dimension in shape):
                dynamic_tensors.append(f"{tensor_kind} {tensor.name!r}: {shape}")
    if dynamic_tensors:
        raise EngineBuildError(
            "InferenceEngine requires static shapes after batch specialization. Dynamic tensors: "
            + ", ".join(dynamic_tensors)
            + "."
        )
