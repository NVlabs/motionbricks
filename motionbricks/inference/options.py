"""Configuration shared by inference backends."""

from dataclasses import asdict, dataclass
from typing import Any


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
