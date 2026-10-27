# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation steps that the SOMA and Unitree G1 engines share."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING
from time import perf_counter

import warp as wp

from motionbricks.inference.errors import TensorContractError

from .config import MotionBricksInferenceConfig
from .contract import (
    MODEL_MAX_TOKEN_COUNT,
    MODEL_MAX_VALID_FRAMES,
    MODEL_MIN_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
)
from .kernels import pad_token_poses_kernel, write_motionbricks_conditions_kernel

if TYPE_CHECKING:
    from motionbricks.animation.animation_buffer import AnimationBuffer
    from motionbricks.inference.engine import InferenceEngine
    from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer

    from .soma.buffers import PoseToken
    from .unitree_g1.buffers import DOFToken


GENERATION_STAGES = ("encode", "inference", "decode", "post_process")


def write_token_poses(poses: wp.array, token_poses: wp.array, stream: wp.Stream | None) -> int:
    """Copy one to four poses into a token, and pad the unused rows with the last pose.

    Args:
        poses: One to four poses laid out as [pose, value], or one pose as [value]. A
            value is a local joint transform for SOMA and a joint coordinate for G1.
        token_poses: The token's storage for four poses, with the same dtype and the same
            number of values per pose.
        stream: Stream for the work. None uses the current stream.

    Returns:
        The number of poses supplied.
    """
    if poses.dtype != token_poses.dtype or poses.ndim not in (1, 2):
        raise ValueError("Token poses must be a vector or matrix with the token's dtype.")
    if poses.ndim == 1:
        poses = poses.reshape((1, poses.shape[0]))
    num_poses, num_values = poses.shape
    if not 1 <= num_poses <= MODEL_POSES_PER_TOKEN:
        raise ValueError(f"A context or target token requires 1 to {MODEL_POSES_PER_TOKEN} poses.")
    if num_values != token_poses.shape[1]:
        raise ValueError(f"Each token pose holds {token_poses.shape[1]} values, got {num_values}.")
    wp.launch(
        pad_token_poses_kernel,
        dim=token_poses.shape,
        inputs=[poses, num_poses],
        outputs=[token_poses],
        device=token_poses.device,
        stream=stream,
    )
    return num_poses


def check_generation_buffers(
    context: PoseToken | DOFToken,
    target: PoseToken | DOFToken,
    output: AnimationBuffer | DOFAnimationBuffer,
) -> None:
    """Check the pose counts and the output buffer that every model needs.

    Each engine also checks the joints of its rig. A mismatch raises ValueError.

    Args:
        context: Context token.
        target: Target token.
        output: Animation buffer that receives the generated motion.
    """
    if context.num_poses != MODEL_POSES_PER_TOKEN:
        raise ValueError(f"Context token must contain {MODEL_POSES_PER_TOKEN} poses.")
    if not 1 <= target.num_poses <= MODEL_POSES_PER_TOKEN:
        raise ValueError(f"Target token must contain 1 to {MODEL_POSES_PER_TOKEN} poses.")
    if output.sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError(f"Output must use a sample rate of {MODEL_SAMPLE_RATE:g} Hz.")
    if output.capacity < MODEL_MAX_VALID_FRAMES:
        raise ValueError(f"Output capacity must be at least {MODEL_MAX_VALID_FRAMES} frames.")
    if output.foot_contacts is None:
        raise ValueError("Output must have foot_contacts. create_output() allocates them.")


def write_conditions(
    engine: InferenceEngine,
    num_target_poses: int,
    config: MotionBricksInferenceConfig,
) -> None:
    """Tell the model which context and target poses are given, and which token counts it may choose.

    Args:
        engine: Engine whose conditioning tensors are written.
        num_target_poses: Number of given target poses, from 1 to 4.
        config: Generation settings with the allowed token counts.
    """
    wp.launch(
        write_motionbricks_conditions_kernel,
        dim=MODEL_MAX_TOKEN_COUNT - MODEL_MIN_TOKEN_COUNT + 1,
        inputs=[num_target_poses, config.allowed_token_bits()],
        outputs=[
            engine.device_tensor("has_global_root_values"),
            engine.device_tensor("has_local_root_values"),
            engine.device_tensor("has_local_poses"),
            engine.device_tensor("allowed_pred_num_tokens"),
        ],
        stream=engine.stream,
    )


def read_token_count(engine: InferenceEngine, config: MotionBricksInferenceConfig) -> int:
    """Return the token count the model chose. Call it after the engine's stream finishes.

    A count outside config.allowed_token_counts raises TensorContractError.

    Args:
        engine: Engine that ran the generation.
        config: Generation settings with the allowed token counts.
    """
    with wp.ScopedStream(engine.stream) if engine.stream is not None else nullcontext():
        token_count = int(engine.device_tensor("pred_num_tokens").numpy()[0])
    if token_count not in config.allowed_token_counts:
        raise TensorContractError(
            f"Model selected {token_count} tokens, outside the allowed set {config.allowed_token_counts}."
        )
    return token_count


class GenerationTimer:
    """Time each stage with CUDA events or the CPU monotonic clock.

    milliseconds holds the times of the last generation, keyed by the names in
    GENERATION_STAGES. A skipped stage reads 0.
    """

    def __init__(self, stream: wp.Stream) -> None:
        """Create timing events for a CUDA stream, or use CPU timing when it is None.

        Args:
            stream: Stream whose work is timed.
        """
        self._stream = stream
        self._cpu_times = {}
        self._events = {
            stage: (
                wp.Event(device=stream.device, enable_timing=True),
                wp.Event(device=stream.device, enable_timing=True),
            )
            for stage in GENERATION_STAGES
        } if stream is not None else {}
        self._timed: set[str] = set()
        self.milliseconds = dict.fromkeys(GENERATION_STAGES, 0.0)

    @contextmanager
    def stage(self, name: str) -> Generator[None]:
        """Time the work that the with block enqueues on the stream.

        Args:
            name: Stage name from GENERATION_STAGES.
        """
        if self._stream is None:
            start_time = perf_counter()
            try:
                yield
            finally:
                self._cpu_times[name] = (perf_counter() - start_time) * 1000
            return
        start, end = self._events[name]
        self._stream.record_event(start)
        yield
        self._stream.record_event(end)
        self._timed.add(name)

    def read(self) -> None:
        """Read the stage times into milliseconds. Call it after the stream finishes."""
        if self._stream is None:
            self.milliseconds = {name: self._cpu_times.get(name, 0.0) for name in GENERATION_STAGES}
            self._cpu_times.clear()
            return
        for name, (start, end) in self._events.items():
            timed = name in self._timed
            self.milliseconds[name] = wp.get_event_elapsed_time(start, end, synchronize=False) if timed else 0.0
        self._timed.clear()
