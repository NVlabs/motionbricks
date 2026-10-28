# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine setup and generation flow that the SOMA and Unitree G1 engines share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import warp as wp

from motionbricks.inference.builder import TensorRTBuildOptions
from motionbricks.inference.engine import InferenceEngine

from .config import MotionBricksInferenceConfig
from .contract import valid_frame_count
from .generation import GenerationTimer, read_token_count

if TYPE_CHECKING:
    from motionbricks.animation.animation_buffer import AnimationBuffer
    from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer

    from .soma.buffers import PoseToken
    from .unitree_g1.buffers import DOFToken


class MotionBricksEngine(InferenceEngine, ABC):
    """Generate motion from recent motion and target poses with a MotionBricks model.

    Each rig engine sets contract, creates its tokens and output buffer, and supplies
    the steps that depend on its pose layout: _initialize_pose_layout(),
    _prepare_generation(), _encode_guide_poses(), _decode_generated_poses(), and
    _post_process_generated_poses().
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        model_to_runtime: wp.transform = wp.transform_identity(),
        backend: str = "auto",
        build_options: TensorRTBuildOptions | None = None,
        stream: wp.Stream | None = None,
        cache_dir: str | Path = ".trt-cache",
        force_rebuild: bool = False,
        bake_cuda_graph: bool = True,
        logger_severity: str = "INFO",
        build_progress_monitor: Any | None = None,
    ) -> None:
        """Load or build the inference plan, then check it against the model's contract.

        Args:
            onnx_path: ONNX model to run.
            model_to_runtime: Transform from model space to world space. The default is
                the identity.
            backend: Inference backend: "auto", "tensorrt", "directml", or "cpu".
            build_options: Build options for the plan. None uses the defaults.
            stream: Stream for inference. None creates a new stream.
            cache_dir: Directory that holds the cached plans (TensorRT only). A relative
                path starts at the working directory.
            force_rebuild: Build a new plan even when a cached one matches.
            bake_cuda_graph: Capture inference into a CUDA graph on the first
                generation (TensorRT only).
            logger_severity: Least severe TensorRT message to forward:
                INTERNAL_ERROR, ERROR, WARNING, INFO, or VERBOSE.
            build_progress_monitor: Progress monitor for the build. None reports no progress.
        """
        super().__init__(
            onnx_path,
            backend=backend,
            build_options=build_options,
            stream=stream,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
            logger_severity=logger_severity,
            build_progress_monitor=build_progress_monitor,
        )
        try:
            self.contract.validate_tensor_specs(self.tensor_specs)
        except Exception:
            self.close()
            raise

        self._use_cuda_graph = bake_cuda_graph
        self._model_to_runtime = wp.transform(model_to_runtime)
        self._runtime_to_model = wp.transform_inverse(self._model_to_runtime)
        self.timer = GenerationTimer(self.stream)
        self._alignment = wp.empty(1, dtype=wp.transform, device=self.device)
        self._generated_frame_count = wp.empty(1, dtype=wp.int32, device=self.device)
        self._initialize_pose_layout()

    def generate_animation(
        self,
        context: PoseToken | DOFToken,
        target: PoseToken | DOFToken,
        output: AnimationBuffer | DOFAnimationBuffer,
        *,
        config: MotionBricksInferenceConfig | None = None,
    ) -> AnimationBuffer | DOFAnimationBuffer:
        """Generate motion at a sample rate of 30 Hz into output, and return output.

        Update the context and target tokens on self.stream before calling. It waits
        for the generation to finish. timer.milliseconds then holds the time of each
        stage.

        Args:
            context: Context token with four poses, their root in world space.
            target: Target token with one to four poses, their root in world space.
            output: Animation buffer from create_output(). Its num_frames gives the number
                of valid frames. Its foot_contacts holds the predicted foot contacts.
            config: Generation settings for the allowed token counts, heading
                alignment, and endpoint correction. None uses the defaults.
        """
        self._ensure_open()
        config = config or MotionBricksInferenceConfig()
        self._prepare_generation(context, target, output)
        output.num_frames = 0

        with self.timer.stage("encode"):
            self._encode_guide_poses(context, target, config)
        if self._use_cuda_graph and not self.cuda_graph_baked:
            self.bake_cuda_graph()
        with self.timer.stage("inference"):
            self.enqueue_async()
        with self.timer.stage("decode"):
            self._decode_generated_poses(output)
        if config.correct_generated_diff:
            with self.timer.stage("post_process"):
                self._post_process_generated_poses(context, target, output)
        self.synchronize()
        self.timer.read()

        token_count = read_token_count(self, config)
        output.num_frames = valid_frame_count(token_count)
        return output

    @abstractmethod
    def _initialize_pose_layout(self) -> None:
        """Allocate what the rig's pose layout needs. The constructor calls it last."""

    @abstractmethod
    def _prepare_generation(self, context, target, output) -> None:
        """Check the tokens and output against the pose layout, and prepare for them."""

    @abstractmethod
    def _encode_guide_poses(self, context, target, config: MotionBricksInferenceConfig) -> None:
        """Align the context and target poses, and write them and the conditions into the inputs."""

    @abstractmethod
    def _decode_generated_poses(self, output) -> None:
        """Decode the prediction into output, and write the valid frame count."""

    @abstractmethod
    def _post_process_generated_poses(self, context, target, output) -> None:
        """Match the generated motion's endpoints to the last context pose and first target pose."""


__all__ = ["MotionBricksEngine"]
