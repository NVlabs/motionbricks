# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load model engines while the viewer shows progress."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import sys
import time
from typing import Any, TypeVar

import newton
import warp as wp

from motionbricks.inference.engine import InferenceEngine
from motionbricks.inference.options import TensorRTBuildOptions

if sys.platform != "darwin":
    import tensorrt as trt
    from motionbricks.inference.builder import create_logger
    from motionbricks.inference.progress import TensorRTBuildProgress, TensorRTBuildProgressMonitor
else:
    class TensorRTBuildProgressMonitor:
        def snapshot(self):
            return None

        def request_cancel(self):
            pass

from motionbricks.model.contract import ModelRig
from motionbricks.model.soma.contract import SOMA_N1_FP16_ONNX_PATH, SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.model.unitree_g1.contract import (
    UNITREE_G1_N1_FP16_ONNX_PATH,
    UNITREE_G1_N1_FP32_ONNX_PATH,
)
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine

from ..runtime.common.rig import RIG_LABELS, model_to_runtime


logger = logging.getLogger(__name__)

_EngineT = TypeVar("_EngineT", bound=InferenceEngine)
# Each rig's engine class and its bundled FP32 and FP16 models.
_ENGINES = {
    ModelRig.SOMA: (MotionBricksSOMAEngine, SOMA_N1_FP32_ONNX_PATH, SOMA_N1_FP16_ONNX_PATH),
    ModelRig.UNITREE_G1: (MotionBricksUnitreeG1Engine, UNITREE_G1_N1_FP32_ONNX_PATH, UNITREE_G1_N1_FP16_ONNX_PATH),
}
_BUILD_PROGRESS_BAR_WIDTH = 16
_BUILD_PROGRESS_UPDATE_INTERVAL = 0.1


def _format_engine_loading_text(progress: TensorRTBuildProgress) -> str:
    if progress is None:
        return "Loading ONNX model..."
    if progress.phase_name is None:
        return "Checking TensorRT plan cache..."

    filled = int(progress.fraction * _BUILD_PROGRESS_BAR_WIDTH)
    progress_bar = "#" * filled + "-" * (_BUILD_PROGRESS_BAR_WIDTH - filled)
    percentage = int(progress.fraction * 100.0)
    phase_name = progress.phase_name
    if len(phase_name) > 36:
        phase_name = phase_name[:33] + "..."
    return (
        f"Building TensorRT plan: {phase_name} "
        f"[{progress_bar}] {percentage}%"
    )


def require_inference_backend() -> None:
    """Check that the current device can run its inference backend."""

    device = wp.get_device()
    if sys.platform == "darwin" and device.is_cpu:
        return
    if device.is_cuda:
        try:
            trt.Runtime(create_logger("ERROR"))
            return
        except Exception:
            pass  # TensorRT has already logged why it cannot start.
    logger.error(
        "TensorRT cannot run on %s. MotionBricks needs an NVIDIA GPU with a recent driver.", device.alias
    )
    raise SystemExit(1)


def create_engine(
    viewer: newton.viewer.ViewerGL, rig: ModelRig, fp16: bool,
) -> MotionBricksSOMAEngine | MotionBricksUnitreeG1Engine:
    """Create a rig's engine from its bundled ONNX model."""
    logger.info(
        "Loading MotionBricks %s in %s.",
        RIG_LABELS[rig], "mixed FP16" if fp16 else "FP32",
    )
    engine_type, fp32_path, fp16_path = _ENGINES[rig]
    return create_model_engine(
        viewer,
        engine_type,
        fp16_path if fp16 else fp32_path,
        model_to_runtime=model_to_runtime(rig),
        build_options=TensorRTBuildOptions(enable_fp16=fp16),
    )


def create_model_engine(
    viewer: newton.viewer.ViewerGL,
    engine_type: type[_EngineT],
    onnx_path: str | Path,
    **engine_options: Any,
) -> _EngineT:
    """Load an engine in a worker while drawing progress on the viewer thread.

    Constructor options are forwarded to the engine, which must accept
    ``build_progress_monitor`` for progress reporting and cancellation.
    The viewer's UI is hidden during the build so it cannot start other work.
    Closing the viewer cancels the load and raises SystemExit.
    """

    progress_monitor = TensorRTBuildProgressMonitor()
    device = wp.get_device()
    viewer_running = True
    show_ui = viewer.show_ui
    viewer.show_ui = False

    def load_engine() -> _EngineT:
        wp.set_device(device)
        return engine_type(
            onnx_path,
            build_progress_monitor=progress_monitor,
            **engine_options,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_engine)
        try:
            while not future.done():
                if viewer_running:
                    viewer.show_loading_splash(
                        _format_engine_loading_text(progress_monitor.snapshot())
                    )
                    viewer.begin_frame(0.0)
                    viewer.end_frame()
                    viewer_running = viewer.is_running()
                    if not viewer_running:
                        progress_monitor.request_cancel()
                time.sleep(_BUILD_PROGRESS_UPDATE_INTERVAL)
            try:
                engine = future.result()
            except Exception:
                if not viewer_running:
                    raise SystemExit(0) from None
                raise
            viewer_running = viewer_running and viewer.is_running()
        finally:
            if viewer_running:
                viewer.hide_loading_splash()
            viewer.show_ui = show_ui

    if not viewer_running:
        engine.close()
        raise SystemExit(0)
    return engine
