# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the examples: load a rig's skeleton, engine, and clip without a viewer.

Only the motionbricks package is used here, so the code carries over to an application that
does not use the demo runtime in app/.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from motionbricks.io.bvh import load_bvh
from motionbricks.io.csv import load_csv
from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.inference.options import TensorRTBuildOptions
from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE, ModelRig
from motionbricks.model.soma.contract import SOMA_N1_FP16_ONNX_PATH, SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP16_ONNX_PATH, UNITREE_G1_N1_FP32_ONNX_PATH
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "motions"
WALK = "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057"
JOG = "locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534"


@dataclass
class Rig:
    """What an example needs from one rig.

    Attributes:
        engine: The engine. Use it as a context manager.
        clip: The clip that the example samples poses from.
        skeleton: The SOMA skeleton, or None for the Unitree G1.
        reference_pose: A pose with the rig's layout.
        model_to_runtime: Transform from the clip's source space to world space, which is Z up.
    """

    engine: MotionBricksSOMAEngine | MotionBricksUnitreeG1Engine
    clip: object
    skeleton: object
    reference_pose: wp.array
    model_to_runtime: wp.transform


def load_rig(model: str, clip_name: str = WALK, *, fp16: bool = False) -> Rig:
    """Load a clip of the bundled locomotion set and create the engine for "soma" or "unitree_g1"."""
    if ModelRig(model) is ModelRig.SOMA:
        skeleton = load_skeleton_from_usd(
            get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH,
        )
        clip = load_bvh(str(ASSETS / "soma" / f"{clip_name}.bvh"), skeleton)
        # SOMA model space and BVH clips are Y up. World space is Z up.
        to_runtime = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
        engine = MotionBricksSOMAEngine(
            SOMA_N1_FP16_ONNX_PATH if fp16 else SOMA_N1_FP32_ONNX_PATH,
            model_to_runtime=to_runtime, build_options=TensorRTBuildOptions(enable_fp16=fp16),
        )
        return Rig(engine, clip, skeleton, skeleton.reference_local_transforms, to_runtime)
    clip = load_csv(ASSETS / "unitree_g1" / f"{clip_name}.csv", device=wp.get_device())
    # G1 model space and CSV clips are already Z up.
    engine = MotionBricksUnitreeG1Engine(
        UNITREE_G1_N1_FP16_ONNX_PATH if fp16 else UNITREE_G1_N1_FP32_ONNX_PATH,
        build_options=TensorRTBuildOptions(enable_fp16=fp16),
    )
    reference = wp.zeros(36, dtype=wp.float32)
    clip.sample(0.0, reference)
    return Rig(engine, clip, None, reference, wp.transform_identity())


def sample_window(rig: Rig, start_time: float, poses: wp.array, root_offset: wp.transform | None = None) -> None:
    """Sample four clip poses at 30 Hz from start_time into poses, then move their roots by root_offset."""
    root_offset = wp.mul(root_offset, rig.model_to_runtime) if root_offset is not None else rig.model_to_runtime
    for index in range(MODEL_POSES_PER_TOKEN):
        rig.clip.sample(
            start_time + index / MODEL_SAMPLE_RATE, poses[index], root_offset=root_offset, stream=rig.engine.stream,
        )


def root_positions(poses: wp.array) -> np.ndarray:
    """Return the root XYZ of each pose of either layout. Both start with the root position."""
    return poses.numpy().reshape(poses.shape[0], -1)[:, :3]
