# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move a character along a steered path with a generation loop. No viewer, no apps.

This is the continuous locomotion loop of the Runtime API page, made complete. Each update plays
the generated motion, records the pose it showed, and decides whether to generate again. The
target poses are one set of jog poses, placed ahead of the character along the steering direction.
The steering turns from +X to +Y, so the path bends left. Run it from the repository root:

    uv run python docs/examples/locomotion_loop.py
    uv run python docs/examples/locomotion_loop.py --model unitree_g1
"""

from contextlib import nullcontext

import argparse
import math

import numpy as np
import warp as wp

from motionbricks.model.contract import MODEL_POSES_PER_TOKEN
from motionbricks.runtime.motion_player import GeneratedDOFMotionPlayer, GeneratedMotionPlayer
from motionbricks.runtime.target_pose import (
    TargetPoseInferenceParams,
    adjust_target_dof_speed,
    adjust_target_pose_speed,
    place_target_dof,
    place_target_poses,
)

from _rig import JOG, load_rig, root_positions, sample_window

UPDATE_RATE = 60.0
DURATION = 9.0
# The jog poses that serve as the target. Any clip time in the jog works.
TARGET_TIME = 1.8
# The ground speed to move at, in m/s. The target poses are scaled to it. Each rig moves at about the
# speed of its jog clip. A higher G1 speed makes its feet slide.
DESIRED_SPEED = {"soma": 1.9, "unitree_g1": 1.45}
# The target sits this many seconds of travel ahead of the character.
LEAD_SECONDS = 1.0
# The transition point, in frames before the end of each generation. The next generation takes over there.
TRANSITION_EARLY_BY = 4
# Generate again when the steering direction turns by more than this many degrees.
TURN_THRESHOLD = 15.0
# The steering direction. It holds +X, turns to +Y between these two times, then holds.
TURN_START, TURN_END = 2.0, 4.0


def steering(time: float) -> np.ndarray:
    """Return the desired ground direction at a time. An application reads its controls here."""
    blend = min(max((time - TURN_START) / (TURN_END - TURN_START), 0.0), 1.0)
    angle = 0.5 * math.pi * blend
    return np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)


def canonical_target(rig, model: str, poses: wp.array) -> None:
    """Fill poses with the jog target at the origin, facing +X, at the model's desired speed.

    The clip's own placement and heading are removed, so each generation can place the poses
    wherever the character is heading.
    """
    sample_window(rig, TARGET_TIME, poses)
    roots = root_positions(poses)
    offset = roots[-1, :2] - roots[0, :2]
    heading = math.atan2(offset[1], offset[0])
    origin = wp.transform(wp.vec3(roots[0, 0], roots[0, 1], 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), heading))
    sample_window(rig, TARGET_TIME, poses, root_offset=wp.transform_inverse(origin))
    adjust = adjust_target_pose_speed if rig.skeleton is not None else adjust_target_dof_speed
    adjust(poses, DESIRED_SPEED[model], stream=rig.engine.stream)


def run(model: str) -> list[np.ndarray]:
    """Move "soma" or "unitree_g1" along the steered path. Return the pose shown at each update."""
    wp.init()
    wp.set_device("cuda:0" if wp.is_cuda_available() else "cpu")
    rig = load_rig(model, JOG)
    shown = []
    with rig.engine as engine, (wp.ScopedStream(engine.stream) if engine.stream is not None else nullcontext()):
        if rig.skeleton is not None:
            player = GeneratedMotionPlayer(engine, rig.skeleton)
            place = place_target_poses
        else:
            player = GeneratedDOFMotionPlayer(engine, rig.reference_pose)
            place = place_target_dof

        jog_poses = wp.empty((MODEL_POSES_PER_TOKEN, *rig.reference_pose.shape), dtype=rig.reference_pose.dtype)
        canonical_target(rig, model, jog_poses)
        target_poses = wp.empty_like(jog_poses)
        settings = TargetPoseInferenceParams(transition_early_by_frames=TRANSITION_EARLY_BY, num_poses_to_use=4)

        # Start at the origin in the jog pose, as if the character had held it.
        current_pose = wp.clone(jog_poses[MODEL_POSES_PER_TOKEN - 1])
        player.seed_history_from_pose(current_pose)
        foot_contacts = wp.zeros(4, dtype=wp.float32)
        generated_direction = None
        generations = 0
        delta_time = 1.0 / UPDATE_RATE

        for step in range(round(DURATION * UPDATE_RATE)):
            time = step * delta_time
            direction = steering(time)
            position = root_positions(current_pose.reshape((1, -1)))[0] * np.array([1.0, 1.0, 0.0])

            turned = generated_direction is not None and math.degrees(
                math.acos(min(1.0, float(direction @ generated_direction)))
            ) > TURN_THRESHOLD
            if generated_direction is None or turned or player.time_to_transition <= delta_time:
                ground_position = position + direction * DESIRED_SPEED[model] * LEAD_SECONDS
                place(wp.vec3(*ground_position), wp.vec3(*direction), jog_poses, target_poses, stream=engine.stream)
                player.generate(target_poses, settings)
                generated_direction = direction
                generations += 1

            player.sample(delta_time, current_pose, foot_contacts)
            # Apply target pose IK, physics, or other changes to current_pose here.
            player.record_pose(delta_time, current_pose)
            shown.append(current_pose.numpy())

            if step % round(UPDATE_RATE) == 0:
                engine.synchronize()
                print(f"t = {time:4.1f} s  steering {direction[:2].round(2)}  root {position[:2].round(2)}")
        engine.synchronize()
        print(f"{generations} generations in {DURATION:.0f} s")
    return shown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("soma", "unitree_g1"), default="soma")
    run(parser.parse_args().model)


if __name__ == "__main__":
    main()
