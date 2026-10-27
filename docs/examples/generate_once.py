# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate motion between clip poses, then chain a second generation. No viewer, no apps.

Each generation takes four context poses and one to four target poses. The first context and both
targets come from one walk clip. The second generation reads its context from the last four poses
of the first, which is the loop that makes continuous motion. Run it from the repository root:

    uv run python docs/examples/generate_once.py
    uv run python docs/examples/generate_once.py --model unitree_g1
"""

from contextlib import nullcontext

import argparse

import numpy as np
import warp as wp

from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE

from _rig import load_rig, sample_window

# The context ends at this clip time. Each generation targets the clip poses at the next time.
CONTEXT_END = 1.0
TARGET_TIMES = (1.8, 3.4)


def generate(model: str) -> list[np.ndarray]:
    """Run both generations for "soma" or "unitree_g1". Return the valid poses of each, as arrays."""
    wp.init()
    wp.set_device("cuda:0" if wp.is_cuda_available() else "cpu")
    rig = load_rig(model)
    motions = []
    # Initial model loading can take several minutes.
    with rig.engine as engine, (wp.ScopedStream(engine.stream) if engine.stream is not None else nullcontext()):
        skeleton_args = (rig.skeleton,) if rig.skeleton is not None else ()
        context = engine.create_input(*skeleton_args)
        target = engine.create_input(*skeleton_args)
        output = engine.create_output(*skeleton_args)
        context_poses = wp.empty_like(context.poses)
        target_poses = wp.empty_like(context.poses)

        # The context is the four poses that end at CONTEXT_END.
        sample_window(rig, CONTEXT_END - (MODEL_POSES_PER_TOKEN - 1) / MODEL_SAMPLE_RATE, context_poses)
        for target_time in TARGET_TIMES:
            sample_window(rig, target_time, target_poses)
            context.update_from_poses(context_poses, stream=engine.stream)
            target.update_from_poses(target_poses, stream=engine.stream)
            engine.generate_animation(context, target, output)
            engine.synchronize()

            # The output buffer has room for the longest result. Only num_frames frames are valid.
            frames = output.num_frames
            motions.append(output.poses[:frames].numpy())
            root = output.root_transforms()[:frames, :3]
            print(
                f"generated {frames} frames ({frames / MODEL_SAMPLE_RATE:.2f} s), "
                f"root from {root[0].round(2)} to {root[-1].round(2)}"
            )

            # Play the whole motion, so the last four poses are what the character showed.
            # An application that moves on early keeps the last four poses it played instead.
            context_poses = wp.clone(output.poses[frames - MODEL_POSES_PER_TOKEN:frames])
    return motions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("soma", "unitree_g1"), default="soma")
    generate(parser.parse_args().model)


if __name__ == "__main__":
    main()
