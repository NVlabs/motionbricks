# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure generation on one bundled walking clip. Run from the repository root.

For example:
    MOTIONBRICKS_ONNX_PROVIDER=cpu uv run --extra metal python tools/benchmark_inference.py --model unitree_g1
    MOTIONBRICKS_ONNX_PROVIDER=mps uv run --extra metal python tools/benchmark_inference.py --model unitree_g1

Run providers separately, on an otherwise idle machine. The measured generation time
includes CPU-side encoding, inference, decoding and endpoint correction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "examples"))
from _rig import load_rig, sample_window  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("soma", "unitree_g1"), default="unitree_g1")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be nonnegative and --runs must be positive")

    wp.init()
    wp.set_device("cuda:0" if wp.is_cuda_available() else "cpu")
    start = perf_counter()
    rig = load_rig(args.model, fp16=args.precision == "fp16")
    load_ms = (perf_counter() - start) * 1000
    with rig.engine as engine:
        skeleton = (rig.skeleton,) if rig.skeleton is not None else ()
        context = engine.create_input(*skeleton)
        target = engine.create_input(*skeleton)
        output = engine.create_output(*skeleton)
        context_poses = wp.empty_like(context.poses)
        target_poses = wp.empty_like(target.poses)
        sample_window(rig, 0.9, context_poses)
        sample_window(rig, 1.8, target_poses)
        context.update_from_poses(context_poses, stream=engine.stream)
        target.update_from_poses(target_poses, stream=engine.stream)

        durations = []
        stage_times = {stage: [] for stage in engine.timer.milliseconds}
        for index in range(args.warmup + args.runs):
            start = perf_counter()
            engine.generate_animation(context, target, output)
            elapsed_ms = (perf_counter() - start) * 1000
            if index >= args.warmup:
                durations.append(elapsed_ms)
                for stage, elapsed in engine.timer.milliseconds.items():
                    stage_times[stage].append(elapsed)

    print(f"Model: {args.model}; precision: {args.precision}; provider: {getattr(engine, 'provider', 'tensorrt')}")
    print(f"Load: {load_ms:.1f} ms; warmup: {args.warmup}; measured: {args.runs}")
    print(f"Generation ms: median {np.median(durations):.1f}, p95 {np.percentile(durations, 95):.1f}")
    print("Median stage ms: " + ", ".join(f"{name} {np.median(times):.1f}" for name, times in stage_times.items()))


if __name__ == "__main__":
    main()
