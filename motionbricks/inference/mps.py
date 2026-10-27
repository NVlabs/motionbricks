# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the released PyTorch models for inference on Apple's Metal GPU."""

from __future__ import annotations

from importlib.util import find_spec
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def is_available_for_model(onnx_path: str | Path) -> bool:
    """Check whether the optional Metal backend is ready for a bundled FP32 model.

    The full checksums are checked by load_model after selection. A partial
    checkpoint download leaves automatic selection on the CPU.
    """
    if find_spec("torch") is None or find_spec("safetensors") is None:
        return False

    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP32_ONNX_PATH

    try:
        from tools.onnx.checksums import CHECKPOINTS
    except ImportError as exc:
        logger.warning("Metal model sources are unavailable; using CPU inference: %s", exc)
        return False

    path = Path(onnx_path).expanduser().resolve()
    if path == UNITREE_G1_N1_FP32_ONNX_PATH.resolve():
        kind = "g1"
    elif path == SOMA_N1_FP32_ONNX_PATH.resolve():
        kind = "soma"
    else:
        return False

    checkpoints = Path(os.environ.get("MOTIONBRICKS_CHECKPOINT_DIR", "checkpoints/MotionBricks")).expanduser()
    if (checkpoints / kind).is_dir():
        checkpoints = checkpoints / kind
    release = CHECKPOINTS[kind]
    if not (checkpoints / release["files"][0]["path"]).is_file():
        checkpoints = checkpoints / release["root_directory"]
    if not all((checkpoints / entry["path"]).is_file() for entry in release["files"]):
        return False

    try:
        import torch

        return torch.backends.mps.is_available()
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Metal backend is unavailable; using CPU inference: %s", exc)
        return False


def load_model(onnx_path: str | Path):
    """Load the matching verified FP32 checkpoint, returning the model and tensor orders.

    Only the bundled models are supported: an arbitrary ONNX file must not silently
    run a different set of weights. The caller owns and releases the returned model.
    """
    try:
        import torch
        from tools.onnx.checkpoints import checkpoint_root, load_sources, sha256
        from tools.onnx.checksums import CHECKPOINTS
    except ImportError as exc:
        raise RuntimeError(
            "Metal inference requires the optional dependencies. Run: uv sync --extra metal"
        ) from exc

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch cannot access an Apple Metal GPU on this system.")
    logger.info("Using Apple Metal GPU: %s", torch.backends.mps.get_name())
    compile_mode = os.environ.get("MOTIONBRICKS_MPS_COMPILE", "0")
    if compile_mode not in {"0", "1"}:
        raise ValueError("MOTIONBRICKS_MPS_COMPILE must be 0 or 1.")

    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP32_ONNX_PATH

    path = Path(onnx_path).expanduser().resolve()
    if path == UNITREE_G1_N1_FP32_ONNX_PATH.resolve():
        from tools.onnx.models.g1 import INPUT_NAMES, OUTPUT_NAMES, build_g1

        kind, builder, input_names, output_names = "g1", build_g1, INPUT_NAMES, OUTPUT_NAMES
        onnx_sha256 = "79258d3613988b74ab84a530352ea600bbe78a1048584f059e231e44641cea16"
    elif path == SOMA_N1_FP32_ONNX_PATH.resolve():
        from tools.onnx.models.soma import INPUT_NAMES, OUTPUT_NAMES, build_soma

        kind, builder, input_names, output_names = "soma", build_soma, INPUT_NAMES, OUTPUT_NAMES
        onnx_sha256 = "856419dc72aeb14e591eebf4dd0ea05734a2506ddb00bcd39dd6dac2ad60f4fb"
    else:
        raise ValueError(
            "Metal inference supports the bundled FP32 SOMA and Unitree G1 models only. "
            "Use the CPU or Core ML provider for other ONNX models."
        )

    # The MPS implementation loads checkpoint weights rather than the ONNX file.
    # Refuse a replaced model at the same path instead of silently using different weights.
    try:
        matches_release = sha256(path) == onnx_sha256
    except OSError as exc:
        raise ValueError(f"Cannot read the bundled ONNX model {path}; run git lfs pull.") from exc
    if not matches_release:
        raise ValueError(
            f"The ONNX model at {path} does not match the release supported by Metal inference. "
            "Restore the bundled model or use the CPU or Core ML provider."
        )

    checkpoints = Path(os.environ.get("MOTIONBRICKS_CHECKPOINT_DIR", "checkpoints/MotionBricks")).expanduser()
    if (checkpoints / kind).is_dir():
        checkpoints = checkpoints / kind
    release = CHECKPOINTS[kind]
    root = checkpoint_root(checkpoints, release)
    logger.info("Loading verified %s checkpoint from %s for Metal inference.", kind, root)
    try:
        sources = load_sources(root, release)
    except ValueError as exc:
        raise ValueError(
            f"{exc} Download the released {kind} checkpoint as described in "
            "tools/onnx/README.md, or set MOTIONBRICKS_CHECKPOINT_DIR."
        ) from exc
    model = builder(sources)
    del sources
    model.eval().to("mps")
    # The runtime intentionally masks the last local-root velocity in each
    # context. This is expected, and does not require a GPU-to-CPU warning check.
    inferencer = model.core.inferencer if kind == "g1" else model.model
    inferencer._warned_missing_frames = True

    # PyTorch's nested-tensor TransformerEncoder fast path is unsupported by MPS.
    # Disable it per encoder rather than changing the process-wide MHA setting.
    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoder):
            module.use_nested_tensor = False

    if compile_mode == "1":
        # Compiling only the neural networks keeps the surrounding, dynamic motion
        # logic in eager mode. First inference can take substantially longer.
        logger.info("Enabling experimental Metal model compilation; the first generation may be slow.")
        if kind == "g1":
            inferencer._root_model.backbone_net = torch.compile(inferencer._root_model.backbone_net)
            inferencer._pose_model.backbone_net = torch.compile(inferencer._pose_model.backbone_net)
        else:
            inferencer._mmm_root_model.mmm_net = torch.compile(inferencer._mmm_root_model.mmm_net)
            inferencer._mmm_pose_model.mmm_net = torch.compile(inferencer._mmm_pose_model.mmm_net)
        inferencer._vqvae_pose_model.decoder = torch.compile(inferencer._vqvae_pose_model.decoder)
    return model, input_names, output_names


def run_model(model, input_names, output_names, inputs):
    """Run GPU inference and return CPU NumPy arrays for the Warp CPU buffers."""
    import torch

    with torch.inference_mode():
        tensors = [torch.from_numpy(inputs[name]).to("mps") for name in input_names]
        outputs = model(*tensors)
        # Copying to CPU also waits for the Metal work before Warp reads the outputs.
        return {
            name: output.to("cpu").numpy()
            for name, output in zip(output_names, outputs, strict=True)
        }
