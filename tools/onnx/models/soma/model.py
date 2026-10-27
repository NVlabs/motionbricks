# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"Build SOMA inference from the released checkpoint states."

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch as t
from torch import nn
from types import SimpleNamespace
from ..representation import GlobalRootGlobalJoints, Stats
from ..vqvae import VQVAE
from .networks import pose_mmm_network, root_mmm_network
from .inference import mmm_inference_class
from .layers import DoubleCondDecoder
from .rotations import (
    cont6d_to_matrix,
    matrix_to_cont6d,
    change_t_pose_global_mats_onnx_compatible,
)

INPUT_NAMES = (
    "global_root_values",
    "has_global_root_values",
    "local_root_values",
    "has_local_root_values",
    "local_poses",
    "has_local_poses",
    "allowed_pred_num_tokens",
)
OUTPUT_NAMES = ("pred_global_motions", "pred_num_tokens")


def replace_contiguous_features(
    values: t.Tensor, replacement: t.Tensor, feature_slice: slice
) -> t.Tensor:
    """Replace a contiguous feature range along the last axis.

    A slice with a step or an open end raises ValueError.
    """
    if (
        feature_slice.start is None
        or feature_slice.stop is None
        or feature_slice.step not in (None, 1)
        or (feature_slice.start < 0)
        or (feature_slice.stop < feature_slice.start)
    ):
        raise ValueError(f"Invalid contiguous feature slice: {feature_slice}.")
    return torch.cat(
        [
            values[..., : feature_slice.start],
            replacement,
            values[..., feature_slice.stop :],
        ],
        dim=-1,
    )


class BatchedSOMA(torch.nn.Module):
    """Batched SOMA graph from encoded inputs to predicted motion features.

    It moves the input joint rotations from the capture T pose to the standard one
    first.
    """

    def __init__(self, model: mmm_inference_class) -> None:
        super().__init__()
        self.model = model
        global_rotation_indices = self.model.global_motion_rep.indices[
            "global_rot_data"
        ]
        local_rotation_start = len(self.model.local_motion_rep.indices["ric_data"])
        self.local_rotation_slice = slice(
            local_rotation_start,
            local_rotation_start
            + len(self.model.local_motion_rep.indices["global_rot_data"]),
        )
        self.global_rotation_slice = slice(
            int(global_rotation_indices[0]), int(global_rotation_indices[-1]) + 1
        )
        self.config = {
            "num_inference_step": 1,
            "allow_pred_out_of_reach_num_tokens": False,
            "pose_token_sampling_use_argmax": True,
            "use_onnx_dynamic_batch_root_conversion": True,
        }
        self.config["final_root_pred_mode"] = "from_height_constraint_violation"
        with np.load(
            Path(__file__).resolve().parents[2] / "assets/soma_model_rig.npz",
            allow_pickle=False,
        ) as data:
            self.register_buffer("capture_offsets", torch.from_numpy(data["capture_offsets"]))
            self.register_buffer("standard_offsets", torch.from_numpy(data["standard_offsets"]))

    def forward(
        self,
        global_root_values: t.Tensor,
        has_global_root_values: t.Tensor,
        local_root_values: t.Tensor,
        has_local_root_values: t.Tensor,
        local_poses: t.Tensor,
        has_local_poses: t.Tensor,
        allowed_pred_num_tokens: t.Tensor,
    ):
        rotation_start = self.local_rotation_slice.start
        rotation_end = self.local_rotation_slice.stop
        global_joints_ric_rot6d = local_poses[:, :, rotation_start:rotation_end]
        global_rot_6d = global_joints_ric_rot6d.reshape(
            local_poses.shape[0],
            local_poses.shape[1],
            self.model.global_motion_rep.num_joints,
            6,
        )
        global_rot_mats = cont6d_to_matrix(global_rot_6d)
        global_rot_mats = change_t_pose_global_mats_onnx_compatible(
            global_rot_mats, self.capture_offsets, self.standard_offsets
        )
        retargeted_local_poses = torch.cat(
            [
                local_poses[..., :rotation_start],
                matrix_to_cont6d(global_rot_mats).reshape(
                    local_poses.shape[0],
                    local_poses.shape[1],
                    self.model.global_motion_rep.num_joints * 6,
                ),
                local_poses[..., rotation_end:],
            ],
            dim=-1,
        )
        pred_global_motions, pred_num_tokens = self.model.predict(
            global_root_values,
            has_global_root_values,
            local_root_values,
            has_local_root_values,
            retargeted_local_poses,
            has_local_poses,
            num_tokens=None,
            allowed_pred_num_tokens=allowed_pred_num_tokens,
            config=self.config,
        )
        batch_size, num_frames, _ = pred_global_motions.shape
        global_joints_ric_rot6d = pred_global_motions[:, :, self.global_rotation_slice]
        global_rot_6d = global_joints_ric_rot6d.reshape(
            batch_size, num_frames, self.model.global_motion_rep.num_joints, 6
        )
        global_rot_mats = cont6d_to_matrix(global_rot_6d)
        global_rot_mats = change_t_pose_global_mats_onnx_compatible(
            global_rot_mats, self.standard_offsets, self.capture_offsets
        )
        retargeted_global_rotations = matrix_to_cont6d(global_rot_mats).reshape(
            pred_global_motions.shape[0],
            pred_global_motions.shape[1],
            self.model.global_motion_rep.num_joints * 6,
        )
        pred_global_motions = replace_contiguous_features(
            pred_global_motions, retargeted_global_rotations, self.global_rotation_slice
        )
        return [pred_global_motions, pred_num_tokens]


class Predictor(nn.Module):
    """One predictor network with its motion representation and settings."""

    def __init__(self, backbone, motion_rep, args, pose_decoder=None):
        super().__init__()
        self.mmm_net = backbone
        self.motion_rep = motion_rep
        self.global_motion_rep = motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep = motion_rep.dual_rep.local_motion_rep
        self.args = args
        self.supporting_nets = {"pose_net": pose_decoder}
        self.vqvae_model_loaded = True


def build_soma(sources):
    """Load the root predictor, the pose predictor, and the VQ-VAE motion decoder."""
    mean = next(value for path, value in sources.items() if path.endswith("mean.npy"))
    std = next(value for path, value in sources.items() if path.endswith("std.npy"))
    stats = Stats()
    stats.register_from_tensors(
        torch.from_numpy(mean).float(), torch.from_numpy(std).float()
    )
    motion_rep = GlobalRootGlobalJoints(
        fps=30,
        skeleton=SimpleNamespace(nbjoints=30),
        name="novaskel30_dual_root_global_joints",
        stats=stats,
    )
    common = dict(
        min_tokens=6,
        max_tokens=16,
        down_t=2,
        local_pose_feature="joint_positions_and_rotations_and_hip_height",
    )
    pose_args = dict(
        common,
        pose_root_mode="pose",
        pose_vqvae_motion_rep="local",
        cond_root_feature="root_without_hip_height",
        cond_root_feature_is_from_motion_rep="global",
    )
    root_args = dict(
        common,
        pose_root_mode="root",
        local_root_feature="root",
        global_root_feature="root",
        use_hard_num_token_emb_for_root_prediction=True,
    )
    pose_network_args = dict(
        pose_args,
        pose_token_mlp_num_layers=2,
        root_feat_width=256,
        pose_feat_width=640,
        token_length_feat_width=128,
        text_emb_dim=4096,
        text_embeddings=None,
        n_embd=1024,
        n_head=16,
        n_layers=16,
        pose_vqvae=dict(nb_code=1048576, code_dim=160, num_heads=5, has_codebook=True),
        root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
    )
    root_network_args = dict(
        root_args,
        pose_feat_dim=256,
        local_root_feat_dim=64,
        global_root_feat_dim=64,
        text_emb_dim=4096,
        text_embeddings=None,
        input_feat_mlp_num_layers=2,
        n_embd=512,
        n_head=16,
        n_layers_shared=6,
        n_layers_root_token=2,
        width=512,
        depth=4,
        dilation_growth_rate=3,
        activation="relu",
        norm="None",
        pose_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
        root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
    )
    pose_backbone = pose_mmm_network(motion_rep, pose_network_args)
    root_backbone = root_mmm_network(root_network_args, motion_rep)
    vqvae = VQVAE(
        decoder_type=DoubleCondDecoder,
        pose_root_mode="pose",
        motion_rep=motion_rep.dual_rep.local_motion_rep,
        feature_mode=[
            "joint_positions_and_rotations_and_hip_height",
            "pose",
            "joint_positions_and_rotations_and_hip_height",
            "root_without_hip_height_without_heading",
        ],
        nb_code=1048576,
        code_dim=160,
        output_emb_width=160,
        down_t=2,
        stride_t=2,
        width=512,
        depth=4,
        dilation_growth_rate=3,
        activation="relu",
        num_heads=5,
        norm="None",
        cond_fusion_last_layer=False,
    )
    for kind, network, prefix in (
        ("pose", pose_backbone, "mmm_net."),
        ("root", root_backbone, "mmm_net."),
        ("vqvae", vqvae, "pose_net."),
    ):
        directory = (
            f"mmm-v6-soma-{kind}-nova-4n8g-ea43dd8-fp32-v1"
            if kind != "vqvae"
            else "vqvae-v6-soma-nova-4n8g-ea43dd8-fp32-v1"
        )
        state = sources[f"{directory}/version_0/checkpoints/model-step=1500000.safetensors"]
        weights = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        network.load_state_dict(weights, strict=True)
    models = dict(
        pose=Predictor(pose_backbone, motion_rep, pose_args, vqvae),
        root=Predictor(root_backbone, motion_rep, root_args),
    )
    return BatchedSOMA(mmm_inference_class(models, pose_args, device="cpu")).eval()
