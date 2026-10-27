# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"SOMA pose and root predictors."

from __future__ import annotations

from functools import cached_property
from typing import Dict, Optional
import numpy as np
import torch
import torch as t
from torch import nn

from ..layers import FCBlock as mlp, PositionEmbedding
from .layers import DoubleCondDecoder
from ..features import (
    extract_feature_from_motion_rep,
    convert_sparse_cond_to_dense_cond_if_needed,
)
from .features import numeric_mask_union
from ..quantizer import resolve_multihead_vq_dimensions


class pose_mmm_network(nn.Module):
    """Transformer that predicts the pose tokens for SOMA."""

    def __init__(self, motion_rep: nn.Module, args: Dict = None):
        super().__init__()
        self.motion_rep = motion_rep
        self.global_motion_rep = motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep = motion_rep.dual_rep.local_motion_rep
        self._args = args
        self._build_transformer_backbone()
        self._build_input_embeddings_projections()
        self.register_buffer("initted", torch.Tensor([False]).to(dtype=t.bool))

    def _semantic_vq_dimensions(self, vqvae: str):
        config = self._args[vqvae]
        if (
            config.get("codes_per_head") is None
            and config.get("code_dim_per_head") is None
        ):
            return None
        return resolve_multihead_vq_dimensions(
            num_heads=int(config["num_heads"]),
            codes_per_head=config.get("codes_per_head"),
            code_dim_per_head=config.get("code_dim_per_head"),
            nb_code=config.get("nb_code"),
            code_dim=config.get("code_dim"),
        )

    def _build_transformer_backbone(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._args["n_embd"], nhead=self._args["n_head"], batch_first=True
        )
        self._transformer_model = nn.TransformerEncoder(
            encoder_layer, num_layers=self._args["n_layers"]
        )

    def _build_input_embeddings_projections(self):
        num_pose_heads, _ = self.get_num_heads()
        num_pose_tokens, _ = self.get_num_codes(include_aug_tokens=False)
        if self._args["pose_vqvae"].get("has_codebook", True):
            pose_dimensions = self._semantic_vq_dimensions("pose_vqvae")
            pose_token_dim = (
                pose_dimensions.code_dim_per_head
                if pose_dimensions is not None
                else self._args["pose_vqvae"]["code_dim"] // num_pose_heads
            )
        else:
            pose_token_dim = self._args["n_embd"] // num_pose_heads
        self._pose_token_emb = t.nn.Embedding(
            num_pose_heads * self.NUM_WITH_AUG_POSE_TOKENS, pose_token_dim
        )
        self._proj_pose_token_emb = mlp(
            num_layers=self._args["pose_token_mlp_num_layers"],
            layer_width=self._args["pose_feat_width"],
            size_in=num_pose_heads * pose_token_dim,
            size_out=self._args["pose_feat_width"],
        )
        num_frames_per_token = self.get_num_frames_per_token()
        if self._args["cond_root_feature_is_from_motion_rep"] == "local":
            self._args["local_root_dim"] = extract_feature_from_motion_rep(
                t.zeros([1, 1, 1000]),
                self.local_motion_rep,
                self._args["cond_root_feature"],
            ).shape[2]
        else:
            self._args["local_root_dim"] = extract_feature_from_motion_rep(
                t.zeros([1, 1, 1000]),
                self.global_motion_rep,
                self._args["cond_root_feature"],
            ).shape[2]
        self._proj_local_root_values = t.nn.Linear(
            self._args["local_root_dim"] * num_frames_per_token,
            self._args["root_feat_width"],
        )
        self._args["local_pose_dim"] = extract_feature_from_motion_rep(
            t.zeros([1, 1, 1000]),
            self.local_motion_rep,
            self._args["local_pose_feature"],
        ).shape[2]
        self._proj_local_pose = t.nn.Linear(
            self._args["local_pose_dim"],
            self._args["pose_feat_width"] // num_frames_per_token,
        )
        assert self._args["pose_feat_width"] % self.get_num_frames_per_token() == 0, (
            "pose_feat_width needs to be divisible by num_frames_per_token"
        )
        self._proj_num_valid_positions = t.nn.Embedding(
            self._args["max_tokens"] - self._args["min_tokens"] + 1,
            self._args["token_length_feat_width"],
        )
        self._proj_input = t.nn.Sequential(
            t.nn.Linear(
                self._args["pose_feat_width"]
                + self._args["root_feat_width"]
                + self._args["token_length_feat_width"],
                self._args["n_embd"],
            ),
            t.nn.ReLU(),
        )
        self._position_emb = PositionEmbedding(
            seq_length=self._args["max_tokens"], dim=self._args["n_embd"]
        )
        self._proj_pose_output_logit = t.nn.Linear(
            self._args["n_embd"], num_pose_heads * num_pose_tokens
        )
        if self.ACCEPT_TEXT_EMB_INPUT:
            self._proj_text_embeddings = mlp(
                num_layers=self._args["pose_token_mlp_num_layers"],
                layer_width=self._args["n_embd"],
                size_in=self._args["text_emb_dim"],
                size_out=self._args["n_embd"],
            )

    def forward(
        self,
        pose_tokens: t.Tensor,
        local_root_values: t.Tensor,
        pose_cond: t.Tensor,
        has_pose_cond: t.Tensor,
        num_tokens: t.Tensor,
        text_embeddings: t.Tensor = None,
        has_text_embeddings: t.Tensor = None,
    ):
        batch_size, num_positions, num_pose_heads = pose_tokens.shape
        device = pose_tokens.device
        num_frames_per_token = self.get_num_frames_per_token()
        local_root_values = local_root_values.reshape(
            [
                batch_size,
                num_positions,
                num_frames_per_token * self._args["local_root_dim"],
            ]
        )
        root_embedding = self._proj_local_root_values(local_root_values)
        dense_pose_cond, dense_has_pose_cond = (
            convert_sparse_cond_to_dense_cond_if_needed(
                pose_cond, has_pose_cond, num_positions * num_frames_per_token
            )
        )
        pose_cond_embedding = self._proj_local_pose(dense_pose_cond)
        pose_cond_embedding = pose_cond_embedding.view(
            [
                batch_size,
                num_positions * num_frames_per_token,
                self._args["pose_feat_width"] // num_frames_per_token,
            ]
        )
        pose_token_id_offsets = (
            torch.arange(num_pose_heads).view([1, 1, num_pose_heads]).to(device=device)
            * self.NUM_WITH_AUG_POSE_TOKENS
        )
        pose_tokens = pose_tokens + pose_token_id_offsets
        pose_token_embedding = self._pose_token_emb(pose_tokens)
        pose_token_embedding = self._proj_pose_token_emb(
            pose_token_embedding.view([batch_size, num_positions, -1])
        )
        pose_token_embedding = pose_token_embedding.view(
            [
                batch_size,
                num_positions * num_frames_per_token,
                self._args["pose_feat_width"] // num_frames_per_token,
            ]
        )
        dense_has_pose_cond = dense_has_pose_cond[:, :, None].float()
        pose_embedding = (
            pose_cond_embedding * dense_has_pose_cond
            + pose_token_embedding * (1 - dense_has_pose_cond)
        )
        pose_embedding = pose_embedding.view(
            [batch_size, num_positions, self._args["pose_feat_width"]]
        )
        num_token_embedding = self._proj_num_valid_positions(
            num_tokens.reshape([batch_size, 1]) - self._args["min_tokens"]
        ).expand([-1, num_positions, -1])
        position_ids = t.arange(num_positions).to(device=device)
        position_emb = self._position_emb.embed[position_ids].view(
            [1, num_positions, self._args["n_embd"]]
        )
        input_embeddings = (
            self._proj_input(
                t.cat([pose_embedding, root_embedding, num_token_embedding], dim=-1)
            )
            + position_emb
        )
        token_mask = t.arange(num_positions).to(device) < num_tokens.view(
            [batch_size, -1]
        )
        if self.ACCEPT_TEXT_EMB_INPUT and text_embeddings is not None:
            text_embeddings = self._proj_text_embeddings(text_embeddings)[:, None, :]
            input_embeddings = t.concat([text_embeddings, input_embeddings], dim=1)
            mask = t.zeros([batch_size, 1 + num_positions], device=device).bool()
            mask[:, :1] = ~has_text_embeddings.bool()
            mask[:, 1:] = ~token_mask.bool()
        else:
            mask = ~token_mask.bool()
        output = self._transformer_model(input_embeddings, src_key_padding_mask=mask)[
            :, -num_positions:
        ]
        pred_logits = self._proj_pose_output_logit(output).reshape(
            [batch_size, num_positions, num_pose_heads, -1]
        )
        return {"pose_logits": pred_logits}

    def get_num_codes(self, include_aug_tokens: bool):
        """Return the code counts of the pose and root codebooks.

        With include_aug_tokens, the counts include the extra tokens, such as the mask
        token.
        """
        num_codes_per_head = {}
        for vqvae in ["pose_vqvae", "root_vqvae"]:
            semantic_dimensions = self._semantic_vq_dimensions(vqvae)
            if semantic_dimensions is not None:
                num_codes_per_head[vqvae] = semantic_dimensions.codes_per_head
                continue
            nb_code = self._args[vqvae]["nb_code"]
            len_nb_code = len(np.array(self._args[vqvae]["nb_code"]).reshape([-1]))
            if len_nb_code > 1:
                assert np.all(np.array(nb_code) == nb_code[0]), (
                    "fsq config should have the same number of codes for each head"
                )
                num_codes_per_head[vqvae] = nb_code[0]
            else:
                num_heads = self._args[vqvae]["num_heads"]
                code_dim = self._args[vqvae]["code_dim"]
                if self._args[vqvae].get("has_codebook", True):
                    assert code_dim % num_heads == 0, (
                        "code dim cannot be divided by the number of heads."
                    )
                    num_codes_per_head[vqvae] = int(
                        round(2 ** (np.log2(nb_code) / num_heads))
                    )
                    assert num_codes_per_head[vqvae] ** num_heads == nb_code, (
                        "the specified number of code is not compatible with the number of heads."
                    )
                else:
                    num_codes_per_head[vqvae] = nb_code
        if include_aug_tokens:
            return (
                num_codes_per_head["pose_vqvae"] + 1,
                num_codes_per_head["root_vqvae"] + 1,
            )
        else:
            return (num_codes_per_head["pose_vqvae"], num_codes_per_head["root_vqvae"])

    def get_num_heads(self):
        """Return the head counts of the pose and root codebooks."""
        return (
            self._args["pose_vqvae"]["num_heads"],
            self._args["root_vqvae"]["num_heads"],
        )

    def get_num_frames_per_token(self):
        """Return the frames per motion token, 2 ** down_t."""
        return 2 ** self._args["down_t"]

    @cached_property
    def NUM_WITH_AUG_POSE_TOKENS(self):
        """Number of pose token values, including the extra tokens."""
        return self.get_num_codes(include_aug_tokens=True)[0]

    @cached_property
    def POSE_MASK_ID(self):
        """Token value that marks a masked pose token."""
        return self.get_num_codes(include_aug_tokens=True)[0] - 1

    @cached_property
    def ACCEPT_TEXT_EMB_INPUT(self):
        """Whether the network takes text embeddings."""
        return True if self._args.get("text_embeddings", None) is not None else False


class root_mmm_network(nn.Module):
    """Transformer that predicts the root trajectory and the token count for SOMA."""

    def __init__(self, args: Dict, motion_rep: nn.Module):
        super().__init__()
        self._args = args
        self.motion_rep = motion_rep
        self.global_motion_rep = motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep = motion_rep.dual_rep.local_motion_rep
        self.IS_MODEL_TOKENIZED = False
        self._build_backbone()
        self._build_input_embeddings()

    def _build_backbone(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._args["n_embd"], nhead=self._args["n_head"], batch_first=True
        )
        self._shared_transformer_model = nn.TransformerEncoder(
            encoder_layer, num_layers=self._args["n_layers_shared"]
        )
        if self._args["n_layers_root_token"] > 0:
            self._root_token_transformer_model = nn.TransformerEncoder(
                encoder_layer, num_layers=self._args["n_layers_root_token"]
            )
        else:
            self._root_token_transformer_model = None
        self._args["local_root_dim"] = extract_feature_from_motion_rep(
            t.zeros([1, 1, 1000]),
            self.local_motion_rep,
            self._args["local_root_feature"],
        ).shape[2]
        self._args["global_root_dim"] = extract_feature_from_motion_rep(
            t.zeros([1, 1, 1000]),
            self.global_motion_rep,
            self._args["global_root_feature"],
        ).shape[2]
        self._args["local_pose_dim"] = extract_feature_from_motion_rep(
            t.zeros([1, 1, 1000]),
            self.local_motion_rep,
            self._args["local_pose_feature"],
        ).shape[2]
        self._conv_output = DoubleCondDecoder(
            self._args["global_root_dim"],
            self._args["n_embd"],
            down_t=self._args["down_t"],
            width=self._args["width"],
            depth=self._args["depth"],
            dilation_growth_rate=self._args["dilation_growth_rate"],
            activation=self._args["activation"],
            norm=self._args["norm"],
            target_cond_dim=self._args["global_root_dim"],
            external_cond_dim=self._args["n_embd"],
        )

    def _build_input_embeddings(self):
        num_frames_per_token = self.get_num_frames_per_token()
        self._proj_local_pose = nn.Linear(
            self._args["local_pose_dim"], self._args["pose_feat_dim"]
        )
        self._proj_local_root_value = nn.Linear(
            self.args["local_root_dim"], self._args["local_root_feat_dim"]
        )
        self._proj_global_root_value = nn.Linear(
            self.args["global_root_dim"], self._args["global_root_feat_dim"]
        )
        self._no_local_pose_emb = nn.Parameter(
            torch.randn([self._args["pose_feat_dim"]])
        )
        self._no_local_root_emb = nn.Parameter(
            torch.randn([self._args["local_root_feat_dim"]])
        )
        self._no_global_root_emb = nn.Parameter(
            torch.randn([self._args["global_root_feat_dim"]])
        )
        self._conv_no_frame_emb = nn.Parameter(torch.randn([self._args["n_embd"]]))
        proj_input_dim = (
            self._args["pose_feat_dim"]
            + self._args["local_root_feat_dim"]
            + self._args["global_root_feat_dim"]
        )
        self._proj_start_input = mlp(
            num_layers=self._args["input_feat_mlp_num_layers"],
            layer_width=self._args["n_embd"],
            size_in=proj_input_dim,
            size_out=self._args["n_embd"],
        )
        self._proj_end_input = mlp(
            num_layers=self._args["input_feat_mlp_num_layers"],
            layer_width=self._args["n_embd"],
            size_in=proj_input_dim,
            size_out=self._args["n_embd"],
        )
        self._input_position_emb = t.nn.Embedding(
            num_frames_per_token * 2, self._args["n_embd"]
        )
        self._proj_input_num_tokens = t.nn.Embedding(
            self._args["max_tokens"] - self._args["min_tokens"] + 1 + 1 + 1,
            self._args["n_embd"],
        )
        self._position_emb = PositionEmbedding(
            seq_length=self._args["max_tokens"], dim=self._args["n_embd"]
        )
        self._proj_num_token_output_logit = t.nn.Linear(
            self._args["n_embd"],
            self._args["max_tokens"] - self._args["min_tokens"] + 1 + 1,
        )
        if self.ACCEPT_TEXT_EMB_INPUT:
            self._proj_text_embeddings = mlp(
                num_layers=self._args["input_feat_mlp_num_layers"],
                layer_width=self._args["n_embd"],
                size_in=self._args["text_emb_dim"],
                size_out=self._args["n_embd"],
            )
        if (
            self.USE_HARD_NUM_TOKEN_EMB_FOR_ROOT_PREDICTION
            and self._root_token_transformer_model is not None
        ):
            self._middle_token_emb = t.nn.Embedding(
                self._args["max_tokens"] - self._args["min_tokens"] + 1 + 1,
                self._args["n_embd"],
            )

    def forward(
        self,
        global_root_values: t.Tensor,
        has_global_root_values: t.Tensor,
        local_root_values: t.Tensor,
        has_local_root_values: t.Tensor,
        poses: t.Tensor,
        has_poses: t.Tensor,
        num_tokens: t.Tensor,
        text_embeddings: Optional[t.Tensor] = None,
        has_text_embeddings: Optional[t.Tensor] = None,
        groundtruth_num_tokens: t.Tensor = None,
        allowed_pred_num_tokens: t.Tensor = None,
        config: dict = {},
    ):
        batch_size, device = (poses.shape[0], poses.device)
        num_frames_per_token = self.get_num_frames_per_token()
        local_pose_emb = self._proj_local_pose(poses)
        local_root_emb = self._proj_local_root_value(local_root_values)
        global_root_emb = self._proj_global_root_value(global_root_values)
        local_pose_emb = local_pose_emb * has_poses[
            :, :, None
        ].float() + self._no_local_pose_emb[None, None, :] * (
            1 - has_poses[:, :, None].float()
        )
        local_root_emb = local_root_emb * has_local_root_values[
            :, :, None
        ].float() + self._no_local_root_emb[None, None, :] * (
            1 - has_local_root_values[:, :, None].float()
        )
        global_root_emb = global_root_emb * has_global_root_values[
            :, :, None
        ].float() + self._no_global_root_emb[None, None, :] * (
            1 - has_global_root_values[:, :, None].float()
        )
        start_frame_emb = torch.cat(
            [
                local_pose_emb[:, :num_frames_per_token, :],
                local_root_emb[:, :num_frames_per_token, :],
                global_root_emb[:, :num_frames_per_token, :],
            ],
            dim=-1,
        )
        start_frame_emb = self._proj_start_input(start_frame_emb)
        end_frame_emb = torch.cat(
            [
                local_pose_emb[:, -num_frames_per_token:, :],
                local_root_emb[:, -num_frames_per_token:, :],
                global_root_emb[:, -num_frames_per_token:, :],
            ],
            dim=-1,
        )
        end_frame_emb = self._proj_end_input(end_frame_emb)
        frame_emb = torch.cat([start_frame_emb, end_frame_emb], dim=1)
        positioned_frame_emb = frame_emb + self._input_position_emb.weight[None, :, :]
        first_stage_time_emb = self._proj_input_num_tokens(
            num_tokens - self._args["min_tokens"]
        )
        first_stage_time_emb = first_stage_time_emb.view(
            [batch_size, 1, self._args["n_embd"]]
        )
        first_stage_input_emb = t.concat(
            [first_stage_time_emb, positioned_frame_emb], dim=1
        )
        if self.ACCEPT_TEXT_EMB_INPUT and text_embeddings is not None:
            text_embeddings = self._proj_text_embeddings(text_embeddings)[:, None, :]
            first_stage_input_emb = t.concat(
                [text_embeddings, first_stage_input_emb], dim=1
            )
            first_stage_mask = t.zeros(
                [batch_size, 1 + 1 + num_frames_per_token * 2], device=device
            ).bool()
            first_stage_mask[:, :1] = ~has_text_embeddings.bool()
        else:
            first_stage_mask = None
        first_stage_output_emb = self._shared_transformer_model(
            first_stage_input_emb, src_key_padding_mask=first_stage_mask
        )
        num_token_logits = self._proj_num_token_output_logit(
            first_stage_output_emb[:, self.TRANSFORMER_TIME_LOGIT_ID, :]
        )
        assert self.USE_HARD_NUM_TOKEN_EMB_FOR_ROOT_PREDICTION, (
            "Only support hard num token emb."
        )
        if groundtruth_num_tokens is not None:
            chosen_token = (
                groundtruth_num_tokens.view([batch_size]) - self._args["min_tokens"]
            )
        else:
            assert not self.training, "groundtruth_num_tokens is required in training."
            if not config.get("allow_pred_out_of_reach_num_tokens", True):
                num_token_logits = num_token_logits.clone()
                num_token_logits[
                    :, self.OUT_OF_REACH_NUM_TOKENS - self._args["min_tokens"]
                ] = -t.inf
            if allowed_pred_num_tokens is not None:
                num_time_tokens = (
                    self._args["max_tokens"] - self._args["min_tokens"] + 1
                )
                modified_num_token_logits = t.where(
                    allowed_pred_num_tokens == 1,
                    num_token_logits[:, :num_time_tokens],
                    t.full([batch_size, num_time_tokens], -t.inf).to(device),
                )
                num_token_logits = t.cat(
                    [modified_num_token_logits, num_token_logits[:, num_time_tokens:]],
                    dim=-1,
                )
            chosen_token = torch.argmax(num_token_logits, dim=-1).int()
            chosen_token = t.where(
                num_tokens.view([batch_size]) == self.MASKED_NUM_TOKENS,
                chosen_token,
                num_tokens.view([batch_size]) - self._args["min_tokens"],
            )
        position_ids = t.arange(self._args["max_tokens"]).to(device=device)
        position_emb = self._position_emb.embed[position_ids].view(
            [1, self._args["max_tokens"], self._args["n_embd"]]
        )
        position_emb = position_emb.expand([batch_size, -1, -1])
        token_mask = (
            t.arange(self._args["max_tokens"]).view([1, -1]).to(device)
            < chosen_token.view([-1, 1]) + self._args["min_tokens"]
        )
        if self._root_token_transformer_model is None:
            second_stage_output_emb = position_emb
        else:
            second_stage_token_emb = self._middle_token_emb(chosen_token)
            second_stage_input_emb = t.concat(
                [
                    second_stage_token_emb[:, None, :],
                    positioned_frame_emb,
                    position_emb,
                ],
                dim=1,
            )
            second_stage_mask = t.cat(
                [
                    t.zeros(
                        [batch_size, 1 + num_frames_per_token * 2], device=device
                    ).bool(),
                    ~token_mask,
                ],
                dim=1,
            )
            second_stage_output_emb = self._root_token_transformer_model(
                second_stage_input_emb, src_key_padding_mask=second_stage_mask
            )
            second_stage_output_emb = second_stage_output_emb[
                :, -self._args["max_tokens"] :
            ]
        pred_num_tokens = chosen_token + self._args["min_tokens"]
        batch = {}
        keys = ["frame", "global_root"]
        num_total_frames = self._args["max_tokens"] * num_frames_per_token
        batch["has_frame"] = numeric_mask_union(
            has_poses, has_global_root_values, has_local_root_values
        )
        batch["frame"] = t.where(
            batch["has_frame"][:, :, None].bool(),
            frame_emb,
            self._conv_no_frame_emb[None, None, :],
        )
        batch["dense_frame"] = t.cat(
            [
                batch["frame"][:, :num_frames_per_token],
                self._conv_no_frame_emb[None, None, :].expand(
                    [batch_size, num_total_frames - num_frames_per_token, -1]
                ),
            ],
            dim=1,
        )
        batch["global_root"], batch["has_global_root"] = (
            global_root_values,
            has_global_root_values,
        )
        batch["dense_global_root"] = t.cat(
            [
                batch["global_root"][:, :num_frames_per_token],
                t.zeros(
                    [
                        batch_size,
                        num_total_frames - num_frames_per_token,
                        self._args["global_root_dim"],
                    ],
                    device=device,
                ),
            ],
            dim=1,
        )
        for key in keys:
            batch["dense_has_" + key] = t.cat(
                [
                    batch["has_" + key][:, :num_frames_per_token],
                    t.zeros(
                        [batch_size, num_total_frames - num_frames_per_token],
                        dtype=batch["has_" + key].dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            for i in range(num_frames_per_token):
                offsets = (
                    pred_num_tokens[:, None] * num_frames_per_token
                    - num_frames_per_token
                    + i
                ).long()
                batch["dense_" + key] = batch["dense_" + key].scatter(
                    1,
                    offsets[:, :, None].repeat([1, 1, batch[key].shape[-1]]),
                    batch[key][:, -num_frames_per_token + i][:, None, :],
                )
                batch["dense_has_" + key] = batch["dense_has_" + key].scatter(
                    1,
                    offsets[:, :],
                    batch["has_" + key][:, -num_frames_per_token + i][:, None],
                )
        pred_global_root_values = self._conv_output(
            second_stage_output_emb.transpose(1, 2),
            external_cond=batch["dense_frame"],
            target_cond=batch["dense_global_root"],
            has_target_cond=batch["dense_has_global_root"].bool(),
            token_mask=token_mask,
        ).transpose(1, 2)
        return {
            "num_token_logits": num_token_logits,
            "pred_num_tokens": pred_num_tokens,
            "pred_global_root_values": pred_global_root_values,
        }

    def get_num_frames_per_token(self):
        """Return the frames per motion token, 2 ** down_t."""
        return 2 ** self._args["down_t"]

    @property
    def args(self):
        """Network settings from the checkpoint."""
        return self._args

    @property
    def OUT_OF_REACH_NUM_TOKENS(self):
        """Token count class for a target that is out of reach."""
        return self._args["max_tokens"] + 1

    @property
    def MASKED_NUM_TOKENS(self):
        """Token count class for a masked token count."""
        return self._args["max_tokens"] + 2

    @cached_property
    def ACCEPT_TEXT_EMB_INPUT(self):
        """Whether the network takes text embeddings."""
        return True if self._args.get("text_embeddings", None) is not None else False

    @property
    def TRANSFORMER_TIME_LOGIT_ID(self):
        """Index of the token count output in the transformer sequence."""
        return 1 if self.ACCEPT_TEXT_EMB_INPUT else 0

    @cached_property
    def USE_HARD_NUM_TOKEN_EMB_FOR_ROOT_PREDICTION(self):
        """Whether root prediction embeds the exact token count."""
        return self._args.get("use_hard_num_token_emb_for_root_prediction", False)
