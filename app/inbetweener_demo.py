# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simple example of generating motion between poses with MotionBricks.

Choose poses from clips, place them in the view, and arrange them in the list.
Press Generate to move from Start through the targets in that order.
Each transition builds on the motion that came before it.

The engine creates the context token, target token, and output buffer once.
Every transition reuses them:

    context.update_from_poses(recent_poses)
    target.update_from_poses(target_poses)
    engine.generate_animation(context=context, target=target, output=output)
    output.sample(time, current_pose)

See reset(), generate_next_segment(), and step() for how this demo does it.

Choose SOMA or the Unitree G1 robot in the Character selector. SOMA uses BVH
clips and G1 uses CSV clips. Both models use the same calls. Only their pose
layouts differ.

This demo uses clips to create target poses, but your app can create them in
other ways.
"""

from __future__ import annotations

from contextlib import nullcontext

from dataclasses import dataclass, field
from pathlib import Path
import time

import newton
import numpy as np
from pyglet.math import Vec3
import warp as wp

from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.animation.foot_contacts import FOOT_CONTACT_COUNT
from motionbricks.model.config import DEFAULT_TOKEN_COUNTS, MotionBricksInferenceConfig
from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE, ModelRig
from motionbricks.renderers.foot_contact_renderer import FootContactRenderer
from motionbricks.renderers.picking import raycast_line
from motionbricks.renderers.vector_arrow_renderer import VectorArrowRenderer
from motionbricks.robotics.dof_animation_buffer import DOFAnimationBuffer
from motionbricks.runtime.target_pose import MAX_TRANSITION_EARLY_BY_FRAMES

from .authoring.common import (
    InlineRename,
    NewtonViewerUIBridge,
    POSES_TO_USE_TOOLTIP,
    accept_drop,
    apply_transform_locks,
    choose_clip,
    drag_source,
    draw_add_remove_buttons,
    draw_character_selector,
    draw_clip_scrubber,
    draw_clip_selector,
    draw_int_slider_property,
    draw_token_range_property,
    draw_transform_editor,
    log_transform_gizmo,
    transforms_equal,
)
from .runtime.common.rig import (
    ASSETS_ROOT,
    CHARACTER_COLORS,
    create_foot_contact_renderer,
    create_mesh_renderer,
    create_skeleton_renderer,
    load_rig_assets,
)
from .utils.cli import add_model_precision_argument, configure_logging, create_gl_parser
from .utils.model_loading import create_engine, require_inference_backend
from .utils.tooltips import show_delayed_tooltip


_MOTION_ROOT = ASSETS_ROOT / "motions"
_DEFAULT_CLIPS = {
    ModelRig.SOMA: _MOTION_ROOT / "soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh",
    ModelRig.UNITREE_G1: _MOTION_ROOT / "unitree_g1/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.csv",
}
_START_COLOR = (0.2, 0.9, 0.4)
_TARGET_COLOR = (0.25, 0.65, 1.0)
_SELECTED_COLOR = (1.0, 0.5, 0.15)
_ORDER_COLOR = (1.0, 0.8, 0.25)
_GROUND_MARKER_HEIGHT = 0.05
_PLAYBACK_MESH_ID = -1
_POSE_PAYLOAD = "INBETWEENER_POSE"
_RING_SEGMENTS = 48
_RING_RADIUS = 0.3
_MAX_FRAME_SECONDS = 0.1


@dataclass(eq=False)
class PoseEntry:
    """Start or a target, sampled from a clip and placed in the view. Start comes first in the list."""

    id: int
    name: str
    path: Path
    clip: AnimationBuffer | DOFAnimationBuffer
    sample_time: float
    placement: wp.transform
    min_tokens: int = DEFAULT_TOKEN_COUNTS[0]
    max_tokens: int = DEFAULT_TOKEN_COUNTS[-1]
    color: tuple[float, float, float] = _TARGET_COLOR
    euler_degrees: list[float] | None = None
    translation_locks: list[bool] = field(default_factory=lambda: [False, False, True])
    rotation_locks: list[bool] = field(default_factory=lambda: [True, True, False])
    show_frames: bool = False
    num_poses_to_use: int = MODEL_POSES_PER_TOKEN
    transition_early_by_frames: int = 0
    show_mesh: bool = True
    show_skeletons: bool = False
    show_velocity: bool = False

    # refresh_poses() samples these from the clip, sample time, and placement.
    sampled_poses: wp.array = field(init=False, repr=False)
    preview_pose: wp.array = field(init=False, repr=False)
    root_position: np.ndarray = field(init=False, repr=False)
    velocity: np.ndarray = field(init=False, repr=False)


def sample_pose_window(
    entry: PoseEntry,
    model_to_runtime: wp.transform,
    poses: wp.array,
    *,
    anchor: int,
) -> None:
    """Sample four poses at 30 Hz into the poses array.

    anchor is the pose that sits at the entry's sample time: 3 for Start, whose
    samples end there, or 0 for a target, whose samples begin there. The anchor's
    root moves to the entry's placement and keeps its height. The other samples move
    with it, so the motion between them stays the same. Only the root needs
    coordinate conversion. SOMA child joints stay relative to their parents, and G1
    joint angles do not change.
    """
    entry.clip.sample(entry.sample_time, poses[anchor], root_offset=model_to_runtime)
    # Both pose layouts start with the root position, in world space.
    root = poses[anchor].numpy().reshape(-1)
    ground_offset = wp.transform(wp.vec3(-float(root[0]), -float(root[1]), 0.0), wp.quat_identity())
    root_offset = wp.mul(entry.placement, wp.mul(ground_offset, model_to_runtime))
    for frame in range(MODEL_POSES_PER_TOKEN):
        sample_time = entry.sample_time + (frame - anchor) / MODEL_SAMPLE_RATE
        entry.clip.sample(sample_time, poses[frame], root_offset=root_offset)


class InbetweenerDemo:
    """Place Start and targets, then play the motion generated through them."""

    def __init__(self, viewer, rig: ModelRig, *, fp16: bool = False):
        """Open the Inbetweener on a viewer with Start and one target.

        fp16 selects the mixed FP16 models.
        """
        self.viewer = viewer
        self.fp16 = fp16
        self.engine = None
        self._pending_clip: PoseEntry | None = None
        self._pending_rig: ModelRig | None = None
        self._rename = InlineRename()
        self.error: str | None = None
        self.show_foot_contacts = False
        self.foot_contact_threshold = FootContactRenderer.DEFAULT_THRESHOLD
        self._setup_viewer()
        self.use_rig(rig)

    def use_rig(self, rig: ModelRig) -> None:
        """Switch to a rig and start over with Start, one target, and the default view.

        The previous rig's engine closes before the new one loads, so only one engine
        holds GPU memory.
        """
        if self.engine is not None:
            # Finish the previous rig's work, remove its drawings, and release its engine.
            self.engine.synchronize()
            self.mesh_renderer.clear(self.viewer)
            self.skeleton_renderer.clear(self.viewer)
            self.velocity_renderer.clear(self.viewer)
            self.foot_contact_renderer.clear(self.viewer)
            self.engine.close()
            self.engine = None
        self.rig, self.engine = rig, create_engine(self.viewer, rig, self.fp16)
        with wp.ScopedStream(self.engine.stream) if self.engine.stream is not None else nullcontext():
            self._prepare_rig(rig)
        camera = self.viewer.camera
        camera.pos = Vec3(-7.0, -4.0, 3.5)
        camera.look_at(np.array([0.0, -1.0, 1.0], dtype=np.float32))

    def _prepare_rig(self, rig: ModelRig) -> None:
        """Create the rig's buffers and renderers, then place Start and one target."""
        engine = self.engine
        assets = load_rig_assets(rig)
        self.clip_registry = assets.clip_registry
        self.model_to_runtime = assets.model_to_runtime

        # Reuse the same context token, target token, and output buffer across all generations.
        if rig is ModelRig.SOMA:
            # A SOMA pose has one local transform per skeleton joint.
            skeleton = assets.model.skeleton
            self.context = engine.create_input(skeleton)
            self.target = engine.create_input(skeleton)
            self.output = engine.create_output(skeleton)
        else:
            # A G1 pose has the root position and XYZW rotation, then 29 joint angles.
            self.context = engine.create_input()
            self.target = engine.create_input()
            self.output = engine.create_output()
        self.mesh_renderer = create_mesh_renderer(rig, assets.model)
        self.skeleton_renderer = create_skeleton_renderer(rig, assets.model, max_instances=MODEL_POSES_PER_TOKEN)
        self.playback_color = CHARACTER_COLORS[rig]
        self.foot_contact_renderer = create_foot_contact_renderer(rig, "generated_foot_contacts", assets.model)
        self.foot_contacts = wp.empty(FOOT_CONTACT_COUNT, dtype=wp.float32, device=engine.device)
        self.current_pose = wp.empty_like(assets.reference_pose)

        path = _DEFAULT_CLIPS[rig]
        clip = self.clip_registry.load(path).animation
        duration = (clip.num_frames - 1) / clip.sample_rate
        self.pose_entries = [
            PoseEntry(
                0, "Start", path, clip, sample_time=min(1.0, duration),
                placement=wp.transform_identity(), color=_START_COLOR,
            ),
            PoseEntry(
                1, "Target 1", path, clip, sample_time=min(7.0, duration),
                placement=wp.transform(wp.vec3(0.0, -2.0, 0.0), wp.quat_identity()),
            ),
        ]
        self.selected_index = 1
        self._next_id = 2
        self.refresh_poses()
        self._last_frame_time = time.perf_counter()

    def reset(self) -> None:
        """Go back to Start and restore its four context poses."""
        self.playing = False
        self.has_segment = False
        self.next_target_index = 1
        self.playback_time = 0.0
        self.segment_end_frame = 0
        start = self.pose_entries[0]
        self.context.update_from_poses(start.sampled_poses, stream=self.engine.stream)
        wp.copy(self.current_pose, start.preview_pose, stream=self.engine.stream)

    def play(self) -> None:
        """Restart from Start and play through the targets. It needs at least one target."""
        self.reset()
        if len(self.pose_entries) > 1:
            self.playing = True

    def generate_next_segment(self) -> None:
        """Generate from the current context toward the next placed target."""
        entry = self.pose_entries[self.next_target_index]
        self.target.update_from_poses(entry.sampled_poses[:entry.num_poses_to_use], stream=self.engine.stream)
        self.engine.generate_animation(
            context=self.context, target=self.target, output=self.output,
            config=MotionBricksInferenceConfig(
                allowed_token_counts=tuple(range(entry.min_tokens, entry.max_tokens + 1)),
            ),
        )

        # Transition Early By skips frames at the end of the motion. At least four frames
        # still play, because the next context is the last four played frames.
        self.segment_end_frame = max(
            MODEL_POSES_PER_TOKEN - 1, self.output.num_frames - 1 - entry.transition_early_by_frames,
        )
        self.playback_time = 0.0
        self.has_segment = True
        self.output.sample(0.0, self.current_pose, stream=self.engine.stream)
        # Don't count generation time as playback time.
        self._last_frame_time = time.perf_counter()

    def step(self, delta_time: float) -> None:
        """Advance playback by delta_time (in seconds).

        A call with no segment ready generates one instead. At the transition point, the last
        four played poses become the next context.
        """
        if not self.playing:
            return
        if not self.has_segment:
            self.generate_next_segment()
            return

        end_time = self.segment_end_frame / MODEL_SAMPLE_RATE
        self.playback_time = min(self.playback_time + delta_time, end_time)
        self.output.sample(self.playback_time, self.current_pose, stream=self.engine.stream)
        if self.playback_time < end_time:
            return

        played_frames = self.segment_end_frame + 1
        recent_poses = self.output.poses[played_frames - MODEL_POSES_PER_TOKEN:played_frames]
        self.context.update_from_poses(recent_poses, stream=self.engine.stream)
        self.next_target_index += 1
        self.has_segment = False
        self.playing = self.next_target_index < len(self.pose_entries)

    def refresh_poses(self) -> None:
        """Resample Start and every target, update their previews, and return to Start."""
        for index, entry in enumerate(self.pose_entries):
            # Start shows its last context pose, and a target shows its first pose.
            anchor = MODEL_POSES_PER_TOKEN - 1 if index == 0 else 0
            entry.sampled_poses = wp.empty_like(self.context.poses)
            sample_pose_window(entry, self.model_to_runtime, entry.sampled_poses, anchor=anchor)
            entry.preview_pose = entry.sampled_poses[anchor]
            # Both pose layouts start with the root position.
            roots = entry.sampled_poses.numpy().reshape(MODEL_POSES_PER_TOKEN, -1)[:, :3]
            entry.root_position = roots[anchor]
            # Use motion leading into Start, or motion leaving a target, for its velocity.
            pair = roots[-2:] if index == 0 else roots[:2]
            entry.velocity = (pair[1] - pair[0]) * MODEL_SAMPLE_RATE
            if entry.num_poses_to_use == 1:
                entry.velocity[:] = 0.0
        self._refresh_order_markers()
        self.reset()

    def add_target(self) -> None:
        """Add a target after the last one.

        It copies the clip and sample time of the selection.
        """
        source = self.pose_entries[self.selected_index]
        previous = self.pose_entries[-1].placement
        placement = wp.transform(previous.p + wp.vec3(0.0, -1.5, 0.0), previous.q)
        self.pose_entries.append(PoseEntry(
            self._next_id, f"Target {self._next_id}", source.path, source.clip,
            source.sample_time, placement,
        ))
        self._next_id += 1
        self.selected_index = len(self.pose_entries) - 1
        self.refresh_poses()

    def remove_target(self) -> None:
        """Remove the selected target. Start cannot be removed."""
        if self.selected_index == 0:
            return
        self.pose_entries.pop(self.selected_index)
        self.mesh_renderer.clear(self.viewer)
        self.skeleton_renderer.clear(self.viewer)
        self.velocity_renderer.clear(self.viewer)
        self._rename.item_id = None
        self.selected_index = min(self.selected_index, len(self.pose_entries) - 1)
        self.refresh_poses()

    def move_target(self, source: int, destination: int) -> None:
        """Move the target at index source to index destination. Start always stays first."""
        if not 1 <= source < len(self.pose_entries) or not 1 <= destination < len(self.pose_entries):
            return
        if source == destination:
            return
        entry = self.pose_entries.pop(source)
        self.pose_entries.insert(destination, entry)
        self.selected_index = destination
        self.refresh_poses()

    def _choose_pending_clip(self) -> None:
        entry = self._pending_clip
        if entry is None:
            return
        self._pending_clip = None
        try:
            chosen = choose_clip(self.clip_registry, entry.path)
            if chosen:
                path = Path(chosen).resolve()
                clip = self.clip_registry.load(path).animation
                entry.path, entry.clip = path, clip
                entry.sample_time = min(
                    (MODEL_POSES_PER_TOKEN - 1) / MODEL_SAMPLE_RATE,
                    (clip.num_frames - 1) / clip.sample_rate,
                )
                self.refresh_poses()
        except Exception as exc:
            self.playing = False
            self.error = f"Open failed: {exc}"
        finally:
            self._last_frame_time = time.perf_counter()

    def _switch_pending_rig(self) -> None:
        rig, self._pending_rig = self._pending_rig, None
        if rig is None:
            return
        try:
            self.use_rig(rig)
        except Exception as exc:
            self.error = f"Character load failed: {exc}"

    def _draw_pose_list(self, ui) -> None:
        move = None
        ui.begin_child("pose_list", ui.ImVec2(0, 160), child_flags=ui.ChildFlags_.borders)
        for index, entry in enumerate(self.pose_entries):
            if self._rename.item_id == entry.id:
                if name := self._rename.draw(ui):
                    entry.name = name
                continue
            label = f"{entry.name}###pose_{entry.id}"
            clicked, _ = ui.selectable(label, self.selected_index == index)
            row_min, row_max = ui.get_item_rect_min(), ui.get_item_rect_max()
            if clicked:
                self.selected_index = index
            if ui.is_item_hovered() and ui.is_mouse_double_clicked(0):
                self.selected_index = index
                self._rename.begin(entry.id, entry.name)
            if index == 0:
                continue
            drag_source(ui, _POSE_PAYLOAD, entry.id, entry.name)
            dropped_id = accept_drop(ui, _POSE_PAYLOAD)
            if dropped_id is not None:
                source = next(i for i, pose_entry in enumerate(self.pose_entries) if pose_entry.id == dropped_id)
                insertion = index + int(ui.get_mouse_pos().y > (row_min.y + row_max.y) / 2)
                move = source, insertion - int(source < insertion)
        ui.end_child()
        if move is not None:
            self.move_target(*move)

    def _draw_properties(self, ui, scale: float) -> None:
        entry = self.pose_entries[self.selected_index]
        ui.text_colored(ui.ImVec4(*entry.color, 1.0), entry.name)

        if ui.collapsing_header("Transform", flags=ui.TreeNodeFlags_.default_open):
            changed, transform, entry.euler_degrees = draw_transform_editor(
                ui, entry.placement, entry.euler_degrees, scale,
                entry.translation_locks, entry.rotation_locks,
            )
            if changed:
                entry.placement = transform
                self.refresh_poses()

        if ui.collapsing_header("Target Pose", flags=ui.TreeNodeFlags_.default_open):
            if draw_clip_selector(ui, entry.path.name, entry.path, entry.id):
                self._pending_clip = entry
            changed, sample_time, entry.show_frames = draw_clip_scrubber(
                ui, entry.clip, entry.sample_time, entry.show_frames, scale,
            )
            if changed:
                entry.sample_time = sample_time
                self.refresh_poses()
            ui.text_unformatted(f"Speed: {np.linalg.norm(entry.velocity):.3f} m/s")
            ui.separator()
            # Start supplies the model's four required context poses.
            ui.begin_disabled(self.selected_index == 0)
            num_poses = draw_int_slider_property(
                ui, "Poses to Use", entry.num_poses_to_use, 1, MODEL_POSES_PER_TOKEN, entry.id,
                tooltip=POSES_TO_USE_TOOLTIP,
            )
            early_frames = draw_int_slider_property(
                ui, "Transition Early By (frames)", entry.transition_early_by_frames,
                0, MAX_TRANSITION_EARLY_BY_FRAMES, entry.id,
                tooltip="Move on to the next target this many frames before the generated motion ends. "
                "At least four frames still play.",
            )
            min_tokens, max_tokens = draw_token_range_property(ui, entry.min_tokens, entry.max_tokens, entry.id)
            ui.end_disabled()
            if (num_poses, early_frames, min_tokens, max_tokens) != (
                entry.num_poses_to_use, entry.transition_early_by_frames, entry.min_tokens, entry.max_tokens,
            ):
                entry.num_poses_to_use, entry.transition_early_by_frames = num_poses, early_frames
                entry.min_tokens, entry.max_tokens = min_tokens, max_tokens
                self.refresh_poses()

        if ui.collapsing_header("Visibility", flags=ui.TreeNodeFlags_.default_open):
            changed, color = ui.color_edit3("Color", entry.color, ui.ColorEditFlags_.no_inputs)
            if changed:
                entry.color = tuple(color)
            _, entry.show_mesh = ui.checkbox("Show Mesh", entry.show_mesh)
            _, entry.show_skeletons = ui.checkbox("Show Skeleton(s)", entry.show_skeletons)
            _, entry.show_velocity = ui.checkbox("Show Velocity", entry.show_velocity)

    def gui(self, ui) -> None:
        """Draw the Inbetweener window."""
        viewport = ui.get_main_viewport()
        scale = ui.get_font_size() / 13.0
        ui.set_next_window_pos(
            ui.ImVec2(viewport.pos.x + viewport.size.x - 12, viewport.pos.y + 12),
            ui.Cond_.first_use_ever, pivot=ui.ImVec2(1, 0),
        )
        ui.set_next_window_size(
            ui.ImVec2(440 * scale, min(740 * scale, viewport.size.y - 24)),
            ui.Cond_.first_use_ever,
        )
        # Double-clicking the title bar collapses the window, as in the Newton viewer.
        expanded, _ = ui.begin("Inbetweener")
        if expanded:
            footer_height = ui.get_frame_height_with_spacing() + ui.get_style().item_spacing.y
            ui.begin_child("pose_settings", ui.ImVec2(0, -footer_height))
            rig = draw_character_selector(ui, self.rig)
            if rig is not self.rig:
                self._pending_rig = rig
            if ui.collapsing_header("Target Poses", flags=ui.TreeNodeFlags_.default_open):
                self._draw_pose_list(ui)
                add, remove = draw_add_remove_buttons(ui, self.selected_index != 0)
                if add:
                    self.add_target()
                if remove:
                    self.remove_target()
            if ui.collapsing_header("Properties", flags=ui.TreeNodeFlags_.default_open):
                ui.push_id(self.pose_entries[self.selected_index].id)
                self._draw_properties(ui, scale)
                ui.pop_id()

            ui.end_child()
            ui.separator()
            ui.begin_disabled(len(self.pose_entries) < 2)
            if ui.button("Generate"):
                self.play()
            ui.end_disabled()
            ui.same_line()
            _, self.show_foot_contacts = ui.checkbox("Show Foot Contacts", self.show_foot_contacts)
            show_delayed_tooltip(ui, "Show outlines for foot contacts above the threshold on the right.")
            ui.same_line()
            ui.set_next_item_width(80 * scale)
            ui.begin_disabled(not self.show_foot_contacts)
            _, self.foot_contact_threshold = ui.slider_float(
                "##FootContactThreshold", self.foot_contact_threshold, 0.0, 1.0, "%.2f",
                flags=ui.SliderFlags_.always_clamp,
            )
            ui.end_disabled()
        if self.error is not None:
            ui.open_popup("Error")
        if ui.begin_popup_modal("Error", flags=ui.WindowFlags_.always_auto_resize)[0]:
            ui.text_unformatted(self.error or "")
            if ui.button("OK"):
                self.error = None
                ui.close_current_popup()
            ui.end_popup()
        ui.end()

    def _select_pose(self, origin: wp.vec3, direction: wp.vec3) -> None:
        """Select the nearest pose whose mesh or ground ring is under the cursor."""
        nearest = self.viewer.camera.far
        ring_starts = self.ring_starts.numpy().reshape(len(self.pose_entries), -1, 3)
        ring_ends = self.ring_ends.numpy().reshape(len(self.pose_entries), -1, 3)
        for index, entry in enumerate(self.pose_entries):
            distance = nearest
            if entry.show_mesh:
                distance = self.mesh_renderer.raycast_pose(
                    entry.preview_pose, wp.transform_identity(), origin, direction, distance,
                )
            for start, end in zip(ring_starts[index], ring_ends[index]):
                distance = raycast_line(wp.vec3(*start), wp.vec3(*end), origin, direction, distance)
            if distance < nearest:
                nearest = distance
                self.selected_index = index

    def _setup_viewer(self) -> None:
        viewer = self.viewer
        self.skeleton_colors = wp.empty(MODEL_POSES_PER_TOKEN, dtype=wp.vec3)
        self.velocity_renderer = VectorArrowRenderer("target_pose_velocity", wp.vec3(1.0, 0.0, 0.0))
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        ground_model = builder.finalize()
        self.state = ground_model.state()
        viewer.set_model(ground_model)
        viewer.renderer.set_title("MotionBricks - Inbetweener")
        self._viewer_ui = NewtonViewerUIBridge(viewer)
        if viewer.ui is not None:
            self._viewer_ui.install(viewer.ui.imgui)
        self._viewer_ui.on_scene_click = self._select_pose
        viewer.register_ui_callback(self.gui, position="free")

    def _refresh_order_markers(self) -> None:
        """Build ground circles and connections in the same order as the pose list."""
        centers = np.array([entry.root_position for entry in self.pose_entries])
        centers[:, 2] = _GROUND_MARKER_HEIGHT
        angles = np.linspace(0, 2 * np.pi, _RING_SEGMENTS + 1)
        circle = _RING_RADIUS * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(_RING_SEGMENTS + 1)))
        rings = centers[:, None, :] + circle[None, :, :]
        self.ring_starts = wp.array(rings[:, :-1].reshape(-1, 3), dtype=wp.vec3)
        self.ring_ends = wp.array(rings[:, 1:].reshape(-1, 3), dtype=wp.vec3)
        self.ring_colors = wp.empty(len(self.pose_entries) * _RING_SEGMENTS, dtype=wp.vec3)
        self.order_starts = wp.array(centers[:-1], dtype=wp.vec3)
        self.order_ends = wp.array(centers[1:], dtype=wp.vec3)
        self.order_colors = wp.array([_ORDER_COLOR] * (len(self.pose_entries) - 1), dtype=wp.vec3)

    def _draw_target_previews(self) -> None:
        for entry in self.pose_entries:
            if entry.show_mesh:
                self.mesh_renderer.draw_pose(
                    self.viewer, entry.preview_pose, wp.transform_identity(),
                    wp.vec3(*entry.color), entry.id,
                )
            else:
                self.mesh_renderer.hide(self.viewer, entry.id)
            if entry.show_skeletons:
                self.skeleton_colors.fill_(wp.vec3(*entry.color))
                self.skeleton_renderer.draw_batch(
                    self.viewer, entry.sampled_poses, entry.num_poses_to_use, wp.transform_identity(),
                    self.skeleton_colors, entry.id,
                )
            else:
                self.skeleton_renderer.hide(self.viewer, entry.id)
            if entry.show_velocity and np.linalg.norm(entry.velocity) > 1e-6:
                self.velocity_renderer.draw(
                    self.viewer, entry.id, wp.transform_identity(),
                    wp.vec3(*entry.root_position), wp.vec3(*entry.velocity),
                )
            else:
                self.velocity_renderer.hide(self.viewer, entry.id)

    def _draw_generated_pose(self) -> None:
        # Hide the playback mesh only when a visible target preview already shows its pose.
        # Time is also zero at each new segment, so it does not always mean playback is at Start.
        target_index = self.next_target_index
        if not self.has_segment:
            target_index -= 1
        endpoint_visible = self.pose_entries[target_index].show_mesh
        at_endpoint = self.playback_time >= self.output.duration
        previous_entry = self.pose_entries[self.next_target_index - 1]
        at_visible_start = (
            self.playback_time == 0.0
            and previous_entry.show_mesh and previous_entry.transition_early_by_frames == 0
        )
        has_playback = self.has_segment or self.next_target_index > 1
        if has_playback and not at_visible_start and not (at_endpoint and endpoint_visible):
            self.mesh_renderer.draw_pose(
                self.viewer, self.current_pose, wp.transform_identity(),
                self.playback_color, _PLAYBACK_MESH_ID,
            )
        else:
            self.mesh_renderer.hide(self.viewer, _PLAYBACK_MESH_ID)

    def _draw_foot_contacts(self) -> None:
        """Show foot contacts for generated motion, including its final pose."""
        if (
            not self.show_foot_contacts or self.output.num_frames == 0
            or not (self.has_segment or self.next_target_index > 1)
        ):
            self.foot_contact_renderer.hide(self.viewer)
            return
        self.output.sample_foot_contacts(self.playback_time, self.foot_contacts, stream=self.engine.stream)
        # The renderer places the outlines itself, because the playback mesh can be hidden at a target.
        self.foot_contact_renderer.threshold = self.foot_contact_threshold
        self.foot_contact_renderer.draw_pose(self.viewer, self.current_pose, self.foot_contacts)

    def render(self) -> None:
        """Draw one frame, then apply any gizmo edit to the selection."""
        entry = self.pose_entries[self.selected_index]
        gizmo = wp.transform(entry.placement)
        before = wp.transform(gizmo)
        colors = [
            _SELECTED_COLOR if index == self.selected_index else pose_entry.color
            for index, pose_entry in enumerate(self.pose_entries)
        ]
        self.ring_colors.assign(np.repeat(colors, _RING_SEGMENTS, axis=0))
        self.viewer.begin_frame(self.playback_time)
        self._draw_target_previews()
        self._draw_generated_pose()
        self._draw_foot_contacts()
        self.viewer.log_lines("pose_rings", self.ring_starts, self.ring_ends, self.ring_colors)
        if len(self.pose_entries) > 1:
            self.viewer.log_arrows("pose_order", self.order_starts, self.order_ends, self.order_colors)
        else:
            self.viewer.log_arrows("pose_order", None, None, None)
        log_transform_gizmo(
            self.viewer, "pose_placement", gizmo,
            translation_locks=entry.translation_locks, rotation_locks=entry.rotation_locks,
        )
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        # The UI runs in end_frame(), so selection or placement may have changed.
        if self.pose_entries[self.selected_index] is entry and not transforms_equal(gizmo, before):
            entry.placement, entry.euler_degrees = apply_transform_locks(
                entry.placement, gizmo, entry.euler_degrees,
                entry.translation_locks, entry.rotation_locks,
            )
            self.refresh_poses()

    def run(self, max_frames: int | None = None) -> None:
        """Play and draw frames until the window closes or max_frames is reached."""
        frame = 0
        try:
            while self.viewer.is_running() and (max_frames is None or frame < max_frames):
                # A switch can build an engine, so it runs before choosing this frame's stream.
                self._switch_pending_rig()
                with wp.ScopedStream(self.engine.stream) if self.engine.stream is not None else nullcontext():
                    self._choose_pending_clip()
                    now = time.perf_counter()
                    delta_time = min(max(now - self._last_frame_time, 0.0), _MAX_FRAME_SECONDS)
                    self._last_frame_time = now
                    try:
                        self.step(delta_time)
                    except Exception as exc:
                        self.playing = False
                        self.error = f"Generation failed: {exc}"
                    self.render()
                frame += 1
        finally:
            if self.engine is not None:
                self.engine.close()
            self.viewer.close()


def main() -> None:
    """Parse the command line and open the Inbetweener."""
    import newton.examples

    parser = create_gl_parser()
    parser.add_argument(
        "--model",
        type=ModelRig,
        choices=list(ModelRig),
        default=ModelRig.SOMA,
        help="Character to start with.",
    )
    add_model_precision_argument(parser)
    viewer, args = newton.examples.init(parser)
    configure_logging(args)
    try:
        require_inference_backend()
        demo = InbetweenerDemo(viewer, args.model, fp16=args.model_precision == "fp16")
        demo.run(args.num_frames if args.test else None)
    finally:
        if viewer.is_running():
            viewer.close()


if __name__ == "__main__":
    main()
