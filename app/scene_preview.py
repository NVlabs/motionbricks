# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Play a character in a saved scene and try its Smart Primitive interactions."""

from __future__ import annotations

from contextlib import nullcontext

from dataclasses import dataclass
from functools import partial
import math
import time
from pathlib import Path

import newton
import numpy as np
from pyglet.math import Vec3
import warp as wp

from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, ModelRig
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine
from motionbricks.renderers.dof_mesh_renderer import DOFMeshRenderer
from motionbricks.renderers.foot_contact_renderer import (
    DOFFootContactRenderer,
    FootContactRenderer,
    SkeletonFootContactRenderer,
)
from motionbricks.renderers.wire_box_renderer import WireBoxRenderer
from motionbricks.renderers.skeletal_mesh_renderer import SkeletalMeshRenderer
from motionbricks.renderers.skeleton_renderer import DOFSkeletonRenderer, SkeletonRenderer
from motionbricks.renderers.solid_box_renderer import SolidBoxRenderer
from motionbricks.renderers.vector_arrow_renderer import VectorArrowRenderer
from motionbricks.runtime.smart_primitive import (
    InteractionBox,
    compute_interaction_box_overlaps,
)
from motionbricks.utils.math_utils import critical_spring_damper_exact

from .authoring.common import WINDOW_MARGIN
from .runtime.common.rig import (
    ASSETS_ROOT,
    CHARACTER_COLORS,
    RigAssets,
    create_foot_contact_renderer,
    create_mesh_renderer,
    create_skeleton_renderer,
    load_rig_assets,
)
from .runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
from .runtime.common.io.scene_document import (
    SCENE_DOCUMENT,
    SceneDefinition,
    SmartPrimitiveInstance,
    check_reference_rigs,
    load_scene_document,
)
from .utils.cli import add_model_precision_argument, configure_logging, create_gl_parser
from .utils.file_dialog import choose_file
from .utils.model_loading import create_engine, require_inference_backend
from .utils.input import InputManager, LocomotionModeInput, camera_relative_movement
from .runtime.soma.character import SOMACharacter
from .runtime.unitree_g1.character import UnitreeG1Character

_CHARACTER_TYPES = {ModelRig.SOMA: SOMACharacter, ModelRig.UNITREE_G1: UnitreeG1Character}
from .runtime.common.motion import LocomotionCommand, ground_direction
from .runtime.common.state_machine import StateMachine
from .runtime.common.smart_primitive_behaviors import (
    SmartPrimitiveBehavior,
    behavior_types,
)


_CAMERA_DISTANCE = 5.0
_CAMERA_ELEVATION_DEGREES = 25.0
_CAMERA_PIVOT_Z_OFFSET = 0.8
_CAMERA_VELOCITY_HALFLIFE = 0.15
_CAMERA_PULL_SECONDS = 0.05
_CONTROLLER_CAMERA_ORBIT_SPEED = 75.0
_VELOCITY_ARROW_HEIGHT = 0.025
_CHARACTER_VELOCITY_ARROW_COLOR = (1.0, 0.25, 0.1)
_INPUT_VELOCITY_ARROW_COLOR = (0.15, 0.45, 1.0)
_DESIRED_VELOCITY_ARROW_COLOR = (0.15, 1.0, 0.35)
_CONTEXT_COLOR = wp.vec3(0.65, 0.3, 0.85)
_TARGET_COLOR = wp.vec3(0.2, 0.75, 1.0)
_BLENDSPACE_DEBUG_COLOR = wp.vec3(0.15, 1.0, 0.35)
_INTERACTION_BOX_COLOR = wp.vec3(1.0, 0.25, 0.85)
_INACTIVE_SMART_PRIMITIVE_BOX_COLOR = wp.vec3(1.0, 0.15, 0.10)
_ACTIVE_SMART_PRIMITIVE_BOX_COLOR = wp.vec3(0.15, 1.0, 0.25)
_PANEL_WIDTH = 340.0
_PANEL_HEIGHT = 390.0
_PANEL_ALPHA = 0.8
_MAX_FRAME_SECONDS = 0.1
_MIN_FRAME_SECONDS = 1.0e-6
_SCENE_LAYER_ID = "scene_preview"
_DEFAULT_SCENE = ASSETS_ROOT / "scenes" / "soma_gym.scene.json"


class _RuntimeSceneSmartPrimitive:
    """A placed Smart Primitive with its interaction boxes and debug previews.

    reference_pose gives the character's pose layout, and target_pose_renderer
    draws the rig's target poses.
    """

    def __init__(
        self,
        definition: SmartPrimitiveInstance,
        runtime: SmartPrimitiveBehavior,
        interaction_box_offset: int,
        reference_pose: wp.array,
        target_pose_renderer: SkeletonRenderer | DOFSkeletonRenderer,
        stream: wp.Stream,
    ) -> None:
        self.definition = definition
        self.runtime = runtime
        self.interaction_box_offset = interaction_box_offset
        self.interaction_box_count = runtime.interaction_box_count
        self.interactable = False
        self._interaction_boxes_dirty = True
        runtime.set_node_world_transform(
            0,
            definition.transform,
            stream=stream,
        )

        self.target_pose_scratch = wp.empty(
            (MODEL_POSES_PER_TOKEN, *reference_pose.shape),
            dtype=reference_pose.dtype,
            device=runtime.device,
        )
        self.target_pose_colors = wp.array([definition.target_pose_color], dtype=wp.vec3, device=runtime.device)
        self.target_pose_renderer = target_pose_renderer

        self.wire_box_renderer = WireBoxRenderer(
            max_boxes=max(1, self.interaction_box_count)
        )

    def distance_squared_to(self, position: np.ndarray) -> float:
        root_position = np.array(self.definition.transform.p, dtype=np.float32)
        offset = root_position - position
        return float(np.dot(offset, offset))

    def refresh_interaction_boxes(
        self,
        scene_interaction_boxes: wp.array,
        stream: wp.Stream,
        nodes_moved: bool = False,
    ) -> None:
        """Update the interaction boxes on the first call, or when nodes_moved says their nodes moved."""

        if self.interaction_box_count == 0 or not (nodes_moved or self._interaction_boxes_dirty):
            return
        start = self.interaction_box_offset
        end = start + self.interaction_box_count
        self.runtime.get_world_interaction_boxes(
            scene_interaction_boxes[start:end],
            stream=stream,
        )
        self._interaction_boxes_dirty = False

    def draw(
        self,
        viewer,
        scene_interaction_boxes: wp.array,
        stream: wp.Stream,
        show_target_poses: bool,
        show_interaction_boxes: bool,
    ) -> None:
        runtime = self.runtime
        instance_id = self.definition.id
        for node_index in range(runtime.node_count):
            if not runtime.has_target_pose(node_index):
                continue
            renderer_id = f"scene_{instance_id}_target_{node_index}"
            if show_target_poses and self.definition.show_target_poses:
                runtime.get_world_target_poses(
                    node_index,
                    self.target_pose_scratch,
                    stream=stream,
                )
                # Only each target's first pose is drawn, so the targets stay readable.
                self.target_pose_renderer.draw_batch(
                    viewer,
                    self.target_pose_scratch,
                    1,
                    wp.transform_identity(),
                    self.target_pose_colors,
                    renderer_id,
                )
            else:
                self.target_pose_renderer.hide(viewer, renderer_id)

        renderer_id = f"scene_{instance_id}_interaction_boxes"
        box_count = self.interaction_box_count
        if (
            show_interaction_boxes
            and self.definition.show_interaction_boxes
            and box_count
        ):
            start = self.interaction_box_offset
            boxes = scene_interaction_boxes[start : start + box_count]
            color = (
                _ACTIVE_SMART_PRIMITIVE_BOX_COLOR
                if self.interactable
                else _INACTIVE_SMART_PRIMITIVE_BOX_COLOR
            )
            self.wire_box_renderer.draw_batch(
                viewer,
                boxes,
                box_count,
                color,
                renderer_id,
            )
        else:
            self.wire_box_renderer.hide(viewer, renderer_id)


@dataclass(slots=True)
class _SceneSession:
    """The character, scene items, and rendering buffers for one loaded scene.

    Poses are local joint transforms for SOMA and joint coordinates for Unitree G1.
    context_poses shows the poses in the character's context token. target_poses shows
    the poses in its target token.
    """

    path: Path
    definition: SceneDefinition
    state_machine: StateMachine
    character: SOMACharacter | UnitreeG1Character
    smart_primitives: tuple[_RuntimeSceneSmartPrimitive, ...]
    interaction_boxes: wp.array
    interaction_overlap_results: wp.array
    interaction_box_count: int
    character_color: wp.vec3 | None
    context_poses: wp.array
    target_poses: wp.array
    blended_target_poses: wp.array
    blendspace_target_poses: wp.array
    solid_box_renderer: SolidBoxRenderer
    mesh_renderer: SkeletalMeshRenderer | DOFMeshRenderer
    context_renderer: SkeletonRenderer | DOFSkeletonRenderer
    target_renderer: SkeletonRenderer | DOFSkeletonRenderer
    blendspace_renderer: SkeletonRenderer | DOFSkeletonRenderer
    interaction_box_renderer: WireBoxRenderer
    foot_contact_renderer: SkeletonFootContactRenderer | DOFFootContactRenderer
    model: object
    simulation_state: object


def _check_scene(scene: SceneDefinition) -> None:
    """Check each Smart Primitive's reference, behavior, and rig before anything loads."""

    for definition in scene.smart_primitives:
        if definition.reference is None:
            raise ValueError(
                f"Smart Primitive {definition.name!r} has no reference."
            )
        if definition.behavior not in behavior_types():
            raise ValueError(
                f"Smart Primitive {definition.name!r} uses unknown behavior "
                f"{definition.behavior!r}."
            )
    check_reference_rigs(scene)


def _camera_pivot_position(ground_position: np.ndarray) -> np.ndarray:
    pivot_position = np.asarray(ground_position, dtype=np.float32).copy()
    pivot_position[2] += _CAMERA_PIVOT_Z_OFFSET
    return pivot_position


def _format_vector3(vector: np.ndarray) -> str:
    return f"({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f})"


class ScenePreview:
    """Move a character through a saved scene and interact with its Smart Primitives.

    The scene's rig chooses the character. One rig and its engine are loaded at a
    time. A scene for the other rig closes the current engine before the new one loads.
    """

    _scene_session: _SceneSession

    def __init__(
        self,
        viewer,
        scene_path: str | Path,
        *,
        fp16: bool = False,
    ) -> None:
        """Open Scene Preview on a graphical viewer and load the scene at scene_path.

        fp16 selects the mixed FP16 models.
        """
        if isinstance(viewer, newton.viewer.ViewerNull):
            raise RuntimeError("Scene Preview requires a graphical viewer.")

        self.viewer = viewer
        self._rig_assets: RigAssets | None = None
        self._engine: MotionBricksSOMAEngine | MotionBricksUnitreeG1Engine | None = None
        self._pending_scene_path: Path | None = None
        self.scene_error: str | None = None

        self.show_context_poses = False
        self.show_target_poses = False
        self.show_blendspace_targets = False
        self.show_probe_box = False
        self.show_actual_velocity = True
        self.show_input_velocity = True
        self.show_desired_velocity = True
        self.show_foot_contacts = False
        self.foot_contact_threshold = FootContactRenderer.DEFAULT_THRESHOLD
        self.show_scene_object_target_poses = False
        self.show_scene_object_interaction_boxes = True
        self.auto_interaction = True
        self.fp16 = fp16

        self.context_colors = wp.array(
            [_CONTEXT_COLOR] * MODEL_POSES_PER_TOKEN,
            dtype=wp.vec3,
        )
        self.target_colors = wp.array(
            [_TARGET_COLOR] * MODEL_POSES_PER_TOKEN,
            dtype=wp.vec3,
        )
        self.blendspace_debug_colors = wp.array(
            [_BLENDSPACE_DEBUG_COLOR],
            dtype=wp.vec3,
        )
        self.character_velocity_renderer = VectorArrowRenderer(
            "actual_velocity", wp.vec3(*_CHARACTER_VELOCITY_ARROW_COLOR),
        )
        self.input_velocity_renderer = VectorArrowRenderer(
            "input_velocity", wp.vec3(*_INPUT_VELOCITY_ARROW_COLOR),
        )
        self.desired_velocity_renderer = VectorArrowRenderer(
            "desired_velocity", wp.vec3(*_DESIRED_VELOCITY_ARROW_COLOR),
        )

        self.viewer.camera_speed = 0.0
        self.viewer.vsync = False
        self.viewer.renderer.set_title("MotionBricks - Scene Preview")
        self._input = InputManager.from_pyglet(self.viewer.is_key_down)
        try:
            self.load_scene(scene_path)
        except BaseException:
            self.close()
            raise
        # The camera follows the character, so wait for the first scene.
        self.viewer.renderer.register_mouse_drag(self._on_mouse_drag)

    def close(self) -> None:
        """Release the input devices and the engine."""

        self._input.close()
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def load_scene(self, path: str | Path) -> None:
        """Load a scene and replace the current one.

        Failures in _prepare_scene() keep the current scene. Failures after it mean a
        broken installation, such as a missing asset or a failed TensorRT build.
        """

        self._start_scene(*self._prepare_scene(path))

    def _prepare_scene(self, path: str | Path) -> tuple[Path, SceneDefinition, RigAssets, SmartPrimitiveAssetPool]:
        """Read and check a scene, then load its rig and every Smart Primitive it uses.

        Nothing here needs an engine, so the current scene keeps running if it fails.
        """

        scene_path = Path(path).expanduser().resolve()
        scene, _, _ = load_scene_document(scene_path)
        # Parenting only helps editing. Runtime items are placed by their world transforms.
        scene = scene.flatten()
        _check_scene(scene)
        assets = self._rig_assets
        if assets is None or assets.clip_registry.rig is not scene.rig:
            assets = load_rig_assets(scene.rig)
        asset_pool = SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)
        for definition in scene.smart_primitives:
            asset_pool.load(definition.reference)
        return scene_path, scene, assets, asset_pool

    def _start_scene(
        self,
        path: Path,
        scene: SceneDefinition,
        assets: RigAssets,
        asset_pool: SmartPrimitiveAssetPool,
    ) -> None:
        self._use_rig(assets)
        with wp.ScopedStream(self._engine.stream) if self._engine.stream is not None else nullcontext():
            session = self._create_scene_session(path, scene, asset_pool)
            self._activate_scene_session(session)

    def _use_rig(self, assets: RigAssets) -> None:
        """Make assets the current rig. For the other rig, close its engine before loading this one."""

        if assets is self._rig_assets:
            return
        if self._engine is not None:
            self._engine.synchronize()
            self._engine.close()
            self._engine = None
        self._rig_assets = assets
        self._engine = create_engine(self.viewer, assets.clip_registry.rig, self.fp16)

    def _create_scene_session(
        self,
        path: Path,
        scene: SceneDefinition,
        asset_pool: SmartPrimitiveAssetPool,
    ) -> _SceneSession:
        engine = self._engine
        assets = self._rig_assets
        start = scene.character_start.transform
        character = _CHARACTER_TYPES[scene.rig](engine, asset_pool=asset_pool, initial_transform=start)
        character_color = CHARACTER_COLORS[scene.rig]
        mesh_renderer = create_mesh_renderer(scene.rig, assets.model)
        skeleton_renderer = partial(create_skeleton_renderer, scene.rig, assets.model)
        foot_contact_renderer = create_foot_contact_renderer(scene.rig, "foot_contacts", assets.model)
        context_poses = character.player.context.poses
        target_poses = character.player.target.poses
        state_machine = character.state_machine
        smart_primitives, interaction_box_count = (
            self._create_smart_primitive_instances(
                scene,
                asset_pool,
                skeleton_renderer,
                engine.stream,
            )
        )
        for instance in smart_primitives:
            state_machine.interaction_state.prepare(character, instance.runtime)
        interaction_box_capacity = max(1, interaction_box_count)
        pose = character.current_pose
        blended_target_poses = wp.empty(
            (MODEL_POSES_PER_TOKEN, *pose.shape),
            dtype=pose.dtype,
            device=engine.device,
        )

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        model = builder.finalize()
        return _SceneSession(
            path=path,
            definition=scene,
            state_machine=state_machine,
            character=character,
            smart_primitives=smart_primitives,
            interaction_boxes=wp.empty(
                interaction_box_capacity,
                dtype=InteractionBox,
                device=engine.device,
            ),
            interaction_overlap_results=wp.empty(
                interaction_box_capacity,
                dtype=wp.int32,
                device=engine.device,
            ),
            interaction_box_count=interaction_box_count,
            character_color=character_color,
            context_poses=context_poses,
            target_poses=target_poses,
            blended_target_poses=blended_target_poses,
            blendspace_target_poses=wp.empty_like(blended_target_poses),
            solid_box_renderer=SolidBoxRenderer(
                max_boxes=max(1, len(scene.solid_cubes)),
                device=engine.device,
            ),
            mesh_renderer=mesh_renderer,
            context_renderer=skeleton_renderer(max_instances=MODEL_POSES_PER_TOKEN),
            target_renderer=skeleton_renderer(max_instances=MODEL_POSES_PER_TOKEN),
            blendspace_renderer=skeleton_renderer(max_instances=1),
            interaction_box_renderer=WireBoxRenderer(),
            foot_contact_renderer=foot_contact_renderer,
            model=model,
            simulation_state=model.state(),
        )

    def _activate_scene_session(self, session: _SceneSession) -> None:
        # The previous scene may run on another engine's stream. Finish its
        # work before its buffers are released.
        wp.synchronize_device(session.character.engine.device)
        if _SCENE_LAYER_ID in self.viewer.layers:
            self.viewer.remove_layer(_SCENE_LAYER_ID)
        self.viewer.activate(_SCENE_LAYER_ID)
        self.viewer.set_model(session.model)
        self.viewer.set_world_offsets([0, 0, 0])

        self._scene_session = session

        self.time = 0.0
        self._mode_input = LocomotionModeInput(session.state_machine.style_modes)
        self._interaction_down = False
        self._active_interaction = None
        self._auto_interaction_blocked: set[int] = set()
        self.camera_facing = session.character.movement.current_facing.copy()
        self.scene_error = None
        self._initialize_camera()
        self._update_smart_primitive_interaction()
        self._last_frame_time = time.perf_counter()

        # Removing a model from ViewerGL also removes our callbacks.
        self.viewer.register_ui_callback(self.gui, position="free")

    def _request_scene_load(self) -> None:
        try:
            selected = choose_file("Open Scene", "Scene documents", SCENE_DOCUMENT.suffix, self._scene_session.path)
        except Exception as exc:
            self.scene_error = str(exc)
            return
        if selected:
            self._pending_scene_path = Path(selected)

    def _load_pending_scene(self) -> None:
        path = self._pending_scene_path
        if path is None:
            return
        self._pending_scene_path = None
        try:
            prepared = self._prepare_scene(path)
        except Exception as exc:
            self.scene_error = str(exc)
            return
        self._start_scene(*prepared)

    def _create_smart_primitive_instances(
        self,
        scene: SceneDefinition,
        asset_pool: SmartPrimitiveAssetPool,
        skeleton_renderer,
        stream: wp.Stream,
    ) -> tuple[tuple[_RuntimeSceneSmartPrimitive, ...], int]:
        instances = []
        interaction_box_offset = 0
        for definition in scene.smart_primitives:
            runtime = asset_pool.instantiate(
                definition.reference,
                behavior_types()[definition.behavior],
            )
            instances.append(
                _RuntimeSceneSmartPrimitive(
                    definition,
                    runtime,
                    interaction_box_offset,
                    asset_pool.reference_pose,
                    skeleton_renderer(max_instances=MODEL_POSES_PER_TOKEN),
                    stream,
                )
            )
            interaction_box_offset += runtime.interaction_box_count
        return tuple(instances), interaction_box_offset

    def _initialize_camera(self) -> None:
        character = self._scene_session.character
        self._camera_target = character.character_ground_position.copy()
        self._camera_velocity = np.zeros(2)
        self._camera_acceleration = np.zeros(2)
        elevation = math.radians(_CAMERA_ELEVATION_DEGREES)
        horizontal_distance = _CAMERA_DISTANCE * math.cos(elevation)
        height = _CAMERA_DISTANCE * math.sin(elevation)
        pivot_position = _camera_pivot_position(
            character.character_ground_position
        )
        position = pivot_position - character.movement.current_facing * horizontal_distance
        position[2] += height

        camera = self.viewer.camera
        camera.pos = Vec3(*position)
        camera.look_at(pivot_position)

    def _follow_character(self, frame_dt: float) -> None:
        """Move the camera's followed point smoothly with the character.

        A damped spring smooths the character's velocity. The point moves at that velocity and
        settles on the character, so the sway of each stride does not reach the camera.
        """
        character = self._scene_session.character
        for axis in range(2):
            self._camera_velocity[axis], self._camera_acceleration[axis] = critical_spring_damper_exact(
                self._camera_velocity[axis],
                self._camera_acceleration[axis],
                float(character.actual_velocity[axis]),
                _CAMERA_VELOCITY_HALFLIFE,
                frame_dt,
            )
        self._camera_target[:2] += self._camera_velocity * frame_dt
        pull = 1.0 - math.exp(-frame_dt / _CAMERA_PULL_SECONDS)
        self._camera_target += (character.character_ground_position - self._camera_target) * pull

    def _update_camera(self) -> None:
        camera = self.viewer.camera
        pivot = np.asarray(camera.pivot, dtype=np.float32)
        pivot_offset = _camera_pivot_position(self._camera_target) - pivot
        if float(np.dot(pivot_offset, pivot_offset)) > 1.0e-12:
            camera.translate(pivot_offset)
        self.camera_facing = ground_direction(
            camera.get_front(),
            self.camera_facing,
        )

    def _on_mouse_drag(self, _x, _y, _dx, _dy, _buttons, _modifiers) -> None:
        self._update_camera()

    def _orbit_camera(
        self,
        camera_orbit: tuple[float, float],
        frame_dt: float,
    ) -> None:
        right, up = camera_orbit
        if right == 0.0 and up == 0.0:
            return
        self.viewer.camera.orbit(
            delta_yaw=-right * _CONTROLLER_CAMERA_ORBIT_SPEED * frame_dt,
            delta_pitch=up * _CONTROLLER_CAMERA_ORBIT_SPEED * frame_dt,
        )
        self._update_camera()

    def _requested_locomotion_mode(self) -> str | None:
        locomotion = self._scene_session.state_machine.active_locomotion_state
        return self._mode_input.poll(self._input, locomotion.active_mode.name)

    def _interaction_pressed(self) -> bool:
        down = self._input.is_interaction_down()
        manually_requested = down and not self._interaction_down
        self._interaction_down = down
        return manually_requested

    def _update_smart_primitive_interaction(self) -> None:
        session = self._scene_session
        character = session.character
        stream = character.engine.stream
        box_count = session.interaction_box_count
        if box_count == 0:
            self._nearest_interactable = None
            return

        for instance in session.smart_primitives:
            # A running behavior can move its nodes, so the active Smart Primitive's boxes refresh every frame.
            instance.refresh_interaction_boxes(
                session.interaction_boxes,
                stream,
                nodes_moved=instance is self._active_interaction,
            )
        compute_interaction_box_overlaps(
            session.interaction_boxes[:box_count],
            character.probe_box,
            session.interaction_overlap_results[:box_count],
            stream,
        )
        overlap_results = session.interaction_overlap_results.numpy()
        for instance in session.smart_primitives:
            start = instance.interaction_box_offset
            end = start + instance.interaction_box_count
            instance.interactable = bool(np.any(overlap_results[start:end]))

        interactable_instances = (
            instance
            for instance in session.smart_primitives
            if instance.interactable
        )
        self._nearest_interactable = min(
            interactable_instances,
            key=lambda instance: instance.distance_squared_to(
                character.character_ground_position
            ),
            default=None,
        )

    def _start_smart_primitive_interaction(
        self,
        instance: _RuntimeSceneSmartPrimitive,
    ) -> None:
        session = self._scene_session
        session.state_machine.start_interaction(
            session.character, instance.runtime,
        )
        self._active_interaction = instance

    def _dispatch_interaction_input(self) -> None:
        """Handle the interaction button, I or the controller's X button.

        A press starts an interaction with the nearest Smart Primitive in range, or
        passes the press to the running one. Auto Interact only starts
        interactions, once per approach.
        """
        session = self._scene_session
        machine = session.state_machine
        pressed = self._interaction_pressed()

        if self._active_interaction is not None and not machine.is_interacting:
            finished = self._active_interaction
            self._auto_interaction_blocked.add(finished.definition.id)
            self._active_interaction = None
        for instance in session.smart_primitives:
            if not instance.interactable:
                self._auto_interaction_blocked.discard(instance.definition.id)

        if machine.is_interacting:
            if pressed:
                machine.interact()
        else:
            instance = self._nearest_interactable
            if instance is not None and (
                pressed or (
                    self.auto_interaction
                    and instance.definition.id not in self._auto_interaction_blocked
                )
            ):
                self._start_smart_primitive_interaction(instance)

    def step(self) -> None:
        """Read input, then advance the character and the camera.

        Each step covers the time since the last frame.
        """
        session = self._scene_session
        frame_time = time.perf_counter()
        frame_dt = min(max(frame_time - self._last_frame_time, _MIN_FRAME_SECONDS), _MAX_FRAME_SECONDS)
        self._last_frame_time = frame_time
        self.time += frame_dt

        input_frame = self._input.poll()
        self._update_camera()
        self._orbit_camera(input_frame.camera_orbit, frame_dt)
        command = LocomotionCommand(
            facing_direction=self.camera_facing,
            movement=camera_relative_movement(self.camera_facing, input_frame.movement),
            use_max_speed=input_frame.use_max_speed,
            requested_mode=self._requested_locomotion_mode(),
        )
        session.character.step(frame_dt, command)
        self._follow_character(frame_dt)
        self._update_camera()

        self._update_smart_primitive_interaction()
        self._dispatch_interaction_input()

        if self.show_blendspace_targets and session.state_machine.is_locomoting:
            session.state_machine.active_locomotion_state.evaluate_current_target_poses(
                session.character,
                session.blended_target_poses,
                session.blendspace_target_poses,
            )

    def _draw_smart_primitives(self) -> None:
        session = self._scene_session
        stream = session.character.engine.stream
        for instance in session.smart_primitives:
            instance.draw(
                self.viewer,
                session.interaction_boxes,
                stream,
                self.show_scene_object_target_poses,
                self.show_scene_object_interaction_boxes,
            )

    def _draw_solid_cubes(self) -> None:
        session = self._scene_session
        cubes = session.definition.solid_cubes
        session.solid_box_renderer.draw_batch(
            self.viewer,
            [cube.transform for cube in cubes],
            [wp.vec3(cube.width, cube.depth, cube.height) for cube in cubes],
            [cube.color for cube in cubes],
            "scene",
        )

    def _draw_character_target_poses(self) -> None:
        session = self._scene_session
        target = session.character.player.target
        if self.show_target_poses and target.num_poses > 0:
            session.target_renderer.draw_batch(
                self.viewer,
                session.target_poses,
                target.num_poses,
                wp.transform_identity(),
                self.target_colors,
                "target",
            )
        else:
            session.target_renderer.hide(self.viewer, "target")

    def _draw_character_context_poses(self) -> None:
        session = self._scene_session
        player = session.character.player
        if self.show_context_poses:
            player.refresh_context()
            session.context_renderer.draw_batch(
                self.viewer,
                session.context_poses,
                player.context.num_poses,
                wp.transform_identity(),
                self.context_colors,
                "context",
            )
        else:
            session.context_renderer.hide(self.viewer, "context")

    def _draw_foot_contacts(self) -> None:
        """Draw the model's predicted foot contacts on the character.

        Outlines show only during generated motion. Clips have no foot contacts.
        """
        session = self._scene_session
        if not self.show_foot_contacts:
            session.foot_contact_renderer.hide(self.viewer)
            return
        character = session.character
        session.foot_contact_renderer.threshold = self.foot_contact_threshold
        session.foot_contact_renderer.draw_pose(self.viewer, character.current_pose, character.current_foot_contacts)

    def render(self) -> None:
        """Draw the character, the scene items, and the debug overlays for one frame."""
        session = self._scene_session
        character = session.character
        self.viewer.begin_frame(self.time)
        session.mesh_renderer.draw_pose(
            self.viewer,
            character.current_pose,
            wp.transform_identity(),
            session.character_color,
            0,
        )
        self._draw_foot_contacts()
        self._draw_solid_cubes()
        self._draw_smart_primitives()
        self._draw_character_context_poses()
        self._draw_character_target_poses()

        if self.show_blendspace_targets and session.state_machine.is_locomoting:
            session.blendspace_renderer.draw_batch(
                self.viewer,
                session.blendspace_target_poses,
                1,
                wp.transform_identity(),
                self.blendspace_debug_colors,
                "blendspace_locomotion",
            )
        else:
            session.blendspace_renderer.hide(
                self.viewer,
                "blendspace_locomotion",
            )

        if self.show_probe_box and session.state_machine.is_locomoting:
            session.interaction_box_renderer.draw_batch(
                self.viewer,
                character.probe_box,
                1,
                _INTERACTION_BOX_COLOR,
                "character_interaction",
            )
        else:
            session.interaction_box_renderer.hide(
                self.viewer,
                "character_interaction",
            )

        origin = wp.vec3(*character.character_ground_position) + wp.vec3(0.0, 0.0, _VELOCITY_ARROW_HEIGHT)
        for renderer, visible, velocity in (
            (self.character_velocity_renderer, self.show_actual_velocity, character.actual_velocity),
            (self.input_velocity_renderer, self.show_input_velocity, character.movement.input_velocity),
            (self.desired_velocity_renderer, self.show_desired_velocity, character.movement.desired_velocity),
        ):
            if visible:
                renderer.draw(self.viewer, 0, wp.transform_identity(), origin, wp.vec3(*velocity))
            else:
                renderer.hide(self.viewer, 0)
        self.viewer.log_state(session.simulation_state)
        self.viewer.end_frame()

    def gui(self, ui) -> None:
        """Draw the Scene Preview window."""
        session = self._scene_session
        dpi_scale = max(1.0, float(ui.get_style().font_scale_dpi))
        viewport = ui.get_main_viewport()
        margin = WINDOW_MARGIN * dpi_scale

        ui.set_next_window_pos(
            ui.ImVec2(
                viewport.pos.x + margin,
                viewport.pos.y + viewport.size.y - margin,
            ),
            ui.Cond_.first_use_ever,
            pivot=ui.ImVec2(0.0, 1.0),
        )
        ui.set_next_window_size(
            ui.ImVec2(_PANEL_WIDTH * dpi_scale, _PANEL_HEIGHT * dpi_scale),
            ui.Cond_.first_use_ever,
        )
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)
        # Double-clicking the title bar collapses the window, as in the Newton viewer.
        expanded, _ = ui.begin("Scene Preview")
        if not expanded:
            ui.end()
            return

        table_flags = (
            ui.TableFlags_.sizing_fixed_fit
            | ui.TableFlags_.no_saved_settings
            | ui.TableFlags_.no_pad_outer_x
        )
        if ui.collapsing_header(
            "Status",
            flags=ui.TreeNodeFlags_.default_open,
        ):
            character = session.character
            mode = session.state_machine.active_locomotion_state.active_mode
            values = (
                ("State:", session.state_machine.state.name),
                ("Locomotion:", mode.label or mode.name),
                ("Time To Transition:", f"{character.time_to_transition():.2f} s"),
                (
                    "Position:",
                    f"{_format_vector3(character.character_ground_position)} m",
                ),
                (
                    "Speed:",
                    f"{np.linalg.norm(character.actual_velocity):.2f} m/s",
                ),
            )
            if ui.begin_table("##StatusValues", 2, table_flags):
                for label, value in values:
                    ui.table_next_row()
                    ui.table_set_column_index(0)
                    ui.text(label)
                    ui.table_set_column_index(1)
                    ui.text(value)
                ui.end_table()

        ui.separator()
        if ui.collapsing_header(
            "Scene",
            flags=ui.TreeNodeFlags_.default_open,
        ):
            ui.align_text_to_frame_padding()
            ui.text("Scene:")
            ui.same_line()
            ui.text(session.path.name)
            ui.same_line()
            if ui.button("...##LoadScene"):
                self._request_scene_load()
            _, self.auto_interaction = ui.checkbox(
                "Auto Interact",
                self.auto_interaction,
            )
            if self.scene_error is not None:
                # The same red as the editors' error status.
                ui.push_style_color(ui.Col_.text, ui.ImVec4(1.0, 0.35, 0.35, 1.0))
                ui.text_wrapped(f"Open failed: {self.scene_error}")
                ui.pop_style_color()

        ui.separator()
        if ui.collapsing_header(
            "Visibility",
            flags=ui.TreeNodeFlags_.default_open,
        ):
            ui.text("Character")
            _, self.show_target_poses = ui.checkbox(
                "Show Target Poses##Character",
                self.show_target_poses,
            )
            _, self.show_context_poses = ui.checkbox(
                "Show Context Poses",
                self.show_context_poses,
            )
            _, self.show_blendspace_targets = ui.checkbox(
                "Show Blend Space Poses",
                self.show_blendspace_targets,
            )
            _, self.show_probe_box = ui.checkbox(
                "Show Probe Box",
                self.show_probe_box,
            )
            _, self.show_actual_velocity = ui.checkbox(
                "Show Actual Velocity",
                self.show_actual_velocity,
            )
            _, self.show_input_velocity = ui.checkbox(
                "Show Input Velocity",
                self.show_input_velocity,
            )
            _, self.show_desired_velocity = ui.checkbox(
                "Show Desired Velocity",
                self.show_desired_velocity,
            )
            _, self.show_foot_contacts = ui.checkbox(
                "Show Foot Contacts",
                self.show_foot_contacts,
            )
            ui.same_line()
            ui.set_next_item_width(80 * dpi_scale)
            ui.begin_disabled(not self.show_foot_contacts)
            _, self.foot_contact_threshold = ui.slider_float(
                "##FootContactThreshold",
                self.foot_contact_threshold,
                0.0,
                1.0,
                "%.2f",
                flags=ui.SliderFlags_.always_clamp,
            )
            ui.end_disabled()

            ui.spacing()
            ui.text("Smart Primitives")
            _, self.show_scene_object_target_poses = ui.checkbox(
                "Show Target Poses##SceneObjects",
                self.show_scene_object_target_poses,
            )
            _, self.show_scene_object_interaction_boxes = ui.checkbox(
                "Show Interaction Boxes",
                self.show_scene_object_interaction_boxes,
            )

        ui.separator()
        if ui.collapsing_header(
            "Model Stats",
            flags=ui.TreeNodeFlags_.default_open,
        ):
            stage_times = session.character.engine.timer.milliseconds
            timings = (
                ("Encode:", stage_times["encode"]),
                ("Inference:", stage_times["inference"]),
                ("Decode:", stage_times["decode"]),
                ("Post-Process:", stage_times["post_process"]),
            )
            total_model_execution_ms = sum(value for _, value in timings)
            if ui.begin_table("##RuntimeTimings", 2, table_flags):
                ui.table_next_row()
                ui.table_set_column_index(0)
                ui.text("Precision:")
                ui.table_set_column_index(1)
                ui.text("Mixed FP16" if self.fp16 else "FP32")

                for label, value in timings:
                    ui.table_next_row()
                    ui.table_set_column_index(0)
                    ui.text(label)
                    ui.table_set_column_index(1)
                    ui.text(f"{value:.3f} ms")

                ui.table_next_row()
                for column in range(2):
                    ui.table_set_column_index(column)
                    ui.separator()

                ui.table_next_row()
                ui.table_set_column_index(0)
                ui.text("Total:")
                ui.table_set_column_index(1)
                ui.text(f"{total_model_execution_ms:.3f} ms")
                ui.end_table()
        ui.end()

    def run(self, max_frames: int | None = None) -> None:
        """Step and draw frames until the window closes or max_frames is reached."""
        try:
            frame = 0
            while self.viewer.is_running() and (
                max_frames is None or frame < max_frames
            ):
                # Loading can build another engine and switch streams, so it
                # runs between frames, outside the active engine's stream.
                self._load_pending_scene()
                with wp.ScopedStream(self._scene_session.character.engine.stream) if self._scene_session.character.engine.stream is not None else nullcontext():
                    self.step()
                    self.render()
                frame += 1
        finally:
            session = self._scene_session
            if session.state_machine.is_interacting:
                with wp.ScopedStream(session.character.engine.stream) if session.character.engine.stream is not None else nullcontext():
                    session.state_machine.interaction_state.exit(
                        session.character, session.state_machine,
                    )
            self.close()
            self.viewer.close()


def main() -> None:
    """Parse the command line and open Scene Preview."""
    import newton.examples

    parser = create_gl_parser()
    add_model_precision_argument(parser)
    parser.add_argument(
        "--scene",
        type=Path,
        default=_DEFAULT_SCENE,
        help="Scene document to open. Its rig selects SOMA or Unitree G1.",
    )
    viewer, args = newton.examples.init(parser)
    configure_logging(args)
    require_inference_backend()
    ScenePreview(viewer, args.scene, fp16=args.model_precision == "fp16").run(
        args.num_frames if args.test else None
    )


if __name__ == "__main__":
    main()
