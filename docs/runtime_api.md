# Runtime API

This page shows how to use the `motionbricks` library in your own application. The sections follow the order of an integration: set up an engine, prepare poses, generate, and chain generations. The `app` package holds demos built on the library. Two of them show it at work. The [Inbetweener](inbetweener.md) chains generations by hand, so each step stays visible. [Scene Preview](scene_preview.md) uses a motion player and a recorder for continuous locomotion and interactions.

## Overview

### Generation flow

MotionBricks builds continuous motion one generation at a time. Each generation takes four context poses and one to four target poses. Your app plays the generated motion up to the transition point and records every pose it shows. The next generation reads its context from those records and generates toward a new target.

![MotionBricks generation flow: the engine generates from context and target poses, and the last four played poses become the next context](media/generation_flow.png)

| Step | API | Section |
| --- | --- | --- |
| Prepare poses | `ClipRegistry`, `sample()` with `root_offset`, or poses from your application | [Place poses in the scene](#place-poses-in-the-scene), [Clips](#clips) |
| Fill the tokens | `create_input()`, `update_from_poses()`, `update_from_recorder()` | [Engines](#engines), [Record what you show](#record-what-you-show) |
| Generate | `generate_animation()`, `MotionBricksInferenceConfig`, `TargetPoseInferenceParams` | [Generation settings](#generation-settings) |
| Play | `sample()`, `output.num_frames`, `sample_foot_contacts()` | [Engines](#engines), [Foot contacts](#foot-contacts) |
| Record and chain | `MotionRecorder`, `GeneratedMotionPlayer`, `time_to_transition` | [Chaining generations](#chaining-generations) |

The Inbetweener runs this loop once for each target in its list. Scene Preview runs it without end, with targets from locomotion and Smart Primitive behaviors.

### Runnable examples

Two short scripts in [examples](examples) call the library without a viewer or the apps. Run them from the repository root, for either rig:

```sh
uv run python docs/examples/generate_once.py
uv run python docs/examples/locomotion_loop.py --model unitree_g1
```

- [generate_once.py](examples/generate_once.py) generates from clip poses, then chains a second generation from the last four poses it played. It shows the [generation flow](#generation-flow) in its simplest form.
- [locomotion_loop.py](examples/locomotion_loop.py) runs the [continuous locomotion loop](#a-continuous-locomotion-loop) with a motion player. It steers a jog around a left turn, scales the target poses to the desired speed, and places them ahead of the character.

[_rig.py](examples/_rig.py) holds the setup that both scripts share. It uses only the `motionbricks` package, so you can copy it. On NVIDIA systems the first launch builds a TensorRT plan. On Apple Silicon, ONNX Runtime initializes its CPU provider, or the experimental Core ML provider when explicitly selected.

### Packages

| Package | Contents |
| --- | --- |
| [motionbricks/model](../motionbricks/model) | Model contracts, engines, context tokens, target tokens, and generation settings |
| [motionbricks/inference](../motionbricks/inference) | Run models through TensorRT or ONNX Runtime |
| [motionbricks/animation](../motionbricks/animation), [motionbricks/robotics](../motionbricks/robotics) | `AnimationBuffer`, `DOFAnimationBuffer`, skeletons, and meshes |
| [motionbricks/io](../motionbricks/io) | BVH, CSV, and USD loading |
| [motionbricks/runtime](../motionbricks/runtime) | Motion players and recorders, target pose helpers, blend spaces, runtime Smart Primitives, and pivots |
| [motionbricks/renderers](../motionbricks/renderers) | Newton viewer renderers for meshes, skeletons, and debug shapes |
| [app/runtime](../app/runtime) | The demo runtime: characters, states, behaviors, documents, and presets |

Each class or function uses its `device` argument, the device of an object it receives, or the current Warp device. For example, `create_input()` uses the engine's device. A recorder uses the device of its reference pose. A renderer uses the device of its skeleton or Newton model.

## Engines

On NVIDIA systems, each engine builds or loads a cached TensorRT plan and orders its work on `engine.stream`. On Apple Silicon, engines use ONNX Runtime and CPU Warp buffers; `engine.stream` is `None` and inference is synchronous. The experimental Core ML provider can delegate unsupported operators to the CPU. Use an engine as a context manager, or call `close()` when you are done.

```python
from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.soma.engine import MotionBricksSOMAEngine

with MotionBricksSOMAEngine(SOMA_N1_FP32_ONNX_PATH, model_to_runtime=model_to_runtime) as engine:
    context = engine.create_input(skeleton)
    target = engine.create_input(skeleton)
    output = engine.create_output(skeleton)

    context.update_from_poses(context_poses, stream=engine.stream)  # four poses
    target.update_from_poses(target_poses, stream=engine.stream)    # one to four poses
    engine.generate_animation(context, target, output)

    # output.num_frames marks the valid range of the reused buffer.
    output.sample(time, current_pose, stream=engine.stream)
    engine.synchronize()
```

```python
from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_FP32_ONNX_PATH
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine

with MotionBricksUnitreeG1Engine(UNITREE_G1_N1_FP32_ONNX_PATH, model_to_runtime=model_to_runtime) as engine:
    context = engine.create_input()
    target = engine.create_input()
    output = engine.create_output()

    context.update_from_poses(context_joint_q, stream=engine.stream)  # four poses
    target.update_from_poses(target_joint_q, stream=engine.stream)    # one to four poses
    engine.generate_animation(context, target, output)

    # output.num_frames marks the valid range of the reused buffer.
    output.sample(time, robot_joint_q, stream=engine.stream)
    engine.synchronize()
```

| | `MotionBricksSOMAEngine` | `MotionBricksUnitreeG1Engine` |
| --- | --- | --- |
| Context and target tokens | `PoseToken`, local joint transforms | `DOFToken`, joint coordinates |
| Output buffer | `AnimationBuffer`, the same type as loaded BVH clips | `DOFAnimationBuffer`, the same type as loaded CSV clips |
| Pose | Local joint transforms as `wp.transform`, root first: an XYZ position and an XYZW quaternion per joint | 36 joint coordinates: the root's XYZ position and XYZW quaternion, then 29 joint angles |
| Model space | Y up, with XZ as the ground plane | Z up, with +X forward |
| World space | Set by `model_to_runtime` | Set by `model_to_runtime` |

The apps never work with model features directly. They sample poses from clips, move the root to place each pose in the scene, and pass the poses to the engine. Joint values are used as sampled. Clips, Smart Primitive documents, states, and scenes all work with either layout, so the same apps drive both characters. The code that is specific to each rig is small. It lives in [app/runtime/soma](../app/runtime/soma) and [app/runtime/unitree_g1](../app/runtime/unitree_g1). Each folder holds the rig's character, its target pose blender, and its presets.

An engine aligns the context and target for inference, then restores the world placement afterward. It also removes the three [boundary frames](#generation-length) at each end of the generated motion. A SOMA engine finds the model's 30 joints in a skeleton by name. The root must be at index zero. The skeleton can hold more joints, such as the SOMA fingers, which the model does not predict. The model uses the SOMA joint layout and proportions, so a character with another skeleton needs a retargeting step. The G1 engine works directly on the robot's joint coordinates, so you can sample straight into one robot's slice of a Newton `model.joint_q`.

Both engines derive from `MotionBricksEngine` in [motionbricks/model/engine.py](../motionbricks/model/engine.py). It holds the constructor and the generation flow. Each rig engine supplies only the steps that depend on its pose layout: checking the buffers, encoding the context and target, decoding the prediction, and endpoint correction.

The apps create their engines with `create_engine()` in [app/utils/model_loading.py](../app/utils/model_loading.py). It selects the bundled FP32 or mixed FP16 model and reports loading progress in the viewer. The apps keep one engine at a time. Switching to the other rig closes the current engine before the new one loads, and the TensorRT plan cache keeps NVIDIA reloads short.

### TensorRT builds

On NVIDIA systems, the first launch of each model, precision, and batch size builds a TensorRT plan and caches it in `.trt-cache/`. `TensorRTBuildOptions` controls the build. On macOS, only `batch_size` applies:

| Option | Default | Meaning |
| --- | --- | --- |
| `enable_fp16` | `False` | Lets TensorRT use FP16. The apps enable it with the mixed FP16 model. |
| `enable_tf32` | `None` | `None` uses the model's default. SOMA permits TF32. G1 disables it, because TF32 changes its motion for little speedup at a batch size of one, as its [model card](unitree_g1_model.md#inference) explains. An explicit value always wins. |
| `batch_size` | 1 | Builds a TensorRT plan for a fixed batch size |

Pass `force_rebuild=True` to an engine to ignore its cached plan. Each option set has its own cache entry. The mixed FP16 models keep FP32 for normalization, motion geometry, and other sensitive operations. Their inputs and outputs stay FP32. Mixed FP16 and FP32 can choose different motion tokens. Differences can grow when generated motion is fed back as context.

For other batch sizes and inference libraries, see [ONNX inference](onnx_inference.md).

### Logging

The library uses Python's standard `logging` module, with one logger per module. TensorRT messages go to the `motionbricks.inference.tensorrt` logger. Configure logging in your application as usual:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("motionbricks").setLevel(logging.DEBUG)
```

The apps log info messages by default. Add `--verbose` to any app command for detailed log messages.

## Pose layouts and model space

The apps work in world space, which is Z up. SOMA model space and the BVH clips are Y up. G1 model space and the CSV clips are already Z up.

`model_to_runtime(rig)` in [app/runtime/common/rig.py](../app/runtime/common/rig.py) converts each rig's space to world space. For SOMA, it turns Y up into Z up. It follows the MuJoCo convention, so SOMA faces the same way as G1. For G1, it is the identity.

The apps pass this transform to the engine and to clip sampling, so generated motion and clips land in the same space. Both engines default to the identity, which suits an app that already works in the model space.

### Place poses in the scene

Your app places every context and target pose in world space. To place a pose, move its root by one world transform and keep the other joint values. The root sets where the pose stands and which way it faces. For SOMA, the root is the first local joint transform. For G1, it is the first seven joint coordinates: the root's XYZ position and XYZW quaternion.

When you sample a clip, pass the placement as `root_offset`. A clip in source coordinates also needs `model_to_runtime`, applied first:

```python
clip.sample(sample_time, pose, root_offset=wp.mul(placement, model_to_runtime), stream=engine.stream)
```

Pass `root_offset` and `stream` by keyword. For a pose that you already have, multiply its root by the placement. `place_target_poses()` and `place_target_dof()` in [target_pose.py](../motionbricks/runtime/target_pose.py) do this for each pose layout.

The apps place poses in four places:

| Where | What it places | Code | Read more |
| --- | --- | --- | --- |
| Inbetweener | Start and each target, from the gizmo and the **Transform** fields | `sample_pose_window()` in [inbetweener_demo.py](../app/inbetweener_demo.py) | [Place a pose in the scene](inbetweener.md#place-a-pose-in-the-scene) |
| Locomotion | Blended target poses, ahead of the character and turned to its facing | `place_target_poses()` and `place_target_dof()` | [Blend and place the target](scene_preview.md#blend-and-place-the-target) |
| Smart Primitives | Authored target poses, through the instance's placement and node hierarchy | `get_world_target_poses()` in [smart_primitive.py](../motionbricks/runtime/smart_primitive.py) | [How a scene places the poses](smart_primitive_editor.md#how-a-scene-places-the-poses) |
| Clips | Idle and interaction clips, ahead of the character or at its current root | `generate_clip_transition()` and `start_clip()` in [character.py](../app/runtime/common/character.py) | [Clips and generated motion](#clips-and-generated-motion) |

Because only the root moves, one set of sampled poses serves every placement. This is why the guided tutorial can move the Entry and Duck-under targets with the gizmo without changing their body poses.

### Clips

[ClipRegistry](../app/runtime/common/clip_registry.py) loads and caches clips for one rig. It reads BVH clips for SOMA and CSV clips for Unitree G1:

```python
from app.runtime.common.rig import load_rig_assets
from motionbricks.model.contract import ModelRig

assets = load_rig_assets(ModelRig.SOMA)
clip = assets.clip_registry.load("assets/motions/soma/locomotion/idle/idle_loop_001__A268.bvh").animation
```

`load_rig_assets()` returns a `RigAssets` with the rig's skeletal mesh or Newton model, its clip registry, a reference pose that sets the buffer layout, and the `model_to_runtime` transform. Clips are only one source of poses. Any pose with the right layout works as a context or a target.

G1 CSV clips store the root in centimeters and angles in degrees at 120 Hz. The loader converts them to meters, XYZW quaternions, and radians.

## Generate

`generate_animation()` reads the context and target tokens and fills the output buffer with the generated motion and its foot contacts. Two kinds of settings tune it: settings for the whole generation, and settings for each target.

### Generation settings

`generate_animation()` accepts an optional `MotionBricksInferenceConfig`:

| Field | Default | Meaning |
| --- | --- | --- |
| `allowed_token_counts` | 8 to 14 | Lengths the model may choose, from 6 to 16 tokens of four poses |
| `correct_generated_diff` | `True` | Spreads small corrections over the motion so it starts at the last context pose and ends at the target |
| `heading_alignment` | (1, 0) | The ground direction in model space that inference turns the last context pose to face: (x, z) for SOMA and (x, y) for G1. (1, 0) faces +X. The [SOMA](soma_model.md) and [Unitree G1](unitree_g1_model.md) model cards describe each model space. |

### Settings for each target

Each target also carries its own settings in `TargetPoseInferenceParams`, from [target_pose.py](../motionbricks/runtime/target_pose.py). They are the settings that you edit for each target in the Inbetweener and the Smart Primitive Editor. The motion players read them in `generate()`.

| Setting | Field | Effect |
| --- | --- | --- |
| **Poses to Use** | `num_poses_to_use` | How many of the target's four poses inform the model, from 1 to 4. One pose sets the destination, with the posture and facing at which the character stops. A second pose adds velocity. A third and fourth pose add acceleration and momentum. |
| **Transition Early By** | `transition_early_by_frames` | Moves the transition point this many frames before the end of the generated motion. A larger value makes this target a guide. The character lines up toward it, then moves on to a more precise next target before arriving. |
| **Tokens to Generate** | `allowed_token_counts` | Replaces the allowed token counts of the generation settings for this target. `None` keeps them. A single count gives the target exact control over how many frames to generate, as [Generation length](#generation-length) explains. |
| **IK Pins** | `ik_pin_mask` | SOMA only. As playback of the generated motion nears the target, `TargetPoseIKCorrector` eases the pinned hands and feet onto it with two-bone IK. See [IK Pins](smart_primitive_editor.md#ik-pins). |

### Generation length

The model decides how long each transition takes. Each token holds four poses at 30 Hz. One generation produces 6 to 16 tokens, or 24 to 64 raw frames. The first token holds the context. The last token holds the target. The engine removes three boundary frames at each end. The generated motion then starts at the last context pose and ends at the first target pose. That leaves 18 to 58 valid frames, or about 0.6 to 1.9 s of motion. `output.num_frames` holds the valid frame count of the latest generation.

### Foot contacts

Both models predict foot contacts with the motion. Each frame holds four values. Each value is a confidence from 0 to 1 that a heel or a toe is planted on the ground. The engines clamp the raw model values to this range. The columns are:

| Column | Contact | SOMA joint | G1 point |
| --- | --- | --- | --- |
| 0 | Left heel | `LeftFoot` | The left ankle roll joint |
| 1 | Left toe | `LeftToeBase` | 0.14 m ahead of the left ankle roll joint and 0.035 m below it |
| 2 | Right heel | `RightFoot` | The right ankle roll joint |
| 3 | Right toe | `RightToeBase` | 0.14 m ahead of the right ankle roll joint and 0.035 m below it |

The G1 toe points are in the local space of each ankle roll link. `SOMA_N1_FOOT_CONTACT_JOINTS` in [soma/contract.py](../motionbricks/model/soma/contract.py) lists the SOMA joints. `UNITREE_G1_N1_FOOT_CONTACT_BODIES` and `UNITREE_G1_N1_FOOT_CONTACT_OFFSETS` in [unitree_g1/contract.py](../motionbricks/model/unitree_g1/contract.py) list the G1 bodies and points.

Generated motion keeps the foot contacts in `output.foot_contacts`. It is an FP32 array of shape `[capacity, 4]`. Its first `output.num_frames` frames are valid. `generate_animation()` requires this array. `create_output()` allocates it. Clips loaded from files have no foot contacts. Their `foot_contacts` is `None`.

`sample_foot_contacts()` blends the four values at a playback time, the way `sample()` blends poses. The **Show Foot Contacts** option in the Inbetweener and Scene Preview draws the sampled values.

`skeletal_mesh_sole_outlines()` and `body_sole_outlines()` in [foot_contact_renderer.py](../motionbricks/renderers/foot_contact_renderer.py) turn these points into heel and toe outlines. `SkeletonFootContactRenderer` draws the outlines on a SOMA pose, and `DOFFootContactRenderer` on a G1 pose. The apps create each rig's mesh, skeleton, and foot contact renderers with `create_mesh_renderer()`, `create_skeleton_renderer()`, and `create_foot_contact_renderer()` in [rig.py](../app/runtime/common/rig.py).

## Chaining generations

A single generation is short. Continuous motion comes from chaining generations, as the [guided tutorial](guided_tutorial.md#chaining-generations) explains. Each new context must hold the poses that were actually shown. Otherwise the next generation starts from another pose, and the character jumps. The Inbetweener chains by hand, which shows each call.

Two pieces in [motionbricks/runtime](../motionbricks/runtime) implement this pattern for real-time use. `MotionRecorder` keeps the poses that were shown. The motion players generate from those poses and play the result. Scene Preview [builds every state](scene_preview.md#generate-from-recorded-motion) on them. Read them as a reference, then rebuild the same pattern in your own application.

### The Inbetweener's approach

The Inbetweener [chains generations](inbetweener.md#playback-and-chained-generation) without a player. At each transition point, it copies the last four played frames of the generated motion into the context. This works because each segment plays to its transition point, and nothing changes the pose on the way.

### Record what you show

[MotionRecorder](../motionbricks/runtime/motion_recorder.py) keeps a fixed number of recent poses, oldest first. It also keeps the time from each pose to the next. Record one pose at each update of your app. The update rate does not need to match the model's 30 Hz.

```python
from motionbricks.runtime.motion_recorder import MotionRecorder

history = MotionRecorder(skeleton.reference_local_transforms, capacity=32, stream=engine.stream)
history.fill(start_pose, stream=engine.stream)

# At each update, after every change to the pose:
history.record(delta_time, current_pose, stream=engine.stream)

# Before each generation:
context.update_from_recorder(history, stream=engine.stream)
```

For G1, pass the robot's reference joint coordinates in place of the skeleton's reference local transforms. `fill()` sets every slot to one pose, as if the character had held it. Call it before the first generation.

`update_from_recorder()` samples four context poses at 30 Hz, ending at the latest recorded pose. It blends the two recorded poses around each sample time. The four poses span 0.1 s. The default 32 poses cover that span at update rates up to about 300 Hz.

Record the pose after every change that your app makes to it. Clip frames, target pose IK, and physics all count. The recorder does not track where a pose came from. That is what lets a generation start from a clip, or from the middle of a segment.

### Play generated motion

[motion_player.py](../motionbricks/runtime/motion_player.py) pairs an engine with a recorder. `GeneratedMotionPlayer` serves SOMA, and `GeneratedDOFMotionPlayer` serves G1. Each player creates its own context token, target token, output buffer, and recorder.

| Member | Use |
| --- | --- |
| `seed_history_from_pose(pose)` | Fills the history and the context with one pose. Call it before the first generation. |
| `generate(target_poses, inference_params)` | Reads the context poses from the history, generates toward the target, and restarts playback. An optional `config` holds the other generation settings. |
| `sample(delta_time, pose, foot_contacts)` | Advances the playback time, then writes the pose and its foot contacts. |
| `record_pose(delta_time, pose)` | Records the pose that you showed. |
| `time_to_transition` | Seconds left until the transition point. |

The player does not decide when to generate. Your code calls `generate()`, usually when `time_to_transition` drops below the time of the next update. Sampling can continue past the transition point. At the end of the valid frames, the pose holds.

### A continuous locomotion loop

This sketch drives a SOMA character with a player. `input_changed()` and `place_target()` stand for your own code.

```python
import warp as wp

from motionbricks.runtime.motion_player import GeneratedMotionPlayer
from motionbricks.runtime.target_pose import TargetPoseInferenceParams

player = GeneratedMotionPlayer(engine, skeleton)
player.seed_history_from_pose(start_pose)
current_pose = wp.empty_like(start_pose)
foot_contacts = wp.zeros(4, dtype=wp.float32, device=engine.device)
target_poses = wp.empty((4, skeleton.num_joints), dtype=wp.transform, device=engine.device)
settings = TargetPoseInferenceParams(transition_early_by_frames=10, num_poses_to_use=4)


def update(delta_time, desired_velocity):
    if player.time_to_transition <= delta_time or input_changed(desired_velocity):
        place_target(target_poses, desired_velocity)
        player.generate(target_poses, settings)
    player.sample(delta_time, current_pose, foot_contacts)
    # Apply target pose IK, physics, or other changes to current_pose here.
    player.record_pose(delta_time, current_pose)
```

Each call to `update()` runs one pass of the [generation flow](#generation-flow). [locomotion_loop.py](examples/locomotion_loop.py) runs this loop with `input_changed()` and `place_target()` filled in. Your app makes two decisions in it:

- **When to generate.** Generate when playback reaches the transition point at the latest. Regenerate earlier when the movement input changes enough, so the character responds quickly. Scene Preview's rules are in [When locomotion generates](scene_preview.md#when-locomotion-generates).
- **Where to place the target.** Sample target poses that match the desired speed. Place them ahead of the character along the desired velocity, and move only the root. Use all four target poses, so the model also matches the velocity at the target. Scene Preview blends its target poses in a blend space, as [Blend and place the target](scene_preview.md#blend-and-place-the-target) shows.

### Clips and generated motion

Clips and generated motion mix freely, because the recorder keeps every pose that was shown. `BaseCharacter` in [character.py](../app/runtime/common/character.py) uses this in both directions:

- **Out of a clip.** `start_clip()` plays a clip, and `step()` keeps recording its frames. The next generation starts from the clip's motion, at any frame.
- **Into a clip.** `generate_clip_transition()` places the clip ahead of the character. It samples the clip's first four poses as the target and generates toward them with a **Transition Early By** of zero. When the clip transition ends, the state starts the clip.

The idle state uses both. It generates into a looping idle clip, and locomotion later generates out of it.

### Choose an approach

| | Inbetweener | Motion player |
| --- | --- | --- |
| Context poses | The last four played frames of the generated motion | Four poses sampled at 30 Hz from the recorded history |
| When to generate | At each transition point | At any update |
| Poses from clips, IK, or physics | Never reach the context | Reach the context like any other shown pose |

Start with the Inbetweener to see each call. Move to a recorder as soon as your app regenerates early, plays clips, or changes the pose after sampling.

## Further reference

- [Inbetweener](inbetweener.md): one integration of this API, step by step.
- [Scene Preview](scene_preview.md): a character runtime built on the motion players, with states, locomotion, and behaviors.
- [Guided tutorial](guided_tutorial.md): the user-facing workflow that uses this API through the apps.
- [ONNX inference](onnx_inference.md): direct ONNX interfaces, batching, and other inference libraries.
- [SOMA model card](soma_model.md): SOMA inputs, outputs, coordinates, and limitations.
- [Unitree G1 model card](unitree_g1_model.md): G1 joint layout, inputs, outputs, and limitations.
