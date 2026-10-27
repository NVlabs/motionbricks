# MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives

<p align="center">
  <a href="https://nvlabs.github.io/motionbricks/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2604.24833"><img src="https://img.shields.io/badge/arXiv-2604.24833-b31b1b.svg" alt="Paper"></a>
  <a href="#checkpoints"><img src="https://img.shields.io/badge/Hugging%20Face-Models-yellow" alt="Hugging Face Models"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg" alt="License"></a>
  <img src="docs/media/teaser.jpg" alt="MotionBricks animates characters and robots with one model" width="100%">
</p>

MotionBricks is a modular latent generative model that animates characters and robots in real time. It fills in the motion between four recent poses and up to four target poses. Three learned modules share the work, as [How MotionBricks works](#how-motionbricks-works) explains. Smart Primitives turn target poses into building blocks for navigation and object interaction, so behaviors like vaulting and sitting snap together like bricks, with no retraining.

This repository provides pretrained MotionBricks models, based on the [SIGGRAPH 2026 paper](https://arxiv.org/abs/2604.24833), and everything you need to put them to work:

- **Pretrained models** for the [SOMA body model](https://github.com/NVlabs/SOMA-X) and the Unitree G1 robot. They ship as ONNX files, with the pretrained [checkpoints](#checkpoints) on Hugging Face.
- **A Python runtime**, built on [Warp](https://github.com/NVIDIA/warp) and [Newton](https://github.com/newton-physics/newton), that chains generations into continuous motion using TensorRT, ONNX Runtime, or PyTorch MPS for inference.
- **Two demos**: an [Inbetweener](#inbetweener) that generates motion between poses you place, and a [Scene Preview](#scene-preview) where a character walks, runs, vaults, ducks, and sits its way through a scene.
- **Two editors** to author your own [Smart Primitives](#editors) and [scenes](#editors).

The demos show one way to drive the model. Your application decides where the targets come from. New to MotionBricks? The [guided tutorial](docs/guided_tutorial.md) walks through all four apps as it builds one interaction scene.

## News

- MotionBricks has its own repository and a pretrained model for SOMA humans. The first release was a Unitree G1 preview in [GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks).

## Setup

On Windows and Linux, inference uses TensorRT and requires an NVIDIA GPU. On Apple Silicon macOS, inference runs on the CPU by default. Once the optional Metal dependencies and checkpoints are installed, the FP32 models use the Apple GPU automatically, as described below. Warp runs animation kernels on the CPU on macOS, and the demos use the Newton OpenGL viewer.

Install [Git LFS](https://git-lfs.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/). Then clone this repository and run these commands from its root:

```sh
git lfs pull
uv sync
```

On NVIDIA systems, the first launch of each model and precision builds a TensorRT plan and caches it in `.trt-cache/`. On Apple Silicon, the experimental Core ML model load may take several minutes. Core ML runs supported portions on the CPU or GPU, and ONNX Runtime executes unsupported operations on the CPU. GPU use and real-time performance depend on the model and system.

Set `MOTIONBRICKS_ONNX_PROVIDER=coreml` to try Core ML. This path is experimental: Core ML or its Metal backend may fail on the bundled model. Core ML uses `CPUAndGPU` by default. Set `MOTIONBRICKS_COREML_COMPUTE_UNITS` to `ALL`, `CPUOnly`, or `CPUAndNeuralEngine` to choose other Core ML compute units. The app renderer uses OpenGL, not Metal, and Warp does not currently offer a Metal backend. Each app section below gives its command.

### Apple GPU inference

Install the optional Metal dependencies and download the released checkpoints using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli). `uvx` runs the CLI without a separate installation:

```sh
uv sync --extra metal
uvx hf download nvidia/MotionBricks --local-dir checkpoints/MotionBricks --include "g1/*" --include "soma/*"
uv run --extra metal python -m app.inbetweener_demo
```

The app selects Metal for bundled FP32 models when the GPU dependencies and complete checkpoints are available; otherwise it uses the CPU. Set `MOTIONBRICKS_ONNX_PROVIDER=cpu` or `mps` to choose explicitly. The GPU backend verifies the checkpoint checksums before loading. To use a different checkpoint location, set `MOTIONBRICKS_CHECKPOINT_DIR` to the downloaded `MotionBricks` folder. It supports the bundled FP32 SOMA and Unitree G1 models with batch size one; `--model-precision fp16` is not supported. The first inference can take longer while Metal compiles kernels. Model inputs and outputs move between CPU Warp buffers and the GPU, so performance depends on the machine and workload. The checkpoints are subject to the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). See [Apple Silicon inference](docs/apple_silicon.md) for validation and current limits.

Include `--extra metal` in every `uv run` command that should use the GPU, including the other app commands below. Running `uv run` without the extra may remove the optional dependencies from the project environment.

The CPU ONNX provider uses up to four threads by default on Mac. Set `MOTIONBRICKS_ONNX_THREADS` to a positive integer to tune it. Run `uv run --extra metal python tools/benchmark_inference.py` with the provider environment variable set to `cpu` or `mps` to compare complete generation times on your machine.

## How MotionBricks works

<p align="center">
  <img src="docs/media/core_idea.gif" alt="Context poses in green, a strobe of generated poses, and a target pose in blue" width="100%">
</p>

Each generation fills in the motion between two sets of poses at 30 Hz:

- The **context** is the four most recent poses of the character (green above).
- The **target** is one to four poses that the motion should reach (blue above). One pose sets where the character ends up and stops. More poses also set how it moves at that point.

The model decides how long the transition takes. One generation produces 6 to 16 tokens of four poses, or about 0.6 to 1.9 s of motion. [Generation length](docs/runtime_api.md#generation-length) explains the frame counts.

Three learned modules share the work:

- The **root module** predicts the duration and the path of the root.
- The **pose module** is a masked token transformer. It predicts the motion tokens along that path.
- The **decoder** turns the tokens into continuous motion that reaches the targets. It uses a VQ-VAE with several codebook heads.

Longer motion comes from chaining generations. The runtime plays the generated motion, records what it played, and uses the last four recorded poses as the context of the next generation. Your application decides the targets. That choice is what turns one inbetweening model into keyframe animation, continuous locomotion, or object interactions. The [paper](https://arxiv.org/abs/2604.24833) covers the method, the training data, and the evaluation.

### One runtime for SOMA and G1

Both models generate the same way. Only the pose layout differs. The apps and the runtime work the same for both rigs, so one code path drives a human skeleton and a robot. [Engines](docs/runtime_api.md#engines) describes the two layouts.

## Repository layout

The `motionbricks` package is the library. The demos and editors in `app` are built on it.

| Folder | Contents |
| --- | --- |
| [motionbricks](motionbricks) | The library: engines, motion players, recorder, blend spaces, pivots, and the ONNX models |
| [app](app) | The demos, the two editors, and the character runtime they share |
| [assets](assets) | Sample motion clips, Smart Primitives, and the gym scenes |
| [docs](docs) | Guides and model cards, with runnable examples in [docs/examples](docs/examples) |
| [tools/onnx](tools/onnx) | The standalone ONNX exporter |

## Inbetweener

Start here to see the core idea. Choose poses from clips, place them in the view, and generate motion through them in order.

<table>
  <tr>
    <th>SOMA</th>
    <th>Unitree G1</th>
  </tr>
  <tr>
    <td><img src="docs/media/inbetweener_soma.gif" alt="SOMA Inbetweener" width="100%"></td>
    <td><img src="docs/media/inbetweener_g1.gif" alt="Unitree G1 Inbetweener" width="100%"></td>
  </tr>
</table>

Run it with this command:

```sh
uv run python -m app.inbetweener_demo
```

[Inbetweener](docs/inbetweener.md) lists its options and things to try. It also walks through the code.

## Scene Preview

Drive a character through a scene with continuous locomotion and interactions. It can walk and run forward, strafe while facing the camera, or strafe while crouching. Six style modes, such as Skipping and Zombie, change how it walks. Scene Preview distills the Unreal Engine demo behind the [paper's videos](https://nvlabs.github.io/motionbricks/) into a small Python runtime. It puts the ideas front and center, so you can carry them into any integration.

<table>
  <tr>
    <th>SOMA</th>
    <th>Unitree G1</th>
  </tr>
  <tr>
    <td><img src="docs/media/scene_preview_soma.gif" alt="SOMA Scene Preview" width="100%"></td>
    <td><img src="docs/media/scene_preview_g1.gif" alt="Unitree G1 Scene Preview" width="100%"></td>
  </tr>
  <tr>
    <td><img src="docs/media/locomotion_targets_soma.gif" alt="SOMA walks, runs, and stops with context, target, and blend space poses shown" width="100%"></td>
    <td><img src="docs/media/locomotion_targets_g1.gif" alt="Unitree G1 walks, runs, and stops with context, target, and blend space poses shown" width="100%"></td>
  </tr>
  <tr>
    <td><img src="docs/media/locomotion_directional_soma.gif" alt="SOMA strafes in four directions, crouches, then stands back up, with context, target, and blend space poses shown" width="100%"></td>
    <td><img src="docs/media/locomotion_directional_g1.gif" alt="Unitree G1 strafes in four directions, crouches, then stands back up, with context, target, and blend space poses shown" width="100%"></td>
  </tr>
  <tr>
    <td><img src="docs/media/locomotion_styles_soma.gif" alt="SOMA walks in each locomotion style, from Skipping to Combat" width="100%"></td>
    <td><img src="docs/media/locomotion_styles_g1.gif" alt="Unitree G1 walks in each locomotion style, from Skipping to Combat" width="100%"></td>
  </tr>
</table>

Run it with the SOMA or the Unitree G1 gym:

```sh
uv run python -m app.scene_preview --scene assets/scenes/soma_gym.scene.json
uv run python -m app.scene_preview --scene assets/scenes/g1_gym.scene.json
```

[Scene Preview](docs/scene_preview.md) lists its options and controls. It also explains the runtime behind it: the character, the state machine, locomotion, and interactions.

## Editors

A Smart Primitive describes an action as target poses placed around an object. Author one in the Smart Primitive Editor. Place it in a scene for Scene Preview with the Scene Editor.

<table>
  <tr>
    <th>Smart Primitive Editor</th>
    <th>Scene Editor</th>
  </tr>
  <tr>
    <td><img src="docs/media/smart_primitive_editor.png" alt="Smart Primitive Editor with a jump primitive" width="100%"></td>
    <td><img src="docs/media/scene_editor.png" alt="Scene Editor with the SOMA gym" width="100%"></td>
  </tr>
</table>

Run them with these commands:

```sh
uv run python -m app.smart_primitive_editor
uv run python -m app.scene_editor
```

Start with the [guided tutorial](docs/guided_tutorial.md#part-2-build-a-smart-primitive) to learn these tools. It builds an interaction step by step with both editors. [Smart Primitive Editor](docs/smart_primitive_editor.md) and [Scene Editor](docs/scene_editor.md) describe each editor and its documents.

These editors show one way to build dynamic sequences and interactions. Your application can follow the same approach or a different one. For example, these editors sample a handful of target poses from clips. A motion generation model, such as [Kimodo](https://github.com/nv-tlabs/kimodo) or [ARDY](https://github.com/nv-tlabs/ardy), could generate those poses instead.

## Checkpoints

MotionBricks checkpoints are available for the SOMA body model and the Unitree G1 robot. Their weights use the SafeTensors format.

> Note: the default inference backends use the ONNX models bundled in [motionbricks/assets/onnx](motionbricks/assets/onnx), so they do not require checkpoints. Download checkpoints for the optional Metal backend, to export your own ONNX models, or to build another runtime. This repository includes the [ONNX exporter](#export-onnx-models).

| Model | Skeleton | Training Data | FPS | Hugging Face | License |
|:-------|:-------------|:------:|:------:|:-------------:|:-------------:|
| **MotionBricks-SOMA-RP-v1** | [SOMA](https://github.com/NVlabs/SOMA-X) | [Bones RigPlay 1 (RP1)](https://bones.studio/datasets#rp01) | 30 | [Link](https://huggingface.co/nvidia/MotionBricks/tree/main/soma) | [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |
| **MotionBricks-G1-RP-v1** | [Unitree G1](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description) | [Bones RigPlay 1 (RP1)](https://bones.studio/datasets#rp01), [retargeted](https://github.com/NVIDIA/soma-retargeter) | 30 | [Link](https://huggingface.co/nvidia/MotionBricks/tree/main/g1) | [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |

The [SOMA model card](docs/soma_model.md) and the [Unitree G1 model card](docs/unitree_g1_model.md) describe each model, its inputs and outputs, and its limitations.

## Export ONNX models

The ONNX models in this repository were exported from these checkpoints with the standalone exporter in [tools/onnx](tools/onnx/README.md). To export them yourself, install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli), then download the checkpoint folders and run the command for your model from the repository root:

```sh
uvx hf download nvidia/MotionBricks --local-dir checkpoints/MotionBricks

# SOMA
uv run --locked --project tools/onnx motionbricks-onnx --model soma --checkpoints checkpoints/MotionBricks/soma

# Unitree G1
uv run --locked --project tools/onnx motionbricks-onnx --model unitree_g1 --checkpoints checkpoints/MotionBricks/g1
```

To download one model, add `--include "soma/*"` or `--include "g1/*"`. Keep the downloaded folder structure intact.

The exporter has its own environment, and uv installs it the first time you run the command. FP32 is the default. Add `--precision fp16` for mixed FP16. The [exporter guide](tools/onnx/README.md) lists every option, and [ONNX inference](docs/onnx_inference.md) explains the tensor interfaces and batching.

## Documentation

New to MotionBricks? Start with the **[guided tutorial](docs/guided_tutorial.md)**. It explains the core concepts, then builds an interaction scene with all four apps.

**Apps**

- **[Inbetweener](docs/inbetweener.md):** Shows how to run the Inbetweener, then walks through its code, from sampling and placing poses to chained playback.
- **[Scene Preview](docs/scene_preview.md):** Lists the options and controls, then describes the example runtime, from the character and the state machine to locomotion and interactions.
- **[Smart Primitive Editor](docs/smart_primitive_editor.md):** Describes what a Smart Primitive holds and how to author one: the node hierarchy, target poses, interaction boxes, pivots, and sequences.
- **[Scene Editor](docs/scene_editor.md):** Shows how to build a scene, and describes the `.scene.json` document that Scene Preview opens.

**Integration**

- **[Runtime API](docs/runtime_api.md):** Start here to integrate MotionBricks in your application. Describes the engines, pose layouts, clips, and chaining generations. It includes two runnable scripts that generate motion without a viewer.
- **[ONNX inference](docs/onnx_inference.md):** Describes the tensor shapes, batching, and other inference libraries, for use without the bundled engines.

**Models**

- **[ONNX exporter](tools/onnx/README.md):** Describes how to export the bundled ONNX models from the checkpoints on Hugging Face.
- **[SOMA model card](docs/soma_model.md):** Describes the SOMA model, its inputs and outputs, its training data, and its limitations.
- **[Unitree G1 model card](docs/unitree_g1_model.md):** Describes the Unitree G1 model, its inputs and outputs, its training data, and its limitations.

## Related humanoid work at NVIDIA

MotionBricks is part of a larger effort to enable humanoid motion for animation, robotics, and physical AI. Check out these related works:

- [Kimodo](https://github.com/nv-tlabs/kimodo): human motion generation model specializing in controllable offline motion authoring.
- [ARDY](https://github.com/nv-tlabs/ardy): builds on Kimodo to enable controllable motion generation in real time for interactive applications.
- [SOMA Body Model](https://github.com/NVlabs/SOMA-X): a unified parametric human body model.
- [BONES-SEED Dataset](https://huggingface.co/datasets/bones-studio/seed): a large motion capture dataset of humans and humanoids in SOMA and G1 format.
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions): simulation and learning framework for training physically simulated humans and humanoids.
- [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter): retargets SOMA motion to humanoid robots such as the Unitree G1.
- [GEM](https://github.com/NVlabs/GEM-X): human motion reconstruction from video.
- [GEAR SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl): humanoid behavior foundation model for physical robots.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2026motionbricks,
  title   = {MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives},
  author  = {Wang, Tingwu and Dionne, Olivier and De Ruyter, Michael and Minor, David and Rempe, Davis and Zhao, Kaifeng and Petrovich, Mathis and Yuan, Ye and Li, Chenran and Luo, Zhengyi and Robison, Brian and Blackwell, Xavier and Antoniazzi, Bernardo and Peng, Xue Bin and Zhu, Yuke and Yuen, Simon},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2026},
  volume  = {45},
  number  = {4},
  doi     = {10.1145/3811334}
}
```

## License

Code, documentation, models, and sample motions have different licenses:

- **Code:** [Apache 2.0](LICENSE), including code examples and snippets in the documentation.
- **Documentation and media:** NVIDIA-authored documentation text and media (images, screenshots, and videos, including the files in `docs/media/`) use [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Models:** [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). This covers the SOMA and G1 checkpoints on Hugging Face and the [ONNX models](motionbricks/assets/onnx/LICENSE.md). It permits commercial use and redistribution of modified models, subject to the agreement's terms. Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement.
- **Sample motions:** [NVIDIA Sample Data Evaluation License](assets/motions/LICENSE.txt). The covered human BVH and retargeted G1 CSV clips are for evaluating and testing NVIDIA technologies. The [sample-data notice](assets/motions/README.md) lists the covered files and credits Bones. These sample terms are separate from the model terms.

Licenses for dependencies and bundled robot assets are in [licenses](licenses). Exporter dependency licenses are in [tools/onnx/licenses](tools/onnx/licenses/).

This project downloads and installs additional components under their own license terms. Review those terms before use.

The [TensorRT OSS notice](licenses/tensorrt-LICENSE.txt) applies to TensorRT OSS components, not the separately installed `tensorrt-cu12`, `tensorrt-cu12-bindings`, and `tensorrt-cu12-libs` runtime packages. Those packages are governed by the NVIDIA TensorRT Software License Agreement distributed with the packages.

## Acknowledgments

This project builds upon excellent open source projects:

- [Newton](https://github.com/newton-physics/newton) for the viewer and scene
- [NVIDIA Warp](https://github.com/NVIDIA/warp) for GPU kernels
- [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT) for inference
- [Dear ImGui](https://github.com/ocornut/imgui) through [imgui_bundle](https://github.com/pthom/imgui_bundle) for the user interface
- [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) for the SOMA character
- [Unitree Robotics](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description) for the Unitree G1 robot model and meshes

## Contact

This project is currently not accepting contributions.

For questions or issues, please open an issue on this repository or reach out directly to the authors.
