# Export MotionBricks checkpoints to ONNX

Export the SOMA and Unitree G1 checkpoints from Hugging Face to batched ONNX models in FP32 or
mixed FP16.

## Export

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
The checkpoints are folders in one
[MotionBricks repository](https://huggingface.co/nvidia/MotionBricks) on Hugging Face:

| Model | Checkpoint folder |
| --- | --- |
| SOMA | [soma/](https://huggingface.co/nvidia/MotionBricks/tree/main/soma) |
| Unitree G1 | [g1/](https://huggingface.co/nvidia/MotionBricks/tree/main/g1) |

Run these commands from the root of this repository. Keep the downloaded folder structure
intact. uv installs the exporter and its dependencies automatically on first use.

```sh
uvx hf download nvidia/MotionBricks --local-dir checkpoints/MotionBricks

# SOMA
uv run --locked --project tools/onnx motionbricks-onnx --model soma --checkpoints checkpoints/MotionBricks/soma

# Unitree G1
uv run --locked --project tools/onnx motionbricks-onnx --model unitree_g1 --checkpoints checkpoints/MotionBricks/g1
```

To download one model, add `--include "soma/*"` or `--include "g1/*"`. Download checkpoints only
for export or another runtime. The demo apps use the ONNX models bundled with the software.

FP32 is the default. The command saves `mmm_v6_soma_fp32.onnx` or `mmm_v6_g1_fp32.onnx` in the
current directory. Add `--precision fp16` for mixed FP16. The file name then ends in `_fp16.onnx`.

Use `--output path/to/model.onnx` to choose a different destination, or `--force` to replace an
existing file. The exporter stages its temporary files in the output folder. Allow several GB of
free disk space there.

Add `--verbose` for detailed log messages.

## Output and checks

The models support different batch sizes, and the demos use a batch size of one (`N=1`).
Floating point inputs and outputs use FP32 for both precisions. For shapes and usage examples, see
[ONNX inference](../../docs/onnx_inference.md), the [SOMA model card](../../docs/soma_model.md), and
the [G1 model card](../../docs/unitree_g1_model.md).

The exporter checks the checkpoint checksums, the exported ONNX graph, and its tensor interface.
Invalid checkpoints stop the export before any weights load. An existing output is replaced only
after the export succeeds.

Each checkpoint holds SafeTensors weights and normalization statistics in `.npy` files. The
exporter reads no other format.

## License

The source uses the repository's [Apache 2.0 license](../../LICENSE). Dependency licenses are in
[licenses/](licenses/).

The downloaded checkpoints are licensed under the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement.
