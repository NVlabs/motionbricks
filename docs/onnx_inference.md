# ONNX inference and batching

This page is for running the MotionBricks ONNX models without the bundled engines, such as in another inference library, or for many characters at once in one batch. The [Runtime API](runtime_api.md) engines cover a single character with TensorRT.

MotionBricks includes one ONNX file per model and precision. All four accept a variable number of independent requests, including a batch of one.

| Model | FP32 | Mixed FP16 | Interface |
| --- | --- | --- | --- |
| SOMA | [mmm_v6_soma_fp32.onnx](../motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx) | [mmm_v6_soma_fp16.onnx](../motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx) | [SOMA model card](soma_model.md) |
| Unitree G1 | [mmm_v6_g1_fp32.onnx](../motionbricks/assets/onnx/unitree_g1/mmm_v6_g1_fp32.onnx) | [mmm_v6_g1_fp16.onnx](../motionbricks/assets/onnx/unitree_g1/mmm_v6_g1_fp16.onnx) | [Unitree G1 model card](unitree_g1_model.md) |

Each folder also holds a model manifest, [soma_n1.json](../motionbricks/assets/onnx/soma/soma_n1.json) and [unitree_g1_n1.json](../motionbricks/assets/onnx/unitree_g1/unitree_g1_n1.json), with the joint names and the tensor specs for a batch of one.

The files use Git LFS. Run `git lfs pull` after cloning to download them. To recreate them from the checkpoints, use the [exporter](../tools/onnx/README.md). The models are licensed under the NVIDIA Open Model Agreement, as stated in [LICENSE.md](../motionbricks/assets/onnx/LICENSE.md).

## Tensors

`N` is the number of requests in the batch. The model cards explain what each tensor holds and how to prepare it.

| Model | Inputs | Outputs |
| --- | --- | --- |
| SOMA | <ul><li><code>global_root_values</code> [N, 8, 5]</li><li><code>local_root_values</code> [N, 8, 4]</li><li><code>local_poses</code> [N, 8, 267]</li><li><code>has_global_root_values</code> [N, 8]</li><li><code>has_local_root_values</code> [N, 8]</li><li><code>has_local_poses</code> [N, 8]</li><li><code>allowed_pred_num_tokens</code> [N, 11]</li></ul> | <ul><li><code>pred_global_motions</code> [N, 64, 366]</li><li><code>pred_num_tokens</code> [N]</li></ul> |
| Unitree G1 | <ul><li><code>context_qpos</code> [N, 4, 36]</li><li><code>target_qpos</code> [N, 4, 36]</li><li><code>has_global_root_values</code> [N, 8]</li><li><code>has_local_root_values</code> [N, 8]</li><li><code>has_local_poses</code> [N, 8]</li><li><code>allowed_pred_num_tokens</code> [N, 11]</li></ul> | <ul><li><code>pred_qpos</code> [N, 64, 36]</li><li><code>pred_foot_contacts</code> [N, 64, 4]</li><li><code>pred_num_tokens</code> [N]</li></ul> |

Floating point inputs and outputs use FP32 in both precisions. The masks use INT32, with 0 for disabled and 1 for enabled. `allowed_pred_num_tokens` holds one INT32 flag for each token count from 6 to 16. Each request must allow at least one token count.

## Prepare a batch

All inputs to one generation must use the same positive `N`. Their other dimensions are fixed. Each request has its own poses, masks, and allowed token counts. Requests are processed independently.

ONNX Runtime is included as a default MotionBricks dependency (with DirectML on Windows). It runs out of the box with `uv run python my_batch.py`.

With ONNX Runtime, prepare each request with a batch size of one. Then concatenate the requests along axis 0:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "motionbricks/assets/onnx/unitree_g1/mmm_v6_g1_fp32.onnx",
    providers=["CPUExecutionProvider"],
)

# requests contains complete input dictionaries, each with a batch size of one.
inputs = {
    name: np.concatenate([request[name] for request in requests], axis=0)
    for name in requests[0]
}
outputs = dict(zip(
    [output.name for output in session.get_outputs()],
    session.run(None, inputs),
))

# Read the raw frames for each request.
motions = [
    outputs["pred_qpos"][i, :4 * int(count)]
    for i, count in enumerate(outputs["pred_num_tokens"])
]
```

The ONNX interface places no fixed upper limit on `N`. Available memory and the inference library, such as ONNX Runtime or TensorRT, determine practical batch sizes.

## Read the outputs

Request `i` produces `4 * pred_num_tokens[i]` raw frames at 30 Hz, between 24 and 64. The rest of its output space is unused. The first token holds the context poses, and the last token holds the target poses.

The G1 model returns its foot contacts in `pred_foot_contacts`, for the same frames as `pred_qpos`. The SOMA model has no separate foot contact output. Its `pred_global_motions` holds the motion and the foot contacts together, and the SOMA engine decodes both.

The ONNX output includes the [boundary frames](runtime_api.md#generation-length) and has no endpoint correction. The SOMA and G1 engines remove the boundary frames and apply endpoint correction. Their motion therefore starts exactly at the last context pose and ends at the first target pose. In a batch, trim, correct, and place each request separately.

## TensorRT and the demos

The demos build TensorRT plans with a batch size of one. They reuse their device buffers and CUDA graph between generations.

The generic `InferenceEngine` accepts `TensorRTBuildOptions(batch_size=N)` to build a TensorRT plan for another fixed batch size. The plan cache stores a separate plan for each size. The SOMA and G1 engine classes check their plan against a batch size of one. With a larger batch, run `InferenceEngine` directly. Your application then prepares the inputs and reads the outputs of each request.

To change the batch size between generations with one TensorRT plan, create optimization profiles and manage the tensor shapes and buffers in your application.

## Inference performance

The table shows the GPU time of the TensorRT plan alone for one batch, with the requests per second in parentheses. Input preparation, output decoding, and endpoint correction are not included.

| Model | N = 1 | N = 8 | N = 16 | N = 32 |
| --- | --- | --- | --- | --- |
| SOMA, FP32 | 2.50 ms (400) | 2.96 ms (2,706) | 3.74 ms (4,280) | 5.28 ms (6,063) |
| SOMA, mixed FP16 | 1.28 ms (778) | 1.54 ms (5,179) | 1.87 ms (8,535) | 3.93 ms (8,138) |
| Unitree G1, FP32 | 2.67 ms (375) | 3.78 ms (2,116) | 5.17 ms (3,096) | 7.81 ms (4,097) |
| Unitree G1, mixed FP16 | 1.63 ms (613) | 1.91 ms (4,194) | 2.36 ms (6,783) | 3.38 ms (9,449) |

The plans were timed with `trtexec` from TensorRT 10.8.0.43, the version the runtime uses, on an NVIDIA GeForce RTX 4090 under Windows. The inputs were already on the GPU, and a CUDA graph replayed each inference, as in the demos. Each batch repeats one real request. The mixed FP16 runs enable FP16 kernels like the demos. Each value comes from the fastest of six plan builds, and every timed plan matched ONNX Runtime on the same request. Timings vary from one build to the next, most for the mixed FP16 models.

The G1 runs disable TF32 like the runtime. Their FP32 convolutions and matrix multiplies then run without Tensor Cores, so G1 FP32 slows down more as the batch grows. For comparison:

| Unitree G1, FP32 | N = 16 | N = 32 |
| --- | --- | --- |
| TF32 disabled, as in the runtime | 5.17 ms (3,096) | 7.81 ms (4,097) |
| TF32 allowed | 4.06 ms (3,936) | 5.78 ms (5,537) |

The [Inference](unitree_g1_model.md#inference) section of the Unitree G1 model card explains why TF32 stays disabled.

To measure your own GPU, run `trtexec` with the batch size in `--shapes`. These commands time the FP32 models with a batch of 8.

SOMA:

```sh
trtexec --onnx=motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx --shapes=global_root_values:8x8x5,has_global_root_values:8x8,local_root_values:8x8x4,has_local_root_values:8x8,local_poses:8x8x267,has_local_poses:8x8,allowed_pred_num_tokens:8x11 --useCudaGraph --noDataTransfers
```

Unitree G1, with TF32 disabled like the runtime:

```sh
trtexec --onnx=motionbricks/assets/onnx/unitree_g1/mmm_v6_g1_fp32.onnx --shapes=context_qpos:8x4x36,target_qpos:8x4x36,has_global_root_values:8x8,has_local_root_values:8x8,has_local_poses:8x8,allowed_pred_num_tokens:8x11 --useCudaGraph --noDataTransfers --noTF32
```

For the mixed FP16 models, use the `_fp16.onnx` file and add `--fp16`. `trtexec` fills the inputs with random values unless `--loadInputs` gives it real requests. `trtexec` comes with the TensorRT packages on the [TensorRT download page](https://developer.nvidia.com/tensorrt/download).

## Further reference

- [SOMA model card](soma_model.md) and [Unitree G1 model card](unitree_g1_model.md): what each tensor holds, the joint order, and the model spaces.
- [Runtime API](runtime_api.md): the engines that prepare these tensors for a batch size of one.
- [ONNX exporter](../tools/onnx/README.md): recreating the ONNX files from the checkpoints.
