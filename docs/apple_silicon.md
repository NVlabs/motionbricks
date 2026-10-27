# Apple Silicon inference

The optional `mps` provider runs the released FP32 SOMA and G1 predictors and decoder with PyTorch on the Apple GPU. Warp's macOS backend still runs the surrounding pose, animation, and picking kernels on the CPU. The viewer uses OpenGL. A context on the tested M3 Max reported the Apple GPU as its OpenGL renderer, but this is not a native Metal rendering backend. Inputs and outputs cross between CPU Warp buffers and GPU tensors at each generation.

Follow [Apple GPU inference](../README.md#apple-gpu-inference) to install the `metal` extra and download the released checkpoints. The backend is selected automatically for bundled FP32 models when the optional dependencies, Apple GPU, and complete checkpoint files are available. Set `MOTIONBRICKS_ONNX_PROVIDER=mps` to require it, or `MOTIONBRICKS_ONNX_PROVIDER=cpu` to use the CPU explicitly. The backend checks both the bundled ONNX model fingerprint and the published checkpoint checksums, so a different file at the same path is not silently ignored. A checksum failure stops loading; use the CPU provider explicitly for a modified ONNX model. Set `MOTIONBRICKS_CHECKPOINT_DIR` if the checkpoint directory is elsewhere.

The GPU backend accepts batch size one and the bundled FP32 models. It does not yet support the bundled mixed-FP16 variants. A PyTorch autocast experiment produced substantially different G1 pose and foot-contact values from the repository's mixed-FP16 ONNX graph, which uses a different precision policy. In one test, using the ONNX root trajectory as input to the Metal pose predictor reduced the divergence but still produced different pose tokens. Both predictor paths therefore need further work. The implementation rejects the FP16 paths rather than silently changing model behavior. A separate ONNX2PyTorch 0.6.0 experiment failed on a batched matrix multiply; after a local converter fix, execution hit mixed CPU and MPS tensor operands. That converter is not a project dependency.

FP32 results are not guaranteed to be identical across CPU and Metal. The model selects discrete pose tokens, so small numerical differences can change a token when two scores are nearly tied. In one SOMA test with a single moved target and 16 tokens, the CPU's top scores differed by about 0.000003 and Metal chose the other token. The largest joint rotation difference in the generated motion was about 5 degrees; foot-contact values differed by up to 0.026, without crossing the 0.5 threshold in that sample.

## Check inference and performance

From the repository root, after installing the extra and checkpoints:

```sh
MOTIONBRICKS_ENGINE_TESTS=1 MOTIONBRICKS_ONNX_PROVIDER=mps uv run --extra metal python -m unittest tests.motionbricks.test_real_engines
MOTIONBRICKS_MPS_PARITY_TESTS=1 uv run --extra metal python -m unittest tests.motionbricks.test_real_mps_parity
MOTIONBRICKS_MPS_APP_TESTS=1 uv run --extra metal python -m unittest tests.app.test_real_mps_loader
MOTIONBRICKS_MPS_VIEWER_TESTS=1 uv run --extra metal python -m unittest tests.app.test_real_mps_viewer
MOTIONBRICKS_ONNX_PROVIDER=mps uv run --extra metal python tools/benchmark_inference.py --model unitree_g1
MOTIONBRICKS_ONNX_PROVIDER=cpu uv run --extra metal python tools/benchmark_inference.py --model unitree_g1
```

The real-engine tests check chained walking, allowed token counts, and motion continuity toward a single moved and rotated target. The parity tests compare FP32 Metal and CPU outputs for both rigs, several walking windows, a moved and rotated four-pose target, and three forced lengths. The loader test creates a G1 engine on the app's worker thread and runs inference on the calling thread. The viewer tests open windows in separate processes and programmatically generate, play, and render motion in the Inbetweener and Scene Preview while switching rigs; they do not test clicking the controls. Benchmark each provider separately on an otherwise idle machine. First inference includes GPU kernel compilation; the benchmark reports warmed timings and includes encoding, inference, decoding, and endpoint correction. Results vary with hardware, load, and CPU/GPU transfers.

For one example on an M3 Max, five warmed SOMA generations had median times of 176.8 ms on CPU, 159.4 ms on Metal, and 109.8 ms on Metal with experimental compilation. These are small samples, not a performance guarantee. In that sample, CPU-side encoding, decoding, and endpoint correction together accounted for about 1 ms per generation.

The CPU ONNX provider uses up to four threads by default on Mac. Set `MOTIONBRICKS_ONNX_THREADS` to a positive integer to tune it. The experimental Core ML provider remains selectable with `MOTIONBRICKS_ONNX_PROVIDER=coreml`; it has not yet produced a verified result for these models on the tested system.

For long-running sessions, `MOTIONBRICKS_MPS_COMPILE=1` enables experimental PyTorch compilation of the GPU networks. The first generation can take substantially longer, while later generations may be faster. Benchmark both settings on your system. Set the same variable when running the real-engine and parity tests above to check the compiled path.

Closing an engine releases its live model tensors. PyTorch may retain unused Metal allocations in its process-wide cache for reuse; applications that need to return that unused memory to the system can call `torch.mps.empty_cache()` after closing the engine.
