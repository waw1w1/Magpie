Optional SR model weights (not included in Magpie's GPL license)

AnimeSharp V4 and Fast: Kim2091, CC BY-NC-SA 4.0
https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV4

4x IllustrationJaNai V1 DAT2: the-database, CC BY-NC-SA 4.0
https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2
The original DirectML DAT2 export uses fixed 128x128 input.

These models require attribution, noncommercial use, and share-alike terms.
See LICENSE-CC-BY-NC-SA-4.0.txt for the full license.

Install from the repository with Python 3.12:
  pip install -r scripts/requirements-sr-models.txt
  python scripts/install_sr_models.py --effects-dir <Magpie folder>/effects

The original models use DirectML on Magpie's selected GPU. Heavy models are intended for static
illustration/CG quality comparisons; real-time frame rates are not guaranteed.
The backend transfers SDR RGB through CPU memory and runs inference on a worker.
While processing, a labelled live bilinear preview is displayed. Only a result
matching the current input replaces it; continuously moving scenes may remain
in preview. Saving effect screenshots is blocked until the model result is ready.
Identical frames are cached; resize invalidates old work and results.
The original presets use tile extent 128 and stride 64. Each 32-pixel halo reserves the outer half
for context and feathers the inner half into neighbouring predictions.
Attention models still differ from whole-image inference; check your CGs.
The D3D11 profiler does not measure worker inference. The first displayed model
result logs inference/conversion time, excluding D3D11 readback and upload.

Optional NVIDIA FP8/BF16 DAT2
---------------------------
Windows x64 builds can also load the separately installed TensorRT RTX plugin.
This uses FP8 E4M3FN for selected matrix operations and BF16 for convolution and
sensitive paths. It requires an NVIDIA GPU with FP8 support (Ada/Blackwell).
It does not silently fall back to DirectML or CPU. The selected graphics adapter
must match a GPU exposed by the NVIDIA plugin.

Install the pinned official runtime next to Magpie.exe:
  python scripts/install_tensorrt_rtx.py --app-dir <Magpie folder>

Create the locally calibrated model in a CUDA-enabled Python 3.12 environment
(WSL is supported for export; use the corresponding /mnt/... output path):
  pip install -r scripts/requirements-sr-fp8.txt
  python scripts/export_dat2_fp8.py --checkpoint <model-cache>/4x_IllustrationJaNai_V1_DAT2_190k.pth --calibration-image <game screenshot.png> --effects-dir <Magpie folder>/effects

The checkpoint is downloaded by install_sr_models.py; pass --cache <model-cache>
to that installer to choose its location. Repeat --calibration-image for several
representative screenshots containing artwork, dialogue and menus. Images are
processed locally. --input-size WIDTH HEIGHT can match the actual capture size.
The exporter reserves separate crops for numerical validation against BF16.
That check is a local numerical screen, not a guarantee for every game scene.

The FP8 preset defaults to a fixed 512-pixel tile with overlap 32 (stride 448);
--tile-size 256 is also available. Fixed attention masks are computed once and
shared in the exported graph, which can increase the ONNX file size.
FP8 matrix dimensions are zero-padded and cropped for aligned GPU execution.
The model and its .onnx.json descriptor must remain paired because the descriptor
records the exported SHA256. Choose ONNX/IllustrationJaNai-DAT2-FP8 in a scaling
mode and add a final screen-fit effect as needed, just like the original preset.

First initialization compiles the model and NVIDIA kernels. Later loads reuse
both the embedded EP context model and kernel cache under
LOCALAPPDATA/Magpie/cache/TensorRT/<model SHA256>. The context cache is invalidated
when ORT or the NVIDIA runtime changes; an unreadable context is rebuilt from
the verified ONNX model. Model initialization is separate from the logged
worker inference time. Keep the NVIDIA DLLs and their
bundled README/acknowledgements with the application; their licenses are separate
from both Magpie and the model. The runtime installer verifies the official wheel
SHA256 and does not require a system CUDA Toolkit on the playback machine.
