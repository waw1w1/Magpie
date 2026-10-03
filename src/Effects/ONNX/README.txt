Optional SR model weights (not included in Magpie's GPL license)

AnimeSharp V4 and Fast: Kim2091, CC BY-NC-SA 4.0
https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV4

4x IllustrationJaNai V1 DAT2: the-database, CC BY-NC-SA 4.0
https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2
DAT2 ONNX export adapts the original checkpoint, with fixed 128x128 input.

These models require attribution, noncommercial use, and share-alike terms.
See LICENSE-CC-BY-NC-SA-4.0.txt for the full license.

Install from the repository with Python 3.12:
  pip install -r scripts/requirements-sr-models.txt
  python scripts/install_sr_models.py --effects-dir <Magpie folder>/effects

Uses DirectML on Magpie's selected GPU. Heavy models are intended for static
illustration/CG quality comparisons; real-time frame rates are not guaranteed.
The backend transfers SDR RGB through CPU memory and runs inference on a worker.
While processing, a labelled live bilinear preview is displayed. Only a result
matching the current input replaces it; continuously moving scenes may remain
in preview. Saving effect screenshots is blocked until the model result is ready.
Identical frames are cached; resize invalidates old work and results.
Tile extent is 128 and stride is 64. Each 32-pixel halo reserves the outer half
for context and feathers the inner half into neighbouring predictions.
Attention models still differ from whole-image inference; check your CGs.
The D3D11 profiler does not measure worker inference. The first displayed model
result logs inference/conversion time, excluding D3D11 readback and upload.
