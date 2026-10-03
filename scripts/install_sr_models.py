"""Install optional, separately licensed SR weights into a local effects folder.

AnimeSharp uses the author's ONNX files. DAT2 is exported at a fixed 128x128
input so its traced attention masks remain valid. Downloads are SHA256 checked;
the installed descriptor records the exact ONNX hash checked by Magpie at load.
No files are uploaded. Model licenses are independent of Magpie's GPL license.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

# ORT initializes telemetry during import, before its runtime opt-out API can run.
os.environ['ORT_DISABLE_TELEMETRY'] = '1'

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verified_download(url, dest, sha256):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(dest.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        if digest(temporary) != sha256:
            temporary.unlink()
            raise ValueError(f'Checksum mismatch: {url}')
        temporary.replace(dest)
    if digest(dest) != sha256:
        raise ValueError(f'Checksum mismatch: {dest}')


def export_dat(model, cache):
    import gdown
    import numpy as np
    import onnx
    import onnxruntime as ort
    import spandrel
    import torch

    archive = cache / 'IllustrationJaNai.zip'
    checkpoint = cache / model['checkpoint']
    if not checkpoint.exists():
        if not archive.exists():
            temporary = archive.with_suffix('.partial')
            if not gdown.download(id=model['driveId'], output=str(temporary), quiet=False):
                raise RuntimeError('Could not download the author\'s DAT2 archive')
            temporary.replace(archive)
        if digest(archive) != model['archiveSha256']:
            raise ValueError('DAT2 archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            # Read only the selected checkpoint; never extract arbitrary paths.
            checkpoint.write_bytes(z.read(model['archiveMember']))
    if digest(checkpoint) != model['checkpointSha256']:
        raise ValueError('DAT2 checkpoint checksum mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(41)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    network = spandrel.ModelLoader().load_from_state_dict(state).eval().model
    sample = torch.rand(1, 3, 128, 128)
    output = cache / model['file']
    with torch.inference_mode():
        expected = network(sample).numpy()
        torch.onnx.export(network, sample, str(output), opset_version=17,
                          input_names=['input'], output_names=['output'], dynamo=False)
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'input': sample.numpy()})[0]
    error = float(np.max(np.abs(actual - expected)))
    if actual.shape != expected.shape or not np.isfinite(actual).all() or error > 0.0003:
        raise RuntimeError(f'DAT2 export verification failed: max error {error}')
    print(f'DAT2 export matches PyTorch: max absolute error {error:.8g}')
    return output


def main():
    manifest = json.loads(Path(__file__).with_name('sr_models.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effects-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT / 'obj' / 'sr-models')
    parser.add_argument('--model', choices=[m['id'] for m in manifest['models']], action='append')
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    target = args.effects_dir / 'ONNX'
    target.mkdir(parents=True, exist_ok=True)
    for model in manifest['models']:
        if args.model and model['id'] not in args.model:
            continue
        print(f'Installing {model["id"]} ({model["license"]}; {model["author"]})')
        if 'url' in model:
            weights = args.cache / model['file']
            verified_download(model['url'], weights, model['sha256'])
        else:
            weights = export_dat(model, args.cache)
        descriptor = {k: model[k] for k in ('name', 'file', 'scale', 'tileSize', 'overlap', 'author', 'license', 'source')}
        descriptor.update(version=1, sha256=digest(weights))
        temporary = target / (model['file'] + '.partial')
        shutil.copyfile(weights, temporary)
        temporary.replace(target / model['file'])
        (target / (model['id'] + '.onnx.json')).write_text(json.dumps(descriptor, indent=2) + '\n', encoding='utf-8')
    for name in ['README.txt', 'LICENSE-CC-BY-NC-SA-4.0.txt']:
        src = ROOT / 'src' / 'Effects' / 'ONNX' / name
        if src.resolve() != (target / name).resolve():
            shutil.copyfile(src, target / name)


if __name__ == '__main__':
    main()
