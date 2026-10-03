"""Compare the production C++ stitcher with whole-image and old hard-tiled SR.

The 128x128 fixture deliberately crosses a tile boundary with gradients and ink.
This checks the known seam regression, not a general Galgame quality benchmark.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    args = parser.parse_args()
    out = ROOT / 'obj' / 'seam-check'
    out.mkdir(parents=True, exist_ok=True)
    x, y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
    image = np.stack([.3 + .35*x, .2 + .4*y, .25 + .2*x + .15*y], axis=-1)
    fixture = Image.fromarray(np.round(image*255).astype(np.uint8))
    draw = ImageDraw.Draw(fixture)
    draw.line((10, 105, 110, 10), fill=(30, 25, 35), width=1)
    draw.text((7, 14), 'SAVE 0123', fill=(245, 245, 245))
    image = np.asarray(fixture).astype(np.float32)/255
    image.tofile(out/'input.f32')
    padded = np.pad(image, ((32,32),(32,32),(0,0)), mode='edge')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    records = []
    for model in json.loads((ROOT/'scripts/sr_models.json').read_text())['models']:
        path = args.models/model['file']
        scale = model['scale']
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        dtype = np.float16 if session.get_inputs()[0].type == 'tensor(float16)' else np.float32

        def infer(pixels):
            return session.run(None, {session.get_inputs()[0].name: pixels.transpose(2,0,1)[None].astype(dtype)})[0][0].transpose(1,2,0).astype(np.float32)

        whole = infer(image).clip(0,1)
        hard = np.empty_like(whole)
        for yy in (0,64):
            for xx in (0,64):
                tile = infer(padded[yy:yy+128,xx:xx+128])
                hard[yy*scale:(yy+64)*scale,xx*scale:(xx+64)*scale] = tile[32*scale:96*scale,32*scale:96*scale]
        hard = hard.clip(0,1)
        subprocess.run([str(args.runner.resolve()),str(path),str(scale),'128','32','128','128',
                        str(out/'input.f32'),str(out/'output.f32')], check=True, capture_output=True)
        blended = np.fromfile(out/'output.f32',np.float32).reshape(whole.shape).clip(0,1)

        def metrics(result):
            cut = 64*scale
            vertical = ((result[:,cut]-result[:,cut-1])-(whole[:,cut]-whole[:,cut-1]))[20*scale:108*scale]
            horizontal = ((result[cut]-result[cut-1])-(whole[cut]-whole[cut-1]))[20*scale:108*scale]
            jumps = np.concatenate([vertical,horizontal])*255
            error = (abs(result-whole)*255)[16*scale:-16*scale,16*scale:-16*scale]
            return {'seam_max':float(abs(jumps).max()),'seam_rms':float(np.sqrt(np.mean(jumps*jumps))),
                    'interior_mean':float(error.mean()),'interior_max':float(error.max())}

        old, new = metrics(hard), metrics(blended)
        record = {'model':model['id'],'hard':old,'blended':new}
        print(json.dumps(record),flush=True)
        records.append(record)
        for label, pixels in [('whole',whole),('hard',hard),('blended',blended)]:
            Image.fromarray(np.round(pixels*255).astype(np.uint8)).save(out/f'{model["id"]}-{label}.png')
        assert new['seam_rms'] < old['seam_rms'], record
        assert new['seam_max'] < old['seam_max'], record
    (out/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
