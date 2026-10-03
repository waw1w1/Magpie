"""Package the authors' pinned LeRF LUTs without changing their quantization.

Requires NumPy. The UNORM atlas stores signed int8 + 128 losslessly. The shader
decodes bytes before interpolation; it must not interpolate the atlas itself.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name('lerf_sources.json')


def dds_rgba8(pixels):
    height, width, _ = pixels.shape
    # DDS_HEADER + DDS_PIXELFORMAT (DX10) + DDS_HEADER_DXT10.
    header = [124, 0x100F, height, width, width * 4, 0, 1] + [0] * 11
    header += [32, 4, int.from_bytes(b'DX10', 'little'), 0, 0, 0, 0, 0]
    header += [0x1000, 0, 0, 0, 0]
    return b'DDS ' + struct.pack('<31I', *header) + struct.pack('<5I', 28, 3, 0, 1, 0) + pixels.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'obj' / 'lerf-sources')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    for model in manifest['models']:
        tables = []
        for src in model['tables']:
            path = args.cache / src['path']
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f'https://raw.githubusercontent.com/{manifest["repository"]}/{manifest["commit"]}/{src["path"]}'
                path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
            if hashlib.sha256(path.read_bytes()).hexdigest() != src['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
            lut = np.load(path, allow_pickle=False).reshape(17 ** 4, -1)
            if lut.dtype != np.int8 or lut.shape[1] not in (1, 3):
                raise ValueError(f'Unexpected LUT layout: {path}')
            rgba = np.full((17 ** 4, 4), 128, np.uint8)
            rgba[:, :lut.shape[1]] = (lut.astype(np.int16) + 128).astype(np.uint8)
            tables.append(rgba.reshape(289, 289, 4))
        data = dds_rgba8(np.concatenate(tables, axis=0))
        dest = ROOT / 'src' / 'Effects' / 'LeRF' / (model['name'] + '.dds')
        if args.check:
            if dest.read_bytes() != data:
                raise SystemExit(f'Generated atlas differs: {dest}')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        print(dest.name, len(data), hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
