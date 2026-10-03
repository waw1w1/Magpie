"""CPU reference checks for translated shader math (not GPU/FP16 execution).

--sources points to pinned ACNetGLSL, ArtCNN and LeRF-PyTorch source folders.
The fixture uses the authors' GLSL, ONNX and NumPy implementations independently
of the translated pass schedule. All sources must match scripts/*_sources.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ORT initializes telemetry during import, before its runtime opt-out API can run.
os.environ['ORT_DISABLE_TELEMETRY'] = '1'

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


def fetch_references(sources):
    neural = json.loads((ROOT / 'scripts/neural_sources.json').read_text())
    lut = json.loads((ROOT / 'scripts/lerf_sources.json').read_text())
    jobs = []
    for model in neural['models'] + neural['references']:
        family = 'ACNetGLSL' if model['family'] == 'acnet' else 'ArtCNN'
        jobs.append((sources / family / model['source'], model['repository'], model['commit'], model['source'], model['sha256']))
    for entry in lut['references'] + [entry for model in lut['models'] for entry in model['tables']]:
        jobs.append((sources / 'LeRF-PyTorch' / entry['path'], lut['repository'], lut['commit'], entry['path'], entry['sha256']))
    def obtain(job):
        path, repo, revision, source, sha256 = job
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/{repo}/{revision}/{source}', timeout=60).read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError(f'Reference source checksum mismatch: {path}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(obtain, jobs))


def shifted(a, x, y, mode):
    height, width = a.shape[:2]
    padded = np.pad(a, ((3, 3), (3, 3), (0, 0)), mode=mode)
    return padded[3 + y:3 + y + height, 3 + x:3 + x + width]


def vector(*v):
    return np.array(v if len(v) > 1 else v[0], dtype=np.float32)


def evaluate(body, values, original=False):
    env = dict(MF=np.float32, MF4=vector, vec4=vector, MF4x4=lambda *v: vector(*v).reshape(4, 4),
               mul=lambda v, m: v @ m, max=np.maximum, min=np.minimum,
               glmul=lambda m, v: np.einsum('ij,...j->...i', np.array(m, np.float32).reshape(4, 4, order='F'), v))
    def sample(m):
        tex, x, y = m.groups()
        key = 'sample' + str(len(env))
        env[key] = shifted(values[tex], int(float(x)), int(float(y)), 'edge' if original or '.SampleLevel' in m[0] else 'constant')
        return key
    if original:
        body = re.sub(r'(\w+)_texOff\(vec2\(([^,]+),\s*([^\)]+)\)\)', sample, body)
        body = re.sub(r'mat4\(([^)]+)\) \* ([^;]+)', r'glmul([\1], \2)', body)
    else:
        body = re.sub(r'(\w+)\.SampleLevel\(SP, pos \+ float2\(([^,]+), ([^)]+)\) \* GetInputPt\(\), 0\)', sample, body)
        body = re.sub(r'(\w+)\.Load\(int3\(int2\(gxy\) \+ int2\((-?\d+), (-?\d+)\), 0\)\)', sample, body)
        body = re.sub(r'Luma\((\w+)\.rgb\)\.xxxx', r'\1', body)
        body = re.sub(r'Luma\((\w+)\.rgb\)', r'\1', body)
    body = re.sub(r'(\w+)\.x\b', r'\1[..., :1]', body).replace('(MF4)0.0', '0.0')
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(('//', '{', '}', 'void ', 'uint2 ', 'if ', 'float2 ')):
            continue
        if line == 'return result;':
            return env['result']
        if '[gxy]' in line:
            tex, expr = line.split('[gxy] = ')
            values[tex] = eval(expr.rstrip(';'), {'__builtins__': {}}, env)
            continue
        line = re.sub(r'^(MF4|MF|vec4) ', '', line).rstrip(';')
        line = re.sub(r'^(\w+) \+= (.*)', r'\1 = \1 + \2', line)
        exec(line, {'__builtins__': {}}, env)


def neural(sources):
    models = json.loads((ROOT / 'scripts/neural_sources.json').read_text())['models']
    rng = np.random.default_rng(41)
    fixtures = [rng.random((11, 13, 1), dtype=np.float32), np.zeros((9, 7, 1), np.float32)]
    fixtures[1][0, 0] = 1
    for model in models:
        family = 'ACNetGLSL' if model['family'] == 'acnet' else 'ArtCNN'
        path = sources / family / model['source']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == model['sha256']
        src = path.read_text()
        shader = (ROOT / 'src/Effects' / (model['effect'] + '.hlsl')).read_text()
        for input in fixtures:
            maps = {'INPUT': input}
            passes = re.split(r'(?=//!PASS \d+)', shader)[1:]
            for part in passes[:-1]:
                evaluate(part, maps)
            last = re.search(r'//!IN INPUT, (\w+)', passes[-1])[1]
            height, width = input.shape[:2]
            actual = maps[last].reshape(height, width, 2, 2).transpose(0, 2, 1, 3).reshape(height * 2, width * 2)
            if family == 'ACNetGLSL':
                original = {'LUMA': input}
                hooks = re.split(r'(?=//!DESC)', src)[1:]
                for hook in hooks[:-1]:
                    result = evaluate(re.search(r'vec4 hook\(\)\s*\{(.*)\}', hook, re.S)[1], original, True)
                    original[re.search(r'//!SAVE (\w+)', hook)[1]] = result
                last = re.search(r'//!BIND (\w+)', hooks[-1])[1]
                expected = original[last].reshape(height, width, 2, 2).transpose(0, 2, 1, 3).reshape(height * 2, width * 2)
            else:
                name = 'ArtCNN_' + model['effect'].split('-')[1] + '.onnx'
                session = ort.InferenceSession(str(sources / family / 'ONNX' / name), providers=['CPUExecutionProvider'])
                expected = session.run(None, {'input': input.transpose(2, 0, 1)[None]})[0][0, 0]
            # Both published reconstructions clamp luma after pixel shuffle.
            error = float(np.max(abs(np.clip(actual, 0, 1) - np.clip(expected, 0, 1))))
            assert error < 0.00002, (model['effect'], error)
            print(model['effect'], input.shape[:2], 'max absolute error', error)


def simplex(table, value):
    grid = value.astype(np.int32) // 16
    fraction = value % 16
    order = np.argsort(-fraction, axis=-1, kind='stable')
    def lookup(g):
        index = ((g[..., 0] * 17 + g[..., 1]) * 17 + g[..., 2]) * 17 + g[..., 3]
        return table[index]
    sorted_fraction = np.take_along_axis(fraction, order, axis=-1)
    result = (16 - sorted_fraction[..., :1]) * lookup(grid)
    for k in range(4):
        grid += np.eye(4, dtype=np.int32)[order[..., k]]
        delta = sorted_fraction[..., k:k + 1] - (sorted_fraction[..., k + 1:k + 2] if k < 3 else 0)
        result += delta * lookup(grid)
    return result / 16


def lerf(sources):
    root = sources / 'LeRF-PyTorch'
    text = (root / 'resample/eval_lut_sr.py').read_text()
    ns = {'np': np}
    exec(text[text.index('def FourSimplexInterpFaster'):text.index('class eltr:')], ns)
    reference = ns['FourSimplexInterpFaster']
    manifest = json.loads((ROOT / 'scripts/lerf_sources.json').read_text())
    rng = np.random.default_rng(47)
    for model in manifest['models']:
        for entry in model['tables']:
            path = root / entry['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
            table = np.load(path, allow_pickle=False).reshape(17 ** 4, -1).astype(np.float32)
            mode = path.stem.split('_')[-1][0]
            for rotation in range(4):
                input = rng.integers(0, 256, (11, 13, 3)).astype(np.float32)
                # Include ties, grid boundaries, saturated black and white.
                input[:2, :2] = 0; input[-2:, -2:] = 255; input[3:5, 3:5] = 128
                rotated = np.rot90(input, rotation)
                h, w = rotated.shape[:2]
                pad = 1 if mode == 's' else 3
                ref = reference(table, np.pad(rotated, ((0, pad), (0, pad), (0, 0)), mode='edge').transpose(2, 0, 1),
                                h, w, 4, 4 - rotation, upscale=1, mode=mode, oC=table.shape[1])
                taps = []
                for i in range(4):
                    x, y = (i % 2, i // 2) if mode == 's' else ((i, 0) if mode == 'c' else (i, i))
                    for _ in range(rotation):
                        x, y = -y, x
                    taps.append(shifted(input, x, y, 'edge'))
                actual = simplex(table, np.stack(taps, axis=-1)).transpose(2, 3, 0, 1).reshape(ref.shape)
                assert np.array_equal(actual, ref), (path.name, rotation, np.max(abs(actual - ref)))
        print(model['name'], 'LUT simplex / all rotations / edge padding: exact match')

    sys.path.insert(0, str(root))
    from resize_right.resize_right2d_numpy import AmplifiedLinearResize2dNumpy, SteeringGaussianResize2dNumpy
    image = rng.integers(0, 256, (3, 7, 9)).astype(np.float32)
    hyper = rng.integers(0, 256, (3, 3, 7, 9)).astype(np.float32) / 255
    for gaussian in (False, True):
        for oh, ow in [(7, 9), (14, 18), (11, 23)]:
            ref = SteeringGaussianResize2dNumpy(support_sz=2, pad_mode='edge') if gaussian else AmplifiedLinearResize2dNumpy(pad_mode='edge')
            ref.set_shape(image.shape, out_shape=[3, oh, ow])
            expected = ref.resize(image, *hyper) if gaussian else ref.resize(image, hyper[0])
            actual = np.zeros((3, oh, ow), np.float32)
            for y in range(oh):
                for x in range(ow):
                    py, px = (y + 0.5) * 7 / oh - 0.5, (x + 0.5) * 9 / ow - 0.5
                    first_y, first_x = np.ceil(np.array([py, px]) - 1 - 1e-6).astype(int)
                    total = np.zeros(3); value = np.zeros(3)
                    for qy in range(first_y, first_y + 2):
                        for qx in range(first_x, first_x + 2):
                            ey, ex = np.clip(qy, 0, 6), np.clip(qx, 0, 8)
                            if gaussian:
                                rho = hyper[0, :, ey, ex] * 2 - 1
                                dy = hyper[1, :, ey, ex] * 10 * (py - qy)
                                dx = hyper[2, :, ey, ex] * 10 * (px - qx)
                                weight = np.exp(-0.5 * (dy * dy - 2 * rho * dy * dx + dx * dx))
                            else:
                                alpha = hyper[0, :, ey, ex] * 2 - 1
                                weight = np.maximum(1 - alpha * abs(px - qx), 0) * np.maximum(1 - alpha * abs(py - qy), 0)
                                if abs(px - qx) > 1 or abs(py - qy) > 1: weight[:] = 0
                            value += weight * image[:, ey, ex]
                            total += weight
                    actual[:, y, x] = value / np.maximum(total, 1e-30)
            error = float(np.max(abs(actual - expected)))
            assert error < 0.0002, (gaussian, oh, ow, error)
            print('LeRF-G' if gaussian else 'LeRF-L', (oh, ow), 'resampling max error in 8-bit units', error)
            # The entire frame, including borders, must preserve a constant
            # prefiltered image. This catches the former 255 -> 142 dark corner.
            for gray in (0,128,255):
                constant=np.full_like(image,gray)
                actual=ref.resize(constant,*hyper) if gaussian else ref.resize(constant,hyper[0])
                assert np.max(abs(actual-gray))<0.0002,(gaussian,gray,actual.min(),actual.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    args = parser.parse_args()
    fetch_references(args.sources)
    neural(args.sources)
    lerf(args.sources)


if __name__ == '__main__':
    main()
