"""Reproducibly translate pinned ACNet/ARNet/ArtCNN weights to MagpieFX.

The GLSL matrix constructors are column-major. HLSL constructors are row-major,
so a GLSL matrix-vector product becomes mul(vector, matrix), not mul(matrix,
vector). Logical feature maps are kept separate from their physical textures;
the lifetime allocator never aliases an input and output in the same dispatch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("neural_sources.json")


def source_file(model, cache):
    target = cache / model["source"]
    if not target.exists():
        url = f'https://raw.githubusercontent.com/{model["repository"]}/{model["commit"]}/{model["source"]}'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != model["sha256"]:
        raise ValueError(f"Source checksum mismatch: {target}")
    return data.decode("utf-8")


def acnet_layers(source):
    hooks = re.split(r"(?=//!DESC)", source)[1:]
    layers, saved = [], {"LUMA": "INPUT"}
    i = 0
    while i < len(hooks) - 1:
        group = [hooks[i]]
        desc = hooks[i].splitlines()[0]
        if "part 0" in desc:
            prefix = desc.rsplit("part ", 1)[0]
            while i + len(group) < len(hooks) - 1:
                next_desc = hooks[i + len(group)].splitlines()[0]
                if next_desc != prefix + "part " + str(len(group)):
                    break
                group.append(hooks[i + len(group)])
        bodies, inputs, outputs, updates = [], [], [], {}
        for part, hook in enumerate(group):
            out = f"F{len(layers)}_{part}"
            name = re.search(r"//!SAVE (\w+)", hook)[1]
            body = re.search(r"vec4 hook\(\)\s*\{(.*)\}", hook, re.S)[1]
            def sample(m):
                binding, x, y = m.groups()
                tex = saved[binding]
                if tex not in inputs:
                    inputs.append(tex)
                expr = f"{tex}.SampleLevel(SP, pos + float2({x}, {y}) * GetInputPt(), 0)"
                return f"Luma({expr}.rgb).xxxx" if tex == "INPUT" else expr
            body = re.sub(r"(\w+)_texOff\(vec2\(([^,]+),\s*([^\)]+)\)\)", sample, body)
            body = re.sub(r"mat4\(([^)]+)\) \* ([^;]+)", r"mul(MF4(\2), MF4x4(\1))", body)
            body = body.replace("vec4", "MF4").replace("MF4(0.0)", "(MF4)0.0").replace("return result;", f"{out}[gxy] = result;")
            bodies.append("{\n" + body.strip() + "\n}")
            outputs.append(out)
            updates[name] = out
        layers.append({"inputs": inputs, "outputs": outputs, "body": "\n".join(bodies)})
        saved.update(updates)
        i += len(group)
    final_binding = re.search(r"//!BIND (\w+)", hooks[-1])[1]
    return layers, saved[final_binding]


def artcnn_layers(source):
    hooks = re.split(r"(?=//!DESC)", source)[1:]
    layers, saved = [], {"LUMA": ["INPUT"]}
    for hook in hooks[:-1]:
        bindings = re.findall(r"//!BIND (\w+)", hook)
        name = re.search(r"//!SAVE (\w+)", hook)[1]
        lines = hook.splitlines()
        result_lines = [s.strip() for s in lines if re.match(r"\s*(?:V4 )?result\d+ (?:=|\+=)", s)]
        count = len(re.findall(r"V4 result\d+ =", hook))
        outputs = [f"F{len(layers)}_{j}" for j in range(count)]
        inputs = list(dict.fromkeys(t for b in bindings for t in saved[b]))
        body = "\n".join(result_lines)
        samples = sorted(set(re.findall(r"inp_(\d+)_(\d+)_(\d+)", body)))
        declarations = []
        for c, x, y in samples:
            expressions = []
            for binding in bindings:
                tex = saved[binding][int(c)]
                sample = f"{tex}.Load(int3(int2(gxy) + int2({int(x)-1}, {int(y)-1}), 0))"
                expressions.append(f"Luma({sample}.rgb)" if tex == "INPUT" else sample)
            ty = "MF" if bindings == ["LUMA"] else "MF4"
            declarations.append(f"{ty} inp_{c}_{x}_{y} = {ty}({' + '.join(expressions)});")
        body = re.sub(r"M4\(([^)]+)\) \* (inp_\d+_\d+_\d+)", r"mul(\2, MF4x4(\1))", body)
        body = body.replace("V4", "MF4")
        writes = []
        for j, out in enumerate(outputs):
            relu = "ReLU" in hook.splitlines()[0]
            value = f"max(result{j}, (MF4)0.0)" if relu else f"result{j}"
            writes.append(f"{out}[gxy] = {value};")
        layers.append({"inputs": inputs, "outputs": outputs, "body": "\n".join(declarations + [body] + writes)})
        saved[name] = outputs
    return layers, layers[-1]["outputs"][0]


def allocate(layers, final):
    last_use = {out: i for i, layer in enumerate(layers) for out in layer["outputs"]}
    for i, layer in enumerate(layers):
        for tex in layer["inputs"]:
            last_use[tex] = i
    last_use[final] = len(layers)
    mapping, slots = {"INPUT": "INPUT"}, []
    for i, layer in enumerate(layers):
        for out in layer["outputs"]:
            slot = next((j for j, until in enumerate(slots) if until < i), len(slots))
            if slot == len(slots):
                slots.append(-1)
            slots[slot] = last_use[out]
            mapping[out] = f"T{slot}"
    return mapping, len(slots)


def generate(model, source):
    layers, final = (acnet_layers if model["family"] == "acnet" else artcnn_layers)(source)
    mapping, count = allocate(layers, final)
    def physical(text):
        return re.sub(r"\bF\d+_\d+\b", lambda m: mapping[m[0]], text)
    license_text = model["license_text"]
    header = "\n".join(("// " + line).rstrip() for line in license_text.splitlines())
    text = [header, f'// Source: https://github.com/{model["repository"]}/blob/{model["commit"]}/{model["source"]}',
            '// Generated by scripts/generate_neural_effects.py; do not edit weights by hand.',
            '//!MAGPIE EFFECT\n//!VERSION 4\n//!CAPABILITY FP16\n//!USE MulAdd',
            '#include "../StubDefs.hlsli"',
            '//!TEXTURE\nTexture2D INPUT;',
            '//!TEXTURE\n//!WIDTH INPUT_WIDTH * 2\n//!HEIGHT INPUT_HEIGHT * 2\nTexture2D OUTPUT;',
            '//!SAMPLER\n//!FILTER POINT\nSamplerState SP;',
            '//!SAMPLER\n//!FILTER LINEAR\nSamplerState SL;']
    for j in range(count):
        text.append(f'//!TEXTURE\n//!WIDTH INPUT_WIDTH\n//!HEIGHT INPUT_HEIGHT\n//!FORMAT R16G16B16A16_FLOAT\nTexture2D T{j};')
    text.append('//!COMMON\nfloat Luma(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }')
    for i, layer in enumerate(layers, 1):
        ins = ', '.join(dict.fromkeys(mapping[t] for t in layer['inputs']))
        outs = ', '.join(mapping[t] for t in layer['outputs'])
        text.append(f'//!PASS {i}\n//!DESC Convolution {i}\n//!IN {ins}\n//!OUT {outs}\n//!BLOCK_SIZE 8\n//!NUM_THREADS 64\n'
                    f'void Pass{i}(uint2 blockStart, uint3 tid) {{\n'
                    '\tuint2 gxy = blockStart + Rmp8x8(tid.x);\n'
                    '\tif (any(gxy >= GetInputSize())) return;\n'
                    '\tfloat2 pos = (gxy + 0.5) * GetInputPt();\n' + physical(layer['body']) + '\n}')
    n = len(layers) + 1
    text.append(f'//!PASS {n}\n//!DESC Reconstruct luma and retain chroma\n//!IN INPUT, {mapping[final]}\n//!OUT OUTPUT\n//!STYLE PS\n'
                f'float4 Pass{n}(float2 pos) {{\n'
                '\tuint2 p = uint2(pos * GetOutputSize());\n'
                f'\tfloat y = saturate({mapping[final]}.Load(int3(p / 2, 0))[(p.y % 2) * 2 + p.x % 2]);\n'
                '\tfloat4 color = INPUT.SampleLevel(SL, pos, 0);\n'
                '\treturn float4(saturate(color.rgb + y - Luma(color.rgb)), color.a);\n}')
    return '\n\n'.join(text) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'obj' / 'neural-sources')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for model in json.loads(MANIFEST.read_text())['models']:
        generated = generate(model, source_file(model, args.cache))
        dest = ROOT / 'src' / 'Effects' / (model['effect'] + '.hlsl')
        if args.check:
            if not dest.exists() or dest.read_text() != generated:
                raise SystemExit(f'Generated effect differs: {dest}')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(generated, encoding='utf-8')
        print(model['effect'], len(generated), 'bytes')


if __name__ == '__main__':
    main()
