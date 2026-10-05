"""Calibrate a DAT2 FP8/BF16 model for Magpie's optional TensorRT RTX backend.

Run in a CUDA-enabled Python environment with requirements-sr-fp8.txt installed.
Calibration images stay local. The original model license also applies to the
derived weights. FP8 is used for selected matrix operations; convolution,
attention, normalization and reconstruction paths retain BF16 precision.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

from install_sr_models import ROOT, digest


def cache_attention_masks(model, tile):
    import torch
    masks = {}
    for module in model.modules():
        if not hasattr(module, 'calculate_mask'):
            continue
        key = (tuple(module.split_size), tuple(module.shift_size))
        if key not in masks:
            masks[key] = tuple(mask.cuda() for mask in
                               module.calculate_mask(tile, tile, dtype=torch.bfloat16))
        module.patches_resolution = tile
        module.attn_mask_0, module.attn_mask_1 = masks[key]


@contextmanager
def preserve_export_precision():
    from modelopt.onnx.export.fp8_exporter import FP8QuantExporter
    original_conv = FP8QuantExporter._quantize_conv_weights_to_fp8
    original_softmax = FP8QuantExporter._insert_qdq_after_softmax

    def selected_convs(graph):
        # ModelOpt 0.47 otherwise also quantizes explicitly excluded Conv
        # weights, creating unsupported W8/A16 paths in TensorRT RTX.
        preserved = []
        for node in graph.nodes:
            if node.op == 'Conv' and not (node.inputs[0].inputs and
                                         node.inputs[0].inputs[0].op == 'DequantizeLinear'):
                preserved.append(node)
                node.op = 'PreserveConvPrecision'
        try:
            return original_conv(graph)
        finally:
            for node in preserved:
                node.op = 'Conv'

    FP8QuantExporter._quantize_conv_weights_to_fp8 = staticmethod(selected_convs)
    # Softmax/attention were not calibrated in FP8. Do not add new Q/DQ there.
    FP8QuantExporter._insert_qdq_after_softmax = staticmethod(lambda graph: 0)
    try:
        yield
    finally:
        FP8QuantExporter._quantize_conv_weights_to_fp8 = staticmethod(original_conv)
        FP8QuantExporter._insert_qdq_after_softmax = staticmethod(original_softmax)


def align_fp8_matrices(model):
    import numpy as np
    from onnx import helper, numpy_helper, TensorProto
    producers = {name: node for node in model.graph.node for name in node.output}
    constants = {value.name: value for value in model.graph.initializer}
    nodes, count = [], 0
    for node in model.graph.node:
        left = producers.get(node.input[0]) if node.op_type == 'MatMul' else None
        right = producers.get(node.input[1]) if node.op_type == 'MatMul' else None
        quantizer = producers.get(left.input[0]) if left and left.op_type == 'DequantizeLinear' else None
        weight = constants.get(right.input[0]) if right and right.op_type == 'DequantizeLinear' else None
        if not (quantizer and quantizer.op_type == 'QuantizeLinear' and weight
                and weight.data_type == TensorProto.FLOAT8E4M3FN and len(weight.dims) == 2):
            nodes.append(node)
            continue
        inner, columns = weight.dims
        padded_inner, padded_columns = (inner+31)//32*32, (columns+31)//32*32
        if (inner, columns) == (padded_inner, padded_columns):
            nodes.append(node)
            continue
        tag = f'magpie_fp8_aligned_{count}'
        values = np.pad(numpy_helper.to_array(weight),
                        ((0, padded_inner-inner), (0, padded_columns-columns)), constant_values=0)
        model.graph.initializer.extend([
            numpy_helper.from_array(values, tag+'_weight'),
            numpy_helper.from_array(np.array([0, padded_inner-inner], np.int64), tag+'_pad'),
            numpy_helper.from_array(np.array([-1], np.int64), tag+'_axis'),
            numpy_helper.from_array(np.array([0], np.int64), tag+'_start'),
            numpy_helper.from_array(np.array([columns], np.int64), tag+'_end'),
        ])
        # Zero padding preserves the computation while giving the FP8 kernels
        # aligned channel dimensions. Crop the padded output before adding bias.
        pad = helper.make_node('Pad', [quantizer.input[0], tag+'_pad', '', tag+'_axis'],
                               [tag+'_input'], name=tag+'_Pad')
        aligned_q, aligned_left, aligned_right = deepcopy(quantizer), deepcopy(left), deepcopy(right)
        aligned_q.name, aligned_q.input[0], aligned_q.output[0] = tag+'_Q', tag+'_input', tag+'_q'
        aligned_left.name, aligned_left.input[0], aligned_left.output[0] = tag+'_DQ', tag+'_q', tag+'_dq'
        aligned_right.name, aligned_right.input[0], aligned_right.output[0] = tag+'_WDQ', tag+'_weight', tag+'_wdq'
        original_output = node.output[0]
        node.input[0], node.input[1], node.output[0] = tag+'_dq', tag+'_wdq', tag+'_result'
        crop = helper.make_node('Slice', [tag+'_result', tag+'_start', tag+'_end', tag+'_axis'],
                                [original_output], name=tag+'_Slice')
        nodes.extend([pad, aligned_q, aligned_left, aligned_right, node, crop])
        count += 1
    # Drop replaced quantizers/weights that no longer contribute to the output.
    needed = {output.name for output in model.graph.output}
    retained_nodes = []
    for node in reversed(nodes):
        if any(output in needed for output in node.output):
            retained_nodes.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(retained_nodes))
    retained_weights = [value for value in model.graph.initializer if value.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_weights)
    return count


def standardize_onnx(path):
    import onnx
    model = onnx.load(path)
    for node in model.graph.node:
        if node.domain == 'trt' and node.op_type in ('QuantizeLinear', 'DequantizeLinear'):
            node.domain = ''
        elif node.domain:
            raise ValueError(f'Unexpected custom operator: {node.domain}:{node.op_type}')
        # TensorRT RTX 1.6 mishandles BF16 Clip with an omitted upper bound.
        # Max(x, lower_bound) expresses the same clamp without that path.
        if (node.op_type == 'Clip' and len(node.input) >= 2 and node.input[1]
                and (len(node.input) == 2 or not node.input[2])):
            node.op_type = 'Max'
            del node.input[2:]
    # BF16 Conv requires opset 22. The original TRT Q/DQ intermediates may be
    # tagged UINT8: infer their actual types from the FLOAT8 zero points.
    del model.opset_import[:]
    model.opset_import.extend([onnx.helper.make_opsetid('', 22)])
    del model.graph.value_info[:]
    print(f'Aligned {align_fp8_matrices(model)} FP8 matrix operations', flush=True)
    onnx.checker.check_model(model, full_check=True)
    if not any(t.data_type == onnx.TensorProto.FLOAT8E4M3FN for t in model.graph.initializer):
        raise ValueError('Export has no FP8 tensors')
    onnx.save_model(model, path, save_as_external_data=False)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError('Export exceeds Magpie\'s model size limit; use a smaller tile')


def image_crops(paths, tile, input_size):
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    calibration, validation = [], []
    for path in paths:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            if input_size:
                image = image.resize(tuple(input_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32) / 255
        pixels = np.pad(pixels, ((0, max(0, tile-pixels.shape[0])),
                                (0, max(0, tile-pixels.shape[1])), (0, 0)), mode='edge')
        for index, (y, x) in enumerate((y, x) for y in (0, .5, 1) for x in (0, .5, 1)):
            left, top = int((pixels.shape[1]-tile)*x), int((pixels.shape[0]-tile)*y)
            crop = torch.from_numpy(pixels[top:top+tile, left:left+tile].transpose(2, 0, 1).copy()).unsqueeze(0)
            (validation if index in (1, 7) else calibration).append(crop)
    if not calibration or not validation:
        raise ValueError('At least one calibration image is required')
    return calibration, validation


def export(args):
    import importlib.metadata
    import numpy as np
    import spandrel
    import torch
    import modelopt.torch.quantization as mtq
    from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata

    if importlib.metadata.version('nvidia-modelopt') != '0.47.0':
        raise ValueError('Install the pinned requirements-sr-fp8.txt dependencies')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Calibration requires a CUDA GPU supporting BF16')
    manifest = json.loads((ROOT / 'scripts/sr_models.json').read_text())
    original = next(m for m in manifest['models'] if m['id'] == 'IllustrationJaNai-DAT2')
    if digest(args.checkpoint) != original['checkpointSha256']:
        raise ValueError('DAT2 checkpoint checksum mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(41)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = spandrel.ModelLoader().load_from_state_dict(state).eval().model.cuda().to(torch.bfloat16)
    cache_attention_masks(model, args.tile_size)
    calibration, validation = image_crops(args.calibration_image, args.tile_size, args.input_size)
    # Spread bounded samples over all supplied images.
    choose = lambda items, count: [items[i] for i in np.linspace(0, len(items)-1, min(count, len(items)), dtype=int)]
    calibration = choose(calibration, args.calibration_samples)
    validation = choose(validation, 4)
    with torch.inference_mode():
        references = [model(c.cuda().to(torch.bfloat16)).float().cpu().numpy() for c in validation]
    config = deepcopy(mtq.FP8_DEFAULT_CFG)
    config['quant_cfg'].append({'quantizer_name': '*', 'parent_class': 'nn.Conv2d', 'enable': False})
    for pattern in ('conv_first.*', 'conv_last.*', 'upsample.*', '*.attns.*.pos.*',
                    '*.channel_interaction.*', '*.spatial_interaction.*'):
        config['quant_cfg'].append({'quantizer_name': pattern, 'enable': False})

    def calibrate(network):
        with torch.inference_mode():
            for index, crop in enumerate(calibration):
                output = network(crop.cuda().to(torch.bfloat16))
                if not bool(torch.isfinite(output).all()):
                    raise ValueError('Non-finite calibration output')
                print(f'Calibrated {index+1}/{len(calibration)}', flush=True)

    model = mtq.quantize(model, config, forward_loop=calibrate)
    scores = []
    with torch.inference_mode():
        for crop, reference in zip(validation, references):
            output = model(crop.cuda().to(torch.bfloat16)).float().cpu().numpy()
            if not np.isfinite(reference).all() or not np.isfinite(output).all():
                raise ValueError('Non-finite validation output')
            mse = float(np.mean((output-reference)**2))
            psnr = float(-10*np.log10(mse)) if mse else 100.0
            scores.append(psnr)
            print(f'FP8 simulation versus BF16 reference: {psnr:.2f} dB', flush=True)
    if min(scores) < 40:
        raise ValueError('FP8 validation fell below 40 dB; supply more representative calibration images')

    class FloatIO(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.model = network

        def forward(self, image):
            return self.model(image.to(torch.bfloat16)).float()

    args.cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.cache, prefix='dat2-fp8-') as working:
        with torch.inference_mode(), preserve_export_precision():
            data, _ = get_onnx_bytes_and_metadata(
                FloatIO(model).eval(), (calibration[0].cuda(),),
                model_name='IllustrationJaNai-DAT2-FP8', weights_dtype='bf16',
                onnx_opset=20, dynamo_export=False)
        OnnxBytes.from_bytes(data).write_to_disk(working, clean_dir=False)
        weights = Path(working) / 'IllustrationJaNai-DAT2-FP8.onnx'
        standardize_onnx(weights)
        target = args.effects_dir / 'ONNX'
        target.mkdir(parents=True, exist_ok=True)
        descriptor = {k: original[k] for k in ('scale', 'overlap', 'author', 'license', 'source')}
        descriptor.update(version=1, name='IllustrationJaNai DAT2 FP8/BF16 (NVIDIA)',
                          file=weights.name, backend='tensorrt-rtx', tileSize=args.tile_size,
                          sha256=digest(weights), precision='FP8 E4M3FN / BF16')
        temporary = target / (weights.name + '.partial')
        shutil.copyfile(weights, temporary)
        temporary.replace(target / weights.name)
        (target / 'IllustrationJaNai-DAT2-FP8.onnx.json').write_text(
            json.dumps(descriptor, indent=2)+'\n', encoding='utf-8')
        for name in ('README.txt', 'LICENSE-CC-BY-NC-SA-4.0.txt'):
            source = ROOT / 'src/Effects/ONNX' / name
            if source.resolve() != (target/name).resolve():
                shutil.copyfile(source, target/name)
    print(f'Installed FP8/BF16 DAT2 into {target}. Requires the optional TensorRT RTX runtime.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration-image', type=Path, action='append', required=True)
    parser.add_argument('--calibration-samples', type=int, default=14)
    parser.add_argument('--input-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--tile-size', type=int, choices=(256, 512), default=512)
    parser.add_argument('--effects-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path(tempfile.gettempdir())/'magpie-fp8-export')
    args = parser.parse_args()
    if not 2 <= args.calibration_samples <= 64 or (args.input_size and min(args.input_size) <= 0):
        parser.error('Use 2-64 calibration samples and positive image dimensions')
    export(args)


if __name__ == '__main__':
    main()
