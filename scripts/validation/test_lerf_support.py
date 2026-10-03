"""CPU checks of LeRF's HLSL support arithmetic; not a GPU execution test."""
from pathlib import Path
import re
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / 'src/Effects/LeRF/LeRFCommon.hlsli').read_text()
BODY = re.search(r'void LeRFSupport\([^{}]+\)\s*\{([^{}]+)\}', SOURCE)[1]


def support(pos, input_size, output_size, float_divide=None):
    # Evaluate the production helper's scalar/vector expressions, preserving its
    # explicit casts. Positive integer division is truncated by the int2 cast.
    env = {'pos': np.asarray(pos, np.float32),
           'GetInputSize': lambda: np.asarray(input_size, np.uint32),
           'GetOutputSize': lambda: np.asarray(output_size, np.uint32),
           'uint2': lambda x: np.asarray(x, np.uint32),
           'int2': lambda x: np.asarray(x, np.int32),
           'int': lambda x: np.asarray(x, np.int32),
           'float2': lambda x: np.asarray(x, np.float32)}
    for line in BODY.splitlines():
        line = line.split('//', 1)[0].strip().rstrip(';')
        if not line:
            continue
        line = re.sub(r'^(uint2|int2|float2) ', '', line)
        name, expression = line.split(' = ', 1)
        expression = re.sub(r'(\d+)u\b', r'\1', expression)
        expression = re.sub(r'(\w+)\.([xy])\b',
                            lambda m: f'{m[1]}[..., {"xy".index(m[2])}]', expression)
        def evaluate(value):
            return eval(value, {'__builtins__': {}}, env)
        if '?' in expression:
            condition, values = expression.split('?', 1)
            yes, no = values.split(':', 1)
            value = np.where(evaluate(condition), evaluate(yes), evaluate(no))
        elif float_divide is not None and name == 'offset' and ' / ' in expression:
            numerator, denominator = expression.split(' / ', 1)
            value = float_divide(evaluate(numerator), evaluate(denominator))
        else:
            value = evaluate(expression)
        if '.' in name:
            vector, component = name.split('.')
            env[vector][..., 'xy'.index(component)] = value
        else:
            env[name] = value
    return env['first'], env['offset']


def positions(width, bottom=False):
    # Match the PS wrapper's TL/TR/BR/BL normalized-position stepping.
    x = np.arange(width, dtype=np.int32)
    base = (x // 16 * 16 + x % 8).astype(np.float32)
    pt = np.float32(1 / width)
    left = (base + np.float32(.5)) * pt
    right = left + np.float32(8) * pt
    px = np.where(x % 16 >= 8, right,
                  right - np.float32(8) * pt if bottom else left)
    return np.stack([px, np.full(width, (8.5 if bottom else .5) / 16, np.float32)], axis=-1)


class LeRFSupportTests(unittest.TestCase):
    def test_two_dimensional_gaussian_endpoint(self):
        # Bottom-left PS quadrant; the exact projection is (50, 23/3).
        size = np.array([573, 27], np.float32)
        point = np.array([151, 24], np.int32)
        base = (point // 16 * 16 + point % 8).astype(np.float32)
        pt = np.float32(1) / size
        pos = (base + np.float32(.5)) * pt
        pos[0] = (pos[0] + 8*pt[0]) - 8*pt[0]
        pos[1] += 8*pt[1]
        first, offset = support(pos, [191, 9], [573, 27])
        np.testing.assert_array_equal(first, [49, 7])
        np.testing.assert_allclose(offset, [1, 2/3], rtol=0, atol=3e-8)
        # Gaussian rho=0, sigma_x=sigma_y=0.1. A retained endpoint has
        # nonzero weight; a one-pixel neighborhood shift changes this ramp.
        total = np.float32(0)
        value = np.float32(0)
        for y in (0, 1):
            for x in (0, 1):
                distance = offset - np.array([x, y], np.float32)
                weight = np.exp(np.float32(-.5)*np.sum(distance*distance))
                total += weight
                value += weight*np.float32((first[0]+x-49)*.5)
        self.assertAlmostEqual(float(value/total), .5/(1+np.exp(-.5)), places=6)

    def test_approximate_reciprocal_keeps_exact_endpoint(self):
        # This reciprocal is within D3D11's allowed 1 ULP error, but 50*r > 1.
        reciprocal = np.nextafter(np.float32(1/50), np.float32(np.inf))
        self.assertGreater(np.float32(50)*reciprocal, 1)
        first, offset = support([[.06, .06]], [25, 25], [25, 25],
                                lambda numerator, denominator: numerator*reciprocal)
        np.testing.assert_array_equal(first, [[0, 0]])
        np.testing.assert_array_equal(offset, [[1, 1]])

    def test_shader_wiring(self):
        for name in ('LeRF-L', 'LeRF-G'):
            text = (ROOT / f'src/Effects/LeRF/{name}.hlsl').read_text()
            self.assertIn('LeRFSupport(pos, first, offset);', text)
            self.assertIn('float2 distance = offset - float2(x, y);', text)
            self.assertNotIn('ceil(projected', text)

    def test_exact_neighborhood_and_local_distances(self):
        sizes = list(range(1, 65)) + [127, 191, 255, 320, 511, 1023, 1920, 4095, 8191, 16383, 16384]
        for source in sizes:
            for width in sorted({source, min(2*source, 16384), min(3*source, 16384), min(2*source+1, 16384)}):
                x = np.arange(width, dtype=np.int64)
                numerator = (2*x+1)*source-width
                expected_first = -np.floor_divide(-numerator, 2*width)-1
                expected_offset = (numerator-expected_first*(2*width))/(2*width)
                for bottom in (False, True):
                    first, offset = support(positions(width, bottom), [source, 16], [width, 16])
                    np.testing.assert_array_equal(first[:, 0], expected_first)
                    np.testing.assert_allclose(offset[:, 0], expected_offset, rtol=0, atol=6e-8)
                    np.testing.assert_array_equal(first[:, 1], 7 if bottom else -1)
                    np.testing.assert_array_equal(offset[:, 1], 1)
                    self.assertTrue(np.all((offset > 0) & (offset <= 1)))

    def test_learned_boundary_weight_is_not_discarded(self):
        first, offset = support([[np.float32(1.5)*np.float32(1/7), .5]], [7, 1], [7, 1])
        np.testing.assert_array_equal(first[0], [0, -1])
        np.testing.assert_array_equal(offset[0], [1, 1])
        # Real DDS prefilter/parameter values from the 1x7 regression fixture.
        prefilter = np.array([4, 255], np.float32) / np.float32(255)
        alpha = np.array([222, 250], np.float32) / np.float32(255)*2-1
        value = np.float32(0)
        total = np.float32(0)
        for y in (0, 1):
            for x in (0, 1):
                distance = offset[0] - np.array([x, y], np.float32)
                weight = np.prod(np.maximum(1-alpha[x]*abs(distance), 0))
                if np.any(abs(distance) > 1):
                    weight = np.float32(0)
                value += weight*prefilter[x]
                total += weight
        self.assertEqual(float(np.rint(value/total*255)), 195)


if __name__ == '__main__':
    unittest.main()
