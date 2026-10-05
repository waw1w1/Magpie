"""Install the optional NVIDIA TensorRT RTX execution provider beside Magpie.exe.

The official Windows x64 wheel contains the plugin, TensorRT RTX and CUDA
runtime DLLs. Python bindings and a system CUDA Toolkit are not needed by Magpie.
These optional NVIDIA components retain their separate licenses/notices.
"""
import argparse
from pathlib import Path
import struct
import tempfile
import zipfile

from install_sr_models import verified_download

URL = ('https://files.pythonhosted.org/packages/0c/7d/'
       '7dae1810328c4ddc611129b66fa11edd14c8c814ac1b206c3fac481541af/'
       'onnxruntime_ep_nv_tensorrt_rtx_cu13-0.4.0-py3-none-win_amd64.whl')
SHA256 = '8aa0db63e31384024409818789eaa8ad136f0e83681dda204b7e3d178df2ff69'
FILES = (
    'onnxruntime_providers_nv_tensorrt_rtx.dll',
    'tensorrt_rtx_1_6.dll',
    'tensorrt_onnxparser_rtx_1_6.dll',
    'tensorrt_plugins.dll',
    'cudart64_13.dll',
    'TRT_RTX_Acknowledgements.txt',
    'TRT_RTX_README.txt',
)


def install(app_dir, cache):
    executable = app_dir / 'Magpie.exe'
    with executable.open('rb') as source:
        source.seek(0x3c)
        pe_offset = struct.unpack('<I', source.read(4))[0]
        source.seek(pe_offset)
        if source.read(4) != b'PE\0\0' or struct.unpack('<H', source.read(2))[0] != 0x8664:
            raise ValueError('This optional TensorRT RTX package requires Windows x64 Magpie')
    archive = cache / 'tensorrt-rtx-ep-0.4.0-win-x64.whl'
    verified_download(URL, archive, SHA256)
    with zipfile.ZipFile(archive) as wheel:
        for name in FILES:
            # Whitelist exact members; never extract arbitrary archive paths.
            data = wheel.read('onnxruntime_ep_nv_tensorrt_rtx/' + name)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=app_dir, prefix=name + '.', suffix='.tmp', delete=False) as output:
                    temporary = Path(output.name)
                    output.write(data)
                temporary.replace(app_dir / name)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
    print(f'Installed TensorRT RTX EP 0.4.0 into {app_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path(tempfile.gettempdir()) / 'magpie-runtime-cache')
    args = parser.parse_args()
    install(args.app_dir.resolve(), args.cache.resolve())


if __name__ == '__main__':
    main()
