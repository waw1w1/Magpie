"""Regression fixtures for missing transitive CRTs and mixed architectures."""
import contextlib
import io
from pathlib import Path
import struct
import tempfile
import unittest

from check_windows_runtime import check


def write_pe(path, dependencies=(), machine=0x8664):
    data = bytearray(2048)
    optional_size = 224 if machine == 0x014c else 240
    directory_offset = 96 if machine == 0x014c else 112
    struct.pack_into('<I',data,0x3c,0x80)
    data[0x80:0x84]=b'PE\0\0'
    struct.pack_into('<HH',data,0x84,machine,1)
    struct.pack_into('<H',data,0x94,optional_size)
    struct.pack_into('<H',data,0x98,0x10b if machine == 0x014c else 0x20b)
    struct.pack_into('<IIII',data,0x98+optional_size+8,0x600,0x1000,0x600,0x200)
    struct.pack_into('<II',data,0x98+directory_offset+8,0x1000,(len(dependencies)+1)*20)
    name_offset=0x200+(len(dependencies)+1)*20
    for i,name in enumerate(dependencies):
        struct.pack_into('<I',data,0x200+i*20+12,0x1000+name_offset-0x200)
        encoded=name.encode('ascii')+b'\0'
        data[name_offset:name_offset+len(encoded)]=encoded
        name_offset+=len(encoded)
    path.write_bytes(data)


class RuntimeChecks(unittest.TestCase):
    def test_dependency_closure_and_architecture(self):
        for arch in (0x8664,0xaa64):
            with tempfile.TemporaryDirectory() as temp:
                folder=Path(temp)
                write_pe(folder/'Magpie.exe',machine=arch)
                write_pe(folder/'onnxruntime.dll',['MSVCP140_1.dll'],arch)
                write_pe(folder/'onnxruntime_providers_shared.dll',machine=arch)
                write_pe(folder/'DirectML.dll',machine=arch)
                with self.assertRaisesRegex(ValueError,'msvcp140_1.dll'):
                    check(folder)
                write_pe(folder/'msvcp140_1.dll',['VCRUNTIME140.dll'],arch)
                with self.assertRaisesRegex(ValueError,'vcruntime140.dll'):
                    check(folder)
                write_pe(folder/'vcruntime140.dll',machine=arch)
                with contextlib.redirect_stdout(io.StringIO()):
                    check(folder)
                write_pe(folder/'vcruntime140.dll',machine=0xaa64 if arch==0x8664 else 0x8664)
                with self.assertRaisesRegex(ValueError,'architecture mismatch'):
                    check(folder)

    def test_application_architecture(self):
        for app_arch, runtime_arch in ((0x8664,0xaa64),(0xaa64,0x8664),(0x014c,0x014c)):
            with self.subTest(app_arch=app_arch, runtime_arch=runtime_arch), tempfile.TemporaryDirectory() as temp:
                folder=Path(temp)
                write_pe(folder/'Magpie.exe',machine=app_arch)
                for name in ('onnxruntime.dll','onnxruntime_providers_shared.dll','DirectML.dll'):
                    write_pe(folder/name,machine=runtime_arch)
                error = 'Unsupported application architecture' if app_arch == 0x014c else 'architecture mismatch'
                with self.assertRaisesRegex(ValueError,error):
                    check(folder)

    def test_missing_application(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in ('onnxruntime.dll','onnxruntime_providers_shared.dll','DirectML.dll'):
                write_pe(folder/name)
            with self.assertRaisesRegex(ValueError,'Missing application executable: Magpie.exe'):
                check(folder)


if __name__=='__main__':
    unittest.main()
