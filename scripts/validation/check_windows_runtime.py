"""Verify app-local VC runtime dependencies without loading Windows DLLs.

Uses the actual PE import tables, so x64/ARM64 builds and changes in the ORT
package cannot silently introduce another missing runtime DLL.
"""
import argparse
from pathlib import Path
import struct


def imports(path):
    data = path.read_bytes()
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    if data[pe:pe+4] != b'PE\0\0':
        raise ValueError(f'Not a PE file: {path}')
    machine, sections = struct.unpack_from('<HH', data, pe+4)
    optional_size = struct.unpack_from('<H', data, pe+20)[0]
    optional = pe+24
    magic = struct.unpack_from('<H', data, optional)[0]
    if magic not in (0x10b, 0x20b):
        raise ValueError(f'Unsupported PE optional header: {path}')
    directories = optional+(112 if magic == 0x20b else 96)
    section_table = optional+optional_size

    def offset(rva):
        for i in range(sections):
            virtual_size, start, raw_size, raw_start = struct.unpack_from('<IIII', data, section_table+i*40+8)
            if start <= rva < start+max(virtual_size, raw_size):
                return raw_start+rva-start
        raise ValueError(f'Invalid PE RVA in {path}: {rva}')

    dependencies = set()
    for directory, stride, name_offset in ((1,20,12),(13,32,4)):
        rva, size = struct.unpack_from('<II', data, directories+directory*8)
        if not rva:
            continue
        entry = offset(rva)
        for cursor in range(entry, entry+size, stride):
            if not any(data[cursor:cursor+stride]):
                break
            # Modern MSVC delay-import descriptors also use RVAs.
            if directory == 13 and struct.unpack_from('<I',data,cursor)[0] != 1:
                raise ValueError(f'Unsupported delay import addressing in {path}')
            name_rva = struct.unpack_from('<I',data,cursor+name_offset)[0]
            name = offset(name_rva)
            dependencies.add(data[name:data.index(b'\0',name)].decode('ascii').lower())
    return machine, dependencies


def check(folder):
    executable = folder/'Magpie.exe'
    if not executable.is_file():
        raise ValueError('Missing application executable: Magpie.exe')
    expected_machine, _ = imports(executable)
    if expected_machine not in (0x8664, 0xaa64):
        raise ValueError(f'Unsupported application architecture: {expected_machine:#x}')
    files = {p.name.lower():p for p in folder.glob('*.dll')}
    pending = ['onnxruntime.dll','onnxruntime_providers_shared.dll','directml.dll']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if name not in files:
            raise ValueError(f'Missing app-local runtime dependency: {name}')
        machine, dependencies = imports(files[name])
        if machine != expected_machine:
            raise ValueError(f'Runtime architecture mismatch: {name}')
        pending.extend(d for d in dependencies if d.startswith(('msvcp','vcruntime','concrt')))
    print('PASS: Windows runtime import closure:',', '.join(sorted(visited)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    check(parser.parse_args().folder)
