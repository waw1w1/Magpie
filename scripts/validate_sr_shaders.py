from pathlib import Path
import re, subprocess, concurrent.futures
import argparse
parser = argparse.ArgumentParser(description='Compile new shader passes with DXC; this does not replace Windows CS5 validation.')
parser.add_argument('--dxc', default='dxc')
args = parser.parse_args()
root=Path(__file__).resolve().parents[1]
dxc=args.dxc
out=root/'obj/shader-check';out.mkdir(parents=True,exist_ok=True)
files=[*root.glob('src/Effects/ACNet2/*.hlsl'),*root.glob('src/Effects/ARNet/*.hlsl'),*root.glob('src/Effects/ArtCNN/*.hlsl'),*root.glob('src/Effects/LeRF/*.hlsl')]
builtins=(root/'src/Effects/StubDefs.hlsli').read_text()
builtins=re.sub(r'uint2 GetInputSize\(\).*?\n','uint2 GetInputSize() { return sizes.xy; }\n',builtins)
builtins=re.sub(r'uint2 GetOutputSize\(\).*?\n','uint2 GetOutputSize() { return sizes.zw; }\n',builtins)
builtins=re.sub(r'float2 GetInputPt\(\).*?\n','float2 GetInputPt() { return 1.0 / sizes.xy; }\n',builtins)
builtins=re.sub(r'float2 GetOutputPt\(\).*?\n','float2 GetOutputPt() { return 1.0 / sizes.zw; }\n',builtins)
builtins=re.sub(r'float2 GetScale\(\).*?\n','float2 GetScale() { return float2(sizes.zw) / sizes.xy; }\n',builtins)
builtins='cbuffer CB : register(b0) { uint4 sizes; };\n'+builtins
jobs=[]
for f in files:
 text=f.read_text();parts=re.split(r'(?=//!PASS \d)',text);head=parts[0]
 common=head.split('//!COMMON\n',1)[1] if '//!COMMON\n' in head else ''
 samplers=re.findall(r'SamplerState (\w+);',head)
 for part in parts[1:]:
  n=int(re.search(r'//!PASS (\d+)',part)[1]); ins=re.search(r'//!IN ([^\n]+)',part)[1].split(', ');outs=re.search(r'//!OUT ([^\n]+)',part)[1].split(', ')
  assert not (set(ins)&set(outs)),(f,n,'alias')
  src=builtins+'\n'+'\n'.join(f'Texture2D<float4> {t} : register(t{i});' for i,t in enumerate(ins))+'\n'+'\n'.join(f'RWTexture2D<float4> {t} : register(u{i});' for i,t in enumerate(outs))+'\n'+'\n'.join(f'SamplerState {t} : register(s{i});' for i,t in enumerate(samplers))+'\n'+common+'\n'+part
  if '//!STYLE PS' in part:
   src+=f'\n[numthreads(8,8,1)] void main(uint3 p : SV_DispatchThreadID) {{ if(all(p.xy < GetOutputSize())) {outs[0]}[p.xy]=Pass{n}((p.xy+0.5)*GetOutputPt()); }}'
  else:
   src+=f'\n[numthreads(64,1,1)] void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {{ Pass{n}(g.xy*8,t); }}'
  for fp16 in [False,True]:
   check=out/f'{f.stem}-{n}-{int(fp16)}.hlsl';check.write_text(src.replace('#define MF float','#define MF min16float').replace('#define MF4 float4','#define MF4 min16float4').replace('#define MF4x4 float4x4','#define MF4x4 min16float4x4') if fp16 else src)
   jobs.append(check)
def compile(p):
 r=subprocess.run([dxc,'-T','cs_6_0','-E','main','-I',str(root/'src/Effects/LeRF'),str(p),'-Fo',str(p.with_suffix('.dxil'))],capture_output=True,text=True)
 if r.returncode: raise RuntimeError(str(p)+'\n'+r.stdout+r.stderr)
 return p
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for _ in pool.map(compile,jobs): pass
print(f'PASS: {len(jobs)} pass/precision variants, DXC CS6 syntax check (not Windows CS5 runtime verification)')
