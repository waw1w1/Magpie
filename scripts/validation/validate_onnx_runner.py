from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, subprocess, time
from onnx import helper as h, TensorProto as T
import argparse
parser=argparse.ArgumentParser(description='Compare the portable C++ tile runner against independent ONNX reference inference.')
parser.add_argument('--models',type=Path,required=True)
parser.add_argument('--runner',type=Path,required=True)
parser.add_argument('--synthetic-only',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]/'obj'
root.mkdir(exist_ok=True)
heavy=args.models
opt=ort.SessionOptions();opt.intra_op_num_threads=2
# A deterministic model verifies stitching, channel order, odd sizes and edges.
model=h.make_model(h.make_graph([h.make_node('Resize',['input','','scales'],['output'],mode='nearest',coordinate_transformation_mode='asymmetric')],'nearest',[h.make_tensor_value_info('input',T.FLOAT,[1,3,'H','W'])],[h.make_tensor_value_info('output',T.FLOAT,[1,3,'OH','OW'])],[h.make_tensor('scales',T.FLOAT,[4],[1,1,2,2])]),opset_imports=[h.make_opsetid('',17)]);model.ir_version=10
onnx.save(model,root/'nearest-test.onnx')
models=[(root/'nearest-test.onnx',2,tile,border,w,h) for tile,border,w,h in [(32,8,43,39),(16,0,1,1),(32,1,33,1),(16,7,17,11)]]
if not args.synthetic_only:
 models += [(heavy/'2x-AnimeSharpV4_Fast_RCAN_PU_fp16_opset17.onnx',2,128,32,67,3),(heavy/'2x-AnimeSharpV4_RCAN_fp16_op17.onnx',2,128,32,67,3),(heavy/'IllustrationJaNai-DAT2.onnx',4,128,32,67,3)]
for path,scale,tile,border,width,height in models:
 rng=np.random.default_rng(13);image=rng.random((height,width,3),dtype=np.float32);image[:,:3]=[1,0,0];image.tofile(root/'model-input.f32')
 before=time.monotonic();p=subprocess.run([str(args.runner.resolve()),str(path),str(scale),str(tile),str(border),str(width),str(height),str(root/'model-input.f32'),str(root/'model-output.f32')],check=True,capture_output=True,text=True);print(path.name,p.stdout.strip(),flush=True)
 actual=np.fromfile(root/'model-output.f32',np.float32).reshape(height*scale,width*scale,3)
 if path.name.startswith('nearest'):
  expected=np.repeat(np.repeat(image,scale,axis=0),scale,axis=1)
 else:
  session=ort.InferenceSession(str(path),opt,providers=['CPUExecutionProvider']);dtype=np.float16 if session.get_inputs()[0].type=='tensor(float16)' else np.float32
  expected=np.zeros_like(actual);weights=np.zeros(actual.shape[:2],np.float32);stride=tile-2*border;feather=border//2
  # Independent reference implementation explicitly pads and slices the frame.
  padded=np.pad(image,((border,border+tile),(border,border+tile),(0,0)),mode='edge')
  for y in range(0,height,stride):
   for x in range(0,width,stride):
    patch=padded[y:y+tile,x:x+tile].transpose(2,0,1)[None].astype(dtype)
    output=session.run(None,{session.get_inputs()[0].name:patch})[0][0].transpose(1,2,0).astype(np.float32)
    y0=max(0,y-feather);x0=max(0,x-feather);y1=min(height,y+stride+feather);x1=min(width,x+stride+feather)
    yy=(np.arange(y0*scale,y1*scale,dtype=np.float32)+.5)/scale-y
    xx=(np.arange(x0*scale,x1*scale,dtype=np.float32)+.5)/scale-x
    wy=np.clip(np.minimum(yy+feather,stride+feather-yy)/(2*feather),0,1)
    wx=np.clip(np.minimum(xx+feather,stride+feather-xx)/(2*feather),0,1)
    weight=wy[:,None]*wx[None,:]
    patch=output[(y0-y+border)*scale:(y1-y+border)*scale,(x0-x+border)*scale:(x1-x+border)*scale]
    expected[y0*scale:y1*scale,x0*scale:x1*scale]+=patch*weight[...,None]
    weights[y0*scale:y1*scale,x0*scale:x1*scale]+=weight
  assert np.all(weights>0)
  expected/=weights[...,None]
 error=np.max(abs(actual-expected));print('max absolute error',error,'finite',np.isfinite(actual).all(),flush=True);assert error<1e-6
# Wrong scale must fail rather than publish an incorrectly shaped texture.
np.zeros((3,67,3),np.float32).tofile(root/'invalid-model-input.f32')
r=subprocess.run([str(args.runner.resolve()),str(root/'nearest-test.onnx'),'4','32','8','67','3',str(root/'invalid-model-input.f32'),str(root/'bad-output.f32')],capture_output=True,text=True);assert r.returncode==1;print('mismatched output scale rejected:',r.stderr.strip(),flush=True)
