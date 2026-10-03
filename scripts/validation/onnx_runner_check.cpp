#include "../../src/Magpie.Core/OnnxModelRunner.h"
#include <onnxruntime_c_api.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <string_view>
int main(int argc, char** argv) {
 if(argc==2 && std::string_view(argv[1])=="--ort-version") {
  std::cout<<OrtGetApiBase()->GetVersionString()<<'\n';return 0;
 }
 if(argc!=9)return 2;
 try {
  Magpie::OnnxModelDesc desc;desc.scale=std::stoul(argv[2]);desc.tileSize=std::stoul(argv[3]);desc.overlap=std::stoul(argv[4]);
  auto width=std::stoul(argv[5]),height=std::stoul(argv[6]);std::vector<float> input(width*height*3);
  std::ifstream f(argv[7],std::ios::binary);if(!f.read(reinterpret_cast<char*>(input.data()),input.size()*4))return 3;
  Magpie::OnnxModelRunner runner(argv[1],desc,-1);auto start=std::chrono::steady_clock::now();auto output=runner.Run(input,width,height);
  std::ofstream out(argv[8],std::ios::binary);out.write(reinterpret_cast<const char*>(output.data()),output.size()*4);
  std::cout<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" seconds\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
