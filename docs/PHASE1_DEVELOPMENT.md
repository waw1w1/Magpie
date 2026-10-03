# 第一阶段本地开发记录

状态：代码及 Linux 上可执行的检查已完成，**尚未通过 Windows 构建和显卡验收，不能作为已验收版本发布**。工作分支为本地 `dev`；不推送、不运行需要上传代码的云端 CI，`main` 保持原提交。

## 已实现的范围

| 新增效果 | 倍率／用途 | 实现 |
| --- | --- | --- |
| ACNet-F8B4、ACNet-F8B8 | 2×，轻量二次元放大 | 内置权重的 D3D11 着色器 |
| ARNet-F8B8 | 2×，残差网络 | 内置权重的 D3D11 着色器 |
| ArtCNN-C4F16、ArtCNN-C4F32 | 2×，中性亮度重建 | 内置权重的 D3D11 着色器 |
| LeRF-L、LeRF-G | 任意放大比例，学习型重采样 | 作者 LUT + 三个着色器阶段 |
| AnimeSharp V4、AnimeSharp V4 Fast | 2× RGB，较高画质候选 | ONNX Runtime / DirectML |
| IllustrationJaNai V1 DAT2 | 4× RGB，插画／CG 画质候选 | 固定输入尺寸的 ONNX 导出 / DirectML |

ACNet、ARNet、ArtCNN 重建亮度并保留双线性插值的色度；这与 RGB 重建模型有区别。ArtCNN 采用作者 ONNX 中的零填充卷积；ACNet、ARNet 保留 GLSL 的边缘钳制。LeRF 保留 LUT 的 int8 量化、四方向集成和作者评估脚本的 support size 2；图像零填充、参数边缘钳制。缩小时 LeRF 返回双线性结果，不将未实现的抗混叠缩小称为 LeRF。

“缩放模式 → 添加效果”分成“推荐模型”和“非推荐模型”。新效果和选定的现有 CuNNy2、Ani4Kv2 ArtCNN、k7、RAVU、FSR、NIS、SGSR 及基础采样算法进入推荐组。唯一名单在 `src/Magpie/RecommendedEffects.h`，其余效果继续可用。现有 ID 和配置引用不变。“非推荐”表示不作为当前默认选择，不表示在所有输入上都更差。

文字／UI 自动分割属于第二阶段，本次未实现；本阶段不能解决所有文字扭曲。

## 重型模型的安装和运行

Windows 构建新增 NuGet 依赖：`Microsoft.ML.OnnxRuntime.DirectML 1.24.4`、`Microsoft.AI.DirectML 1.15.4`，支持 x64 和 ARM64。`src/OnnxRuntime.props` 配置头文件、链接、运行库复制和部署内容。完整 Windows C++／WinRT 工具链仍沿用仓库构建说明。

模型权重独立安装，不提交大体积 ONNX 文件。使用 Python 3.12：

```sh
python -m pip install -r scripts/requirements-sr-models.txt
python scripts/install_sr_models.py --effects-dir <编译输出目录>/effects
```

可用 `--model AnimeSharpV4-Fast`、`--model AnimeSharpV4` 或 `--model IllustrationJaNai-DAT2` 只安装指定模型。仅下载 AnimeSharp ONNX 时，安装脚本不加载 PyTorch 等导出依赖。缓存位于 `obj/sr-models`，安装到编译后的效果目录。

下载验证 SHA256；DAT2 额外验证原始 checkpoint、导出图和 PyTorch／ONNX 数值一致性。安装后的 `.onnx.json` 记录精确哈希，程序加载权重时再次校验。缺失或错误的权重会显示加载失败，不会用其他算法冒充选中的模型。输入／输出仅支持单张 NCHW RGB、FP32 或 FP16，并验证倍率和维度。

DirectML 匹配 Magpie 当前选用显卡的 LUID。ORT 可把形状／控制算子安排在 CPU；创建 DirectML 失败会报错。初版通过 CPU 内存连接 D3D11 和模型推理，存在整帧读回及上传开销。每块网络输入为 128×128，每边保留 32 像素上下文，取中心区域拼接；可在块间响应停止请求。

**RCAN 的通道注意力和 DAT2 的注意力使分块结果不等同于整图推理，重叠不能保证完全消除接缝。** 这些模型需要实际 CG、渐变和细线画面验收，也没有“实时 60 FPS”承诺。当前 D3D11 性能面板不足以独立衡量整个 DirectML 路径；日志另外记录首帧读回、推理、上传的墙钟时间。首帧数值不代表稳定帧率。

## 来源及许可

- ACNet／ARNet：`TianZerL/ACNetGLSL`，提交 `c7d2d8dbb5364c550e5bfb738860fc9bcc0ea424`，MIT。
- ArtCNN：`Artoriuz/ArtCNN`，提交 `7d6955141b88047a983e3a424e87c5a5e3df3e3c`，MIT；保留 Joao Chrisostomo 和 Kacper Michajłow 的署名。
- LeRF：`ddlee-cn/LeRF-PyTorch`，提交 `8099191508de171c0d7c16fb42a11c5848bb94d0`，MIT。
- AnimeSharp V4／Fast：Kim2091 官方 release；IllustrationJaNai V1 DAT2：the-database 官方模型链接。三者均为 **CC BY-NC-SA 4.0**，有署名、非商业、相同方式共享条件，不被 Magpie 的 GPL 许可替代。

源码／权重链接与哈希分别在 `scripts/neural_sources.json`、`scripts/lerf_sources.json`、`scripts/sr_models.json`；许可证和运行库第三方声明随效果目录保留。固定版本的生成脚本可重新生成 CNN 着色器和 LUT DDS。

## 本地验证结果

环境：Linux、Python 3.12、CPU PyTorch 2.14.1、ONNX Runtime CPU 1.30.0、GCC、DXC 1.9.2609.5。CPU ORT 用于独立检查；Windows 应用依赖的是前述 1.24.4 DirectML 包，两者不是同一执行环境。

| 检查 | 结果／边界 |
| --- | --- |
| 七个着色器的 120 个 pass／精度组合 | DXC CS6 语法编译通过；不等同于 Magpie 的 Windows CS5 编译 |
| 五个 CNN 的 CPU 数学解释与作者 GLSL／ONNX | 随机输入及边角脉冲测试，最大绝对误差 < 6.3×10⁻⁷；未验证实际 GPU FP16 误差 |
| LeRF 18 张表、全部旋转、边缘和量化边界 | 四维单纯形插值与作者参考完全相同 |
| LeRF 重采样 | 1×、2×、非整数且各向异性的尺寸与参考一致；最大误差 < 0.00012 个 8-bit 灰阶单位 |
| C++ ONNX 分块器 | 最近邻基准验证通道、边界、奇数尺寸及跨块拼接；三个真实模型与独立的同分块 ONNX 参考输出相同 |
| 模型异常处理 | 错误输出倍率被拒绝 |
| DAT2 导出／安装 | 官方 checkpoint → 128×128 静态图，PyTorch／ONNX 最大绝对误差 < 2.4×10⁻⁶；安装哈希一致 |
| 工程和菜单 | XML／资源解析、源文件打包条目、29 个推荐 ID 的存在性和唯一性、C++ 分类函数检查通过 |

真实模型的上述分块测试输入为 67×3 彩色合成图，用于触发边缘填充和跨块边界，**不是 Galgame 画质或 1080p 性能基准**。CPU 两块推理的观察时间约为 Fast 1.20 秒、完整版 10.10 秒、DAT2 21.54 秒，不应据此推断显卡帧率。

复现生成和检查：

```sh
python scripts/generate_neural_effects.py --check
python scripts/generate_lerf_effects.py --check
python scripts/validate_sr_shaders.py --dxc <dxc可执行文件>
python scripts/validate_sr_numerics.py --sources obj/reference-sources
```

最后一个命令下载并校验固定来源的参考程序和权重，执行 CPU 数学检查。分块测试使用 `scripts/validation/onnx_runner_check.cpp` 与 `src/Magpie.Core/OnnxModelRunner.cpp`，以 C++20 编译，包含 ORT C++ 头文件并链接本机 CPU ORT 库，再运行：

```sh
python scripts/validation/validate_onnx_runner.py --runner <编译好的检查程序> --models obj/sr-models
```

## 尚未完成的验收

当前工作环境没有可执行完整工程构建的 Windows MSVC／SDK／C++/WinRT 环境，也没有可用于 DirectML 实测的 Windows 显卡。按本地提交要求，未上传开发代码、未调用 GitHub Windows 构建。

后续必须在本地 Windows 环境完成：完整 Release 构建；原生 CS5／FP16 编译；DirectML x64／ARM64 运行与 DLL 部署；缺失权重、调整窗口大小、停止和截图流程；现有配置与菜单回归；真实 Galgame／CG 的边缘、文字、渐变、分块接缝、显存和帧率检查。只有这些检查通过，才能确认第一阶段验收完成和最终推荐名单。
