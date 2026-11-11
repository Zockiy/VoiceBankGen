# DirectML 分支源码许可复核

日期：2026-11-11；公开分支：codex/directml；程序基线：`d16f0ec86067fa09caab8bbb55c58495cc7ac5b5`。

## 发布结论

维护者已选择 MIT 用于有权授权的自研代码与文档。根 LICENSE 保留 MIT 原文，第三方内容按各自许可保留。
本仓库包含自研程序、安装器源码、必要的小型发音表/图标、公开说明及依赖元数据；不包含运行库、完整第三方仓库、模型或录音。
此源码范围可在保留以下声明与独立许可的前提下公开；该结论不是对任意 EXE 或离线资源包的分发批准。

## 必须保留的独立授权

| 内容 | 来源与许可 | 处理 |
|---|---|---|
| 日语 CVVC 兼容规则参考 | [OpenUtau MIT](https://github.com/openutau/OpenUtau/blob/master/LICENSE.txt)，Copyright 2014 StAkira | 保留 LICENSES/OpenUtau-MIT.txt |
| TIFA 接口/配置适配 | [固定 commit MIT](https://github.com/openvpi/TIFA/blob/32a0a13aebce580844ad1dc281ea11ef1be94f14/LICENSE)，Copyright 2026 Team OpenVPI | 保留 LICENSES/TIFA-MIT.txt；完整上游另取 |
| 普通话音节表 | [CC-CEDICT / MDBG](https://www.mdbg.net/chinese/dictionary?page=cc-cedict)，CC BY-SA 4.0 | 保留同目录许可、.license、署名、修改说明及完整 CC BY-SA 文本，不改标 MIT |
| 模型与完整词典 | 不随源码提供 | 资源锁与下载地址不授予额外权利 |

## 外部资源的重要限制

- [TIFA v1.0.0](https://github.com/openvpi/TIFA/releases/tag/v1.0.0)：代码 MIT，模型 CC BY-NC-SA 4.0。后者不是 OSI 开源许可，不能随程序改标 MIT；上游说明模型许可不约束流经模型的数据，原录音与声库仍有独立权利。
- [Whisper small 固定模型卡](https://huggingface.co/Systran/faster-whisper-small/blob/536b0662742c02347bc0e980a01041f333bce120/README.md)：MIT；本源码仍排除权重。
- [UniDic](https://github.com/polm/unidic-py#license)：Python 下载器与词典本体分别授权，锁文件选择词典 BSD 路径；完整词典不随源码上传。
- [Qt/PySide6](https://www.qt.io/development/open-source-lgpl-obligations)：使用 LGPL 路径仍需满足准确模块的义务；部分 Qt 模块 GPL-only。分发应用时须落实适用的对应源码、替换/重链接及安装信息等要求。
- [FFmpeg](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)：按实际构建检查 LGPL/GPL；enable-nonfree 构建不可按该开源许可再分发，不能将本机 GPL 构建称作 LGPL 构建。
- [PyInstaller](https://pyinstaller.org/en/stable/license.html)：GPL 带打包例外，不自动要求应用代码 GPL，也不豁免被打包依赖的许可。
- CUDA/cuDNN/NVIDIA 组件存在独立许可；SoundFile 所带 libsndfile、PyAV 所带 FFmpeg、PyTorch 和数值库的原生组件各有 notices，不能仅看顶层 Python 包标签。
- Parselmouth/Praat 等 GPL 可选依赖不因 TIFA 顶层 MIT 而免除义务；本仓库不提供这些分发物。

## 证据范围

docs/DEPENDENCIES.md 与 dependency_inventory.json 列出基线的 161 个包/版本条目，包含开发、可选与平台依赖，不等于 161 个直接依赖。
PyPI 元数据原采集日期为 2026-11-03；没有声称所有包都在 2026-11-11 重新联网验证。
本轮复核核心上游官方许可页面，并在维护者本地保存了 109 个精确版本包的许可原文及相关元数据，共 354 份证据文件。
其余条目仍有原文/精确分发物核对缺口；这些第三方代码与二进制不随源码上传，不能据此批准它们的二进制重分发。

首次公开快照通过文件白名单、媒体/权重/二进制排除、常见凭据模式与哈希检查。这不等于完整安全审计或真实素材质量验收。
本项目不提供实验录音，测试代码可以生成合成数据；本地试验许可不能视为公开素材授权。

## DirectML / ONNX 补充（2026-11-11）

- torch-directml 0.2.5.dev240914 官方 wheel 的 MIT 与完整 NOTICE 保留在 LICENSES；CPU torch / torchvision 仍分别适用原始许可。本次没有上传 wheel / DLL。
- ONNX Runtime 1.24.4 使用 MIT，图转换工具 ONNX 使用 Apache-2.0；推理库和转换工具均另行安装，不能用顶层许可替代精确二进制附属 notices。
- openai/whisper-small 官方模型卡声明 Apache-2.0；固定 ONNX 转换来源、拆分图与计算精度改编说明见 LICENSES/Whisper-ONNX-NOTICE.txt，并保留 Apache-2.0 全文。不将其改标为本项目 MIT。
- 本轮重新读取官方 PyPI torch-directml、Whisper 官方模型卡和 ONNX Runtime 1.24.4 LICENSE；DirectML 历史环境/哈希记录的日期仍为 2026-11-05。没有重新进行全部传递依赖或全部硬件验收。
- 上述历史 CPU 清点不等于本分支的实际安装版本；DirectML 固定版本见 runtime_directml_requirements.txt 和 docs/directml_dependencies.json。

官方来源：[torch-directml](https://pypi.org/project/torch-directml/0.2.5.dev240914/)、[Whisper 模型卡](https://huggingface.co/openai/whisper-small/blob/main/README.md)、[ORT 1.24.4 MIT](https://github.com/microsoft/onnxruntime/blob/v1.24.4/LICENSE)。
