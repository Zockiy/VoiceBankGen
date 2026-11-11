# 第三方组件与许可

源码发布复核：2026-11-11。此说明对应 **`codex/directml` 独立分支的仅源码公开版本**；旧依赖清点保留其原采集日期。许可证原文、文件范围及来源见 [LICENSES](LICENSES/README.md)，详细证据和分发限制见 [许可审查](docs/LICENSE_AUDIT.md)。

## 随源码保留的内容

| 内容 | 来源 / 许可 | 本项目处理 |
| --- | --- | --- |
| 自研 Python 代码、文档和图标 | VoiceBankGen contributors，MIT | 根 LICENSE；不覆盖下列第三方内容 |
| 日语 CVVC 兼容规则 / 别名表参考 | OpenUtau，MIT，Copyright (c) 2014 StAkira | 保留 [MIT 原文](LICENSES/OpenUtau-MIT.txt)；不包含 OpenUtau EXE 或声库 |
| TIFA 适配所参考的接口 / 配置 | openvpi/TIFA commit 32a0a13aebce580844ad1dc281ea11ef1be94f14，MIT，Copyright (c) 2026 Team OpenVPI | 保留 [MIT 原文](LICENSES/TIFA-MIT.txt)；完整上游代码自行下载，不 vendor |
| 普通话音节表 `src/utau_forge/data/zh_cv_inventory.json` | TIFA 中文音素表与 cpp-pinyin / CC-CEDICT 读音数据的改编，CC BY-SA 4.0 | 保留同目录许可和 .license 侧文件；2026-10-02 过滤非标准读音，保留 411 个拼音；不包含完整字词典 |
| 下载目录 / 资源锁 | 官方 URL、版本、大小、SHA-256，VoiceBankGen 生成 | 只是元数据，不包含权重、词典本体或第三方代码的编码副本 |

CC-CEDICT 为社区维护、MDBG 发布，继承 Paul Andrew Denisowski 的 CEDICT（1997、1998）；来源与完整署名见 [音节表许可](src/utau_forge/data/zh_inventory_LICENSE.txt)。此表及其改编部分采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)，独立于程序代码的 MIT。打包元数据用 `MIT AND CC-BY-SA-4.0` 表示随包存在两种许可，不表示所有 Python 代码都变为 CC BY-SA。

## 安装时另行取得的依赖

| 组件 | 用途 | 主要许可 / 注意 |
| --- | --- | --- |
| NumPy / SciPy | 数组、信号处理 | BSD 为主；wheel 内 OpenBLAS 等另有 notices |
| Pydantic / PyYAML / Typer | 配置、CLI | MIT |
| PySide6 / Qt / Shiboken | GUI、音频播放 | 当前 GUI 可选 LGPLv3；Qt 部分模块为 GPL-only。只能按实际使用模块和二进制确认 |
| pyqtgraph | 波形与频谱 | MIT |
| SoundFile / libsndfile | WAV I/O | Python 层 BSD-3-Clause；libsndfile 为 LGPL-2.1-or-later；编解码库各有许可 |
| faster-whisper / CTranslate2 | 转写 | MIT；PyAV 及所链接 FFmpeg 另查 |
| Silero VAD / ONNX Runtime | 分段 | MIT；模型不随源码包提供 |
| PyWORLD / WORLD | F0 与重合成 | PyWORLD MIT；WORLD modified BSD |
| PyTorch / Lightning / librosa | 独立 TIFA 推理 | PyTorch 按固定 wheel 的复合 notices；Lightning Apache-2.0；librosa ISC |
| fugashi / MeCab / UniDic Python 包 | 日语前端 | fugashi MIT、MeCab 独立许可；UniDic Python 包 MIT，完整词典另算 |
| Gradio / Plotly | 可选旧审核界面 | Apache-2.0 / MIT |
| PyInstaller | 开发打包工具 | GPL-2.0-or-later 带 bootloader 分发例外；不是使用它就自动要求自研代码 GPL |
| CPython | 运行环境 | PSF 及其附属 notices |
| FFmpeg / ffprobe | 解码 | 依构建为 LGPL / GPL；带 enable-nonfree 的构建不可按开源条款再分发 |
| torch-directml 0.2.5.dev240914 / torchvision 0.19.1 | 本分支的 TIFA 加速 / 依赖 | torch-directml 官方 wheel 的 MIT 与完整 NOTICE 见 LICENSES/torch-directml-*；torchvision 为 BSD；保留配套二进制的实际许可。此分支不打包 CUDA / cuDNN |
| ONNX Runtime DirectML / ONNX | ONNX Whisper 推理 / 仅开发机的图转换 | MIT / Apache-2.0，准确分发物另有附带 notices；源码不包含其 wheel 或 DLL |
| MFA | 可选旧对齐接口 | MFA 代码 MIT；Kaldi、模型、词典等另有许可。本源码包不提供 MFA 环境 |

准确版本与哈希见 `uv.lock`、`packaging/runtime_requirements.txt`、`packaging/runtime_directml_requirements.txt`、下载 catalog；DirectML 核对见 [依赖记录](docs/directml_dependencies.json)。原清点见 [依赖清单](docs/dependency_inventory.json)，它早于本分支且不代替 DirectML 的固定版本记录。清单包含开发 / 可选 / 平台依赖，不代表运行时全部都会安装，也不是二进制分发批准。

## 不随源码分发的模型与词典

| 资源 | 固定来源 | 许可与处理 |
| --- | --- | --- |
| TIFA-1.0-ST 1.0.0 | [TIFA release](https://github.com/openvpi/TIFA/releases/tag/v1.0.0) | CC BY-NC-SA 4.0，非 OSI 开源许可；不上传、不放入源码 ZIP，不作为无条件商用资源 |
| Whisper small CTranslate2 | [Systran](https://huggingface.co/Systran/faster-whisper-small)，revision 536b0662742c02347bc0e980a01041f333bce120 | 模型卡 MIT；本次仍排除权重，使用者自行下载 |
| Whisper small ONNX | [onnx-community](https://huggingface.co/onnx-community/whisper-small)，revision 36050c46d777d46dc4b5f43f6d90574fc38f8732；基础模型 openai/whisper-small | 原始基础模型卡声明 Apache-2.0；本仓库仅保留来源与下载清单，不包含权重。本项目拆分解码图并提升计算精度，详见 LICENSES/Whisper-ONNX-NOTICE.txt |
| UniDic 2.3.0 | [unidic-py v1.0.2](https://github.com/polm/unidic-py/releases/tag/v1.0.2) | 词典 GPL / LGPL / BSD 三选一中的 BSD 路径；实际包内 notices 为准，Python 包 MIT 不能替代词典许可 |

## 录音、声库与输出

测试语音、角色语音、个人录音、示例声库、OpenUtau 实验工程和渲染 WAV 均未随源码发布。本机测试许可不等于公开分发许可。软件 MIT 不授予录音、人物声音、角色素材或第三方 resampler 的使用权；输出声库由素材、模型等实际权利条件决定。

## 二进制发行边界

此轮只准备源码。现有 Full / Online EXE、便携目录、DLL、GPU 组件和离线模型包 **不在本次审核放行范围**。发布这些内容前，应针对确切构建收集全部许可证与署名，履行 Qt / LGPL 库的对应源代码、替换 / 重链接与安装信息义务，审查 FFmpeg 构建、PyAV、libsndfile、NVIDIA 条款及模型限制。开发环境发现的 GPL Parselmouth 也不得被误当成 MIT 依赖混入发行包。
