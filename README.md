# VoiceBankGen

将单人语音素材整理为可校对的 UTAU / OpenUtau CV、CVVC 声库工程。

当前源码为独立分支 **`codex/directml`**，保留新版 UI 与 DirectML 支持；`main` 使用 CPU / NVIDIA CUDA。界面按「素材与切分 → 校对与选音 → 同步与导出」组织操作。默认深色界面，也可切换浅色。无需先制作完整录音表，可从已有语音中寻找可用候选，再逐音校对。

本分支以 DirectML 替代 CUDA：**ONNX Whisper 转写与 TIFA 神经网络均可使用 DirectML**，自动模式失败会回退 CPU。ONNX Whisper 使用拆分解码图、共享权重和 FP32 计算，磁盘权重保留 FP16；旧 faster-whisper 格式仍使用 CPU。精简便携版共用依赖，排除 CUDA / cuDNN、编译文件和安装缓存。已在一台 RTX 4060 Laptop 上完成短语音实际转写及对齐检查，其他显卡兼容性和速度未经全面验证。准备环境与构建方法见 [DirectML 说明](docs/DIRECTML.md)。

VoiceBankGen 提供本地自动分析、逐音候选选择、oto 标线编辑和拼接工作台。自动切分用于建立草稿，最终发音与歌唱效果需要人工试听。

> **本仓库仅提供源码**：不包含测试录音、角色语音、声库、模型权重、完整词典、运行环境、EXE 或演示工程。首次使用需自行安装依赖并下载模型。自研代码采用 MIT；普通话音节表单独采用 CC BY-SA 4.0；TIFA 模型有非商业限制，详见下文。

## 功能

- 日语 / 普通话来源与目标语言设置，CV / CVVC 版本管理；对应版本共用基础 CV 调整，VC / VV 单独保存。
- VAD 分段、Whisper 转写、TIFA 音素对齐，保留候选来源和缺音提示。
- 波形、频谱、五标线与 oto 数值校对；空格试听实际使用区间，无 oto 时播放完整切片，A / B 比较候选。
- 候选试听与正式选音分开；一次完整拖动记录一步，最多 8 步撤销 / 重做。
- 拼接工作台：整音、辅音、元音片段，移动 / 裁剪 / 淡化，S 键切分，保存到候选或导出 WAV。
- 编号备选音、自定义标签、导入音频、可选呼吸音与自动补缺音。
- 默认保留原音高，可选标准音高；日语 oto 默认平假名，音频文件名使用罗马音 / 拼音。
- 编辑保存草稿，点击同步修改统一应用；只改 oto 时复用音频缓存，后台任务显示进度、耗时和取消状态。
- 批量优化 VC / VV 衔接；强制标注已有切片但未通过稳定元音判定的缺失音，处理后仍需试听。
- 导出音源 ZIP、安装至 OpenUtau，或导出各音节最佳 X 个切片：保留已选用候选，其余优先稳定元音更长的候选，无稳定元音时按切片长度排序。
- 模型路径设置、官方 / 自填镜像下载源、HTTP / HTTPS 代理；搜索停止输入约 250 ms 后筛选，减少输入卡顿。

## 基本流程

1. **素材与切分**：新建工程，导入有权使用的单人录音，选择来源语言与 TIFA，开始自动切片；核对识别文本。
2. **校对与选音**：选择左侧发音、试听右侧候选，点击「使用这个音」确定选用。拖动标线或编辑 oto；需要时使用拼接工作台。
3. **同步与导出**：点击底部「同步修改」应用草稿，再安装或导出音源，在 OpenUtau 重新载入并试听。

详细操作见 [使用指南](docs/USAGE.md) 和 [界面手册](docs/UI_WORKFLOW_FORK.md)。本项目不包含 OpenUtau、resampler 或第三方声库。

## 从源码运行（Windows x64）

需要 Python **3.11**、Git、FFmpeg / ffprobe。其他系统尚未全面验证。以下命令在仓库根目录的 PowerShell 执行。

### 1. 获取此分支并创建应用环境

```powershell
git clone --branch codex/directml --single-branch https://github.com/Zockiy/VoiceBankGen.git
cd VoiceBankGen
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r packaging/runtime_requirements.txt
.venv/Scripts/python.exe -m pip install -e ".[desktop,asr,vad,pitch]"
```

此步骤先准备应用与 CPU 后端，不会安装 CUDA。不要执行 `uv sync --all-extras`：历史锁文件同时包含可选和开发依赖，并不是本分支 DirectML 推理环境的安装清单。PyWORLD 等原生扩展若要求编译，按其官方说明准备 Microsoft C++ Build Tools。

### 2. 启用 ONNX Whisper 的 DirectML 后端

以下命令只操作上一步新建的 `.venv`。CPU 版与 DirectML 版 ONNX Runtime 提供相同的 Python 模块，不能在同一环境混装：

```powershell
.venv/Scripts/python.exe -m pip uninstall -y onnxruntime
.venv/Scripts/python.exe -m pip install -e ".[asr-directml]"
```

之后不要再向此环境安装 CPU 版 `onnxruntime`，也不要重复安装带 `[vad]` 的 extra；Silero VAD 已在第一步安装，仍可使用 DirectML 版 ORT 提供的 CPU provider。如果不需要 Whisper GPU 转写，可以跳过本步骤并使用旧 faster-whisper 模型和 CPU。

### 3. 准备独立 TIFA CPU / DirectML 环境

```powershell
py -3.11 -m venv .directml-env
.directml-env/Scripts/python.exe -m pip install --upgrade pip
.directml-env/Scripts/python.exe -m pip install -r packaging/runtime_directml_requirements.txt
git clone https://github.com/openvpi/TIFA.git resources/upstream/TIFA
git -C resources/upstream/TIFA checkout 32a0a13aebce580844ad1dc281ea11ef1be94f14
```

主界面与 Whisper 使用 `.venv`，TIFA 子进程自动查找项目根目录的 `.directml-env`。后者固定使用 CPU PyTorch 2.4.1、torchvision 0.19.1 和 torch-directml 0.2.5.dev240914；不要安装 CUDA PyTorch 或直接升级旧环境。两个环境各自独立，源码安装步骤不代表便携版必须携带两套环境，便携版构建器会共享依赖。

### 4. 安装 FFmpeg 并准备模型

从 [FFmpeg 官方列出的下载来源](https://ffmpeg.org/download.html)安装 Windows 构建，将包含 ffmpeg / ffprobe 的目录加入 PATH，保留配套 DLL 与许可，然后检查：

```powershell
ffmpeg -version
ffprobe -version
.venv/Scripts/voicebankgen.exe
```

打开 **工具 → 模型 / 环境下载**，准备 TIFA-1.0-ST v1.0.0 和完整 UniDic 2.3.0；按面板提示准备 VAD 资源。UniDic 不要改用 unidic-lite。源码的 Python 环境已由上面命令安装，不必下载打包版环境。模型下载可使用 HTTP / HTTPS 代理或自填镜像；旧下载清单提供的 Whisper 是 CTranslate2 格式，只支持 CPU，不会自动变为 ONNX。

若要使用 DirectML Whisper，先关闭主界面，下载并转换固定版本模型：

```powershell
.venv/Scripts/python.exe -m pip install "onnx>=1.17,<2"
.venv/Scripts/python.exe packaging/download_onnx_whisper.py --output build/whisper-small-original
.venv/Scripts/python.exe packaging/prepare_onnx_whisper.py --source build/whisper-small-original --output resources/whisper-small-onnx
.venv/Scripts/voicebankgen.exe
```

需要代理时，在下载命令末尾加 `--proxy http://127.0.0.1:7890`；镜像使用 `--mirror 镜像基地址`。转换输出目录必须不存在；`onnx` 仅是模型准备工具，不是日常推理必需库。

打开 **工具 → 模型设置**，将 Whisper 模型路径设为生成的 `resources/whisper-small-onnx`，保存后选择计算设备「DirectML GPU」或「自动」。目录应包含 `voicebankgen-onnx.json`、配置/tokenizer，以及 `onnx/` 下的 encoder、两份解码图和 `decoder.weights`。原始 merged ONNX 不能直接代替准备后的拆分模型。

「自动」优先 DirectML，失败可回退 CPU；明确选择 DirectML 后初始化失败会报错。部分不支持的算子仍可能在 CPU 执行，预处理、束搜索等也在 CPU。DirectML 不能保证比 CPU/CUDA 更快，识别结果也不保证与旧后端逐字一致。完整设备、模型格式和构建限制见 [DIRECTML](docs/DIRECTML.md)。

### 常见问题

- **缺少模型 / 无法连接 Hub**：先完成下载，或在模型设置指定完整本地目录。切片工作不需要把原始录音上传到识别服务。
- **下载不通**：填写自己的 HTTP / HTTPS 代理或可靠镜像。模型下载面板的镜像地址是资源转发地址，不是 pip 索引；安装 Python 依赖时需另外配置 pip 网络。
- **有切片但音节仍显示缺失**：可能没有检测到足够稳定的元音。可以人工调整或使用「校对 → 强制标注 oto」，不要据此假定音质合格。
- **在 OpenUtau 看不到修改**：先同步，再在 OpenUtau 重新载入音源；确认使用了正确的音源版本和 phonemizer。

## 开发与目录

**安装器代码也包含在仓库中**：支持 Full / Online 两种版本，构建主程序与安装器的步骤见 [BUILDING](docs/BUILDING.md)。运行源码无需先构建 EXE。

```text
src/utau_forge/       桌面程序、音频处理、工程与导出逻辑
src/utau_forge/data/  内置音节表、图标、下载 URL / 哈希清单（非模型）
packaging/           推理入口、下载清单、安装器及源码发布工具
tests/               自动化测试代码；运行时生成临时合成数据
docs/                使用、许可审查和发布说明
LICENSES/            第三方署名与许可范围
```

```powershell
.venv/Scripts/python.exe -m pytest -q
```

本项目没有附带可用于复现真实发音效果的录音数据。单元测试不等于整库歌唱质量验收；长录音、噪声、跨语种近似发音和缺音拼接均需要校对。MFA 仅保留可选旧接口，默认流程不需要其环境。

## 开源许可与分发

- **自研代码与文档：MIT**，允许在保留许可和署名的条件下使用、修改和分发，见 [LICENSE](LICENSE)。
- **普通话音节表 `zh_cv_inventory.json`：CC BY-SA 4.0**，保留 CC-CEDICT / MDBG 来源、署名和改编说明。修改该表后分享须遵守同许可，不能将它改标为 MIT。
- **TIFA：代码 MIT；当前 TIFA-1.0-ST v1.0.0 模型 CC BY-NC-SA 4.0**。源码许可不授予模型的商用权。不包含模型不代表可以忽略实际使用或再分发模型时的限制；商业使用此模型前请向权利人确认。上游明确模型许可不约束流经模型的数据，但录音与输出声库仍有独立权利问题。
- Qt / PySide6、FFmpeg、libsndfile 和 DirectML / ONNX Runtime 等依赖各有分发条件。上传本项目源码不等于已经完成 EXE / 离线整合包的分发合规。

完整说明：[第三方组件](THIRD_PARTY_NOTICES.md) · [许可审查](docs/LICENSE_AUDIT.md) · [源码发布步骤](docs/SOURCE_RELEASE.md)。本次源码快照不沿用本机开发历史。下载脚本和资源 URL 可以公开，使用这些脚本取得的模型 / 运行库仍须遵守各自许可。

## 参与开发

提交 Issue 时附上系统、Python 版本、操作步骤和去掉个人路径的错误信息。请勿提交录音、角色素材、模型、声库、工程数据库、密钥或打包产物；问题复现优先使用自行生成的短合成信号。修改代码时同步更新相关文档和必要测试，第三方来源与许可声明必须保留。

## 致谢

感谢 [TIFA](https://github.com/openvpi/TIFA)、[OpenUtau](https://github.com/stakira/OpenUtau)、[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[Silero VAD](https://github.com/snakers4/silero-vad)、[Qt for Python](https://doc.qt.io/qtforpython-6/)、[WORLD / PyWORLD](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) 及其依赖项目。
