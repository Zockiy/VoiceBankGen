# VoiceBankGen

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](pyproject.toml)

**从同一说话人的语音素材中提取、筛选和校对音节，制作 UTAU / OpenUtau CV、CVVC 声库。**

VoiceBankGen 将语音分段、文本转写、音素对齐、候选筛选、oto 编辑、音频拼接和声库导出放在同一个桌面工作流中。你可以从已有录音开始，先生成可用候选，再逐音试听和修正，不必先录制一套完整的音节表。

> 本仓库发布的是 **v0.1.0 首个 Release 源码快照**，包括主程序与安装器源码。仓库不包含模型权重、完整词典、Python 运行环境、测试录音、角色语音、声库或 EXE。自动分析结果是待校对草稿，不能替代逐音试听。本版本以语音素材处理为主，后续开发中的歌声＋LRC 导入功能不包含在本次发布中。

## 能做什么

- **自动建立候选**：VAD 分段、Whisper 转写、TIFA 音素对齐，提取音节并分析可用元音区。
- **日语与普通话**：选择来源语言与目标版本，管理日语 CV / CVVC、普通话拼音 CV / CVVC。
- **统一管理 CV 与 CVVC**：对应版本共用基础 CV 的选音与调整，VC / VV 连接音单独保存。
- **可视化校对 oto**：波形、可展开频谱、五条标线和高级 oto 数值互相对应；支持锁定标线。
- **试听与选用分开**：点击候选用于试听；点击「使用这个音」才改变正式选音。空格试听当前音的使用范围，没有 oto 时播放完整切片。
- **撤销与重做**：一次完整拖动记为一步，最多保留 8 步编辑历史。
- **拼接工作台**：选择已有片段，移动、裁剪、淡化、按 S 切分；可保存为目标音候选或单独导出 WAV。
- **补充与替换**：自定义音标签、导入单独音频、编号备选音、可选呼吸音及补缺音。
- **衔接与标线工具**：批量优化 VC / VV；对已有切片但未检出稳定元音的缺失音，可尝试强制标注 oto，再人工校对。
- **同步与导出**：编辑先暂存，点击「同步修改」统一应用；只改 oto 时复用音频缓存。支持安装到独立声库目录、导出音源 ZIP，以及导出各音节最佳 X 个切片。
- **本地模型与下载设置**：模型路径可自定义；支持官方资源、自填镜像与 HTTP / HTTPS 代理，后台任务显示进度并可取消。

默认深色界面，可切换浅色；默认保留原音高。日语 oto 默认使用平假名，音频文件名使用罗马音；普通话使用拼音文件名，便于校对。

## 工作流程

```text
单人语音 → VAD 分段 → Whisper 转写 → TIFA 音素对齐
        → 音节候选 → 试听、选音与 oto 校对 → 同步、安装或导出
```

1. **素材与切分**：新建说话人工程，选择来源语言与目标 CV/CVVC 版本，添加同一说话人的录音；选择 TIFA，首次使用建议选 CPU，再开始自动切片。
2. **校对与选音**：在左侧选择发音，试听右侧候选，比较稳定元音、噪声和边界；点击「使用这个音」确定选音，拖动标线或编辑 oto。
3. **补充缺音**：可以添加同一人的更多素材、导入单音，或在拼接工作台制作候选。拼接和近似补音需要特别试听，不应自动视为合格。
4. **同步与导出**：点击底部「同步修改」应用草稿，再安装到独立的声库子目录或导出。OpenUtau 中重新载入音源后试听。

素材应具有明确的使用权限，尽量使用单人、清晰、少噪声和少混响的录音。背景音乐、多人重叠、强噪声以及转写错误都会影响后续对齐。有限素材没有覆盖所有发音时，出现缺音是正常现象。

详细操作见 [使用指南](docs/USAGE.md) 和 [界面手册](docs/UI_WORKFLOW_FORK.md)。

## 本地运行源码：Windows x64

本版本要求 **64 位 Python 3.11**。以下命令均在 **PowerShell** 中执行。其他系统尚未完成完整桌面流程验证。

需要准备：

| 项目 | 用途 |
|---|---|
| Python 3.11 x64 | 运行主程序与 CPU 推理 |
| Git | 获取本项目与固定版本 TIFA 源码 |
| FFmpeg / ffprobe | 解码和转换素材，两个命令都需要在 PATH 中 |
| 可用网络与足够磁盘空间 | 首次安装依赖、模型与完整日语词典；Git 仓库大小不代表完整运行环境大小 |
| 可选 NVIDIA GPU | GPU 加速需要额外匹配的驱动、CUDA 推理环境，不是运行源码的必需条件 |

### 1. 获取源码

```powershell
git clone https://github.com/Zockiy/VoiceBankGen.git
cd VoiceBankGen
py -3.11 --version
```

如果从 GitHub 下载源码 ZIP，请先解压，再进入包含 `pyproject.toml` 的目录。后面的步骤相同。

### 2. 创建虚拟环境并安装依赖

下面使用虚拟环境内 Python 的完整路径，不要求激活环境，也不需要修改 PowerShell 执行策略。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r packaging/runtime_requirements.txt
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,asr,vad,pitch]"
```

`runtime_requirements.txt` 提供固定的 **CPU TIFA 推理依赖**，包括 CPU PyTorch；安装过程可能较长。第二条安装命令将主程序、桌面界面、转写、VAD 和声码器安装到同一个 `.venv`。

不要为启动桌面版直接安装全部 optional extras；`review` 是可选旧浏览器审核界面，`packaging` 和 `test` 分别用于打包与开发测试。

如果 PyWORLD 等扩展在本机没有适配的预编译 wheel，pip 可能要求 Microsoft C++ Build Tools。按该扩展的官方说明安装编译工具，然后重试。

### 3. 获取固定版本的 TIFA 上游代码

在本项目根目录执行：

```powershell
git clone https://github.com/openvpi/TIFA.git resources/upstream/TIFA
git -C resources/upstream/TIFA checkout 32a0a13aebce580844ad1dc281ea11ef1be94f14
```

TIFA 源码用于音素对齐。保持上述固定 commit，不要在首次安装时随意改为上游最新版本。此目录已被 Git 忽略，不要将完整 TIFA 仓库提交到本项目。

### 4. 安装并检查 FFmpeg

从 [FFmpeg 官方列出的下载来源](https://ffmpeg.org/download.html)选择 Windows 构建，解压后将包含 `ffmpeg.exe` 与 `ffprobe.exe` 的目录加入 PATH，重新打开 PowerShell 并执行：

```powershell
ffmpeg -version
ffprobe -version
```

两条命令都应返回版本信息。若使用带独立 DLL 的构建，须保留其配套 DLL。第三方发行的 FFmpeg 可能采用不同许可，重新分发时应检查实际构建，不能一概视为 LGPL。

### 5. 启动界面

在仓库根目录执行：

```powershell
.\.venv\Scripts\voicebankgen.exe
```

也可通过 Python 模块入口启动：

```powershell
.\.venv\Scripts\python.exe -m utau_forge.app
```

`utau_forge` 是保留的内部包名，产品名是 VoiceBankGen。

### 6. 下载模型并检查路径

在界面中打开 **工具 → 模型 / 环境下载**，准备下面的资源：

| 资源 | 作用 |
|---|---|
| Whisper small（CTranslate2 格式） | 语音转写 |
| TIFA-1.0-ST v1.0.0 | 音素对齐 |
| 完整 UniDic 2.3.0 | 日语文本与读音前端；不要用 unidic-lite 替代 |
| Silero VAD 资源 | 语音活动检测，按下载面板提示准备 |

源码运行已通过第 2 步准备 CPU Python 依赖，不需要先构建 EXE 或安装完整便携版。下载面板还支持打包版所需的独立环境组件，它们与 `.venv` 中的依赖不是同一个概念。

有现成资源时，打开 **工具 → 模型设置** 指定本地目录，保存后记住；留空时自动检测。典型模型检查项：

- Whisper：`model.bin`、`config.json`、`tokenizer.json`。
- TIFA：`model.pt`、`config.yaml`、`vocabulary.json` 和配套 `dictionaries/`。
- UniDic：`sys.dic`、`unk.dic`、`char.bin`、`matrix.bin`、`dicrc`、`mecabrc`。

首次下载需要联网。资源准备完整后，转写与对齐使用本地模型，不需要上传原录音到在线识别服务。TIFA 模型的许可条件见下文。

### 7. 先用 CPU 验证自己的短素材

新建工程，选择实际素材语言，计算设备选 **CPU**，对齐工具选 **TIFA**，导入一段有权使用的短录音并开始自动切片。完成后检查候选、试听并校对，再逐步处理长录音。

没有素材时，界面可以打开，但无法据此验证真实转写与对齐质量。本仓库不附带实验录音。

### GPU 与代理说明

- 本项目主线使用 CPU / NVIDIA CUDA。CUDA 是否可用要分别检查 CTranslate2（Whisper）与 PyTorch（TIFA）；其中一个可用不代表另一个可用。
- 上述快速开始固定安装 CPU PyTorch。GPU 使用者需按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)准备兼容组合，并注意本版本固定依赖的兼容性。
- 源码模式的 TIFA CUDA 回退入口会查找 `.tifa-env\Scripts\python.exe`，或使用已有的独立 TIFA GPU 组件。只在主 `.venv` 中安装 CUDA PyTorch，不等于这个独立环境已经准备好。首次使用建议先完成 CPU 路线。
- 模型下载面板可填写自己的 HTTP / HTTPS 代理和镜像地址。它们仅影响组件下载；Git 和 pip 的网络配置需要分别设置。
- pip 使用代理时可在相应安装命令后增加 `--proxy http://127.0.0.1:7890`，端口替换成自己的。不要把含账号、密码或 token 的代理地址提交到仓库。

## 工程数据与缓存

每个说话人工程单独保存素材记录、候选、选音、标线与构建结果。工程文件与软件源码目录是不同的概念，建议选择单独的工作目录。

编辑会暂存到工程；「同步修改」才将调整应用到生成或关联安装的音源。不要把 `.venv`、模型、工程数据库、素材、切片和安装缓存加入 Git。迁移工程前应保存并退出程序，连同工程所需的素材和衍生文件一起备份。

## 常见问题

| 现象 | 检查方法 |
|---|---|
| Python 版本不符或依赖无法安装 | 确认是 Python 3.11 x64，并始终使用同一个 `.venv\Scripts\python.exe` |
| 找不到 `voicebankgen.exe` | 确认 editable install 成功；可改用上面的 Python 模块入口 |
| 找不到 FFmpeg / ffprobe | 检查 PATH，重开 PowerShell，分别运行两条版本检查命令 |
| 找不到模型或尝试访问 Hub | 检查下载是否完成，在模型设置中指定完整本地目录；不要只选择单个权重文件 |
| 缺少 TIFA 推理模块 | 检查第 2 步依赖和第 3 步固定上游代码；只下载 model.pt 不足以运行 |
| CUDA 不可用 | 先切换 CPU，再分别检查 CTranslate2、TIFA 环境和驱动兼容性 |
| 有切片但音节仍显示缺失 | 可能未检出稳定元音；可校对候选或尝试「校对 → 强制标注 oto」，之后继续试听 |
| 在 OpenUtau 中没有看到修改 | 先同步并重新载入音源，确认使用了正确的版本、安装路径与 phonemizer |
| 提示无效历史绑定 | 旧安装路径不存在、缺少关联标记，或工程/版本 ID 不匹配；被跳过的目录不会收到本次更新 |

## 构建与开发

**运行源码不需要先打包。** 若要生成桌面 EXE：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[packaging]"
.\.venv\Scripts\python.exe packaging/build_desktop.py
```

构建结果包含 GUI、后台 Worker 与 `_internal`；不要只移动 GUI EXE。Full / Online 安装器和便携包构建见 [BUILDING](docs/BUILDING.md)。这些脚本不会赋予第三方模型或运行库额外的分发许可。

安装开发测试依赖并运行测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
```

测试代码会使用合成数据或明确的外部依赖；单元测试不是声库发音或歌唱质量验收。真实素材仍需人工检查。

```text
src/utau_forge/       应用、音频处理、工程管理与导出
src/utau_forge/data/  内置音节表、图标和资源下载元数据
packaging/           构建、安装器、下载清单与推理入口
tests/               测试代码
docs/                使用、构建、发布与许可说明
LICENSES/            独立版权与许可文本
```

首次源码发布只包含 `main` 和 `v0.1.0`；不发布本地实验分支与开发历史。后续版本和分支以本仓库实际内容为准。

## 开源许可与素材权利

**本项目自研代码与文档采用 [MIT License](LICENSE)**，允许在保留版权与许可声明的条件下使用、修改和分发，包括商业使用。第三方内容分别遵守其原许可：

- 普通话内置音节表 `zh_cv_inventory.json` 是 CC-CEDICT 等读音数据的改编，采用 **CC BY-SA 4.0**；须保留 [来源与改编声明](src/utau_forge/data/zh_inventory_LICENSE.txt)，不能将该表改标为 MIT。
- OpenUtau 与 TIFA 参考内容保留原 MIT 版权声明，见 [LICENSES](LICENSES/README.md)。
- TIFA 上游代码为 MIT，但当前 **TIFA-1.0-ST v1.0.0 模型为 CC BY-NC-SA 4.0**。源码 MIT 不替代模型的独立许可，也不授予其额外商用权。上游说明该模型许可约束模型文件而非流经它的数据；原录音与输出声库仍有独立权利问题。
- PySide6/Qt、FFmpeg、libsndfile、CUDA 等各有分发要求。仅公开本项目源码，并不等于任意现成 EXE 或离线整合包都可以直接再分发。
- 仅处理与发布你有权使用的录音和声库；本地试验许可不等于公开素材的授权。

完整说明见 [第三方组件](THIRD_PARTY_NOTICES.md)、[许可审查](docs/LICENSE_AUDIT.md) 和 [依赖索引](docs/DEPENDENCIES.md)。

## 反馈与贡献

请在 [Issues](https://github.com/Zockiy/VoiceBankGen/issues) 提供系统、Python 版本、操作步骤和去掉私人路径的错误信息。不要提交录音、歌词、模型权重、工程数据库、API 密钥或打包环境；优先使用可公开的最小复现。

欢迎通过 Pull Request 改进代码和文档。修改功能时同步更新相关说明与必要测试，保留第三方署名，并在加入新依赖或复制外部代码前检查其许可证。
