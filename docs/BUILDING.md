# 从源码构建桌面程序与安装器

安装器源代码随公开仓库提供。仅上传代码、依赖声明和下载 URL / 哈希清单；第三方模型、运行环境、EXE、DLL 和本机打包目录均不随源码上传。

本说明对应独立分支 `codex/directml`，保留新版 UI 与 DirectML；`main` 为 CPU / CUDA；先按 [DirectML 说明](DIRECTML.md) 生成共享环境的固定版本下载清单，再构建安装器。历史 CPU 清单不能用于默认 DirectML 构建。仅制作 CPU 版时明确传入 `--acceleration cpu`。

## 构建内容

| 文件 | 作用 |
| --- | --- |
| `packaging/build_desktop.py` | 从当前源码生成 Windows GUI 和后台 Worker，使用新输出目录，不要求旧缓存或实验工程 |
| `packaging/desktop_entry.py` | GUI / Worker 共用入口 |
| `packaging/build_single_exe.py` | 把桌面主程序封装为 Full / Online 单文件安装器 |
| `packaging/bootstrap_entry.py` | 安装器窗口、目录选择、进度、资源安装及启动 |
| `packaging/create_shortcut.ps1` | 创建桌面 / 开始菜单快捷方式 |
| `src/utau_forge/resource_download.py` | 下载、代理 / 镜像、断点续传、校验、解包、备份和组件安装 |
| `src/utau_forge/resource_download_dialog.py` | 主程序中的模型 / 环境下载窗口 |
| `packaging/prepare_download_catalog.py`、`prepare_runtime_catalog.py` | 维护者更新固定版本下载清单；普通使用者无需运行 |

## 1. 准备构建环境

在 Windows x64 上按 [README](../README.md) 安装 Python 3.11 和应用依赖，再安装打包工具：

```powershell
.venv/Scripts/python.exe -m pip install -e ".[packaging]"
```

优先使用专用构建环境，避免无关的开发库、GPU 库或 GPL 组件被收集。当前构建脚本显式排除 Parselmouth；这不能替代对最终依赖树和原生库的许可检查。

## 2. 生成主程序目录

```powershell
.venv/Scripts/python.exe packaging/build_desktop.py
```

脚本在新的 `dist/desktop_时间戳/VoiceBankGen/` 中生成 `VoiceBankGen.exe`、`VoiceBankGenWorker.exe` 和 `_internal/`。GUI 与 Worker 使用相同应用代码，二者需一起保留。推理环境、FFmpeg 和模型由下载清单另行取得；不要单独移动 GUI EXE。

脚本收集构建环境中的 dist-info 许可文本，保留根 LICENSE、LICENSES、第三方声明和公开手册；这只是分发材料的起点，不能据此认定全部 native / Qt 许可义务已完成。具体义务见 [许可审查](LICENSE_AUDIT.md)。

## 3. 制作 Online 安装器

下面路径中的时间戳替换为上一步实际输出。`--desktop-build` 指向包含两个 EXE 的目录；输出必须是尚不存在的新目录。

```powershell
.venv/Scripts/python.exe packaging/build_single_exe.py `
  dist/desktop_时间戳/VoiceBankGen `
  --desktop-build dist/desktop_时间戳/VoiceBankGen `
  --variants online --version 0.1.0 --output dist/VoiceBankGen-Online-0.1.0
```

得到 `VoiceBankGen-Online-Setup.exe`、校验文件和 `online/install-catalog.json`。主程序 payload 已嵌入安装器；模型与独立 TIFA / FFmpeg 环境从固定的官方地址下载，也可填写镜像和 HTTP / HTTPS 代理。安装时按文件哈希校验；镜像只改变下载地址，不改变预期哈希。无需预先提供 VoiceBankGen 自己的 GitHub Release 地址。

## 4. 制作 Full 安装器

Full 使用相同主程序，加上本机已准备好的 CPU 推理环境、FFmpeg、TIFA / UniDic / Whisper / VAD 资源。先启动上一步生成的 `VoiceBankGen.exe`，通过 **工具 → 模型 / 环境下载** 完整安装并校验目录中的全部组件，再运行：

```powershell
.venv/Scripts/python.exe packaging/build_single_exe.py `
  dist/desktop_时间戳/VoiceBankGen `
  --desktop-build dist/desktop_时间戳/VoiceBankGen `
  --variants full --version 0.1.0 --output dist/VoiceBankGen-Full-0.1.0
```

Full 构建会核对固定模型文件，不接纳本机自定义模型作为默认权重。可用 `--variants both` 同时生成两种安装器。构建会占用较多磁盘空间；生成物均在被 Git 忽略的目录中。

已有由本项目安装器生成的完整缓存时，可增加 `--component-cache 软件目录/.component-cache`；脚本按该软件目录的 `components.json` 核对归档哈希，并核对固定模型记录后复用，避免重复下载。发布安装器只附简短 TXT 使用说明及必要许可，排除内部 Markdown 手册、开发审查报告和本机验证记录。额外的固定版本许可文本与源码来源可用 `--release-notices 目录` 一起保留。

此分支不再打包独立 CUDA 组件；TIFA 的 CPU 与 DirectML 共用 `runtime/tifa-cpu`。`build_single_exe.py` 默认 `--acceleration directml`，可通过 `--runtime-catalog 路径` 选择已生成的固定清单；会拒绝设备类型不匹配的旧缓存。CPU-only 构建使用单独的 CPU 清单及 `--acceleration cpu`。

## 发布边界

便携发布可使用 `packaging/build_portable.py --catalog 离线构建/full/install-catalog.json --payload-dir 离线构建目录 --output dist/VoiceBankGen-Windows-x64.zip`。它将全部组件展开为同一个 VoiceBankGen 目录，生成解压即用的 ZIP；DirectML 发行中的 CPU 与 DirectML 使用同一份 TIFA 环境，旧 faster-whisper 转写使用 CPU；ONNX Whisper 可使用 DirectML，模型准备见 DIRECTML.md。不需要执行安装器，不附内部手册或重复的安装缓存；便携版使用说明通过 generated 文件记录保存。压缩优先使用已安装的 7-Zip，没有时使用 Python ZIP64。

较小的便携版使用 `.7z` 输出，必须安装 7-Zip。`--cpu-only` 为真正的 CPU-only 清单生成 CPU 使用说明；共享 DirectML 环境不能简单删除其中的文件转为 CPU 版。`--prepared-dir 已准备的VoiceBankGen目录` 可替代 `--payload-dir`，通过硬链接复用文件，避免再次解包和重复占用磁盘。两版分别压缩，CPU 版保留全部必要模型。

当前默认 `--compression balanced`：LZMA2 普通级别、32 MiB 字典、256 MiB 固实块和 4 个线程，兼顾体积、压缩耗时、解压内存和单文件提取。极限压缩需要显式选择 `--compression smallest`。如果目录已完成对应 CPU / GPU 配置，使用 `--prepared-dir 目录 --reuse-in-place` 可直接重新压缩；会拒绝含旧 CUDA 目录或重复组件缓存的输入。

可以上传上述安装器代码。**本次源码审查没有放行任何既有 Full / Online EXE**：即使 Online 在运行时下载模型，其 EXE 仍内置 Qt、Python 库等主程序依赖，不能视为没有第三方分发义务。Full 还涉及 TIFA 模型的 CC BY-NC-SA 4.0 条件。

发布二进制前按实际构建逐项完成许可 / NOTICE、Qt / LGPL 对应源码与替换说明、FFmpeg 配置及原生依赖检查，确认模型分发条件。不要将工作目录中的实验工程、素材和任意旧 `_internal` 作为公开构建输入。
