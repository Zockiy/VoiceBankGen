# DirectML 支持

DirectML 在独立分支 `codex/directml` 维护；`main` 已恢复 CPU / NVIDIA CUDA，并保留新版 UI。本分支最初从 `codex/ui-workflow` 的 `06e8de2` 创建，原分支继续保留。2026-11-05 已构建精简便携目录，并在 RTX 4060 Laptop 上完成启动、短语音转写与 DirectML 对齐检查。

## 设备与范围

| 阶段 | 执行设备 |
| --- | --- |
| Whisper / CTranslate2 转写 | CPU，int8；此后端没有 DirectML 支持 |
| ONNX Whisper 转写 | 编码器与解码器用 DirectML / CPU，FP32 计算、FP16 权重存储；信号预处理和束搜索在 CPU |
| TIFA 的模型 forward | DirectML，float32 |
| TIFA 频谱、读音评分中的非网络计算、路径解码、统计和 TextGrid | CPU |
| oto、编辑、拼接、导出 | 保持现有实现 |

计算设备可选「自动」「CPU」「DirectML GPU」。自动模式先尝试 DirectML；Whisper 失败仅重试当前语句，已完成文本保留。明确选择 DirectML 时失败报错，不静默切换为纯 CPU 会话；DirectML 内不支持的单个算子仍可由 CPU 执行。旧 faster-whisper 模型继续使用 CPU。进度显示实际设备。兼容旧任务中的 `cuda` 值，将其映射为 DirectML，不加载旧 CUDA DLL。

## ONNX Whisper

新增 `onnx_whisper` 后端，由模型格式自动选择。模型设置接受原 CTranslate2 目录，或本项目准备的 ONNX 目录。ONNX 权重来自 `onnx-community/whisper-small` 固定修订 `36050c46d777d46dc4b5f43f6d90574fc38f8732`，原模型为 `openai/whisper-small`。该原模型的官方模型卡声明 Apache-2.0；来源、修改说明和许可全文见 `LICENSES/Whisper-ONNX-NOTICE.txt` 和 `Apache-2.0.txt`。

原始 FP16 合并解码图在本机 DirectML 上产生过非有限数值；ORT 1.24 的高阶 FP16 融合也曾在 CPU 初始化失败。构建时使用 `prepare_onnx_whisper.py` 拆开首轮/带缓存解码图，两个图共用一份外部权重；权重继续用 FP16 保存，计算提升到 FP32。关闭高阶图融合，保留基础优化。转换工具 `onnx` 仅在开发机使用，不进入便携环境。

本后端固定来源语言为日语或普通话，5 路束搜索、448 token 上限；语句超过 30 秒时分块。跳过已有转写或修订文本，按语句保存检查点。取消仍由既有 Worker 进程终止处理。模型只放软件资源目录，工程只存引用和哈希。

2026-11-05 的真实长录音回归修复了束搜索结束条件：默认 patience 为 1，收集足够的完成序列后结束，避免继续扩展重复字符。对压缩比超过 2.4 或平均对数概率低于 −1 的结果，以 0.2、0.4、0.6、0.8、1.0 的温度重试采样；编码器结果在重试中复用。高无语音概率且置信度低时返回空文本；所有尝试仍不理想时提示校对。采样使用局部固定种子以便复现。搜索结束和重试设计参考 [Whisper 官方解码实现](https://github.com/openai/whisper/blob/main/whisper/decoding.py) 与旧后端的默认策略；并不保证与 CTranslate2 逐字一致。

```powershell
# 维护者先安装转换工具 onnx；仅用于准备模型，不打进便携包。
.venv/Scripts/python.exe packaging/download_onnx_whisper.py `
  --output build/whisper-small-original --proxy http://127.0.0.1:7890
.venv/Scripts/python.exe packaging/prepare_onnx_whisper.py `
  --source build/whisper-small-original --output build/whisper-small-prepared
.venv/Scripts/python.exe packaging/build_directml_portable.py `
  --base dist/VoiceBankGen --runtime build/directml-release-runtime `
  --onnx-model build/whisper-small-prepared --onnx-runtime build/onnxruntime-directml-wheel `
  --output build/whisper-directml-portable
```

`--onnx-runtime` 指解压后的官方 `onnxruntime-directml==1.24.4` wheel，替换旧 CPU ORT，不能同时混装两种 onnxruntime wheel。CPU 和 DirectML 复用同一套模型。通用上游下载清单仍提供历史 faster-whisper 组件；准备后的拆分模型由此次便携发行内置，源码用户按上述方式构建，不能把原始 merged 图直接当作准备完毕的模型。

必要验收：已授权的 3.25 秒日语片段，在 ONNX CPU 与 DirectML 均输出「私、一人でいるの嫌だな」；ORT 执行记录确认 DmlExecutionProvider 实际运行计算节点。不是性能基准，也不代表已完成普通话真实录音或多显卡验证。

Lightning 没有内置的 DirectML accelerator。本适配让其保留 CPU trainer，仅将上游 `backend.model` 的 forward 输入和模型参数移到 DirectML，输出回 CPU。频谱 FFT 和复杂解码无需在 DirectML 上运行。每次重试使用独立输出目录，避免把失败尝试的旧 TextGrid 当作新结果。未修改 TIFA 上游代码或模型权重。

DirectML 面向支持 DirectX 12 的 Windows GPU，包括部分 NVIDIA、AMD、Intel 设备；实际算子支持和性能取决于硬件、驱动与模型，不能保证比 CPU 或 CUDA 快。只验证了本机短片段，没有运行多显卡兼容性或速度基准。

## 依赖

2026-11-05 完成 PyPI 官方元数据和官方 Windows wheel 核对：

- `torch-directml==0.2.5.dev240914` 要求 `torch==2.4.1`、`torchvision==0.19.1`。
- 使用官方 PyPI 的 Windows **CPU wheel**（文件名版本为 2.4.1，内部为 2.4.1+cpu、cuda=None），并固定 NumPy 1.26.4；不要混入 CUDA 版 PyTorch。
- DirectML 扩展 wheel 为 8,955,958 字节，约 8.54 MiB。这不是整个环境大小；仍需 CPU PyTorch、torchvision 和 TIFA 其他依赖。
- 确切 URL、SHA-256、此次便携包的实际依赖版本与核对范围见 [依赖记录](directml_dependencies.json)。

源码开发环境（在仓库根目录的 PowerShell 执行；下载和安装会占用额外空间）：

```powershell
py -3.11 -m venv .directml-env
.directml-env/Scripts/python.exe -m pip install -r packaging/runtime_directml_requirements.txt
```

主程序仍使用 README 的 `.venv`，TIFA 子进程对 CPU/DirectML 共用 `.directml-env`。不要直接升级原 `.tifa-env`，两者的 PyTorch 版本不同。`.directml-env` 已加入忽略和源码导出禁止清单。未安装新环境时，自动模式可继续通过主程序的 CPU 环境对齐；明确选择 DirectML 会提示缺少环境。

## 下载清单与构建

内置旧 `runtime-catalog.json` 仍是历史 CPU 清单，不把它改名伪装成 DirectML 环境。维护者先使用一个全新的下载目录生成固定版本 DirectML 清单：

```powershell
.venv/Scripts/python.exe packaging/prepare_runtime_catalog.py `
  --backend directml --cache build/directml-assets `
  --pip-python .venv/Scripts/python.exe `
  --output src/utau_forge/data/runtime-catalog.json
```

此操作会下载官方依赖、Python、固定 TIFA 源码、FFmpeg 和 VAD 文件；主程序环境需已有 `silero-vad`、`faster-whisper` 以确定 VAD 版本。不要混用旧 CPU/CUDA 下载目录。若网络需代理，通过当前终端的 HTTP/HTTPS 代理配置连接上游；用户安装器仍提供代理和镜像输入。

随后按 [构建说明](BUILDING.md) 生成桌面主程序和 Online 安装器。`build_single_exe.py` 默认 `--acceleration directml`，会拒绝没有 DirectML 文件的旧清单；可用 `--runtime-catalog 路径` 指定新清单。新生成的主程序数据应包含该清单。Full 构建需先用该清单准备完整的共享环境和模型，拒绝误用原 CPU-only 缓存；不再接受 `--include-gpu`。

CPU-only 发行使用单独的 `--backend cpu` 清单，以及 `build_single_exe.py --acceleration cpu`，不与 DirectML 环境混装。便携版沿用 `build_portable.py` 的均衡 7z 设置；`--cpu-only` 不会从共享环境中粗暴移除依赖，需提供真正的 CPU-only 清单。旧 `tifa-gpu` 清单和带旧 CUDA 目录的原地压缩输入会被拒绝。

共享环境继续安装到 `runtime/tifa-cpu`，这是兼容历史工程的目录/组件 ID；其组件名称明确写成「TIFA CPU / DirectML 共享运行环境」。同一份 CPU PyTorch 可承担 CPU 回退，无需再附一整套 GPU PyTorch。新 fork 不自动清理原软件目录或环境；若要节省现有磁盘空间，先取得并确认新包后另行处理旧版本。

## 精简便携构建

`packaging/build_directml_portable.py` 从现有固定版本桌面发行和独立的嵌入式 DirectML 环境生成全新的目录。保留原发行依赖的代码对象及原生库，重新编译本项目代码；已共享的依赖从两个 EXE 的 PYZ 和 `_internal` 中移除，启动钩子在其他运行时钩子之前加载共享路径。这样无需把主程序的旧 PyTorch 与 TIFA 的 2.4.1 混用。

嵌入式 Python 使用官方 3.11.9 Windows x64 ZIP，`python311._pth` 包含 `Lib/site-packages` 和 `import site`。维护者将 `runtime_directml_requirements.txt` 安装到这个目录。某些 pip 的 `--target` 安装会漏放 fugashi wheel 的 `.data/data/lib/site-packages/fugashi/libmecab.dll`；必须按 wheel 原路径映射至 `Lib/site-packages/fugashi/libmecab.dll`，构建器会检查此文件。

```powershell
.venv/Scripts/python.exe packaging/build_directml_portable.py `
  --base dist/VoiceBankGen --runtime build/directml-release-runtime `
  --output build/directml-portable-release
.venv/Scripts/python.exe packaging/build_portable.py `
  --catalog build/directml-portable-release/install-catalog.json `
  --prepared-dir build/directml-portable-release/VoiceBankGen --reuse-in-place `
  --output dist/VoiceBankGen-DirectML.7z
```

输出目录和压缩文件必须不存在。构建记录和临时组件 ZIP 放在便携目录外，不进入最终 7z。发布目录排除重复环境、CUDA、静态 `.lib`、编译头文件、调试符号、测试、字节码缓存、内部 Markdown，以及 UniDic 的训练文件 `model.bin`；保留完整的推理词典和原 Whisper small / TIFA 模型。许可与第三方 NOTICE 保留。压缩使用均衡配置，不以更换小模型或极慢压缩换取体积。

进一步精简会从冻结 PYZ 和 `_internal` 同时移除仅用于模型导出的 `onnx` / `onnxscript` / `onnx_ir` / `ml_dtypes`，以及桌面推理和编辑流程未使用的 Pandas / Dask。实际推理使用的 `onnxruntime-directml` 保留。旧 faster-whisper / CTranslate2 / PyAV 仍保留兼容，避免改变已有手动模型路径的使用方式。此项约节省 20 MiB 的主程序 ZIP 组件；最终 7z 收益以实际发行大小为准。

## 许可与验证边界

保留 torch-directml wheel 的 MIT、完整 NOTICE 以及所有传递依赖自带的许可。未把 DirectML 扩展的 MIT 推广到模型、Qt、FFmpeg 或任意其他 DLL。TIFA 模型的 CC BY-NC-SA 4.0 不变。

2026-11-05 必要验收范围：精简目录的 GUI 启动成功；使用已授权的 3.25 秒日语片段，经打包 Worker 完成 CPU/int8 转写、显式 DirectML 对齐和候选分析，产生 2 份 TextGrid、26 个已识别音素及候选。测试数据、工程和日志仅保存在 `verification_runs`，不进入发行。不代表所有编辑/导出操作的完整回归，也没有速度对比或其他显卡测试。

## 真实工程回归（2026-11-05）

上述短片段是早期最小验收。后续使用一份 34 分 26 秒日语原录音及其既有工程完成了更广的回归：

- 全文件导入和 VAD 得到 598 段；取消后保留已完成文本，可继续处理剩余语句。
- 对 596 段已有非空文本运行全量 DirectML 对齐与候选分析。新旧均对齐同一组 548 段，未新增未知音素或越界音素；48 段未产生可用结果，旧工程中同样没有对齐结果。对全部 31 段与旧保存结果不同的标签序列及 3 个对照句，用当前 CPU 环境复测，34 段的标签和边界与当前 DirectML 完全一致。
- 旧工程的 CV 选音和共享 oto 保留，对应导出记录的 WAV 哈希全部一致。实际 GUI 控件操作覆盖拖动暂存、8 步撤销重做、备选音、导入音、拼接切分/保存/导出、搜索、发音格式切换；实际 Worker 覆盖增量同步、VC/VV 优化、ZIP 和最佳候选切片导出。实际 resampler 渲染单音成功。
- 原录音重新分析后，批量强制标注 12 个缺音，执行可选补音，得到 101 个 CV 选音；关联 CVVC 保留这些选音并导出 281 条选音。此为本地草稿验收，没有替换用户已安装的音源。
- 对原录音的前、中、后部及困难语句共 7 段，修复后的打包 Worker 完成了 ONNX DirectML 转写、DirectML 对齐和候选提取；同一组片段也通过旧 faster-whisper/CPU 流程。存在识别文字差异，困难语句重试仍较慢。

完整自动用例在修复前为 178 项通过；解码修复后的专项用例为 7 项通过，并重新构建 EXE 验证启动及上述真实流程。未在修复后重新转写全部 598 段，也未进行 OpenUtau 连续歌唱的人工听感评审、普通话真实录音或多显卡回归。功能兼容通过不等于识别准确率、速度或听感与旧后端完全等价。模型转换没有修改原录音、既有工程或原安装音源；测试数据与详细报告留在忽略目录，不进入源码发布包。
