# DirectML 源码发布范围

日期：2026-11-11。公开分支：`codex/directml`。
源码基线：`d16f0ec86067fa09caab8bbb55c58495cc7ac5b5`；该标识来自维护者本地仓库，不是公开仓库中的可 checkout 提交。

此分支从公开 main 首个 Release 建立新的源码提交，不包含私有开发历史。
Python 程序和测试源码保持上述基线；公开 README、许可审查、发布元数据及资源选择说明经过整理。
与 main 的 CPU / CUDA 路线分别维护；不包含 main 后续的歌声 / LRC 导入等开发功能。

包含桌面程序、安装器和下载/模型转换工具源码、测试代码、必要表与图标、公开文档和许可。
不包含模型、录音、歌词、完整词典、声库、工程数据库、虚拟环境、EXE、DLL 或本地试验记录。
下载地址和 SHA-256 只是资源元数据，不是模型的编码副本。

项目自研部分 MIT；普通话音节表 CC BY-SA 4.0。ONNX Whisper 权重的来源和改编声明见 LICENSES/Whisper-ONNX-NOTICE.txt；模型不随源码提供。
TIFA 模型的 CC BY-NC-SA 4.0、Qt / FFmpeg 等依赖的独立条件仍适用。
此次源码审核不代替离线安装包、便携包或模型包的二进制分发审核。

SOURCE_MANIFEST.json 记录此发布快照的文件 SHA-256（不包含清单自身）。修改后需重新生成。
后续使用 packaging/export_source.py 按白名单导出；新增文件先审核，禁止直接上传工作目录或未审核历史。
