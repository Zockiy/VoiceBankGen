# 主线源码许可复核

日期：2026-11-11；公开目标分支：main；当前程序基线：`e0036a8091af67d5cb11d64c22fe85b163de6b3e`。

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

## 歌声导入更新

本轮新增歌声＋LRC 时间匹配与波形检查、素材右键删除及相关测试/打包工具源码，未引入新的运行依赖或附带外部素材。实际歌声录音、歌词文件和试验工程全部排除。MIT 不授予用户上传或分发第三方歌词、录音的权利。依赖原文复核范围保持前述记录，本轮不重新声称完成全部传递依赖或二进制许可审计。
