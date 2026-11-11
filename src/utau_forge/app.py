"""Windows desktop controller and workflow-fork entry point."""
from __future__ import annotations
import json
import os
import sys
import time
import codecs
from .startup import begin_startup, startup_progress, finish_startup, icon_path
if __name__=="__main__":
    if "--worker" in sys.argv:
        from .startup import launch
        launch()
        raise SystemExit(0)
    begin_startup()
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

startup_progress(15,"加载界面基础组件…")
from PySide6.QtCore import QProcess, QSettings, QTimer, Qt, QUrl, QRectF, QObject, Signal, QRunnable, QThreadPool, QEvent
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QKeySequence, QShortcut, QIcon, QActionGroup
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QApplication,QCheckBox,QComboBox,QDialog,QDialogButtonBox,QDoubleSpinBox,QFileDialog,QFormLayout,QGroupBox,QHBoxLayout,QInputDialog,QLabel,QLineEdit,QListWidget,QListWidgetItem,QMainWindow,QMessageBox,QProgressBar,QPushButton,QSplitter,QStackedWidget,QTableWidget,QTableWidgetItem,QTextEdit,QVBoxLayout,QWidget,QMenu)
startup_progress(25,"加载音频计算组件…")
import numpy as np
startup_progress(40,"加载波形绘图组件…")
import pyqtgraph as pg
startup_progress(50,"加载音频文件组件…")
import soundfile as sf

startup_progress(55,"加载工程与音源服务…")
from .desktop_store import ProjectStore, visible_recent_projects, forget_recent_project, user_data_dir
from .desktop_sync import installations, resolve_installation
from .desktop_service import safe_path, cached_preview, preview_file, audition_file, preview_interval
from .languages import language, profile
from .models import Landmarks, Utterance
from .desktop_jobs import BackgroundProcess, FunctionTask
from .issue_labels import issue_text, issue_details
from .bank_aliases import split_variant, export_aliases, DEFAULT_ALIAS_FORMAT, to_katakana
startup_progress(75,"准备主界面…")

PROFILE_CHOICES={"日语 CV":"ja-cv","日语 CVVC":"ja-cvvc","普通话拼音 CV":"zh-cv","普通话拼音 CVVC":"zh-cvvc"}

MARK_NAMES={"start":"使用起点","overlap":"重叠","preutter":"对拍点","fixed_end":"固定区结束","end":"使用终点"}
COLORS=["#737f91","#46b8a6","#efb34c","#ec6f83","#779cec"]
JOB_NAMES={"sync":"同步修改","build":"制作音源","optimize_vc":"批量优化 VC / VV 音素","analyze":"自动切片","transitions":"准备 CVVC 版本","force_oto":"强制标注 oto","preview":"加载试听","recover":"恢复音源","repair":"可选补音","slices":"导出切片"}
MARK_HELP={"start":"片段开始使用的位置；对应 oto 的 offset。", "overlap":"与相邻音重叠衔接的参考位置；oto overlap 是它与使用起点的时间差。", "preutter":"与乐谱音符起点对齐的位置；oto preutterance 是它与使用起点的时间差。", "fixed_end":"固定区结束的位置；后方元音可由渲染引擎按音符时长伸缩。", "end":"片段停止使用的位置；对应 oto 的 cutoff，尾部剩余音频不使用。"}

BUTTON_HELP={
    "跨语种音素拼接":("素材与制作","用另一种语言的真实音素，为当前目标版本生成拼接候选。","先按实际来源语言自动切片，再新增目标语言版本。选择供体语言、是否允许近似及重叠时长；已有目标原音和人工选音保留。拼接结果需试听，缺少供体会列入报告；点击同步修改才导出/安装。"),
    "确定（跨语种拼接）":("对话框","开始为当前版本生成跨语种候选。","兼容模式只使用兼容规则；允许近似会标出发音差异。后台可取消，尚未发布的结果不会改变选音。"),
    "新增音":("标线与选音","在左侧新增一个自定义音标签。","填写标签，可设置罗马音/拼音文件名前缀；新增后选择该音，再用导入音添加候选。标签原样用于 oto，不能与已有导出别名重复；可撤销。"),
    "导入音":("标线与选音","将一个或多个本地音频加入左侧当前音的右侧候选列表。","不运行转写或对齐；复制原文件并转为 44.1kHz 单声道，提供可编辑初始标线。试听校对后点击使用这个音；原选音不变，更新声库需同步。"),
    "确定（新增音）":("对话框","保存新音标签及文件名前缀。","新增到当前版本的左侧列表；尚无候选，接着用导入音添加音频。标签或文件名不合法时给出原因。"),
    "新增备选音":("标线与选音","把当前候选另存为独立编号的选音，如 a_1、a_2 或 br_1。","主音保持原选音；在左侧选择编号音可更换候选，编号音共享所选候选的 oto 标线。保存只暂存，点击同步修改后导出。"),
    "移除备选音":("标线与选音","移除左侧选中的编号备选音。","保留原候选及主音，可撤销；同步后才从受管理的 oto 中移除该编号别名。"),
    "切分片段（S）":("拼接工作台","在红色时间光标处切分所选片段。","点击时间线或填写光标位置，选中片段后按 S（时间线获得焦点）或点击按钮；保留原音频、音量及淡化，可撤销。"),
    "音源工作台":("工程与版本","从首页返回当前已打开的工程。","显示当前版本的候选、波形与 oto；不重新运行对齐。"),
    "保存拼接草稿":("拼接工作台","保存可继续编辑的时间线配方。","写入工程 assemblies；Ctrl+S。不会改变正式选音，关闭时还会保留自动副本。"),
    "打开拼接草稿":("拼接工作台","恢复此前保存的片段、位置、裁剪和淡化设置。","只能载入当前工程的配方；支持撤销载入操作。"),
    "添加到时间线":("拼接工作台","追加所选完整音节、辅音或元音。","双击素材也可添加；与前片段按所设时长重叠。之后可移动或裁剪。"),
    "试听素材":("拼接工作台","播放所选素材部分；左侧单击也会自动播放。","按整音/辅音/元音选择试听；重复单击从头播放。不会移动时间线光标或改变选音。"),
    "移除选中片段":("拼接工作台","从时间线移除选中的片段。","Delete，也可在片段右键菜单操作；原候选保留，可撤销。"),
    "适合窗口":("拼接工作台","缩放时间线以显示全部片段。","Ctrl+滚轮可手动缩放，拖动底部滚动条查看其他时间范围。"),
    "播放 / 暂停（空格）":("拼接工作台","试听整条时间线的混音与淡化效果。","空格也可播放或暂停；红线指示位置，点击空白时间线可定位。"),
    "保存到候选":("拼接工作台","把时间线音频保存为目标音的未审核备选。","保留配方和父候选；回主工作台校对 oto、试听接受、正式使用，再手动同步。"),
    "单独导出 WAV":("拼接工作台","将时间线混合为一个独立 WAV。","默认罗马音文件名，不覆盖已有文件；右键片段或素材可只导出该部分。"),
    "拼接工作台":("拼接","从已有音节选择完整音、辅音或元音，自由拼接。","拖动移动、拖边裁剪、顶角调淡化；保存候选或单独导出。保存候选后需校对、接受和正式使用，再同步。"),
    "单独导出当前候选":("安装与导出","导出右侧当前候选的 UTAU 使用区间。","右键候选也可使用；保留原音高，文件名为罗马音。没有标线时请先初始化手动标线。"),
    "一键解锁标线":("标线与选音","解除当前候选五条标线的锁定。","保留位置，重新允许输入与拖动；修改仅暂存。无可用标线时先初始化。"),
    "初始化手动标线":("标线与选音","为没有自动稳定区的候选创建可编辑标线。","请试听并校对 oto；草稿可选用后同步，严格导出还需试听接受候选。质量提示仍保留。"),
    "强制标注 oto":("标线与选音","一键处理当前音源全部已有切片、因未找到稳定元音而缺失的音节。","每个未选用音节优先用最长切片，忽略稳定元音门槛生成 oto 并暂存选音；无切片、已选用、已排除或有其他严重问题的音节跳过。整批占一步撤销，保留质量提示；请试听校对后同步修改。"),
    "新建说话人工程":("工程与版本","为同一说话人的录音创建独立工程。","填写名称、选择 CV/CVVC 和音高模式，再选择保存位置；完成后可添加素材。"),
    "打开工程":("工程与版本","继续编辑已有工程。","选择包含 project.json 和 project.sqlite 的工程文件夹，加载已保存的选音、标线和安装绑定。"),
    "打开已安装音源":("工程与版本","从 OpenUtau 音源目录定位可编辑工程。","选择音源根目录；有工程标记时关联原工程，无标记时可导入 WAV/oto 为恢复工程。"),
    "首页":("工程与版本","返回新建、打开和最近工程页面。","当前工程仍保留在工作台；修改自动暂存，返回首页不会执行安装同步。"),
    "新增 CV/CVVC 版本":("工程与版本","创建 CVVC 视图，与对应 CV 共用基础音节，连接音单独保存。","选音、oto、备选、自定义音与音源设置在两边共用；同步修改会更新已绑定的关联版本。名称及安装路径各自保留。"),
    "批量优化 VC / VV 音素":("标线与选音","为当前 CVVC 的 VC、VV 及配套 CV 设置衔接重叠；基础 CV 修改也会作用于共用的 CV 库。","默认 30 ms，按可用前置区缩短；保留对拍点、音频、选音和锁定字段。VC/VV 参数各自在本版本保存；整批占一步撤销，仅暂存，点击同步修改应用。"),
    "音源设置":("工程与版本","设置当前版本的名称、音高、呼吸音、oto 发音格式、补音和导出规则。","oto 默认平假名，可改罗马音或片假名；呼吸音仅提取明确的 TIFA AP/br 标签。保存只暂存，制作或同步才应用。"),
    "模型设置":("模型与设备","查看自动识别的模型位置，或指定本地模型与词典路径。","留空自动识别；保存后下次启动保留，下一次任务使用。支持 Whisper、TIFA、UniDic 和 MFA。"),
    "模型 / 环境下载":("模型与设备","下载缺少的模型，或修复随单文件安装器提供的运行环境。","选择官方 GitHub/Hugging Face 或自定义镜像，勾选组件后下载；支持续传、哈希校验，下载源会记忆。CPU 环境随 EXE 提供，默认不安装 MFA。"),
    "按钮说明":("工程与版本","集中查看全部按钮的用途、条件和操作结果。","打开可滚动的说明窗口；也可把鼠标停在任一按钮上查看该按钮说明。"),
    "添加素材":("素材与制作","添加同一说话人的音频，供后续切片使用。","选择 WAV/FLAC/MP3 等音频，加入待处理列表；点击自动切片才开始处理和导入。"),
    "自动切片":("素材与制作","对素材执行语音检测、转写、音素对齐和候选提取。","使用上方所选设备和对齐方式；已有素材可继续已保存的转写进度，耗时取决于录音长度。"),
    "可选补音":("素材与制作","尝试为缺失的 CV 音生成元音补尾或辅音与元音拼接候选。","需要已分析的素材；更新候选和推荐，供试听校对。没有合适供体时保留缺音，安装仍需手动同步。"),
    "制作音源":("素材与制作","按当前选音、标线和设置准备可用音源。","在工程内生成 WAV、oto 和清单；已有音源复用未变化的音。生成后可试听处理音、渲染、导出或安装。"),
    "安装到目录":("安装与导出","首次安装音源或绑定新的安装位置。","选择该音源的根文件夹，准备最新音源并直接写入；完成后在 OpenUtau 重新载入音源。"),
    "同步修改":("安装与导出","统一应用暂存的选音、标线和设置修改。","更新当前安装及共用 CV 的关联安装；无绑定时先准备工程内音源。只改标线不重做 WAV，换一个候选只处理该音。关联目录缺失、无效或外部冲突会显示待处理结果。"),
    "导出 ZIP":("安装与导出","把最近一次制作的音源打包，方便分享或手动安装。","选择新的 ZIP 文件名；如需包含暂存修改，先制作音源或同步修改。已有同名 ZIP 不会被覆盖。"),
    "导出切片":("安装与导出","导出每音最佳 X 个切片或全部原始候选，X 可输入。","最佳 X 个包含各音已选用的主候选，再补最多 X−1 个；稳定元音优先并按时长降序，其余按切片总长度降序。未选用时取前 X 个，不足则全部导出。保留现有音高，文件名使用罗马音。"),
    "打开生成目录":("安装与导出","在文件管理器中打开当前版本最近的制作结果。","需要先制作音源；可查看 WAV、oto.ini、缺音清单及工程关联文件。"),
    "备份工程":("安装与导出","保存包含数据库、素材和派生音频的恢复副本。","选择备份父目录，生成新的备份文件夹；素材较多时备份耗时和占用空间也会增加。"),
    "重新自动标注（保留锁定）":("标线与选音","将当前候选未锁定的标线恢复为已有自动标注。","保留锁定字段的位置；只暂存标线，不重新运行语音对齐，应用到音源需同步。"),
    "恢复全部自动":("标线与选音","恢复当前候选全部五条自动标线。","同时取消字段锁定；修改仅暂存，可撤销，点击同步修改后才更新音源。"),
    "撤销":("标线与选音","撤销当前版本最近一次可撤销的编辑。","每次完成拖动记录一步，当前版本最多保留 8 步；也支持选音、编号音、自定义标签和校对。结果只暂存，Ctrl+Z。"),
    "重做":("标线与选音","恢复刚刚撤销的编辑。","最多恢复 8 步；新的有效编辑清除当前版本重做记录。结果只暂存，Ctrl+Shift+Z。"),
    "使用这个音":("标线与选音","把右侧选中的候选设为该音的正式选音。","单击候选仅试听；此按钮才保存正式选择。应用到安装目录需同步；候选表内 Enter 也可使用。"),
    "排除此候选":("标线与选音","标记当前候选不可用于制作。","状态会暂存，可撤销；若该候选正被使用，请另选其他候选，否则制作时该音会成为缺音。"),
    "试听接受候选":("标线与选音","记录当前短音、补尾或拼接候选已经试听接受。","严格导出需要该接受状态；接受不会自动更换正式选音，还需点击使用这个音。"),
    "播放 / 暂停":("试听与渲染","播放或暂停播放器当前加载的音频。","暂停后再次点击从暂停位置继续；空格键从使用起点播放右侧选中候选，到使用终点停止。"),
    "从使用起点播放":("试听与渲染","试听使用起点至使用终点；没有 oto 时播放完整切片。","请先加载候选原音；播放区间使用中栏当前标线，适合检查切片边界。"),
    "试听正式使用音（A）":("试听与渲染","选择并试听当前音的正式候选原音。","该音需已有正式选音；会把右侧选择切到使用中的候选，不更换正式选音。"),
    "试听当前备选（B / 空格）":("试听与渲染","试听当前候选；有 oto 时播放使用区间，没有 oto 时播放完整切片。","优先直接读取缓存；首次无缓存会后台生成。空格键也可使用，试听不会替换正式选音。"),
    "试听处理后":("试听与渲染","试听当前候选最近一次制作后的 WAV。","该候选必须已经制作；按当前标线裁剪处理后 WAV 试听。音高处理采用最近制作结果，新预设需先同步。"),
    "短音实际渲染":("试听与渲染","调用 UTAU resampler 渲染并试听约 180ms 的短音。","当前候选需已制作，并有可用 resampler；默认寻找 moresampler，否则提示选择引擎。测试音高为 A3。"),
    "长音实际渲染":("试听与渲染","调用 UTAU resampler 渲染并试听约 1000ms 的长音。","当前候选需已制作，并有可用 resampler；用于检查长元音及延长效果。测试音高为 A3。"),
    "取消任务":("任务与外部修改","中止正在执行的后台制作、分析、同步等任务。","保留已保存的阶段和工程编辑；同步若被中断，可点击同步修改按日志恢复。"),
    "任务详情":("任务与外部修改","查看后台进度、实际设备、错误和失败原因。","打开只读日志窗口，排查耗时、设备回退和处理失败时使用。"),
    "采用外部修改":("任务与外部修改","把安装目录中外部编辑的 oto/WAV 读入工程。","需要选择已绑定安装目录；生成新的候选或标线并暂存，校对后点击同步修改统一应用。"),
    "保留工程并覆盖（先备份）":("任务与外部修改","发生外部冲突时，选择以当前工程为准更新安装目录。","需要选择安装绑定；此按钮会立即准备并同步，先备份被替换的文件，再覆盖受管理文件。"),
    "暂缓同步":("任务与外部修改","保留当前暂存编辑，稍后再处理安装冲突。","不会写入安装目录或取消正在运行的任务；要中止后台任务，请使用取消任务。"),
    "确定（新建工程）":("对话框按钮","确认新工程名称、格式和音高模式。","随后选择保存位置；取消保存位置选择时不创建工程。"),
    "确定（音源设置）":("对话框按钮","保存当前音源设置。","设置暂存到工程；点击制作或同步才应用到音源文件。"),
    "取消（对话框）":("对话框按钮","退出当前设置或新建对话框。","不应用这个对话框中尚未确认的输入；先前已暂存的工程编辑仍保留。"),
    "关闭说明":("对话框按钮","关闭按钮说明窗口，返回工作台。","查看说明不会改变选音、标线或安装目录。"),
}

def button_help_text(name):
    _,purpose,result=BUTTON_HELP[name]
    return f"用途：{purpose}\n使用与结果：{result}"

def button_reference():
    sections=[]
    for group in dict.fromkeys(value[0] for value in BUTTON_HELP.values()):
        rows=[f"### {group}","","| 按钮 | 用途 | 使用条件与结果 |","| --- | --- | --- |"]
        hidden={"从使用起点播放","试听当前备选（B / 空格）","暂缓同步"}
        rows += [f"| {name} | {purpose} | {result} |" for name,(category,purpose,result) in BUTTON_HELP.items() if category==group and name not in hidden]
        sections.append("\n".join(rows))
    return "\n\n".join(sections)

def romaji_labels(profile_id):
    export_profile=profile(profile_id)
    labels={}
    for unit in export_profile.required_units(language(export_profile.language_id).inventory,"core"):
        roman=export_profile.filename_stem(unit)
        if unit.unit_type in {"VC","VV"}:
            roman=roman.split("_",1)[1].replace("_"," ")
        for alias in export_profile.aliases(unit):
            labels[alias]=roman
    return labels

class PreviewSignals(QObject):
    finished = Signal(int, object, str)

class PreviewTask(QRunnable):
    def __init__(self, store, identifier, token, spectrum=False):
        super().__init__()
        self.store, self.identifier, self.token, self.spectrum = store, identifier, token, spectrum
        self.signals = PreviewSignals()

    def run(self):
        try:
            result = preview_file(self.store, self.identifier, self.spectrum)
            self.signals.finished.emit(self.token, result, "")
        except Exception as error:
            self.signals.finished.emit(self.token, None, str(error))

class MarkLine(pg.InfiniteLine):
    def __init__(self, owner, **kwargs):
        super().__init__(**kwargs)
        self.owner=owner

    def mouseDragEvent(self,event):
        if self.movable and event.button()==Qt.MouseButton.LeftButton and event.isStart():
            self.owner.begin_mark_drag()
        super().mouseDragEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(str(icon_path())))
        self.setWindowTitle("VoiceBankGen · 音源工作台")
        self.resize(1420,900)
        self.setMinimumSize(1150,780)
        self.settings=QSettings("UTAUForge","DesktopWorkflowFork")
        self.settings.setFallbacksEnabled(False)
        from .ui_theme import apply_theme,THEME_KEY,DEFAULT_THEME
        self.theme_name=apply_theme(QApplication.instance(),self.settings.value(THEME_KEY,DEFAULT_THEME))
        self.store=None
        self.bank_id=None
        self.current_candidate=None
        self.preview_token=0
        self.process=None
        self.applying_task=None
        self.job_started_at=None
        self.last_job_report=None
        self.job_stage="空闲"
        self.preview_pool=QThreadPool(self)
        self.preview_pool.setMaxThreadCount(2)
        self.job_pool=QThreadPool(self)
        self.job_pool.setMaxThreadCount(1)
        self.preview_tasks={}
        self.wave_cache=OrderedDict()
        self.preview_result=None
        self.preview_ready=False
        self.romaji={}
        self.romaji_profile=None
        self.splice_windows={}
        self.splice_source_cache={}
        self.request=None
        self.pending_files=[]
        self.installation_id=None
        self.loading=False
        self.mark_dragging=False
        self.window_start=0
        self.wave_samples=0
        self.buffer=""
        self.preview_buffer=""
        self.player=QMediaPlayer(self)
        self.output=QAudioOutput(self)
        self.output.setVolume(.65)
        self.player.setAudioOutput(self.output)
        self.player.errorOccurred.connect(lambda *args:self.status("播放失败："+self.player.errorString()))
        self.pages=QStackedWidget()
        self.setCentralWidget(self.pages)
        self.create_home()
        self.create_workbench()
        self.create_menus()
        self.debounce=QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(400)
        self.debounce.timeout.connect(self.save_marks)
        self.activity_timer=QTimer(self)
        self.activity_timer.setInterval(250)
        self.activity_timer.timeout.connect(self.update_job_activity)
        self.shortcut("Ctrl+Z",lambda:self.undo(False))
        self.shortcut("Ctrl+Shift+Z",lambda:self.undo(True))
        self.shortcut("Ctrl+S",self.save_marks)
        self.shortcut("Space",self.play_current_selected)
        QApplication.instance().installEventFilter(self)
        self.setAcceptDrops(True)
        geometry=self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def shortcut(self,key,callback):
        shortcut=QShortcut(QKeySequence(key),self)
        shortcut.activated.connect(callback)

    def button(self,text,callback,layout):
        button=QPushButton(text)
        button.setToolTip(button_help_text(text))
        button.setAccessibleDescription(button.toolTip())
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def describe_dialog_buttons(self,buttons,confirmation):
        for kind,name in ((QDialogButtonBox.StandardButton.Ok,confirmation),(QDialogButtonBox.StandardButton.Cancel,"取消（对话框）")):
            button=buttons.button(kind)
            button.setText("确定" if kind==QDialogButtonBox.StandardButton.Ok else "取消")
            button.setToolTip(button_help_text(name))
            button.setAccessibleDescription(button.toolTip())

    def show_button_help(self):
        dialog=QDialog(self)
        dialog.setWindowTitle("操作说明 · VoiceBankGen")
        dialog.resize(1000,720)
        layout=QVBoxLayout(dialog)
        text=QTextEdit()
        text.setReadOnly(True)
        text.setMarkdown("# 工作流界面操作说明\n\n"
            "1. **素材与切分**：添加录音，选择语言，开始自动切片。\n"
            "2. **校对与选音**：左侧选音节，中间选候选，右侧试听和拖动标线；点击「使用这个音」确定正式选音。\n"
            "3. **同步与导出**：底部「同步修改」应用暂存结果；本页可安装音源或展开「导出」菜单。\n\n"
            "试听入口统一为「试听 / 停止 · 空格」；没有 oto 时播放完整切片。高级编辑集中于「标线操作」，"
            "导入、编号备选等集中于「候选操作」，处理音和实际渲染集中于「对比与渲染」。\n\n"
            "鼠标停在控件或菜单项上可查看说明。以下列出各操作用途，其中部分操作位于菜单内。\n\n"+button_reference())
        layout.addWidget(text)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close=buttons.button(QDialogButtonBox.StandardButton.Close)
        close.setText("关闭说明")
        close.setToolTip(button_help_text("关闭说明"))
        close.setAccessibleDescription(close.toolTip())
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def create_home(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        layout.setContentsMargins(70,55,70,55)
        title=QLabel("从语音制作你的 UTAU 音源")
        title.setStyleSheet("font-size:30px;font-weight:600")
        layout.addWidget(title)
        layout.addWidget(QLabel("同一说话人的素材 → 自动切片 → 校对候选 → 制作与安装"))
        row=QHBoxLayout()
        self.button("新建说话人工程",self.new_project,row).setProperty("primary",True)
        self.button("打开工程",self.open_project,row)
        self.button("打开已安装音源",self.open_installed,row)
        layout.addLayout(row)
        layout.addWidget(QLabel("最近工程"))
        self.recent=QListWidget()
        self.recent.setObjectName("recentProjects")
        self.recent.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.recent.setSpacing(0)
        self.recent.itemDoubleClicked.connect(lambda item:self.open_directory(Path(item.data(Qt.ItemDataRole.UserRole))))
        self.recent.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent.customContextMenuRequested.connect(self.recent_menu)
        self.recent.setToolTip("单击选中（蓝色边框），双击打开；右键删除最近工程记录，工程文件会保留。")
        layout.addWidget(self.recent)
        self.refresh_recent()
        self.home_status=QLabel("单击选择最近工程，双击打开；右键可移除最近记录，工程文件保留。")
        self.home_status.setWordWrap(True)
        layout.addWidget(self.home_status)
        self.pages.addWidget(page)

    def refresh_recent(self):
        self.recent.clear()
        for identifier,path in reversed(list(visible_recent_projects().items())):
            item=QListWidgetItem(f"{Path(path).name}    {path}")
            item.setData(Qt.ItemDataRole.UserRole,path)
            item.setData(Qt.ItemDataRole.UserRole+1,identifier)
            self.recent.addItem(item)

    def recent_menu(self,position):
        item=self.recent.itemAt(position)
        if not item:
            return
        self.recent.setCurrentItem(item)
        menu=QMenu(self)
        action=menu.addAction("删除最近工程记录")
        action.setToolTip("从最近工程列表移除；保留工程文件和音源安装关联。重新打开工程后会再次显示。")
        if menu.exec(self.recent.viewport().mapToGlobal(position))==action:
            try:
                forget_recent_project(item.data(Qt.ItemDataRole.UserRole+1))
                self.refresh_recent()
                self.home_status.setText("已删除最近工程记录；工程文件保留，可通过打开工程再次添加。")
            except (OSError,ValueError) as error:
                self.error(error)

    def create_workbench(self):
        from .workflow_ui import build_workbench
        build_workbench(self)

    def show_stage(self,index):
        if self.mark_dragging:
            for i,button in enumerate(self.step_buttons):
                button.setChecked(i==self.workflow_pages.currentIndex())
            return
        self.save_marks()
        if index!=1:
            self.player.stop()
            self.advanced_oto.close()
        self.pages.setCurrentIndex(1)
        self.workflow_pages.setCurrentIndex(index)
        for i,button in enumerate(self.step_buttons):
            button.setChecked(i==index)
        if index==0:
            from .workflow_ui import refresh_sources
            refresh_sources(self)
        elif index==2:
            self.refresh_installations()

    def remove_pending_source(self):
        if self.process or self.applying_task:
            return
        rows={item.row() for item in self.source_table.selectedItems()}
        indices={self.source_table.item(row,0).data(Qt.ItemDataRole.UserRole) for row in rows}
        previous=len(self.pending_files)
        self.pending_files=[path for i,path in enumerate(self.pending_files) if i not in indices]
        from .workflow_ui import refresh_sources
        refresh_sources(self)
        self.status(f"已移出 {previous-len(self.pending_files)} 个待处理素材；剩余 {len(self.pending_files)} 个，原文件保留")

    def create_menus(self):
        self.menu_actions=[]
        groups={
            "工程": [("新建说话人工程",self.new_project,False),("打开工程",self.open_project,False),("打开已安装音源",self.open_installed,False),None,("备份工程",self.backup,True),("首页",lambda:self.pages.setCurrentIndex(0),False),("音源工作台",lambda:self.show_stage(1),True)],
            "校对": [("撤销",lambda:self.undo(False),True),("重做",lambda:self.undo(True),True),None,("强制标注 oto",self.force_oto,True)],
            "音源": [("新增 CV/CVVC 版本",self.new_bank,True),("音源设置",self.bank_settings,True),None,("批量优化 VC / VV 音素",self.optimize_vc,True),("制作音源",lambda:self.start_job("build"),True),("打开生成目录",self.open_build,True)],
            "工具": [("拼接工作台",self.open_splice,True),("可选补音",self.repair,True),("跨语种音素拼接",self.cross_language,True),None,("模型设置",self.model_settings,False),("模型 / 环境下载",self.resource_downloads,False)],
            "帮助": [("按钮说明",self.show_button_help,False),("任务详情",self.job_details,False)]}
        busy_names={"新增音","导入音","新建说话人工程","打开工程","打开已安装音源","新增 CV/CVVC 版本","音源设置","模型设置","模型 / 环境下载","拼接工作台","自动切片","可选补音","制作音源","安装到目录","同步修改","备份工程","导出 ZIP","导出切片"}
        for title, entries in groups.items():
            menu=self.menuBar().addMenu(title)
            menu.setToolTipsVisible(True)
            for entry in entries:
                if entry is None:
                    menu.addSeparator()
                    continue
                name,callback,needs_project=entry
                action=menu.addAction(name)
                action.setToolTip(button_help_text(name))
                action.setStatusTip(action.toolTip())
                action.triggered.connect(callback)
                self.menu_actions.append((action,needs_project,name in busy_names))
            menu.aboutToShow.connect(self.update_menus)
            if title=="工具":
                menu.addSeparator()
                appearance=menu.addMenu("界面主题")
                appearance.setToolTipsVisible(True)
                self.theme_group=QActionGroup(self)
                self.theme_group.setExclusive(True)
                self.theme_actions={}
                for key,label in (("dark","深色（默认）"),("light","浅色")):
                    action=appearance.addAction(label)
                    action.setCheckable(True)
                    action.setChecked(self.theme_name==key)
                    action.setToolTip("立即切换界面配色并记住选择；下次启动沿用。不改变工程或音频。")
                    action.triggered.connect(lambda checked,k=key:self.set_theme(k) if checked else None)
                    self.theme_group.addAction(action)
                    self.theme_actions[key]=action
        self.update_menus()

    def set_theme(self,name):
        from .ui_theme import apply_theme,THEME_KEY
        self.theme_name=apply_theme(QApplication.instance(),name)
        self.settings.setValue(THEME_KEY,self.theme_name)
        self.settings.sync()
        for key,action in self.theme_actions.items():
            action.setChecked(key==self.theme_name)

    def update_menus(self):
        busy=self.process is not None or self.applying_task is not None
        self.sync_button.setEnabled(self.store is not None and not busy)
        self.add_unit_button.setEnabled(self.store is not None and not busy)
        self.import_sound_button.setEnabled(self.store is not None and not busy)
        self.cancel_button.setEnabled(self.process is not None and not self.cancelled if hasattr(self,"cancelled") else False)
        self.task_panel.setVisible(busy or self.job_started_at is not None)
        self.cancel_button.setVisible(busy)
        for control in self.workflow_controls:
            control.setEnabled(self.store is not None and not busy)
        candidate_actions={"新增备选音","试听接受候选","排除此候选","单独导出当前候选","一键解锁标线","初始化手动标线","重新自动标注（保留锁定）","恢复全部自动","高级 oto 数值","试听处理后","短音实际渲染","长音实际渲染"}
        for action in self.workflow_actions:
            action.setEnabled(self.store is not None and not busy and (action.text() not in candidate_actions or self.selected_id() is not None))
        for action,needs_project,blocks_busy in self.menu_actions:
            action.setEnabled((not needs_project or self.store is not None) and (not blocks_busy or not busy)
                              and (not self.applying_task or not needs_project or action.text()=="任务详情"))
            if action.text()=="强制标注 oto":
                action.setEnabled(self.store is not None and not busy)
            if action.text()=="批量优化 VC / VV 音素":
                action.setEnabled(self.store is not None and not busy and self.store.bank(self.bank_id)["profile_id"].endswith("-cvvc"))

    def open_splice(self):
        if not self.store or self.process:
            return
        self.save_marks()
        self.player.stop()
        from .splice_workbench import SpliceWorkbench
        key=(str(self.store.directory),self.bank_id)
        if key in self.splice_windows:
            dialog=self.splice_windows[key]
            dialog.show()
            dialog.load_source_index()
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog=SpliceWorkbench(self)
        self.splice_windows[key]=dialog
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.destroyed.connect(lambda *_:self.splice_windows.pop(key,None))
        dialog.show()

    def model_settings(self):
        if self.process:
            self.status("后台任务完成或取消后可修改模型设置")
            return
        from .desktop_engine import application_root
        from .model_settings_dialog import ModelSettingsDialog
        try:
            dialog=ModelSettingsDialog(application_root(),self.store.directory if self.store else None,self)
            if dialog.exec():
                self.status("模型设置已保存；下一次任务生效")
        except (OSError,ValueError) as error:
            self.error(error)

    def resource_downloads(self, checked=False, *, first_run=False):
        if self.process:
            self.status("后台任务完成或取消后可下载模型与环境")
            return
        from .desktop_engine import application_root
        from .resource_download_dialog import ResourceDownloadDialog
        try:
            ResourceDownloadDialog(application_root(), self, first_run=first_run).exec()
        except (OSError, ValueError) as error:
            self.error(error)

    def candidate_menu(self,position):
        row=self.candidates.rowAt(position.y())
        if row<0:
            return
        self.candidates.selectRow(row)
        menu=QMenu(self)
        action=menu.addAction("单独导出当前候选")
        action.setToolTip(button_help_text("单独导出当前候选"))
        add=menu.addAction("新增备选音")
        add.setToolTip(button_help_text("新增备选音"))
        import_sound=menu.addAction("导入音")
        import_sound.setToolTip(button_help_text("导入音"))
        chosen=menu.exec(self.candidates.viewport().mapToGlobal(position))
        if chosen==action:
            self.export_candidate()
        elif chosen==add:
            self.add_variant()
        elif chosen==import_sound:
            self.import_sound()

    def target_menu(self,position):
        item=self.targets.itemAt(position)
        if item:
            self.targets.setCurrentItem(item)
        menu=QMenu(self)
        new=menu.addAction("新增音")
        new.setToolTip(button_help_text("新增音"))
        imported=menu.addAction("导入音")
        imported.setToolTip(button_help_text("导入音"))
        imported.setEnabled(self.selected_alias() is not None and self.process is None)
        new.setEnabled(self.store is not None and self.process is None)
        action=menu.exec(self.targets.viewport().mapToGlobal(position))
        if action==new:
            self.add_unit()
        elif action==imported:
            self.import_sound()

    def add_unit(self):
        if not self.store or self.process:
            return
        self.save_marks()
        from .custom_units import default_stem
        dialog=QDialog(self)
        dialog.setWindowTitle("新增音")
        form=QFormLayout(dialog)
        label=QLineEdit()
        label.setObjectName("new_sound_label")
        label.setPlaceholderText("例如 breath_soft、笑声")
        label.setToolTip("左侧显示并作为 oto 自定义别名；与已有音不能重名。自定义标签不随罗马音/平假名设置转换。")
        form.addRow("音标签",label)
        stem=QLineEdit()
        stem.setObjectName("new_sound_stem")
        stem.setPlaceholderText("留空自动生成")
        stem.setToolTip("WAV 文件名前缀，使用罗马音/拼音、数字、下划线或连字符。非英文标签无法推断发音时，留空使用稳定编号；可填写 xiaosheng 等拼音便于校对。")
        label.textChanged.connect(lambda text:stem.setPlaceholderText(default_stem(text.strip()) if text.strip() else "留空自动生成"))
        form.addRow("文件名前缀（可选）",stem)
        form.addRow(QLabel("新增后选择该音，点击“导入音”添加音频候选。"))
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.describe_dialog_buttons(buttons,"确定（新增音）")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if not dialog.exec():
            return
        try:
            unit=self.store.create_custom_unit(self.bank_id,label.text(),stem.text())
            self.search.clear()
            self.kind.setCurrentText("全部")
            self.only_missing.setChecked(False)
            self.refresh_targets()
            for row in range(self.targets.count()):
                if self.targets.item(row).data(Qt.ItemDataRole.UserRole)==unit["alias"]:
                    self.targets.setCurrentRow(row)
                    break
            self.status("已新增音“"+unit["label"]+"”；请点击“导入音”添加候选，修改已暂存")
        except Exception as error:
            self.error(error)

    def import_sound(self):
        if not self.store or self.process:
            return
        alias=self.selected_alias()
        if not alias:
            self.status("请先选择左侧的音，或点击“新增音”创建标签")
            return
        name=self.store.display_alias(self.bank_id,alias)
        files,_=QFileDialog.getOpenFileNames(self,"导入音 → "+name,filter="音频 (*.wav *.flac *.mp3 *.ogg *.m4a);;所有文件 (*)")
        if files:
            self.start_job("import_candidates",selection_alias=alias,files=files)

    def export_candidate(self):
        identifier=self.selected_id()
        if not identifier:
            self.status("请先选择候选")
            return
        self.save_marks()
        if self.oto_invalid:
            self.status("请先修正无效标线再导出")
            return
        try:
            from .desktop_splice import clip_from_candidate,export_audio
            candidate=self.store.candidate(identifier,self.bank_id)
            if candidate.landmarks is None:
                raise ValueError("候选没有使用标线，请先初始化手动标线")
            clip=clip_from_candidate(self.store,identifier,self.bank_id)
            stem=self.romaji.get(candidate.alias,"clip").replace(" ","_")
            filename,_=QFileDialog.getSaveFileName(self,"单独导出当前候选",stem+"_"+identifier[:8]+".wav","WAV (*.wav)")
            if filename:
                export_audio(self.store,[clip],filename)
                self.status("已单独导出："+filename)
        except Exception as error:
            self.error(error)

    def status(self,text):
        self.message.setText(text)
        self.message.setToolTip(text)
        self.home_status.setText(text)

    def error(self,error):
        self.status(str(error))

    def new_project(self):
        dialog=QDialog(self)
        dialog.setWindowTitle("新建说话人工程")
        form=QFormLayout(dialog)
        name_field=QLineEdit()
        form.addRow("音源名称",name_field)
        format_field=QComboBox()
        format_field.setToolTip("CV 使用单音节；CVVC 增加 VC/VV 连接过渡。以后可在同一工程中新增另一种版本。")
        for label,pid in PROFILE_CHOICES.items(): format_field.addItem(label,pid)
        form.addRow("目标声库格式",format_field)
        source_field=QComboBox()
        source_field.addItem("日语","ja"); source_field.addItem("普通话","zh")
        source_field.setToolTip("选择实际录音语言；与目标不同需在切片后执行跨语种音素拼接。")
        form.addRow("来源语音语言",source_field)
        pitch=QCheckBox("调整到标准音高（默认关闭，可在音源设置中开启）")
        pitch.setToolTip("制作时把选音调整到统一音高；取消则保留原音高。目标音高可在音源设置中更改。")
        pitch.setChecked(False)
        form.addRow(pitch)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.describe_dialog_buttons(buttons,"确定（新建工程）")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if not dialog.exec() or not name_field.text().strip():
            return
        name=name_field.text().strip()
        parent=QFileDialog.getExistingDirectory(self,"选择工程保存位置")
        if not parent:
            return
        try:
            if any(character in name for character in '<>:"/\\|?*'):
                raise ValueError("名称不能包含文件路径字符")
            store=ProjectStore.create(Path(parent)/name.strip(),name.strip(),profile_id=format_field.currentData(),source_language=source_field.currentData())
            bank=store.banks()[0]
            with store.connection() as connection:
                connection.execute("UPDATE banks SET profile_id=? WHERE id=?",(format_field.currentData(),bank["id"]))
            store.update_bank(bank["id"],name,{"pitch_mode":"flatten" if pitch.isChecked() else "preserve","target_note":"auto","repair":False,"release_mode":"draft"})
            self.open_directory(store.directory)
            self.add_files()
        except Exception as error:
            self.error(error)

    def open_project(self):
        directory=QFileDialog.getExistingDirectory(self,"选择工程文件夹")
        if directory:
            self.open_directory(Path(directory))

    def open_directory(self,directory):
        if self.process:
            self.status("请等待当前任务完成或取消后再打开工程")
            return
        try:
            self.save_marks()
            self.preview_token+=1
            self.preview_ready=False
            self.preview_result=None
            self.wave_cache.clear()
            self.store=ProjectStore(directory)
            self.current_candidate=None
            self.pending_files=[]
            self.loading=True
            self.banks.clear()
            for bank in self.store.banks():
                self.banks.addItem(bank["name"]+" · "+bank["profile_id"].upper(),bank["id"])
            self.loading=False
            self.bank_id=self.banks.currentData()
            self.source_language.blockSignals(True)
            source=self.settings.value("source_language/"+self.store.project_id,self.store.meta("source_languages")[0])
            self.source_language.setCurrentIndex(max(0,self.source_language.findData(source)))
            self.source_language.blockSignals(False)
            if source=="zh": self.backend.setCurrentIndex(self.backend.findData("tifa"))
            self.refresh_installations()
            self.project_label.setText(self.store.meta("display_name"))
            self.pages.setCurrentIndex(1)
            self.refresh_targets()
            self.refresh_recent()
            self.status("工程已保存 · "+str(self.store.directory))
            self.update_menus()
            with self.store.connection() as connection:
                has_candidates=connection.execute("SELECT EXISTS(SELECT 1 FROM candidates)").fetchone()[0]
            self.show_stage(1 if has_candidates else 0)
        except Exception as error:
            self.error(error)

    def open_installed(self):
        path=QFileDialog.getExistingDirectory(self,"选择已安装音源根目录")
        if not path:
            return
        root=Path(path)
        if (root/"uforge.project-link.json").exists():
            try:
                store,identifier=resolve_installation(root)
            except FileNotFoundError:
                original=QFileDialog.getExistingDirectory(self,"定位原工程；取消后可导入为恢复工程")
                if original:
                    try:
                        store,identifier=resolve_installation(root,Path(original))
                    except Exception as error:
                        self.error(error)
                        return
                else:
                    self.recover(root)
                    return
            except Exception as error:
                self.error(error)
                return
            self.open_directory(store.directory)
            with store.connection() as connection:
                bank_id=connection.execute("SELECT bank_id FROM installations WHERE id=?",(identifier,)).fetchone()[0]
            self.banks.setCurrentIndex(self.banks.findData(bank_id))
            self.installation_id=identifier
            self.refresh_installations()
            self.status("已关联安装目录："+str(root)+"；编辑将暂存，点击“同步修改”统一应用")
        else:
            self.recover(root)

    def recover(self,root):
        suggested=root.name
        character=root/"character.txt"
        if character.exists():
            try:
                text=character.read_bytes().decode("cp932")
                suggested=next((line[5:].strip() for line in text.splitlines() if line.startswith("name=")),suggested)
            except UnicodeDecodeError:
                pass
        name,ok=QInputDialog.getText(self,"恢复音源","恢复工程的音源名称",text=suggested)
        if not ok:
            return
        parent=QFileDialog.getExistingDirectory(self,"选择恢复工程保存位置")
        if parent:
            destination=Path(parent)/(name+"_recovered_"+uuid4().hex[:6])
            saved=json.loads((root/"uforge.manifest.json").read_text(encoding="utf-8")) if (root/"uforge.manifest.json").exists() else {}
            pid=saved.get("profile_id")
            if pid not in PROFILE_CHOICES.values():
                choice,ok=QInputDialog.getItem(self,"恢复格式","原音源语言及格式",list(PROFILE_CHOICES),0,False)
                if not ok: return
                pid=PROFILE_CHOICES[choice]
            self.start_job("recover",project=str(destination),source=str(root),name=name,profile=pid)

    def switch_bank(self):
        if self.loading or not self.store:
            return
        self.save_marks()
        self.preview_ready=False
        self.current_candidate=None
        self.bank_id=self.banks.currentData()
        self.installation_id=None
        self.refresh_installations()
        self.refresh_targets()

    def new_bank(self):
        if not self.store:
            return
        if self.process:
            self.status("后台任务完成后可新增版本")
            return
        self.save_marks()
        choice,ok=QInputDialog.getItem(self,"新增音源版本","格式（共享已有音素，无需重跑转写）",list(PROFILE_CHOICES),0,False)
        if ok:
            bank=self.store.create_bank(PROFILE_CHOICES[choice],copy_units_from=self.bank_id)
            self.store.recommend(bank["id"],preserve_shared_cv=True)
            self.banks.blockSignals(True)
            self.banks.clear()
            for version in self.store.banks():
                self.banks.addItem(version["name"]+" · "+version["profile_id"].upper(),version["id"])
            self.banks.setCurrentIndex(self.banks.findData(bank["id"]))
            self.banks.blockSignals(False)
            self.switch_bank()
            if PROFILE_CHOICES[choice].endswith("-cvvc"):
                self.start_job("transitions")

    def optimize_vc(self):
        if not self.store or self.process or self.mark_dragging:
            return
        self.save_marks()
        fade,ok=QInputDialog.getDouble(self,"批量优化 VC / VV 音素","交叉淡化时长（ms，默认 30）：\n同时修正当前 VC、VV 及共用 CV 的重叠，基础 CV 修改也影响关联版本。\n保持对拍点与锁定参数；整批可撤销，完成后点击同步修改。",30,5,100,1)
        if ok:
            self.start_job("optimize_vc",fade_ms=fade)

    def bank_settings(self):
        if not self.store:
            return
        bank=self.store.bank(self.bank_id)
        self.save_marks()
        dialog=QDialog(self)
        dialog.setWindowTitle("音源设置")
        form=QFormLayout(dialog)
        name=QLineEdit(bank["name"])
        form.addRow("音源名称",name)
        notation=QComboBox()
        notation.setObjectName("alias_format")
        notation.addItem("平假名（默认）：あ / し / あ_1", "hiragana")
        notation.addItem("罗马音：a / shi / a_1", "romaji")
        notation.addItem("片假名：ア / シ / ア_1", "katakana")
        if bank["profile_id"].startswith("zh-"):
            notation.clear(); notation.addItem("拼音（ü 写 v）：a / shi / lv / a_1","romaji")
        notation.setCurrentIndex(max(0,notation.findData(bank["preset"].get("alias_format",DEFAULT_ALIAS_FORMAT))))
        notation.setToolTip("只改变 oto 发音别名；WAV 文件名始终为罗马音，不重新处理音频。JA CVVC（legacy）使用平假名；罗马音需选择保留别名的音素器。呼吸音三种格式均为 br。")
        form.addRow("oto 发音格式",notation)
        from .breaths import is_breath
        from .models import Phone
        breath_count=sum(is_breath(phone) for phone in self.store.records("alignments",Phone))
        breaths=QCheckBox(f"加入呼吸音（已对齐标记 {breath_count} 段）")
        breaths.setObjectName("include_breaths")
        breaths.setChecked(bool(bank["preset"].get("include_breaths")))
        breaths.setToolTip("从现有 TIFA AP/br 标签提取，无需重跑对齐；静音和 EP 不算呼吸。启用后在左侧“呼吸”选择候选，可新增 br_1 等备选。呼吸保留原音高与音量；关闭仅停止导出，已选候选保留。")
        form.addRow(breaths)
        if not breath_count:
            form.addRow(QLabel("当前记录没有明确呼吸标签；启用后 br 将显示缺音。"))
        flatten=QCheckBox("调整到标准音高（WORLD）")
        flatten.setToolTip("制作或同步时用 WORLD 将选音调整到目标音高；取消则保留原音高。更改后可能需要处理所有选音。")
        flatten.setChecked(bank["preset"].get("pitch_mode")=="flatten")
        form.addRow(flatten)
        note=QLineEdit(bank["preset"].get("target_note","auto"))
        note.setToolTip("输入 auto、C4、A3 等音名；auto 根据选音的音高选择最近半音并固定此版本目标。仅开启标准音高时使用。")
        note.setPlaceholderText("auto / C4 / A3")
        form.addRow("目标音高",note)
        repair=QCheckBox("草稿允许使用标记的拼接候选")
        repair.setToolTip("允许推荐已生成的补尾或拼接候选；不会在保存设置时生成补音。需要生成时点击可选补音。")
        repair.setChecked(bank["preset"].get("repair",False))
        form.addRow(repair)
        auto_fade=QCheckBox("自动选择拼接交叉淡化时长（10–40 ms）")
        auto_fade.setChecked(bank["preset"].get("splice_crossfade_ms") is None)
        auto_fade.setToolTip("自动补音根据音高选择重叠长度；关闭后使用下方自定义时长。已有拼接不会在保存设置时重做。")
        form.addRow(auto_fade)
        crossfade=QDoubleSpinBox()
        crossfade.setRange(1,200)
        crossfade.setDecimals(1)
        crossfade.setSuffix(" ms")
        crossfade.setValue(bank["preset"].get("splice_crossfade_ms") or 20)
        crossfade.setEnabled(not auto_fade.isChecked())
        crossfade.setToolTip("辅音/元音或两个片段重叠并淡出淡入的时长。用于下一次自动补音及手动工作台追加的默认值；片段太短会拒绝自动拼接。已有拼接可在工作台拖动调整。")
        auto_fade.toggled.connect(lambda enabled:crossfade.setEnabled(not enabled))
        form.addRow("拼接重叠 / 淡化时长",crossfade)
        strict=QCheckBox("严格导出（补音必须先试听接受）")
        strict.setToolTip("补尾、拼接、中文待校对或未找到稳定元音区的手动 oto 候选需要先点击试听接受候选，并正式使用。其他原音不需要逐个接受。")
        strict.setChecked(bank["preset"].get("release_mode")=="strict")
        form.addRow(strict)
        shared_cv=None
        if bank["profile_id"].endswith("-cvvc"):
            shared_cv=QComboBox()
            shared_cv.addItem("请选择基础 CV 版本",None)
            for candidate_bank in self.store.banks():
                if candidate_bank["profile_id"]==bank["profile_id"].replace("-cvvc","-cv"):
                    shared_cv.addItem(candidate_bank["name"],candidate_bank["id"])
            shared_cv.setCurrentIndex(max(0,shared_cv.findData(bank["preset"].get("cv_base_bank_id"))))
            shared_cv.setToolTip("基础 CV 的设置、选音和 oto 双向共用；VC/VV 单独保存。切换来源后采用新基础版本现有设置，保存后可重新打开调整。")
            form.addRow("共用基础 CV",shared_cv)
        form.addRow(QLabel("CVVC 的基础 CV 在两边共用；修改会影响关联版本，VC 连接音单独保存。"))
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.describe_dialog_buttons(buttons,"确定（音源设置）")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            try:
                if shared_cv is not None and shared_cv.currentData() is None:
                    raise ValueError("请选择要共用的基础 CV 版本")
                if shared_cv is not None and shared_cv.currentData()!=bank["preset"].get("cv_base_bank_id"):
                    if shared_cv.currentData() is None:
                        raise ValueError("请选择要共用的基础 CV 版本")
                    self.store.link_cvvc(self.bank_id,shared_cv.currentData())
                    self.current_candidate=None
                    self.preview_ready=False
                    self.refresh_targets()
                    self.status("已关联共用 CV，采用该 CV 的现有设置；需要调整时重新打开音源设置。原版本数据已保留备份。")
                    return
                from .build import _note_to_hz
                # Validation uses the same parser as production rendering.
                if flatten.isChecked() and note.text().strip()!="auto":
                    _note_to_hz(note.text().strip())
                preset={**bank["preset"],"pitch_mode":"flatten" if flatten.isChecked() else "preserve","target_note":note.text().strip(),"repair":repair.isChecked(),"release_mode":"strict" if strict.isChecked() else "draft","splice_crossfade_ms":None if auto_fade.isChecked() else crossfade.value(),"alias_format":notation.currentData(),"include_breaths":breaths.isChecked()}
                if preset["pitch_mode"]!=bank["preset"].get("pitch_mode") or preset["target_note"]!=bank["preset"].get("target_note"):
                    preset.pop("resolved_hz",None)
                self.store.update_bank(self.bank_id,name.text(),preset)
                self.store.recommend(self.bank_id)
                self.banks.setItemText(self.banks.currentIndex(),name.text()+" · "+bank["profile_id"].upper())
                self.current_candidate=None
                self.preview_ready=False
                self.refresh_targets()
                self.status("设置已暂存并在共用 CV / CVVC 中生效；点击“同步修改”统一应用到已绑定版本")
            except Exception as error:
                self.error(error)

    def filter_targets(self):
        self.target_search_timer.stop()
        query=self.search.text().strip().casefold()
        first=None
        for row in range(self.targets.count()):
            item=self.targets.item(row)
            hidden=bool(query and query not in item.data(Qt.ItemDataRole.UserRole+1))
            item.setHidden(hidden)
            if not hidden and first is None:
                first=item
        current=self.targets.currentItem()
        if current is not None and not current.isHidden():
            return
        self.targets.setCurrentItem(first)
        if first is None:
            self.show_alias(None)

    def refresh_targets(self):
        if not self.store or not self.bank_id:
            return
        self.target_search_timer.stop()
        previous=self.targets.currentItem()
        previous_alias=previous.data(Qt.ItemDataRole.UserRole) if previous else None
        self.targets.blockSignals(True)
        self.targets.clear()
        selections=self.store.selections(self.bank_id)
        target_language=profile(self.store.bank(self.bank_id)["profile_id"]).language_id
        with self.store.connection() as connection:
            counts={row[0]:row[1] for row in connection.execute("SELECT alias,count(*) FROM candidates WHERE coalesce(json_extract(payload,'$.target_language'),json_extract(payload,'$.source_language'))=? GROUP BY alias",(target_language,))}
        targets=self.store.targets(self.bank_id)
        bank=self.store.bank(self.bank_id)
        profile_id=bank["profile_id"]
        if self.romaji_profile!=profile_id:
            self.romaji={**romaji_labels("ja-cvvc"),**romaji_labels("zh-cvvc"),**romaji_labels(profile_id)}
            self.romaji_profile=profile_id
        custom_units=self.store.custom_units(self.bank_id)
        custom_labels={unit["alias"]:unit["label"] for unit in custom_units}
        self.romaji={key:value for key,value in self.romaji.items() if not key.startswith("custom:")}
        self.romaji.update({unit["alias"]:unit["filename_stem"] for unit in custom_units})
        restore=None
        for alias,kind in targets:
            selected=selections.get(alias)
            base,suffix=split_variant(alias)
            roman=self.romaji.get(base,base)+suffix
            display_base=to_katakana(base) if bank["preset"].get("alias_format",DEFAULT_ALIAS_FORMAT)=="katakana" else base
            label=custom_labels.get(base,display_base)+suffix
            if self.only_missing.isChecked() and selected:
                continue
            if self.kind.currentText()=="CV/V" and kind not in {"CV","V"}:
                continue
            if self.kind.currentText() in {"VC","VV","呼吸","自定义"} and kind!=self.kind.currentText():
                continue
            if self.kind.currentText()=="编号备选" and not suffix:
                continue
            name=f"{label} ({roman})" if roman!=label else label
            if kind=="呼吸":
                name="呼吸音 ("+roman+")"
            item=QListWidgetItem(f"{name}  · {kind}{' / 编号备选' if suffix else ''}  {'已选' if selected else '缺音'}  / {counts.get(base,0)} 候选")
            item.setToolTip(f"音名：{label}\n罗马音 / 文件前缀：{roman}\n音型：{kind}\n单击查看候选，试听只播放使用起点到使用终点。")
            item.setData(Qt.ItemDataRole.UserRole,alias)
            item.setData(Qt.ItemDataRole.UserRole+1,(label+" "+alias+" "+roman).casefold())
            item.setHidden(bool(self.search.text().strip() and self.search.text().strip().casefold() not in item.data(Qt.ItemDataRole.UserRole+1)))
            self.targets.addItem(item)
            if alias==previous_alias and not item.isHidden():
                restore=item
        self.targets.blockSignals(False)
        selected_count=sum(alias in selections for alias,kind in targets)
        self.coverage.setText(f"已选 {selected_count} / {len(targets)}；缺音 {len(targets)-selected_count}\n工程修订 {self.store.revision}")
        if restore:
            self.targets.setCurrentItem(restore)
        else:
            first=next((self.targets.item(row) for row in range(self.targets.count()) if not self.targets.item(row).isHidden()),None)
            self.targets.setCurrentItem(first)
            if first is None:
                self.show_alias(None)
        if self.workflow_pages.currentIndex()==2:
            from .workflow_ui import refresh_export
            refresh_export(self)

    def show_alias(self,current,*args):
        if not current:
            self.save_marks()
            self.preview_token+=1
            self.preview_ready=False
            self.current_candidate=None
            self.player.stop()
            self.loading=True
            self.candidates.setRowCount(0)
            self.wave.setData([],[])
            self.load_marks(None,[])
            self.source_label.setText("没有匹配的音节，请修改或清空搜索")
            return
        self.save_marks()
        self.preview_ready=False
        self.preview_token+=1
        alias=current.data(Qt.ItemDataRole.UserRole)
        items=self.store.candidates(alias,bank_id=self.bank_id)
        chosen=self.store.selections(self.bank_id).get(alias,{}).get("candidate_id")
        self.loading=True
        self.candidates.setRowCount(len(items))
        source_names={"original":"原音","vowel_extended":"补尾","cv_spliced":"拼接"}
        chosen_row=0
        for row,candidate in enumerate(items):
            state="使用中" if candidate.candidate_id==chosen else ("推荐" if row==0 else "备选")
            if candidate.candidate_id==chosen:
                chosen_row=row
            source_name="导入" if candidate.phonetic_context.get("manual_import") else source_names.get(candidate.provenance_kind.value,candidate.provenance_kind.value)
            cross=candidate.phonetic_context.get("cross_language")
            if cross: source_name="跨语种近似" if cross.get("grade")=="approximate" else "跨语种拼接"
            item=QTableWidgetItem(state+" · "+source_name)
            if cross:
                item.setToolTip("方向："+cross.get("direction",candidate.source_language+"->"+(candidate.target_language or candidate.source_language))+"\n"+"；".join(cross.get("losses",[]))+"\n供体来源："+json.dumps(cross.get("donors",[]),ensure_ascii=False))
            if candidate.phonetic_context.get("manual_import_name"):
                item.setToolTip("导入文件："+candidate.phonetic_context["manual_import_name"])
            item.setData(Qt.ItemDataRole.UserRole,candidate.candidate_id)
            self.candidates.setItem(row,0,item)
            duration=(candidate.span.end-candidate.span.start)/44.1 if candidate.unit_role=="breath" or candidate.phonetic_context.get("manual_import") else (candidate.metrics.stable_vowel_ms or 0)
            duration_item=QTableWidgetItem(str(round(duration)))
            duration_item.setToolTip("导入音频片段时长（不是稳定元音测量）" if candidate.phonetic_context.get("manual_import") else ("呼吸标签片段时长" if candidate.unit_role=="breath" else "自动分析得到的稳定元音区时长"))
            self.candidates.setItem(row,1,duration_item)
            problem=QTableWidgetItem(issue_text(candidate.quality_flags,candidate.phonetic_context.get("ranking_reason","边界可用")))
            problem.setToolTip(issue_details(candidate.quality_flags) or problem.text())
            self.candidates.setItem(row,2,problem)
        self.loading=False
        if items:
            self.candidates.blockSignals(True)
            self.candidates.selectRow(chosen_row)
            self.candidates.blockSignals(False)
            self.preview_selected()
        else:
            self.player.stop()
            self.current_candidate=None
            self.source_label.setText(self.store.display_alias(self.bank_id,alias)+"：缺音，可点击“导入音”添加候选")
            self.wave.setData([],[])

    def selected_id(self):
        row=self.candidates.currentRow()
        item=self.candidates.item(row,0) if row>=0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def selected_alias(self):
        item=self.targets.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def preview_selected(self):
        if self.loading:
            return
        identifier=self.selected_id()
        if identifier:
            self.save_marks()
            self.preview_ready=False
            self.current_candidate=self.store.candidate(identifier,self.bank_id)
            self.play_preview()

    def play_preview(self):
        identifier=self.selected_id()
        if not identifier:
            return
        self.save_marks()
        if self.preview_ready and self.oto_invalid:
            self.status("使用区间无效，请先修正标线再试听")
            return
        self.preview_token+=1
        token=self.preview_token
        result=cached_preview(self.store,identifier)
        if result:
            self.load_preview(result,True)
        else:
            self.player.stop()
            self.preview_ready=False
            self.source_label.setText(self.store.display_alias(self.bank_id,self.current_candidate.alias)+"：首次生成预览缓存…")
            self.start_preview_task(identifier,token)

    def start_preview_task(self,identifier,token,spectrum=False):
        for pending in self.preview_tasks.values():
            if pending.store.directory==self.store.directory and pending.identifier==identifier and pending.spectrum==spectrum:
                pending.token=token
                return
        task=PreviewTask(self.store,identifier,token,spectrum)
        key=(token,spectrum)
        self.preview_tasks[key]=task
        def finished(received,result,error):
            self.preview_tasks.pop(key,None)
            if received!=self.preview_token:
                return
            if error:
                self.error(error)
            elif spectrum:
                self.load_spectrum(result)
            else:
                self.load_preview(result,True)
        task.signals.finished.connect(finished)
        self.preview_pool.start(task)

    def load_preview(self,result,update_wave):
        path=Path(result["path"])
        candidate=self.store.candidate(result["candidate_id"],self.bank_id)
        if not update_wave:
            self.play_candidate_preview(candidate,result)
            return
        self.current_candidate=candidate
        self.window_start=result["start"]
        cache_key=str(path)
        if cache_key not in self.wave_cache:
            audio,sr=sf.read(path,dtype="float32")
            stride=max(1,len(audio)//18000)
            self.wave_cache[cache_key]=(len(audio),np.arange(0,len(audio),stride)/44.1,audio[::stride].copy())
            while len(self.wave_cache)>64:
                self.wave_cache.popitem(last=False)
        self.wave_cache.move_to_end(cache_key)
        self.wave_samples,x,y=self.wave_cache[cache_key]
        self.wave.setData(x,y)
        self.plot.setXRange(0,self.wave_samples/44.1,padding=.01)
        self.plot.setYRange(-1,1)
        with self.store.connection() as connection:
            row=connection.execute("SELECT payload FROM utterances WHERE id=?",(candidate.utterance_id,)).fetchone()
        utterance=Utterance.model_validate_json(row[0]) if row else None
        text=(utterance.revised_text or utterance.asr_text or "") if utterance else candidate.phonetic_context.get("manual_import_name","旧音源 oto 恢复")
        alias=self.selected_alias() or candidate.alias
        chosen=self.store.selections(self.bank_id).get(alias,{}).get("candidate_id")==candidate.candidate_id
        self.source_label.setText(f"{'正式使用' if chosen else '当前试听'}：{self.store.display_alias(self.bank_id,alias)} · {text}\n来源 {candidate.source_id[:8]} · {candidate.span.start/44100:.3f}–{candidate.span.end/44100:.3f}s · {candidate.provenance_kind.value}")
        self.load_marks(candidate.landmarks,candidate.locked_fields)
        self.preview_ready=True
        self.preview_result=result
        # A cache request can finish after the user has left the review page.
        if self.pages.currentIndex()==1 and self.workflow_pages.currentIndex()==1:
            self.play_candidate_preview(candidate,result)
        if self.spectrum_toggle.isChecked():
            self.toggle_spectrum(True)

    def toggle_spectrum(self,visible):
        self.spectrum.setVisible(visible)
        if visible and self.preview_ready and self.preview_result:
            if Path(self.preview_result["spectrum"]).is_file():
                self.load_spectrum(self.preview_result)
            else:
                self.spectrum_image.clear()
                self.start_preview_task(self.selected_id(),self.preview_token,True)

    def load_spectrum(self,result):
        if self.spectrum_toggle.isChecked() and Path(result["spectrum"]).is_file():
            with np.load(result["spectrum"]) as data:
                self.spectrum_image.setImage(data["power"].T,levels=(-90,-25))
                self.spectrum_image.setRect(QRectF(0,0,max(1,float(data["times"][-1])),float(data["frequencies"][-1])))
            self.spectrum.setXRange(0,self.wave_samples/44.1,padding=0)

    def play_candidate_preview(self,candidate,result):
        start,end=preview_interval(candidate,result)
        label="原音使用区间" if candidate.landmarks else "完整切片（未标注 oto）"
        self.play_usage(Path(result["path"]),start,end,label+" · "+self.store.display_alias(self.bank_id,candidate.alias))

    def load_marks(self,marks,locks):
        self.loading=True
        for field in MARK_NAMES:
            self.fields[field].setRange(0,self.wave_samples/44.1)
            value=(getattr(marks,field)-self.window_start)/44.1 if marks else 0
            self.fields[field].setValue(value)
            self.fields[field].setEnabled(marks is not None and field not in locks)
            self.locks[field].setChecked(field in locks)
            self.lines[field].setValue(value)
            self.lines[field].setMovable(marks is not None and field not in locks)
            self.lines[field].setBounds([0,self.wave_samples/44.1])
        for value in self.oto_fields.values():
            value.setEnabled(marks is not None and not locks)
        if marks is None:
            self.edit_reason.setText("没有 oto，试听播放完整切片。「校对 → 强制标注 oto」批量处理因未找到稳定元音而缺失的音节；当前切片可在「标线操作 → 初始化手动标线」自行校对。")
        elif locks:
            self.edit_reason.setText(f"已锁定 {len(locks)} 条标线；取消对应锁定或点击“一键解锁标线”。高级 oto 在全部解锁后可编辑。")
        else:
            self.edit_reason.setText("内部标线只需位于起点与终点之间，顺序可自由调整；修改暂存，同步时应用。")
        self.loading=False
        self.update_oto_label()
        if marks is None:
            self.oto_label.setText("oto：未标注；点击候选或按空格试听完整切片。")

    def update_oto_label(self):
        values={field:value.value() for field,value in self.fields.items()}
        start=values["start"]
        self.oto_label.setText(f"oto（ms）：offset {start:.3f} · overlap {values['overlap']-start:.3f} · preutterance {values['preutter']-start:.3f} · consonant {values['fixed_end']-start:.3f} · cutoff {-(values['end']-start):.3f}")
        self.loading=True
        for field,value in {"offset":start,"overlap":values["overlap"]-start,"preutterance":values["preutter"]-start,"consonant":values["fixed_end"]-start,"cutoff":-(values["end"]-start)}.items():
            self.oto_fields[field].setValue(value)
        self.oto_invalid=False
        self.loading=False

    def oto_changed(self):
        if self.loading:
            return
        values={field:widget.value() for field,widget in self.oto_fields.items()}
        start=values["offset"]
        positions={"start":start,"overlap":start+values["overlap"],"preutter":start+values["preutterance"],"fixed_end":start+values["consonant"],"end":start-values["cutoff"] if values["cutoff"]<0 else self.wave_samples/44.1-values["cutoff"]}
        if not (0<=positions["start"]<positions["end"]<=self.wave_samples/44.1) or any(not positions["start"]<=positions[field]<=positions["end"] for field in ("overlap","preutter","fixed_end")):
            self.oto_invalid=True
            self.status("oto 输入无效：起点必须早于终点，三条内部标线均须在起点与终点之间；不限制内部顺序，尚未保存")
            return
        self.loading=True
        for field,value in positions.items():
            self.fields[field].setValue(value)
            self.lines[field].setValue(value)
        self.loading=False
        self.update_oto_label()
        self.debounce.start()

    def lines_changed(self):
        if self.loading:
            return
        self.loading=True
        for field,line in self.lines.items():
            self.fields[field].setValue(line.value())
        self.loading=False
        self.update_oto_label()

    def numbers_changed(self):
        if self.loading:
            return
        self.loading=True
        for field,value in self.fields.items():
            self.lines[field].setValue(value.value())
        self.loading=False
        self.update_oto_label()
        self.debounce.start()

    def lock_changed(self):
        if not self.loading:
            self.save_marks()

    def unlock_marks(self):
        if not self.current_candidate or not self.preview_ready:
            return
        current=self.store.candidate(self.current_candidate.candidate_id,self.bank_id)
        if current.landmarks is None:
            self.status("候选没有标线，请先点击“初始化手动标线”")
            return
        try:
            self.store.edit_timing(self.bank_id,current.candidate_id,current.landmarks,[])
            self.current_candidate=self.store.candidate(current.candidate_id,self.bank_id)
            self.load_marks(self.current_candidate.landmarks,[])
            self.status("标线已解锁，位置保留；修改已暂存")
        except Exception as error:
            self.error(error)

    def initialize_marks(self):
        if not self.current_candidate or not self.preview_ready:
            return
        current=self.store.candidate(self.current_candidate.candidate_id,self.bank_id)
        if current.landmarks:
            self.status("此候选已有标线；可直接解锁编辑，或恢复自动标线")
            return
        try:
            start=max(self.window_start,current.span.start)
            end=min(self.window_start+self.wave_samples,current.span.end)
            vowel=current.phonetic_context.get("vowel_span",{})
            boundary=max(start,min(end-1,vowel.get("start",start)))
            marks=Landmarks(start=start,overlap=max(start,boundary-441),preutter=boundary,fixed_end=min(end-1,boundary+441),end=end)
            self.store.edit_timing(self.bank_id,current.candidate_id,marks,[])
            self.current_candidate=self.store.candidate(current.candidate_id,self.bank_id)
            self.load_marks(marks,[])
            self.status("手动初始标线已暂存；请试听并校对 oto，再使用这个音并同步修改。严格导出还需点击“试听接受候选”。")
        except Exception as error:
            self.error(error)

    def force_oto(self):
        if not self.store or self.process or self.mark_dragging:
            self.status("请打开工程，松开标线并等待当前任务完成后批量强制标注")
            return
        self.save_marks()
        if self.start_job("force_oto",scope="missing_sustain"):
            self.status("批量处理当前音源中已有切片、因未找到稳定元音而缺失的音节：优先最长切片，补标 oto 并暂存选音；已选音保留，完成后请同步修改。")

    def begin_mark_drag(self):
        # Flush an earlier numeric edit before the pointer gesture begins.
        self.save_marks()
        self.debounce.stop()
        self.mark_dragging=True

    def finish_mark_drag(self,*args):
        self.mark_dragging=False
        self.save_marks()

    def save_marks(self):
        self.debounce.stop()
        if self.applying_task or self.loading or self.mark_dragging or not self.preview_ready or self.oto_invalid or not self.current_candidate or not self.current_candidate.landmarks:
            return
        try:
            marks=Landmarks(**{field:self.window_start+round(value.value()*44.1) for field,value in self.fields.items()})
            locks=[field for field,value in self.locks.items() if value.isChecked()]
            current=self.store.candidate(self.current_candidate.candidate_id,self.bank_id)
            if current.landmarks==marks and current.locked_fields==locks:
                return
            self.store.edit_timing(self.bank_id,current.candidate_id,marks,locks)
            self.current_candidate=self.store.candidate(current.candidate_id,self.bank_id)
            self.load_marks(marks,locks)
            self.status(f"修改已暂存 · 修订 {self.store.revision} · 点击“同步修改”统一应用")
        except Exception as error:
            self.oto_invalid=True
            self.error(str(error)+"；起点需早于终点，三条内部标线须在使用区间内，顺序不限")

    def reset_marks(self,all_fields):
        if not self.current_candidate or not self.current_candidate.automatic_landmarks:
            self.status("此候选没有自动标线；请使用“初始化手动标线”")
            return
        current=self.store.candidate(self.current_candidate.candidate_id,self.bank_id)
        values=current.automatic_landmarks.model_dump()
        if not all_fields:
            for field in current.locked_fields:
                values[field]=getattr(current.landmarks,field)
        self.load_marks(Landmarks(**values),[] if all_fields else current.locked_fields)
        self.save_marks()

    def use_candidate(self):
        self.save_marks()
        identifier=self.selected_id()
        if identifier:
            try:
                self.store.choose(self.bank_id,identifier,self.selected_alias())
                self.refresh_targets()
                self.status("正式选音已保存 · 安装待同步")
            except Exception as error:
                self.error(error)

    def exclude_candidate(self):
        identifier=self.selected_id()
        if identifier:
            self.store.exclude(self.bank_id,identifier)
            self.refresh_targets()
            self.status("候选已排除；正式选音需换用其他候选")

    def add_variant(self):
        if not self.store or self.process:
            return
        self.save_marks()
        identifier=self.selected_id()
        if not identifier or self.oto_invalid:
            self.status("请先选择候选并校对有效标线")
            return
        try:
            alias=self.store.add_variant(self.bank_id,identifier)
            self.refresh_targets()
            for row in range(self.targets.count()):
                if self.targets.item(row).data(Qt.ItemDataRole.UserRole)==alias:
                    self.targets.setCurrentRow(row)
                    break
            output=export_aliases(self.store.bank(self.bank_id)["profile_id"],alias,self.store.bank(self.bank_id)["preset"].get("alias_format",DEFAULT_ALIAS_FORMAT),self.store.custom_units(self.bank_id))[0]
            self.status(f"已新增 {output}，主音保留；修改已暂存，点击“同步修改”导出。编号音共享所选候选的标线。")
        except Exception as error:
            self.error(error)

    def remove_variant(self):
        if not self.store or self.process:
            return
        self.save_marks()
        try:
            self.store.remove_variant(self.bank_id,self.selected_alias() or "")
            self.refresh_targets()
            self.status("编号备选已移除，原候选保留；可撤销，点击“同步修改”应用")
        except Exception as error:
            self.error(error)

    def accept_candidate(self):
        identifier=self.selected_id()
        if identifier:
            # ingest deliberately preserves prior review state; explicit review
            # uses its own update instead of pretending analysis accepted it.
            self.store.review(self.bank_id,identifier,"accepted")
            self.status("试听接受已保存；点击“使用这个音”正式选用")

    def undo(self,redo):
        if self.applying_task:
            self.status("正在暂存批量结果；完成后可整批撤销")
            return
        if self.mark_dragging:
            self.status("请松开鼠标后撤销或重做；一次拖动记录一步")
            return
        identifier=self.selected_id()
        if self.oto_invalid and self.current_candidate:
            self.debounce.stop()
            current=self.store.candidate(self.current_candidate.candidate_id,self.bank_id)
            self.load_marks(current.landmarks,current.locked_fields)
            self.status("未保存的无效标线已还原；撤销历史保留")
            return
        self.save_marks()
        if self.store and self.store.history(self.bank_id,redo):
            # Refresh invokes save_marks while changing rows. First load the
            # restored DB marks so it cannot re-save the pre-undo positions.
            if identifier and self.current_candidate:
                self.current_candidate=self.store.candidate(identifier,self.bank_id)
                if self.preview_ready:
                    self.load_marks(self.current_candidate.landmarks,self.current_candidate.locked_fields)
            self.refresh_targets()
            for row in range(self.candidates.rowCount()):
                if self.candidates.item(row,0).data(Qt.ItemDataRole.UserRole)==identifier:
                    self.candidates.selectRow(row)
                    break
            self.status(("重做" if redo else "撤销")+"已暂存 · 当前版本最多保留 8 步 · 点击同步修改应用")
        elif self.store:
            self.status("没有可重做步骤" if redo else "没有可撤销步骤（最多保留最近 8 步）")

    def add_files(self):
        if not self.store or self.process or self.applying_task:
            return
        files,_=QFileDialog.getOpenFileNames(self,"导入同一人的语音",filter="音频 (*.wav *.flac *.mp3 *.ogg *.m4a);;所有文件 (*)")
        self.pending_files=list(dict.fromkeys([*self.pending_files,*files]))
        if files:
            self.status(f"待处理素材共 {len(self.pending_files)} 个（已去重）；点击自动切片开始")
            self.show_stage(0)

    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self,event):
        if self.store and not self.process and not self.applying_task:
            files=[url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile() and Path(url.toLocalFile()).is_file() and Path(url.toLocalFile()).suffix.lower() in {".wav",".flac",".mp3",".ogg",".m4a"}]
            self.pending_files=list(dict.fromkeys([*self.pending_files,*files]))
            self.status(f"待处理素材共 {len(self.pending_files)} 个（已去重）；点击自动切片开始")
            self.show_stage(0)

    def save_source_language(self):
        if self.store:
            self.settings.setValue("source_language/"+self.store.project_id,self.source_language.currentData())
        if self.source_language.currentData()=="zh": self.backend.setCurrentIndex(self.backend.findData("tifa"))

    def analyze(self):
        files=self.pending_files.copy()
        if self.start_job("analyze",files=files,device=self.device.currentData(),backend=self.backend.currentData(),source_language=self.source_language.currentData()):
            self.pending_files=[]
            from .workflow_ui import refresh_sources
            refresh_sources(self)

    def cross_language(self):
        if not self.store or self.process: return
        bank=self.store.bank(self.bank_id)
        target=profile(bank["profile_id"]).language_id
        dialog=QDialog(self); dialog.setWindowTitle("跨语种音素拼接")
        form=QFormLayout(dialog)
        form.addRow(QLabel("目标版本："+bank["name"]+" · "+bank["profile_id"]))
        source=QComboBox(); source.addItem("日语 → 普通话" if target=="zh" else "普通话 → 日语","ja" if target=="zh" else "zh")
        form.addRow("来源语言 / 方向",source)
        approximate=QCheckBox("允许近似（可能丢失送气、浊音、卷舌或圆唇差异）")
        approximate.setChecked(bool(bank["preset"].get("cross_allow_approximate",False)))
        approximate.setToolTip("关闭时只用兼容规则，缺音较多；开启会在每个候选记录近似原因，不计为原生覆盖。")
        form.addRow(approximate)
        fade=QDoubleSpinBox(); fade.setRange(1,200); fade.setSuffix(" ms"); fade.setValue(bank["preset"].get("crossfade_ms",15))
        fade.setToolTip("拼接重叠和线性交叉淡化时长；短片段自动缩短，实际时长保存在来源记录。")
        form.addRow("片段重叠时长",fade)
        form.addRow(QLabel("仅使用同一说话人已对齐的真实音素。已有原音优先，人工选音保留。\n结果先暂存；试听校对后同步，完整缺音报告保存在工程 analyses。"))
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.describe_dialog_buttons(buttons,"确定（跨语种拼接）")
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        if dialog.exec(): self.start_job("cross_language",source_language=source.currentData(),allow_approximate=approximate.isChecked(),crossfade_ms=fade.value())

    def repair(self):
        if not self.store:
            return
        bank=self.store.bank(self.bank_id)
        preset={**bank["preset"],"repair":True}
        self.store.update_bank(self.bank_id,bank["name"],preset)
        self.start_job("repair")

    def install(self):
        path=QFileDialog.getExistingDirectory(self,"选择音源根目录（文件将直接写入此文件夹）")
        if path:
            self.start_job("sync",destination=path)

    def sync_now(self):
        if not self.store:
            return
        self.save_marks()
        if self.oto_invalid:
            self.status("标线输入无效，请先修正再同步")
            return
        if self.installation_id:
            self.start_job("sync",installation_id=self.installation_id,sync_related=True)
        else:
            self.start_job("build",sync_related=True)

    def refresh_installations(self):
        if not self.store or not self.bank_id:
            return
        bindings=installations(self.store,self.bank_id)
        saved=self.installation_id or self.settings.value("installation/"+self.bank_id,"")
        self.installation_combo.blockSignals(True)
        self.installation_combo.clear()
        for binding in bindings:
            state={"synced":"已同步","pending":"待同步","conflict":"外部修改冲突","failed":"同步失败"}.get(binding["state"],binding["state"])
            self.installation_combo.addItem(binding["path"]+" · "+state,binding["id"])
            if Path(binding["path"]).name.casefold()=="singers":
                row=self.installation_combo.count()-1
                self.installation_combo.setItemText(row,binding["path"]+" · 无效总目录绑定")
                self.installation_combo.model().item(row).setEnabled(False)
        index=self.installation_combo.findData(saved)
        valid=[row for row,binding in enumerate(bindings) if Path(binding["path"]).name.casefold()!="singers"]
        self.installation_combo.setCurrentIndex(index if index in valid else (valid[-1] if valid else -1))
        self.installation_combo.blockSignals(False)
        self.select_installation()

    def select_installation(self):
        self.installation_id=self.installation_combo.currentData()
        if self.installation_id and self.bank_id:
            self.settings.setValue("installation/"+self.bank_id,self.installation_id)
        from .workflow_ui import refresh_export
        refresh_export(self)

    def adopt_external(self):
        if self.installation_id:
            self.start_job("adopt",installation_id=self.installation_id)
        else:
            self.status("请先选择绑定的安装目录")

    def force_sync(self):
        if self.installation_id:
            self.start_job("sync",installation_id=self.installation_id,force=True,sync_related=True)
        else:
            self.status("请先选择绑定的安装目录")

    def export_zip(self):
        path,_=QFileDialog.getSaveFileName(self,"导出 ZIP",filter="ZIP (*.zip)")
        if path:
            self.start_job("zip",destination=path)

    def export_slices(self):
        choice,ok=QInputDialog.getItem(self,"导出切片","范围",["每音最佳 X 个（含已选用）","全部原始候选"],0,False)
        if not ok:
            return
        top_k=1
        if choice!="全部原始候选":
            top_k,ok=QInputDialog.getInt(self,"导出每音最佳 X 个","每个音节最多导出的数量 X：\n包含已选用的一个，再补 X−1 个；稳定元音越长越优先，\n无稳定元音的候选按切片总长度排序，不足 X 个时全部导出。",1,1,9999)
            if not ok:
                return
        self.start_job("slices",all_candidates=choice=="全部原始候选",top_k=top_k)

    def backup(self):
        parent=QFileDialog.getExistingDirectory(self,"选择备份保存位置")
        if parent:
            self.start_job("backup",destination=str(Path(parent)/(self.store.directory.name+"_backup_"+uuid4().hex[:8])))

    def open_build(self):
        bank=self.store.bank(self.bank_id)
        if bank["build_path"]:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(safe_path(self.store.directory,bank["build_path"]))))

    def play_used(self):
        current=self.targets.currentItem()
        if current:
            selected=self.store.selections(self.bank_id).get(current.data(Qt.ItemDataRole.UserRole))
            if selected:
                for row in range(self.candidates.rowCount()):
                    if self.candidates.item(row,0).data(Qt.ItemDataRole.UserRole)==selected["candidate_id"]:
                        self.candidates.blockSignals(True)
                        self.candidates.selectRow(row)
                        self.candidates.blockSignals(False)
                        self.preview_selected()
                        break

    def play_processed(self):
        if not self.current_candidate:
            return
        bank=self.store.bank(self.bank_id)
        if not bank["build_path"]:
            self.status("请先制作音源后试听处理结果")
            return
        from .desktop_service import load_manifest
        root=safe_path(self.store.directory,bank["build_path"])
        _,manifest=load_manifest(root)
        entry=next((entry for entry in manifest["entries"] if entry["candidate_id"]==self.current_candidate.candidate_id),None)
        if entry:
            self.save_marks()
            if self.oto_invalid:
                self.status("使用区间无效，请先修正标线再试听")
                return
            candidate=self.store.candidate(self.current_candidate.candidate_id,self.bank_id)
            self.play_usage(safe_path(root,entry["filename"]),candidate.landmarks.start-entry["window_start"],candidate.landmarks.end-entry["window_start"],"处理后使用区间 · "+entry["alias"])
        else:
            self.status("此候选尚未制作；先使用候选并制作音源")

    def play_range(self):
        self.play_preview()

    def play_usage(self,path,start,end,label):
        try:
            for editor in self.splice_windows.values():
                editor.player.stop()
            source=QUrl.fromLocalFile(str(audition_file(self.store,Path(path),start,end)))
            if self.player.source()!=source:
                self.player.setSource(source)
            else:
                self.player.setPosition(0)
            self.player.play()
            self.play_label.setText(label)
            self.play_label.setToolTip(f"使用区间：{start/44.1:.3f}–{end/44.1:.3f} ms；时长 {(end-start)/44.1:.3f} ms")
        except (ValueError,OSError,RuntimeError) as error:
            self.player.stop()
            self.error(error)

    def render_note(self,length):
        if not self.current_candidate:
            return
        engine=self.settings.value("resampler","")
        if not Path(engine).is_file():
            roots=[Path.home()/"Documents/OpenUtau/Resamplers",Path("D:/Documents/OpenUtau/Resamplers")]
            engine=next((str(root/"moresampler.exe") for root in roots if (root/"moresampler.exe").is_file()),"")
        if not engine:
            engine,_=QFileDialog.getOpenFileName(self,"选择已安装 UTAU Resampler",filter="程序 (*.exe)")
        if engine:
            self.settings.setValue("resampler",engine)
            self.start_job("render",candidate_id=self.current_candidate.candidate_id,engine=engine,length_ms=length)

    def toggle_play(self):
        if self.player.playbackState()==QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def play_current_selected(self):
        focus=QApplication.focusWidget()
        # Keep spaces available in names/text/search; numeric oto fields can
        # still use Space to audition while calibrating.
        if isinstance(focus,QLineEdit) and not isinstance(focus.parentWidget(),QDoubleSpinBox):
            if not focus.isReadOnly():
                focus.insert(" ")
            return
        if not self.store or self.pages.currentIndex()!=1 or self.workflow_pages.currentIndex()!=1 or not self.selected_id():
            return
        self.save_marks()
        self.play_preview()

    def eventFilter(self,watched,event):
        numeric=isinstance(watched,QDoubleSpinBox) or (isinstance(watched,QLineEdit) and isinstance(watched.parentWidget(),QDoubleSpinBox))
        if numeric and watched.window() in (self,self.advanced_oto) and self.pages.currentIndex()==1 and event.type() in (QEvent.Type.ShortcutOverride,QEvent.Type.KeyPress) and event.key()==Qt.Key.Key_Space:
            event.accept()
            if event.type()==QEvent.Type.KeyPress:
                self.play_current_selected()
            return True
        return super().eventFilter(watched,event)

    def make_request(self,action,**kwargs):
        request={"job_id":str(uuid4()),"action":action}
        if action=="analyze":
            from .model_settings import load_model_paths
            request["model_paths"]=load_model_paths()
        if self.store:
            request.update(project=str(self.store.directory),bank_id=self.bank_id,generation=self.store.bank(self.bank_id)["generation"])
        request.update(kwargs)
        directory=user_data_dir()/"jobs"
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/(request["job_id"]+".json")
        path.write_text(json.dumps(request,ensure_ascii=False),encoding="utf-8")
        return request,path

    def launch(self,process,request):
        _,path=request
        if getattr(sys,"frozen",False):
            arguments=["--worker",str(path)]
            program=str(Path(sys.executable).parent/"VoiceBankGenWorker.exe")
            if not Path(program).exists():
                program=str(Path(sys.executable).parent/"UTAUForgeWorker.exe")
        else:
            arguments=["-m","utau_forge.app","--worker",str(path)]
            program=sys.executable
        process.setProgram(program)
        process.setArguments(arguments)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.start()

    def start_job(self,action,**kwargs):
        if self.process or self.applying_task:
            self.status("已有后台任务；可等待完成或取消")
            return False
        if not self.store and action!="recover":
            return False
        self.save_marks()
        if self.oto_invalid:
            self.status("标线输入无效，请修正后再执行任务")
            return False
        request=self.make_request(action,**kwargs)
        self.request=request[0]
        self.cancelled=False
        self.banks.setEnabled(False)
        self.process=BackgroundProcess(self)
        self.update_menus()
        self.buffer=""
        self.job_decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.job_result=None
        self.job_error=None
        self.progress.setRange(0,0)
        self.job_started_at=time.monotonic()
        self.last_job_report=None
        self.job_stage="正在启动后台程序（首次启动可能较慢）"
        self.activity_timer.start()
        self.update_job_activity()
        self.status("正在执行："+JOB_NAMES.get(action,action))
        if self.store:
            with self.store.connection() as connection:
                connection.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)",(self.request["job_id"],self.bank_id,self.request.get("generation"),action,"running",json.dumps(self.request)))
        self.process.readyReadStandardOutput.connect(self.read_progress)
        self.process.finished.connect(self.job_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.started.connect(self.worker_started)
        self.launch(self.process,request)
        return True

    def read_progress(self):
        if not self.process:
            return
        data=bytes(self.process.readAllStandardOutput())
        if data:
            self.last_job_report=time.monotonic()
        self.buffer+=self.job_decoder.decode(data)
        while "\n" in self.buffer:
            line,self.buffer=self.buffer.split("\n",1)
            self.detail_text+=line+"\n"
            self.detail_text=self.detail_text[-40000:]
            try:
                event=json.loads(line)
            except ValueError:
                continue
            if event.get("job_id")!=self.request["job_id"]:
                continue
            if event.get("kind")=="result":
                self.job_result=event["result"]
            elif event.get("kind")=="error":
                self.job_error=event["error"]
            elif "stage" in event:
                self.job_stage=event["stage"]
                total=event.get("total",0)
                self.progress.setRange(0,total)
                self.progress.setValue(event.get("current",0))
                self.status(f"{event['stage']} · {event.get('current',0)}/{total or '…'} · {event.get('device','')} {event.get('reason','')}")
        self.update_job_activity()

    def worker_started(self):
        if not self.cancelled:
            self.job_stage="后台进程已启动，正在初始化组件"
        self.update_job_activity()

    def update_job_activity(self):
        if self.job_started_at is None:
            return
        elapsed=time.monotonic()-self.job_started_at
        text=f"任务：{JOB_NAMES.get((self.request or {}).get('action'), '后台处理')} · {self.job_stage} · 已用 {elapsed:.1f} 秒"
        if self.process and not self.cancelled:
            if self.last_job_report is None:
                text+=" · 等待首条进度，窗口可操作"
            else:
                lag=time.monotonic()-self.last_job_report
                text+=f" · 最近回报 {lag:.1f} 秒前"
                if lag>=10:
                    text+="；后台暂未回报，可查看详情或取消"
        self.job_activity.setText(text)

    def finish_job_activity(self,state):
        self.job_stage=state
        self.update_job_activity()
        self.activity_timer.stop()

    def job_finished(self,exit_code,*args):
        self.read_progress()
        action=self.request["action"]
        completed_bank=self.request.get("bank_id")
        process=self.process
        self.process=None
        self.finish_job_activity("已取消" if self.cancelled else ("失败，请查看任务详情" if exit_code!=0 or self.job_error else "已完成"))
        self.update_menus()
        self.banks.setEnabled(True)
        process.deleteLater()
        self.progress.setRange(0,100)
        self.progress.setValue(100 if exit_code==0 else 0)
        if exit_code!=0 and not self.job_error:
            self.job_error="任务已取消；已保存的阶段可以继续" if self.cancelled else "后台进程异常退出；详情见任务详情\n"+self.detail_text[-900:]
        if self.store:
            with self.store.connection() as connection:
                state="cancelled" if self.cancelled else ("applying" if action=="optimize_vc" and exit_code==0 and not self.job_error else ("complete" if exit_code==0 else "failed"))
                connection.execute("UPDATE jobs SET status=? WHERE id=?",(state,self.request["job_id"]))
        result=self.job_result or {}
        if result.get("installation_id"):
            self.installation_id=result["installation_id"]
        if self.cancelled:
            self.finish_job_activity("后台已结束，取消请求已处理")
            self.status("后台已结束；已保存或已提交的修改保留，可查看任务详情后重试")
            if action in {"sync","adopt","build","timings"}:
                self.refresh_installations()
            return
        if self.job_error:
            self.error(self.job_error+" · 工程已保存，可重试；详情见任务详情")
        elif result.get("stale"):
            self.status("旧任务结果已忽略，保留当前编辑")
        elif action=="recover":
            self.open_directory(Path(result["project"]))
            self.banks.setCurrentIndex(self.banks.findData(result["bank_id"]))
            self.status(f"已恢复 {result['imported']} 行；原安装文件保留。可安装绑定后点击“同步修改”。")
        else:
            if action=="optimize_vc":
                self.status("衔接参数已计算，正在校验并暂存；完成后可整批撤销")
            else:
                self.status("完成 · "+json.dumps({key:value for key,value in result.items() if key not in {"manifest","transformations"}},ensure_ascii=False,default=str)[:650])
            if action=="force_oto":
                self.apply_forced_oto(result)
            if action=="optimize_vc":
                self.apply_vc_optimization(result)
            if action=="transitions":
                self.status(f"CVVC 已准备：新增 {result.get('added',0)} 个连接候选，设置 {result.get('vc',0)} 个 VC / {result.get('vv',0)} 个 VV 的衔接；基础音节采用共用 CV 的现有调整，请试听后同步或制作新版本。" + (" 未找到对齐音素，请先自动切片。" if result.get("no_alignments") else ""))
            if action=="render" and result.get("path"):
                self.player.setSource(QUrl.fromLocalFile(result["path"]))
                self.player.play()
                self.play_label.setText("实际渲染 · "+result["engine"])
            if action in {"analyze","repair","build","transitions","cross_language"}:
                self.refresh_targets()
            if action=="import_candidates":
                self.refresh_targets()
                ids=result.get("candidate_ids",[])
                for row in range(self.candidates.rowCount()):
                    if ids and self.candidates.item(row,0).data(Qt.ItemDataRole.UserRole)==ids[0]:
                        self.candidates.selectRow(row)
                        break
                self.status(f"已导入 {result.get('imported',0)} 个候选，已有 {result.get('existing',0)} 个，失败 {len(result.get('failures',[]))} 个；原选音保留，请校对标线后点击“使用这个音”，再同步修改")
            if action in {"sync","adopt","build","timings"}:
                self.refresh_installations()

            if "linked_sync" in result:
                linked=result["linked_sync"]
                okay=sum(item["state"]=="synced" for item in linked)
                skipped=sum(item["state"]=="skipped" for item in linked)
                pending=len(linked)-okay-skipped
                self.status(f"同步完成，已更新 {okay} 个关联安装，跳过 {skipped} 个无效历史绑定，待处理 {pending} 个。基础 CV 共用，VC 单独保存。"+(" 请切换相应版本查看冲突或任务详情。" if pending else " 请在 OpenUtau 重新载入音源。"))
            if action=="analyze":
                from .workflow_ui import refresh_sources
                refresh_sources(self)
                self.show_stage(1)

    def apply_vc_optimization(self,result):
        self.save_marks()
        if self.mark_dragging or self.oto_invalid:
            self.status("当前正在拖动或有无效标线输入，未应用批量优化；请修正后重试")
            self.progress.setValue(0)
            self.finish_job_activity("未应用，当前编辑保留")
            return
        identifier=self.selected_id()
        store,bank_id,generation=self.store,self.bank_id,self.request["generation"]
        self.applying_task=FunctionTask(lambda:store.apply_timing_batch(bank_id,result["changes"],generation))
        self.vc_apply_result=result
        self.vc_apply_identifier=identifier
        self.job_stage=f"正在校验并暂存 {len(result['changes'])} 项标线，完成后可整批撤销"
        self.progress.setRange(0,0)
        self.splitter.setEnabled(False)
        self.banks.setEnabled(False)
        self.update_menus()
        self.activity_timer.start()
        self.update_job_activity()
        self.applying_task.signals.finished.connect(self.finish_vc_optimization)
        self.job_pool.start(self.applying_task)

    def finish_vc_optimization(self,count,error):
        result=self.vc_apply_result
        identifier=self.vc_apply_identifier
        self.applying_task=None
        self.splitter.setEnabled(True)
        self.banks.setEnabled(True)
        self.update_menus()
        self.progress.setRange(0,100)
        self.progress.setValue(0 if error else 100)
        self.finish_job_activity("暂存失败，请查看提示" if error else "已完成并暂存，等待同步修改")
        with self.store.connection() as connection:
            connection.execute("UPDATE jobs SET status=? WHERE id=?",("failed" if error else "complete",self.request["job_id"]))
        try:
            if error:
                raise ValueError(error)
            if identifier and self.current_candidate:
                self.current_candidate=self.store.candidate(identifier,self.bank_id)
                if self.preview_ready:
                    self.load_marks(self.current_candidate.landmarks,self.current_candidate.locked_fields)
            self.refresh_targets()
            for row in range(self.candidates.rowCount()):
                if self.candidates.item(row,0).data(Qt.ItemDataRole.UserRole)==identifier:
                    self.candidates.selectRow(row)
                    break
            self.status(f"已暂存 VC / VV 衔接优化 {count} 项（VC {result['vc']} / VV {result.get('vv',0)} / 配套 CV {result['cv']}）；锁定跳过 {result['locked']}，无标线或前置区不足 {result['insufficient']}，已合适 {result['unchanged']}。整批可一步撤销，请同步修改后在 OpenUtau 重新载入音源；已有乐谱手动包络需自行重置。")
        except Exception as error:
            self.error(error)

    def apply_forced_oto(self,result):
        self.save_marks()
        if self.mark_dragging or self.store.bank(self.bank_id)["generation"]!=self.request["generation"]:
            self.status("分析期间工程发生修改，未覆盖当前编辑；可再次选择强制标注 oto")
            return
        try:
            if "proposals" in result:
                identifier=self.selected_id()
                count=self.store.apply_forced_missing(self.bank_id,result["proposals"],self.request["generation"])
                if identifier and self.current_candidate:
                    self.current_candidate=self.store.candidate(identifier,self.bank_id)
                    if self.preview_ready:
                        self.load_marks(self.current_candidate.landmarks,self.current_candidate.locked_fields)
                self.refresh_targets()
                for row in range(self.candidates.rowCount()):
                    if self.candidates.item(row,0).data(Qt.ItemDataRole.UserRole)==identifier:
                        self.candidates.selectRow(row)
                        break
                failures=len(result.get("failures",[]))
                self.status(f"批量强制标注完成：{count} 个缺音已补标并选用，失败 {failures} 个。只处理已有切片且未找到稳定元音的缺音，已选音保留；整批可一步撤销。请试听校对后同步修改。")
                return
            identifier=result["candidate_id"]
            marks=Landmarks.model_validate(result["marks"])
            self.store.edit_timing(self.bank_id,identifier,marks,[])
            if self.selected_id()==identifier and self.preview_ready:
                self.current_candidate=self.store.candidate(identifier,self.bank_id)
                self.load_marks(self.current_candidate.landmarks,[])
            self.status("强制 oto 已暂存，可撤销；请试听校对，使用这个音后点击同步修改。严格导出仍需试听接受。")
        except Exception as error:
            self.error(error)

    def cancel_job(self):
        if self.process:
            self.cancelled=True
            self.process.kill()
            self.job_stage="正在取消；若后台仍在启动，将在启动返回后停止"
            self.update_menus()
            self.update_job_activity()
            self.status("已请求取消，正在等待后台安全停止；已保存或提交的修改保留")

    def process_error(self,error):
        if self.process and not self.cancelled:
            self.job_error="后台进程无法运行："+self.process.errorString()
            self.status(self.job_error)

    def job_details(self):
        dialog=QDialog(self)
        dialog.setWindowTitle("后台任务详情")
        dialog.resize(900,550)
        layout=QVBoxLayout(dialog)
        text=QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self.detail_text)
        layout.addWidget(text)
        dialog.exec()

    def closeEvent(self,event):
        if self.process or self.applying_task:
            self.status("请先取消后台任务或等待完成后关闭")
            event.ignore()
            return
        self.save_marks()
        self.advanced_oto.close()
        self.player.stop()
        self.preview_token+=1
        for editor in list(self.splice_windows.values()):
            editor.close()
        self.preview_pool.waitForDone()
        self.settings.setValue("geometry",self.saveGeometry())
        self.settings.setValue("splitter",self.splitter.saveState())
        event.accept()

def main():
    for stream in (sys.stdout,sys.stderr):
        if stream and hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="replace")
    if "--worker" in sys.argv:
        from .desktop_worker import main as worker
        worker(sys.argv[sys.argv.index("--worker")+1])
        return
    begin_startup()
    application=QApplication.instance()
    application.setWindowIcon(QIcon(str(icon_path())))
    startup_progress(82,"读取界面设置…")
    application.setFont(QFont("Microsoft YaHei",10))
    application.setStyle("Fusion")
    startup_progress(90,"创建主界面…")
    window=MainWindow()
    startup_progress(95,"准备工程视图…")
    if "--project" in sys.argv:
        window.open_directory(Path(sys.argv[sys.argv.index("--project")+1]))
    window.show()
    application.processEvents()
    finish_startup()
    if getattr(sys, "frozen", False) and "--smoke" not in sys.argv:
        from .desktop_engine import application_root
        from .model_settings import detect_model_paths, load_model_paths
        detected, explicit = detect_model_paths(application_root()), load_model_paths()
        if any(not (explicit[key] or detected[key]) for key in ("asr", "tifa", "unidic")):
            QTimer.singleShot(100, lambda: window.resource_downloads(first_run=True))
    if "--smoke" in sys.argv:
        window.resize(1420,900)
        if window.store:
            chosen=window.store.selections(window.bank_id)
            for index in range(window.targets.count()):
                item=window.targets.item(index)
                if item.data(Qt.ItemDataRole.UserRole) in chosen:
                    window.targets.setCurrentRow(index)
                    break
        smoke_attempts=0
        def capture():
            nonlocal smoke_attempts
            smoke_attempts+=1
            if window.store and (window.wave.xData is None or not len(window.wave.xData)) and smoke_attempts<45:
                QTimer.singleShot(1000,capture)
                return
            path=Path(os.environ.get("UFORGE_SMOKE_IMAGE","desktop-smoke.png")).resolve()
            if "--smoke-downloads" in sys.argv:
                from .desktop_engine import application_root
                from .resource_download_dialog import ResourceDownloadDialog
                dialog=ResourceDownloadDialog(application_root(),window)
                dialog.show()
                application.processEvents()
                dialog.grab().save(str(path))
                dialog.close()
            elif "--smoke-models" in sys.argv:
                from .desktop_engine import application_root
                from .model_settings_dialog import ModelSettingsDialog
                dialog=ModelSettingsDialog(application_root(),window.store.directory if window.store else None,window)
                dialog.show()
                application.processEvents()
                dialog.grab().save(str(path))
                dialog.close()
            elif "--smoke-splice" in sys.argv and window.store:
                from .splice_workbench import SpliceWorkbench
                from .desktop_splice import candidate_recipe
                dialog=SpliceWorkbench(window)
                candidate=next((c for c in window.store.candidates(bank_id=window.bank_id) if c.phonetic_context.get("manual_assembly")),None)
                if candidate:
                    dialog.restore(candidate_recipe(window.store,window.bank_id,candidate.candidate_id))
                dialog.show()
                dialog.fit()
                application.processEvents()
                dialog.grab().save(str(path))
                dialog.close()
            else:
                window.grab().save(str(path))
            application.quit()
        QTimer.singleShot(4500,capture)
    sys.exit(application.exec())

if __name__=="__main__":
    main()
