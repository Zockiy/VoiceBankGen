"""Workflow-first desktop fork. Presentation and navigation only."""
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QKeySequence,QShortcut
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QFormLayout,
    QLabel,QPushButton,QToolButton,QMenu,QStackedWidget,QSplitter,QListWidget,
    QListWidgetItem,QTableWidget,QTableWidgetItem,QComboBox,QLineEdit,QCheckBox,
    QDoubleSpinBox,QDialog,QProgressBar,QHeaderView)

def menu_button(window,title,entries,layout):
    from .app import BUTTON_HELP,button_help_text
    button=QToolButton()
    button.setText(title)
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu=QMenu(button)
    menu.setToolTipsVisible(True)
    for name,callback in entries:
        action=menu.addAction(name)
        action.setToolTip(button_help_text(name) if name in BUTTON_HELP else "直接编辑当前候选的 oto 数值；修改自动暂存。")
        action.triggered.connect(callback)
        window.workflow_actions.append(action)
    menu.aboutToShow.connect(window.update_menus)
    button.setMenu(menu)
    button.setToolTip("展开"+title+"，选择所需操作。")
    layout.addWidget(button)
    return button

def label(text,layout,role=None):
    widget=QLabel(text)
    widget.setWordWrap(True)
    if role: widget.setProperty("role",role)
    layout.addWidget(widget)
    return widget

def build_workbench(self):
    from .app import MarkLine,MARK_NAMES,MARK_HELP,COLORS
    self.setWindowTitle("VoiceBankGen · 工作流 UI Fork")
    self.settings.setFallbacksEnabled(False)
    self.workflow_actions=[]
    self.workflow_controls=[]
    page=QWidget()
    layout=QVBoxLayout(page)
    layout.setContentsMargins(18,14,18,12)
    layout.setSpacing(10)
    heading=QHBoxLayout()
    self.project_label=QLabel("VoiceBankGen")
    self.project_label.setProperty("role","title")
    heading.addWidget(self.project_label,1)
    heading.addWidget(QLabel("音源版本"))
    self.banks=QComboBox()
    self.banks.setMinimumWidth(235)
    self.banks.setToolTip("CV 与对应 CVVC 共用基础音；VC/VV 独立保存。切换前自动暂存。")
    self.banks.currentIndexChanged.connect(self.switch_bank)
    heading.addWidget(self.banks)
    self.workflow_controls.append(menu_button(self,"版本设置",[
        ("音源设置",self.bank_settings),("新增 CV/CVVC 版本",self.new_bank)],heading))
    layout.addLayout(heading)
    steps=QHBoxLayout()
    self.step_buttons=[]
    for index,title in enumerate(("01  素材与切分","02  校对与选音","03  同步与导出")):
        button=QPushButton(title)
        button.setCheckable(True)
        button.setMinimumHeight(42)
        button.setToolTip(("添加同一说话人的录音，设置语言，再开始自动切片。",
                           "选择音节与候选，试听、调整标线并确定正式选音。",
                           "把暂存的修改应用到音源，或安装、导出。")[index])
        button.clicked.connect(lambda checked=False,i=index:self.show_stage(i))
        self.step_buttons.append(button)
        steps.addWidget(button,1)
    layout.addLayout(steps)
    self.workflow_pages=QStackedWidget()
    layout.addWidget(self.workflow_pages,1)

    prep=QWidget()
    body=QHBoxLayout(prep)
    files_column=QVBoxLayout()
    label("添加录音，开始制作",files_column,"title")
    label("同一说话人的日语或普通话录音。可拖入音频；已经处理的素材会复用缓存。",files_column,"muted")
    self.source_table=QTableWidget(0,3)
    self.source_table.setHorizontalHeaderLabels(["素材文件","时长","状态"])
    self.source_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    self.source_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    self.source_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
    self.source_table.verticalHeader().hide()
    files_column.addWidget(self.source_table,1)
    controls=QHBoxLayout()
    self.workflow_controls.append(self.button("添加素材",self.add_files,controls))
    remove=QPushButton("移出待处理列表")
    remove.setToolTip("只移除选中的待处理素材；磁盘音频和已导入素材均保留。")
    remove.clicked.connect(self.remove_pending_source)
    controls.addWidget(remove)
    self.workflow_controls.append(remove)
    controls.addStretch()
    files_column.addLayout(controls)
    body.addLayout(files_column,3)
    settings=QVBoxLayout()
    label("切分设置",settings,"title")
    label("确认素材语言后开始。模型任务在后台执行，进度显示在下方。",settings,"muted")
    form=QFormLayout()
    self.source_language=QComboBox()
    self.source_language.setObjectName("source_language")
    self.source_language.addItem("日语","ja")
    self.source_language.addItem("普通话","zh")
    self.source_language.setToolTip("用于新增素材的识别与对齐；目标声库版本由上方选择。")
    self.source_language.currentIndexChanged.connect(self.save_source_language)
    self.device=QComboBox()
    for text,value in (("自动（推荐）","auto"),("CPU","cpu"),("NVIDIA GPU","cuda")): self.device.addItem(text,value)
    self.device.setToolTip("自动优先使用可用 GPU；任务详情显示实际计算设备。")
    self.backend=QComboBox()
    self.backend.addItem("TIFA（默认）","tifa")
    self.backend.addItem("MFA（旧版可选）","mfa")
    self.backend.setToolTip("选择音素对齐工具。中文使用 TIFA；编辑 oto 不会重新对齐。")
    for name,widget in (("素材语言",self.source_language),("计算设备",self.device),("对齐工具",self.backend)):
        form.addRow(name,widget)
        self.workflow_controls.append(widget)
    settings.addLayout(form)
    analyze=self.button("自动切片",self.analyze,settings)
    analyze.setProperty("primary",True)
    analyze.setText("开始自动切片")
    self.workflow_controls.append(analyze)
    label("完成后进入「校对与选音」，检查推荐候选和缺音。",settings,"muted")
    settings.addStretch()
    body.addLayout(settings,1)
    self.workflow_pages.addWidget(prep)

    review=QWidget()
    review_layout=QVBoxLayout(review)
    review_layout.setContentsMargins(0,0,0,0)
    self.splitter=QSplitter()
    left=QWidget()
    column=QVBoxLayout(left)
    left.setMinimumWidth(205)
    self.add_unit_button=self.button("新增音",self.add_unit,column)
    self.search=QLineEdit()
    self.search.setToolTip("输入平假名或罗马音（例如 shi、kya）筛选左侧目标音；清空可恢复列表。仅改变显示范围。")
    self.search.setPlaceholderText("搜索音名 / 罗马音")
    self.target_search_timer=QTimer(self)
    self.target_search_timer.setSingleShot(True)
    self.target_search_timer.setInterval(250)
    self.target_search_timer.timeout.connect(self.filter_targets)
    self.search.textChanged.connect(lambda *_:self.target_search_timer.start())
    self.search.returnPressed.connect(self.filter_targets)
    column.addWidget(self.search)
    self.kind=QComboBox()
    self.kind.setToolTip("按音型筛选：CV/V 为主体音节或元音，VC 为前元音到辅音过渡，VV 为元音之间的过渡。")
    self.kind.addItems(["全部","CV/V","VC","VV","呼吸","自定义","编号备选"])
    self.kind.currentTextChanged.connect(self.refresh_targets)
    column.addWidget(self.kind)
    self.only_missing=QCheckBox("只看缺音")
    self.only_missing.setToolTip("只显示还没有正式选音的目标音；这里只筛选列表，不自动补音。")
    self.only_missing.toggled.connect(self.refresh_targets)
    column.addWidget(self.only_missing)
    self.targets=QListWidget()
    self.targets.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.targets.setTextElideMode(Qt.TextElideMode.ElideRight)
    self.targets.currentItemChanged.connect(self.show_alias)
    self.targets.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.targets.customContextMenuRequested.connect(self.target_menu)
    column.addWidget(self.targets)
    self.coverage=QLabel()
    self.coverage.setWordWrap(True)
    column.addWidget(self.coverage)
    self.splitter.addWidget(left)
    center=QWidget()
    column=QVBoxLayout(center)
    self.source_label=QLabel("选择目标音和备选以试听")
    self.source_label.setWordWrap(True)
    column.addWidget(self.source_label)
    toolbar=QHBoxLayout()
    self.button("撤销",lambda:self.undo(False),toolbar)
    self.button("重做",lambda:self.undo(True),toolbar)
    toolbar.addStretch()
    column.addLayout(toolbar)
    self.plot=pg.PlotWidget()
    self.plot.setBackground("#151c26")
    self.plot.setLabel("bottom","片段时间",units="ms",color="#bac8da")
    self.plot.showGrid(x=True,y=True,alpha=.2)
    self.plot.setMenuEnabled(False)
    self.plot.setMouseEnabled(x=True,y=False)
    self.wave=self.plot.plot(pen=pg.mkPen("#b1c9e8",width=1))
    self.lines={}
    for index,(field,name) in enumerate(MARK_NAMES.items()):
        line=MarkLine(self,pos=0,movable=True,pen=pg.mkPen(COLORS[index],width=2,style=Qt.PenStyle.DashLine if index%2 else Qt.PenStyle.SolidLine),label=name,labelOpts={"position":.95-index*.14,"color":COLORS[index]})
        line.sigPositionChanged.connect(self.lines_changed)
        line.sigPositionChangeFinished.connect(self.finish_mark_drag)
        self.plot.addItem(line)
        self.lines[field]=line
    column.addWidget(self.plot,1)
    self.spectrum_toggle=QCheckBox("展开频谱")
    self.spectrum_toggle.setToolTip("显示当前候选的时间与频率分布，辅助观察元音、辅音和噪声；首次展开才生成频谱缓存。")
    column.addWidget(self.spectrum_toggle)
    self.spectrum=pg.PlotWidget()
    self.spectrum.setLabel("bottom","片段时间",units="ms",color="#bac8da")
    self.spectrum.setLabel("left","频率",units="Hz",color="#bac8da")
    self.spectrum.setMaximumHeight(180)
    self.spectrum_image=pg.ImageItem()
    self.spectrum.addItem(self.spectrum_image)
    for graph in (self.plot,self.spectrum):
        graph.setBackground("#151c26")
        for axis in ("bottom","left"):
            graph.getAxis(axis).setTextPen("#bac8da")
            graph.getAxis(axis).setPen("#52657c")
    self.spectrum.hide()
    self.spectrum_toggle.toggled.connect(self.toggle_spectrum)
    column.addWidget(self.spectrum)
    form=QGridLayout()
    self.fields={}
    self.locks={}
    for index,(field,name) in enumerate(MARK_NAMES.items()):
        row=QHBoxLayout()
        value=QDoubleSpinBox()
        value.setToolTip(MARK_HELP[field]+"\n数值单位为毫秒；调整后自动暂存，空格可试听当前候选，点击同步修改应用。")
        value.setDecimals(3)
        value.setRange(0,1e9)
        value.setSuffix(" ms")
        value.setKeyboardTracking(True)
        value.valueChanged.connect(self.numbers_changed)
        value.editingFinished.connect(self.save_marks)
        row.addWidget(value)
        lock=QCheckBox("锁")
        lock.setToolTip("锁定“"+name+"”的位置，禁止直接拖动或输入；重新自动标注时保留。恢复全部自动会取消此锁定。")
        lock.toggled.connect(self.lock_changed)
        row.addWidget(lock)
        form.addWidget(QLabel(name),index//3*2,index%3)
        form.addLayout(row,index//3*2+1,index%3)
        self.fields[field]=value
        self.locks[field]=lock
    column.addLayout(form)
    self.oto_label=QLabel()
    self.oto_label.setWordWrap(True)
    column.addWidget(self.oto_label)
    self.edit_reason=QLabel()
    self.edit_reason.setWordWrap(True)
    column.addWidget(self.edit_reason)
    advanced=QDialog(self)
    advanced.setWindowTitle("高级 oto 数值（ms）")
    self.advanced_oto=advanced
    advanced.setToolTip("展开直接编辑 oto 的 offset、overlap、preutterance、consonant 和 cutoff；它们与五条标线互相转换，修改只暂存。")
    advanced_layout=QVBoxLayout(advanced)
    advanced_body=QWidget()
    advanced_form=QFormLayout(advanced_body)
    self.oto_fields={}
    self.oto_invalid=False
    for field in ("offset","overlap","preutterance","consonant","cutoff"):
        value=QDoubleSpinBox()
        descriptions={"offset":"WAV 前方不使用的时长。", "overlap":"相对 offset 的重叠位置。", "preutterance":"相对 offset 的对拍位置。", "consonant":"相对 offset 的固定区长度。", "cutoff":"负数表示从 offset 起使用的长度，正数表示从 WAV 末尾裁去的长度。"}
        value.setToolTip(descriptions[field]+"单位为毫秒；需符合五条标线顺序及当前 WAV 范围。")
        value.setDecimals(3)
        value.setRange(-1e9,1e9)
        value.valueChanged.connect(self.oto_changed)
        value.editingFinished.connect(self.save_marks)
        advanced_form.addRow(field,value)
        self.oto_fields[field]=value
    advanced_layout.addWidget(advanced_body)
    tools=menu_button(self,"标线操作",[
        ("一键解锁标线",self.unlock_marks),("初始化手动标线",self.initialize_marks),
        ("重新自动标注（保留锁定）",lambda:self.reset_marks(False)),
        ("恢复全部自动",lambda:self.reset_marks(True)),
        ("高级 oto 数值",self.advanced_oto.show)],toolbar)
    transport=QHBoxLayout()
    play=QPushButton("试听 / 停止 · 空格")
    play.setToolTip("点击试听当前候选的 UTAU 使用部分；没有 oto 时播放完整切片。播放中再次点击停止。空格始终从头试听当前候选。")
    play.clicked.connect(lambda:self.player.stop() if self.player.playbackState()==QMediaPlayer.PlaybackState.PlayingState else self.play_current_selected())
    transport.addWidget(play)
    menu_button(self,"对比与渲染",[
        ("播放 / 暂停",self.toggle_play),
        ("试听正式使用音（A）",self.play_used),("试听处理后",self.play_processed),
        ("短音实际渲染",lambda:self.render_note(180)),("长音实际渲染",lambda:self.render_note(1000))],transport)
    self.play_label=QLabel("选择候选后试听")
    self.play_label.setWordWrap(True)
    transport.addWidget(self.play_label,1)
    column.insertLayout(3,transport)
    self.splitter.addWidget(center)
    right=QWidget()
    column=QVBoxLayout(right)
    right.setMinimumWidth(280)
    column.addWidget(QLabel("单击只试听；“使用”才改变正式选音"))
    self.candidates=QTableWidget(0,3)
    self.candidates.setHorizontalHeaderLabels(["状态 / 来源","时长 ms","主要问题"])
    self.candidates.verticalHeader().hide()
    self.candidates.setColumnWidth(0,100)
    self.candidates.setColumnWidth(1,70)
    self.candidates.horizontalHeader().setMinimumSectionSize(60)
    self.candidates.horizontalHeaderItem(1).setToolTip("元音候选显示稳定元音区时长；呼吸候选显示呼吸标签片段时长。")
    self.candidates.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    self.candidates.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    self.candidates.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    self.candidates.itemSelectionChanged.connect(self.preview_selected)
    self.candidates.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.candidates.customContextMenuRequested.connect(self.candidate_menu)
    self.candidates.horizontalHeader().setStretchLastSection(True)
    column.addWidget(self.candidates,1)
    use=self.button("使用这个音",self.use_candidate,column)
    use.setProperty("primary",True)
    operations=menu_button(self,"候选操作",[
        ("导入音",self.import_sound),("新增备选音",self.add_variant),("移除备选音",self.remove_variant),
        ("试听接受候选",self.accept_candidate),("排除此候选",self.exclude_candidate),
        ("单独导出当前候选",self.export_candidate)],column)
    self.import_sound_button=operations.menu().actions()[0]
    self.splitter.insertWidget(1,right)
    self.splitter.setSizes([220,320,740])

    self.splitter.setStretchFactor(2,1)
    saved=self.settings.value("splitter")
    if saved: self.splitter.restoreState(saved)
    review_layout.addWidget(self.splitter,1)
    self.workflow_pages.addWidget(review)
    self.workflow_controls.append(self.advanced_oto)

    export=QWidget()
    export_layout=QVBoxLayout(export)
    export_layout.setContentsMargins(36,24,36,24)
    label("把校对结果应用到音源",export_layout,"title")
    self.export_summary=label("选择音源版本后查看当前状态。",export_layout)
    label("编辑会自动暂存。点击底部「同步修改」统一制作并更新关联安装；OpenUtau 中需重新载入音源。",export_layout,"muted")
    destination=QFormLayout()
    self.installation_combo=QComboBox()
    self.installation_combo.setMinimumContentsLength(20)
    self.installation_combo.setToolTip("选择当前同步目标；关联 CV/CVVC 安装也会一起更新。")
    self.installation_combo.currentIndexChanged.connect(self.select_installation)
    destination.addRow("安装位置",self.installation_combo)
    export_layout.addLayout(destination)
    actions=QHBoxLayout()
    self.workflow_controls.append(self.button("安装到目录",self.install,actions))
    self.workflow_controls.append(menu_button(self,"导出",[
        ("导出 ZIP",self.export_zip),("导出切片",self.export_slices),
        ("打开生成目录",self.open_build)],actions))
    actions.addStretch()
    export_layout.addLayout(actions)
    self.conflict_panel=QWidget()
    conflicts=QVBoxLayout(self.conflict_panel)
    label("检测到外部修改，请选择处理方式",conflicts,"title")
    label("采用外部修改后可继续校对；保留工程并覆盖会先备份当前文件。",conflicts)
    conflict_buttons=QHBoxLayout()
    self.workflow_controls.append(self.button("采用外部修改",self.adopt_external,conflict_buttons))
    self.workflow_controls.append(self.button("保留工程并覆盖（先备份）",self.force_sync,conflict_buttons))
    conflicts.addLayout(conflict_buttons)
    export_layout.addWidget(self.conflict_panel)
    self.conflict_panel.hide()
    export_layout.addStretch()
    self.workflow_pages.addWidget(export)

    footer=QHBoxLayout()
    self.message=QLabel("编辑自动暂存")
    self.message.setWordWrap(True)
    self.message.setMaximumHeight(42)
    footer.addWidget(self.message,1)
    self.sync_button=self.button("同步修改",self.sync_now,footer)
    self.sync_button.setProperty("primary",True)
    layout.addLayout(footer)
    self.task_panel=QWidget()
    task_layout=QHBoxLayout(self.task_panel)
    task_layout.setContentsMargins(0,0,0,0)
    self.job_activity=QLabel("任务：空闲")
    self.job_activity.setWordWrap(True)
    task_layout.addWidget(self.job_activity,1)
    self.progress=QProgressBar()
    self.progress.setMaximumWidth(180)
    self.progress.setFixedHeight(16)
    task_layout.addWidget(self.progress)
    self.cancel_button=self.button("取消任务",self.cancel_job,task_layout)
    self.button("任务详情",self.job_details,task_layout)
    layout.addWidget(self.task_panel)
    self.task_panel.hide()
    self.detail_text=""
    self.pages.addWidget(page)
    enter=QShortcut(QKeySequence("Return"),self.candidates)
    enter.setContext(Qt.ShortcutContext.WidgetShortcut)
    enter.activated.connect(self.use_candidate)
    self.workflow_pages.setCurrentIndex(0)
    self.step_buttons[0].setChecked(True)

def refresh_sources(self):
    from .models import SourceAsset
    assets=self.store.records("assets",SourceAsset) if self.store else []
    entries=[(Path(a.original_path).name,f"{a.original_duration_s:.1f} 秒","已导入",None) for a in assets]
    entries += [(Path(p).name,"—","待切片",i) for i,p in enumerate(self.pending_files)]
    self.source_table.setRowCount(len(entries))
    for row,values in enumerate(entries):
        for column,text in enumerate(values[:3]):
            item=QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole,values[3])
            item.setToolTip(self.pending_files[values[3]] if values[3] is not None else assets[row].original_path)
            self.source_table.setItem(row,column,item)

def refresh_export(self):
    if not self.store: return
    from .desktop_sync import installations
    bindings=installations(self.store,self.bank_id)
    conflict=any(b["id"]==self.installation_id and b["state"]=="conflict" for b in bindings)
    other_conflict=any(b["id"]!=self.installation_id and b["state"]=="conflict" for b in bindings)
    self.conflict_panel.setVisible(conflict)
    self.export_summary.setText(self.coverage.text().replace("\n","  ·  ")+"\n"+
        ("检测到外部修改，请先处理下方冲突。" if conflict else
         ("其他安装位置存在冲突，请在上方列表选择对应位置处理。" if other_conflict else
          ("已绑定安装目录，点击同步修改即可更新。" if bindings else "尚未绑定安装目录。可以先同步生成音源，再导出；也可选择安装目录。"))))
