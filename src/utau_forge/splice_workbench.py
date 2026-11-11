"""Four-track manual splice editor with movable, trimmable waveform clips."""
from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, QRectF, QPointF, QUrl, QTimer, QObject, Signal, QRunnable, QThreadPool, QAbstractListModel, QModelIndex, QEvent
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QShortcut, QKeySequence
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QWidget, QLineEdit, QListView, QComboBox, QPushButton,
    QLabel, QGraphicsObject, QGraphicsScene, QGraphicsView, QDoubleSpinBox,
    QFormLayout, QFileDialog, QMenu)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from .desktop_splice import clip_from_candidate, assembly_preview, export_audio, save_candidate, validate_clips, candidate_recipe, split_clip, timeline_fades, clip_envelope
from .desktop_service import safe_path
from .storage import write_json
from .desktop_store import UNDO_LIMIT
from .bank_aliases import DEFAULT_ALIAS_FORMAT,to_katakana


class SourceListModel(QAbstractListModel):
    """Virtual list: Qt asks only for visible text/tooltips, not every row."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []

    def replace(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.rows):
            return None
        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return row["text"]
        if role == Qt.ItemDataRole.UserRole:
            return row["candidate_id"]
        if role == Qt.ItemDataRole.ToolTipRole:
            from .issue_labels import issue_details
            flags = json.loads(row["quality_flags"] or "[]")
            return (f"素材语言 {row['source_language']} · 来源 {row['source_id']}\n"
                    f"{row['start']/44100:.3f}–{row['end']/44100:.3f}s\n" + issue_details(flags))
        return None


class SourceIndexSignals(QObject):
    ready = Signal(object)
    error = Signal(str)


class SourceIndexTask(QRunnable):
    def __init__(self, store, revision, labels):
        super().__init__()
        self.store, self.revision, self.labels = store, revision, labels
        self.signals = SourceIndexSignals()

    def run(self):
        try:
            rows = self.store.splice_source_index()
            for row in rows:
                label = self.labels.get(row["alias"], row["alias"])
                row["text"] = f"{label} · {row['candidate_id'][:8]} · {(row['end']-row['start'])/44.1:.0f}ms"
                row["search"] = (row["text"] + " " + row["alias"] + " " + row["candidate_id"]).casefold()
            rows.sort(key=lambda row:(self.labels.get(row["alias"],row["alias"]),row["alias"],row["start"]))
            self.signals.ready.emit((self.revision, rows))
        except Exception as error:
            try:
                self.signals.error.emit(str(error))
            except RuntimeError:
                # The application may have closed during a read-only task.
                pass


class RecipeTask(QRunnable):
    def __init__(self, store, bank_id, candidate_id):
        super().__init__()
        self.store, self.bank_id, self.candidate_id = store, bank_id, candidate_id
        self.signals = SourceIndexSignals()

    def run(self):
        try:
            clips = candidate_recipe(self.store, self.bank_id, self.candidate_id)
            audio = {clip["path"]: sf.read(safe_path(self.store.directory,clip["path"]),dtype="float32")[0]
                     for clip in clips}
            self.signals.ready.emit((clips, audio))
        except Exception as error:
            try:
                self.signals.error.emit(str(error))
            except RuntimeError:
                pass


class ClipItem(QGraphicsObject):
    def __init__(self, editor, data):
        super().__init__()
        self.editor, self.data = editor, copy.deepcopy(data)
        self.audio = editor.clip_audio(data["path"])
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setPos(data["position"]/44.1, 35+data["track"]*96)
        self.mode = None
        self.setToolTip("拖动中间移动片段；左右边缘裁剪；左上/右上拖动调淡入淡出。右键可单独导出。")

    def boundingRect(self):
        return QRectF(0, 0, (self.data["source_end"]-self.data["source_start"])/44.1, 84)

    def paint(self, painter, option, widget=None):
        rect = self.boundingRect()
        color = QColor("#345c80" if not self.isSelected() else "#477faf")
        painter.fillRect(rect, color)
        pen=QPen(QColor("#d9e8f5"), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(rect)
        segment = self.audio[self.data["source_start"]:self.data["source_end"]]
        stride = max(1, len(segment)//400)
        wave = QPainterPath()
        for n in range(0, len(segment), stride):
            x = n/44.1
            low, high = float(segment[n:n+stride].min()), float(segment[n:n+stride].max())
            wave.moveTo(x, 52-low*25)
            wave.lineTo(x, 52-high*25)
        painter.drawPath(wave)
        painter.setPen(QColor("#ffc767"))
        items=[item for item in self.editor.scene.items() if isinstance(item,ClipItem)]
        groups=timeline_fades([item.data for item in items],strict=False)
        indices=np.linspace(self.data["source_start"],self.data["source_end"]-1,48)
        envelope=clip_envelope(self.data,groups[items.index(self)],indices)
        fade_path=QPainterPath()
        for index,(sample,level) in enumerate(zip(indices,envelope)):
            point=QPointF((sample-self.data["source_start"])/44.1,80-56*level)
            if index:
                fade_path.lineTo(point)
            else:
                fade_path.moveTo(point)
        painter.drawPath(fade_path)
        # Keep labels readable when the timeline is stretched horizontally.
        transform=painter.worldTransform()
        top_left=transform.map(QPointF(0,2))
        bottom_right=transform.map(QPointF(rect.width(),26))
        painter.save()
        painter.resetTransform()
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(top_left.x()+6,top_left.y(),max(0,bottom_right.x()-top_left.x()-12),24),self.data["label"]+" · "+self.data.get("part","片段"))
        painter.restore()

    def hoverMoveEvent(self, event):
        self.setCursor(Qt.CursorShape.SizeHorCursor if event.pos().x()<8 or event.pos().x()>self.boundingRect().width()-8 else Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        self.editor.scene.clearSelection()
        self.setSelected(True)
        self.editor.selected_changed()
        self.editor.set_edit_cursor(round(event.scenePos().x()*44.1))
        self.before = self.editor.recipe()
        self.original = copy.deepcopy(self.data)
        self.press = event.scenePos()
        x, width = event.pos().x(), self.boundingRect().width()
        self.mode = ("fade_in" if x < 14 else "fade_out" if x > width-14 else "move") if event.pos().y()<22 else ("left" if x<8 else "right" if x>width-8 else "move")
        event.accept()

    def mouseMoveEvent(self, event):
        if not self.mode:
            return
        dx = round((event.scenePos().x()-self.press.x())*44.1)
        old = self.original
        data = copy.deepcopy(old)
        self.prepareGeometryChange()
        if self.mode == "move":
            data["position"] = max(0, min(120*44100-(data["source_end"]-data["source_start"]), old["position"]+dx))
            data["track"] = max(0, min(3, round(old["track"]+(event.scenePos().y()-self.press.y())/96)))
        elif self.mode == "left":
            dx = max(-old["source_start"], -old["position"], min(dx, old["source_end"]-old["source_start"]-44))
            data["source_start"] += dx
            data["position"] += dx
        elif self.mode == "right":
            data["source_end"] = max(old["source_start"]+44, min(len(self.audio), old["source_end"]+dx, old["source_start"]+120*44100-old["position"]))
        else:
            duration = old["source_end"]-old["source_start"]
            data[self.mode] = max(0, min(duration, old[self.mode]+(dx if self.mode=="fade_in" else -dx)))
            for fade in data.get("inherited_fades",[]):
                fade[self.mode]=0
        duration = data["source_end"]-data["source_start"]
        data["fade_in"] = min(data["fade_in"], duration)
        data["fade_out"] = min(data["fade_out"], duration)
        self.data = data
        self.setPos(data["position"]/44.1, 35+data["track"]*96)
        self.update()
        self.editor.scene.update()
        self.editor.selected_changed()

    def mouseReleaseEvent(self, event):
        if self.mode:
            self.mode = None
            self.editor.record(self.before)
            self.editor.resize_scene()
        event.accept()

    def contextMenuEvent(self, event):
        self.editor.scene.clearSelection()
        self.setSelected(True)
        self.editor.selected_changed()
        menu = QMenu()
        export = menu.addAction("单独导出这个片段…")
        export.setToolTip("只导出此片段当前裁剪区间与淡入淡出，文件名默认使用罗马音。")
        remove = menu.addAction("从时间线移除")
        split = menu.addAction("在时间光标处切分（S）")
        split.setToolTip("选中片段在红色时间光标处切分，保留原音频、音量与淡化；可撤销。")
        remove.setToolTip("移除时间线片段；原始候选文件保留，可撤销。")
        selected = menu.exec(event.screenPos())
        if selected == export:
            self.editor.export_selected()
        elif selected == remove:
            self.editor.remove_selected()
        elif selected == split:
            self.editor.split_selected()


class Timeline(QGraphicsView):
    def __init__(self, scene, editor):
        super().__init__(scene, editor)
        self.editor = editor
        self.pan_origin = None

    def drawForeground(self, painter, rect):
        painter.setPen(QPen(QColor("#ff796d"), 1))
        x = self.editor.edit_cursor/44.1
        painter.drawLine(QPointF(x, 0), QPointF(x, 425))

    def mousePressEvent(self, event):
        if event.button()==Qt.MouseButton.MiddleButton:
            self.pan_origin = event.position().toPoint()
            self.pan_scroll = (self.horizontalScrollBar().value(), self.verticalScrollBar().value())
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button()==Qt.MouseButton.LeftButton and not isinstance(self.itemAt(event.pos()), ClipItem):
            self.editor.set_edit_cursor(round(self.mapToScene(event.pos()).x()*44.1))
            self.setFocus()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan_origin is not None:
            delta = event.position().toPoint() - self.pan_origin
            self.horizontalScrollBar().setValue(self.pan_scroll[0] - delta.x())
            self.verticalScrollBar().setValue(self.pan_scroll[1] - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button()==Qt.MouseButton.MiddleButton and self.pan_origin is not None:
            self.pan_origin = None
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y()>0 else 1/1.2
            current = self.transform().m11()
            if .08 <= current*factor <= 8:
                self.scale(factor, 1)
            event.accept()
        else:
            super().wheelEvent(event)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor("#1c2430"))
        painter.setPen(QColor("#607083"))
        for track in range(5):
            painter.drawLine(QPointF(rect.left(), 30+track*96), QPointF(rect.right(), 30+track*96))
        scale = self.transform().m11()
        step = 100 if scale>.5 else 500
        for x in range(max(0, int(rect.left()/step)*step), int(rect.right())+step, step):
            painter.drawLine(QPointF(x, 0), QPointF(x, 420))
            point=painter.worldTransform().map(QPointF(x+4,22))
            painter.save()
            painter.resetTransform()
            painter.drawText(point, f"{x/1000:.2f}s")
            painter.restore()


class SpliceWorkbench(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.host = parent
        self.store, self.bank_id = parent.store, parent.bank_id
        from .app import romaji_labels
        self.romaji={**romaji_labels("ja-cvvc"),**romaji_labels("zh-cvvc"),**parent.romaji}
        bank=self.store.bank(self.bank_id)
        self.custom_labels={unit["alias"]:unit["label"] for unit in self.store.custom_units(self.bank_id)}
        self.alias_format=bank["preset"].get("alias_format",DEFAULT_ALIAS_FORMAT)
        self.setWindowTitle("拼接工作台 · "+bank["name"]+" · VoiceBankGen")
        self.setModal(False)
        self.player=QMediaPlayer(self)
        self.audio_output=QAudioOutput(self)
        self.audio_output.setVolume(parent.output.volume())
        self.player.setAudioOutput(self.audio_output)
        self.player.errorOccurred.connect(lambda *_:self.message.setText("播放失败："+self.player.errorString()))
        self.audio_cache=OrderedDict()
        self.source_records=[]
        self.index_revision=None
        self.index_pending=False
        self.source_labels={alias:self.sound_label(alias) for alias in self.romaji.keys() | self.custom_labels.keys()}
        self.cache_key=(str(self.store.directory),self.bank_id,self.alias_format,tuple(self.custom_labels.items()))
        self.resize(1350, 820)
        self.finished.connect(self.save_recovery)
        self.history, self.future = [], []
        self.loading = False
        self.edit_cursor=0
        self.playback_kind=None
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        for label, callback, tip in (
            ("打开拼接草稿", self.load_recipe, "载入本工程保存的时间线位置、裁剪和淡化参数。"),
            ("保存拼接草稿", self.save_recipe, "保存可继续编辑的拼接配方；不修改正式选音。"),
            ("撤销", self.undo, "撤销时间线编辑。Ctrl+Z。"),
            ("重做", self.redo, "重做时间线编辑。Ctrl+Shift+Z。"),
            ("移除选中片段", self.remove_selected, "只从时间线移除，原候选保留。Delete。"),
            ("切分片段（S）", self.split_selected, "在红色时间光标处切分所选片段；点击时间线定位，支持撤销。时间线获得焦点后可按 S。"),
            ("适合窗口", self.fit, "缩放以显示全部片段；Ctrl+滚轮可手动缩放。"),
            ("播放 / 暂停（空格）", self.play, "播放整条拼接时间线，包括重叠和淡化；再次点击暂停。")):
            self.button(toolbar, label, callback, tip)
        layout.addLayout(toolbar)
        split = QSplitter()
        left = QWidget()
        col = QVBoxLayout(left)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索平假名 / 罗马音 / 拼音 / 候选 ID")
        self.search.setToolTip("在内存索引中筛选原始候选，停止输入约 160 毫秒后刷新；单击试听所选部分，双击添加到时间线。")
        col.addWidget(self.search)
        self.sources = QListView()
        self.source_model=SourceListModel(self.sources)
        self.sources.setModel(self.source_model)
        self.sources.setUniformItemSizes(True)
        col.addWidget(self.sources, 1)
        self.source_count=QLabel("正在后台读取素材索引…")
        col.addWidget(self.source_count)
        self.part = QComboBox()
        self.part.addItem("完整音节 / 使用区间", "whole")
        self.part.addItem("辅音部分（至元音起点）", "consonant")
        self.part.addItem("元音部分（从元音起点）", "vowel")
        self.part.setToolTip("依据 TIFA 音素分界选择 C/V；添加后可自由拖边裁剪成任意部分。")
        col.addWidget(self.part)
        self.join = QDoubleSpinBox()
        self.join.setRange(0, 200)
        self.join.setValue(bank["preset"].get("splice_crossfade_ms") or 20)
        self.join.setSuffix(" ms")
        self.join.setToolTip("追加到末尾时与前一个片段的重叠时长；长度自动限制为较短片段的一半。拖动可另行调整。")
        col.addWidget(QLabel("追加时交叉淡化"))
        col.addWidget(self.join)
        self.button(col, "添加到时间线", self.add_source, "将所选 C/V 或完整音节追加到时间线；双击候选也可添加。")
        self.button(col, "试听素材", self.play_source, "试听所选素材部分，不改变时间线或正式选音。")
        self.sources.doubleClicked.connect(self.add_source)
        self.sources.clicked.connect(lambda *_:self.play_source())
        self.part.currentIndexChanged.connect(lambda *_:self.play_source() if self.sources.currentIndex().isValid() else None)
        self.sources.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sources.customContextMenuRequested.connect(self.source_menu)
        split.addWidget(left)
        right = QWidget()
        col = QVBoxLayout(right)
        col.addWidget(QLabel("拖动移动 · 中键拖动视图 · 左右边缘裁剪 · 顶角调整淡化 · 同轨重叠自动交叉淡化 · Ctrl+滚轮缩放"))
        self.scene = QGraphicsScene(self)
        self.view = Timeline(self.scene,self)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.player.positionChanged.connect(self.playback_position_changed)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.view.scale(.8, 1)
        self.scene.selectionChanged.connect(self.selected_changed)
        col.addWidget(self.view, 1)
        cursor_row=QHBoxLayout()
        cursor_row.addWidget(QLabel("时间光标 / 切分位置"))
        self.cursor_field=QDoubleSpinBox()
        self.cursor_field.setRange(0,120000)
        self.cursor_field.setDecimals(3)
        self.cursor_field.setSuffix(" ms")
        self.cursor_field.setToolTip("红色光标位置。点击时间线或输入精确位置，再选中片段并点击切分片段（S）。试听左侧素材不会改变这个位置。")
        self.cursor_field.valueChanged.connect(lambda value:self.set_edit_cursor(round(value*44.1)))
        cursor_row.addWidget(self.cursor_field)
        cursor_row.addStretch()
        col.addLayout(cursor_row)
        fields = QHBoxLayout()
        self.fields = {}
        for key, name in (("position","时间线位置"),("source_start","素材起点"),("source_end","素材终点"),("fade_in","淡入"),("fade_out","淡出")):
            form = QFormLayout()
            value = QDoubleSpinBox()
            value.setRange(0, 120000)
            value.setDecimals(3)
            value.setSuffix(" ms")
            value.setToolTip(name+"：精确调整选中片段，单位毫秒。素材坐标相对于候选预览 WAV。")
            value.editingFinished.connect(self.apply_fields)
            self.fields[key] = value
            form.addRow(name, value)
            fields.addLayout(form)
        self.track = QComboBox()
        self.track.addItems(["轨道 1", "轨道 2", "轨道 3", "轨道 4"])
        self.track.setToolTip("同轨重叠自动淡化；不同轨道直接叠加混音。")
        self.track.currentIndexChanged.connect(self.apply_fields)
        fields.addWidget(self.track)
        self.gain=QDoubleSpinBox()
        self.gain.setRange(-24,24)
        self.gain.setSuffix(" dB")
        self.gain.setToolTip("选中片段音量；自动拼接重新打开时保留原供体的响度匹配增益。")
        self.gain.editingFinished.connect(self.apply_fields)
        fields.addWidget(self.gain)
        col.addLayout(fields)
        output = QHBoxLayout()
        output.addWidget(QLabel("保存目标"))
        self.alias_search=QLineEdit()
        self.alias_search.setObjectName("splice_target_search")
        self.alias_search.setClearButtonEnabled(True)
        self.alias_search.setPlaceholderText("搜索音名 / 罗马音 / 拼音")
        self.alias_search.setToolTip("筛选右侧目标发音列表；支持平假名、罗马音、中文拼音及自定义标签。保存使用下拉框中选中的目标音。")
        self.alias_search.setMaximumWidth(220)
        output.addWidget(self.alias_search)
        self.alias = QComboBox()
        self.alias.setPlaceholderText("没有匹配的目标音")
        self.alias.setMinimumContentsLength(12)
        self.alias.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.target_records=self.store.targets(self.bank_id)
        for alias, kind in self.target_records:
            self.alias.addItem(self.sound_label(alias)+" · "+kind, alias)
        current = parent.current_candidate.alias if parent.current_candidate else None
        if current:
            self.alias.setCurrentIndex(max(0, self.alias.findData(current)))
        self.last_target_alias=self.alias.currentData()
        self.alias.currentIndexChanged.connect(self.remember_target)
        self.alias.setToolTip("保存为候选时指定目标发音；保存后请校对 oto、试听接受并正式使用。")
        output.addWidget(self.alias, 1)
        self.button(output, "保存到候选", self.save_candidate, "加入当前工程的备选，不自动替换选音或同步。初始 oto 是手动估计，需校对。")
        self.button(output, "单独导出 WAV", self.export_all, "把整个时间线混合成独立 WAV；默认罗马音文件名，不改变工程选音。")
        col.addLayout(output)
        split.addWidget(right)
        split.setSizes([300, 1050])
        layout.addWidget(split, 1)
        self.message = QLabel("选择左侧候选并添加；所有拼接编辑均在此工作台暂存。")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.search_timer=QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(250)
        self.search_timer.timeout.connect(self.refresh_sources)
        self.search.textChanged.connect(lambda *_:self.search_timer.start())
        self.alias_search_timer=QTimer(self)
        self.alias_search_timer.setSingleShot(True)
        self.alias_search_timer.setInterval(250)
        self.alias_search_timer.timeout.connect(self.refresh_aliases)
        self.alias_search.textChanged.connect(lambda *_:self.alias_search_timer.start())
        self.alias_search.returnPressed.connect(self.refresh_aliases)
        QTimer.singleShot(0,self.load_source_index)
        self.resize_scene()
        if parent.current_candidate and parent.current_candidate.provenance_kind.value in {"cv_spliced","vowel_extended"}:
            self.recipe_task=RecipeTask(self.store,self.bank_id,parent.current_candidate.candidate_id)
            self.recipe_task.signals.ready.connect(self.initial_recipe_ready)
            self.recipe_task.signals.error.connect(self.initial_recipe_error)
            self.message.setText("正在后台载入当前候选的拼接配方；可继续使用音源工作台。")
            QThreadPool.globalInstance().start(self.recipe_task)
        for key, action in (("Space",self.play_shortcut),("Ctrl+Z",self.undo),("Ctrl+Shift+Z",self.redo),("Delete",self.remove_selected),("Ctrl+S",self.save_recipe)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
        self.split_shortcut=QShortcut(QKeySequence("S"),self.view)
        self.split_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.split_shortcut.activated.connect(self.split_selected)

    def button(self, layout, text, callback, tip):
        button = QPushButton(text)
        button.setToolTip(tip)
        button.setAccessibleDescription(tip)
        button.clicked.connect(callback)
        layout.addWidget(button)

    def sound_label(self,alias):
        from .bank_aliases import split_variant
        base,suffix=split_variant(alias)
        label=self.custom_labels.get(base,to_katakana(base) if self.alias_format=="katakana" else base)+suffix
        roman=self.romaji.get(base,label)+suffix
        return f"{label} ({roman})" if roman!=label else label

    def refresh_sources(self):
        query = self.search.text().strip().casefold()
        previous = self.sources.currentIndex().data(Qt.ItemDataRole.UserRole)
        rows = [row for row in self.source_records if (row["unit_role"] != "custom" or row["alias"] in self.custom_labels)
                and (not query or query in row["search"])]
        self.source_model.replace(rows)
        if previous:
            index = next((i for i,row in enumerate(rows) if row["candidate_id"] == previous), None)
            if index is not None:
                self.sources.setCurrentIndex(self.source_model.index(index))
        self.source_count.setText(f"显示 {len(rows)} / {len(self.source_records)} 个素材")

    def load_source_index(self):
        revision=self.store.revision
        if self.index_pending or self.index_revision==revision:
            return
        self.custom_labels={unit["alias"]:unit["label"] for unit in self.store.custom_units(self.bank_id)}
        self.alias_format=self.store.bank(self.bank_id)["preset"].get("alias_format",DEFAULT_ALIAS_FORMAT)
        self.source_labels={alias:self.sound_label(alias) for alias in self.romaji.keys() | self.custom_labels.keys()}
        self.cache_key=(str(self.store.directory),self.bank_id,self.alias_format,tuple(self.custom_labels.items()))
        self.target_records=self.store.targets(self.bank_id)
        self.refresh_aliases()
        cached=self.host.splice_source_cache.get(self.cache_key)
        if cached and cached[0]==revision:
            self.source_index_ready(cached)
            return
        self.index_pending=True
        self.source_count.setText("正在后台更新素材索引…")
        self.source_task=SourceIndexTask(self.store,revision,self.source_labels)
        self.source_task.signals.ready.connect(self.source_index_ready)
        self.source_task.signals.error.connect(self.source_index_error)
        QThreadPool.globalInstance().start(self.source_task)

    def source_index_ready(self, result):
        self.index_pending=False
        self.index_revision,self.source_records=result
        cache=self.host.splice_source_cache
        cache[self.cache_key]=result
        if len(cache)>8:
            cache.pop(next(iter(cache)))
        self.refresh_sources()
        if self.index_revision!=self.store.revision:
            QTimer.singleShot(0,self.load_source_index)

    def source_index_error(self, error):
        self.index_pending=False
        self.source_count.setText("素材索引读取失败："+error)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type()==QEvent.Type.ActivationChange and self.isActiveWindow() and hasattr(self,"source_model"):
            QTimer.singleShot(0,self.load_source_index)

    def initial_recipe_ready(self, result):
        clips,audio=result
        if not clips:
            return
        # Never replace timeline edits made while the recipe was loading.
        if self.recipe() or self.history or self.future:
            self.message.setText("原配方已读取；时间线已有编辑，保留当前内容。")
            return
        self.audio_cache.update(audio)
        self.restore(clips)
        self.fit()
        self.message.setText("已载入当前拼接候选的组成片段；可调整重叠、淡化，再保存为新候选。")

    def initial_recipe_error(self, error):
        self.message.setText("原拼接配方未能完整载入："+error+"。可从左侧素材重新拼接。")

    def clip_audio(self, path):
        if path not in self.audio_cache:
            self.audio_cache[path]=sf.read(safe_path(self.store.directory,path),dtype="float32")[0]
        self.audio_cache.move_to_end(path)
        if len(self.audio_cache)>32:
            self.audio_cache.popitem(last=False)
        return self.audio_cache[path]

    def refresh_aliases(self):
        self.alias_search_timer.stop()
        query=self.alias_search.text().strip().casefold()
        previous=self.alias.currentData() or self.last_target_alias
        self.alias.blockSignals(True)
        self.alias.clear()
        for alias,kind in self.target_records:
            label=self.sound_label(alias)+" · "+kind
            if not query or query in (label+" "+alias).casefold():
                self.alias.addItem(label,alias)
        self.alias.setCurrentIndex(max(0,self.alias.findData(previous)))
        self.alias.blockSignals(False)
        self.remember_target()

    def remember_target(self):
        if self.alias.currentData() is not None:
            self.last_target_alias=self.alias.currentData()

    def target_alias(self):
        if self.alias_search_timer.isActive():
            self.refresh_aliases()
        alias=self.alias.currentData()
        if alias is None:
            raise ValueError("没有匹配的目标发音；请修改搜索并选择目标音")
        return alias

    def source_clip(self):
        item = self.sources.currentIndex()
        if not item.isValid():
            raise ValueError("请先选择素材")
        return clip_from_candidate(self.store, item.data(Qt.ItemDataRole.UserRole), self.bank_id, self.part.currentData())

    def recipe(self):
        return sorted((copy.deepcopy(item.data) for item in self.scene.items() if isinstance(item, ClipItem)), key=lambda c:(c["position"],c["track"],c["candidate_id"],c["source_start"],c["source_end"],c["fade_in"],c["fade_out"],c.get("gain_db",0)))

    def restore(self, clips):
        self.scene.clear()
        for clip in clips:
            self.scene.addItem(ClipItem(self, clip))
        self.resize_scene()

    def record(self, before):
        if before != self.recipe():
            self.history.append(before)
            self.history = self.history[-UNDO_LIMIT:]
            self.future.clear()
            self.player.stop()
            self.message.setText("时间线修改已暂存；保存草稿可继续编辑，保存到候选后仍需手动同步。")

    def add_source(self, *args):
        try:
            clip = self.source_clip()
            before = self.recipe()
            if before:
                previous = max(before, key=lambda c: c["position"]+c["source_end"]-c["source_start"])
                crossfade = min(round(self.join.value()*44.1), (clip["source_end"]-clip["source_start"])//2, (previous["source_end"]-previous["source_start"])//2)
                clip["position"] = previous["position"]+previous["source_end"]-previous["source_start"]-crossfade
                clip["track"] = previous["track"]
            validate_clips(self.store, [*before, clip])
            self.scene.addItem(ClipItem(self, clip))
            self.record(before)
            self.resize_scene()
            self.fit()
        except Exception as error:
            self.message.setText(str(error))

    def selected(self):
        return next((i for i in self.scene.selectedItems() if isinstance(i, ClipItem)), None)

    def selected_changed(self):
        selected = self.selected()
        for item in self.scene.items():
            if isinstance(item,ClipItem):
                item.setZValue(1 if item is selected else 0)
        self.loading = True
        for key, value in self.fields.items():
            value.setEnabled(selected is not None)
            if selected:
                value.setValue(selected.data[key]/44.1)
        self.track.setEnabled(selected is not None)
        self.gain.setEnabled(selected is not None)
        if selected:
            self.track.setCurrentIndex(selected.data["track"])
            self.gain.setValue(selected.data.get("gain_db",0))
        self.loading = False

    def apply_fields(self):
        item = self.selected()
        if self.loading or not item:
            return
        before = self.recipe()
        original = dict(item.data)
        data = {**original, **{k:round(v.value()*44.1) for k,v in self.fields.items()}, "track":self.track.currentIndex()}
        if self.gain.value()!=original.get("gain_db",0):
            data["gain_db"]=self.gain.value()
        data["inherited_fades"]=copy.deepcopy(original.get("inherited_fades",[]))
        for key in ("fade_in","fade_out"):
            if data[key]!=original[key]:
                for fade in data["inherited_fades"]:
                    fade[key]=0
        try:
            validate_clips(self.store, [data])
            item.prepareGeometryChange()
            item.data = data
            item.setPos(data["position"]/44.1, 35+data["track"]*96)
            item.update()
            self.record(before)
            self.resize_scene()
        except Exception as error:
            self.message.setText(str(error))
            self.selected_changed()

    def remove_selected(self):
        before = self.recipe()
        for item in list(self.scene.selectedItems()):
            self.scene.removeItem(item)
        self.record(before)
        self.resize_scene()

    def set_edit_cursor(self,samples):
        self.edit_cursor=max(0,min(120*44100,samples))
        self.cursor_field.blockSignals(True)
        self.cursor_field.setValue(self.edit_cursor/44.1)
        self.cursor_field.blockSignals(False)
        if self.playback_kind=="timeline":
            self.player.setPosition(round(self.edit_cursor/44.1))
        else:
            self.player.stop()
        self.view.viewport().update()

    def playback_position_changed(self,milliseconds):
        if self.playback_kind=="timeline":
            self.edit_cursor=round(milliseconds*44.1)
            self.cursor_field.blockSignals(True)
            self.cursor_field.setValue(milliseconds)
            self.cursor_field.blockSignals(False)
            self.view.viewport().update()

    def split_selected(self):
        selected=self.selected()
        if not selected:
            self.message.setText("请先选择时间线片段，并将红色光标移到片段内部")
            return
        try:
            before=self.recipe()
            left,right=split_clip(selected.data,self.edit_cursor)
            result=[item.data for item in self.scene.items() if isinstance(item,ClipItem) and item is not selected]+[left,right]
            validate_clips(self.store,result)
            timeline_fades(result)
            self.playback_kind=None
            self.player.stop()
            self.scene.removeItem(selected)
            self.scene.addItem(ClipItem(self,left))
            following=ClipItem(self,right)
            self.scene.addItem(following)
            self.scene.clearSelection()
            following.setSelected(True)
            self.record(before)
            self.resize_scene()
            self.message.setText("片段已在时间光标处切分；音频和淡化保留，可撤销。")
        except Exception as error:
            self.message.setText(str(error))

    def undo(self):
        if self.history:
            self.future.append(self.recipe())
            self.future=self.future[-UNDO_LIMIT:]
            self.restore(self.history.pop())
            self.player.stop()

    def redo(self):
        if self.future:
            self.history.append(self.recipe())
            self.history=self.history[-UNDO_LIMIT:]
            self.restore(self.future.pop())
            self.player.stop()

    def resize_scene(self):
        width = max([1200]+[(c["position"]+c["source_end"]-c["source_start"])/44.1+200 for c in self.recipe()])
        self.scene.setSceneRect(0, 0, width, 425)

    def fit(self):
        self.view.resetTransform()
        width = max(300, self.scene.itemsBoundingRect().right()+50)
        self.view.scale(max(.08, min(3, self.view.viewport().width()/width)), 1)

    def play_clips(self, clips,kind="timeline"):
        path = assembly_preview(self.store, clips)
        info = sf.info(path)
        self.playback_kind=kind
        self.play_audio(path, 0, info.frames, "拼接素材试听" if kind=="source" else "拼接工作台试听")

    def play_audio(self, path, start, end, label):
        # Workbench playback belongs to its pinned project, even while the
        # main window changes bank/project. Each window has its own cursor.
        self.host.player.stop()
        for editor in self.host.splice_windows.values():
            if editor is not self:
                editor.player.stop()
        source=QUrl.fromLocalFile(str(path))
        if self.player.source()!=source:
            self.player.setSource(source)
        else:
            self.player.setPosition(0)
        self.player.play()
        self.message.setText(label+f" · {(end-start)/44.1:.1f} ms")

    def play(self):
        try:
            if self.playback_kind=="timeline" and self.player.playbackState() == self.player.PlaybackState.PlayingState:
                self.player.pause()
            elif self.playback_kind=="timeline" and self.player.playbackState() == self.player.PlaybackState.PausedState:
                self.player.play()
            else:
                self.play_clips(self.recipe())
        except Exception as error:
            self.message.setText(str(error))

    def play_shortcut(self):
        from PySide6.QtWidgets import QApplication
        focus=QApplication.focusWidget()
        if isinstance(focus,QLineEdit) and not isinstance(focus.parentWidget(),QDoubleSpinBox):
            focus.insert(" ")
            return
        self.play()

    def play_source(self):
        try:
            self.player.stop()
            self.play_clips([self.source_clip()],kind="source")
        except Exception as error:
            self.player.stop()
            self.message.setText(str(error))

    def source_menu(self, position):
        item = self.sources.indexAt(position)
        if not item.isValid():
            return
        self.sources.setCurrentIndex(item)
        menu = QMenu(self)
        action = menu.addAction("单独导出所选素材部分…")
        action.setToolTip("导出当前选择的完整音节、辅音或元音部分。")
        if menu.exec(self.sources.viewport().mapToGlobal(position)) == action:
            try:
                clip = self.source_clip()
                self.export_clips([clip], self.romaji.get(clip["label"], "clip"))
            except Exception as error:
                self.message.setText(str(error))

    def export_clips(self, clips, name):
        filename, _ = QFileDialog.getSaveFileName(self, "单独导出 WAV（请选择新文件名）", name.replace(" ","_")+".wav", "WAV (*.wav)")
        if filename:
            path = export_audio(self.store, clips, filename)
            self.message.setText("已导出："+str(path))

    def export_selected(self):
        try:
            selected = self.selected()
            if selected:
                clip = {**selected.data, "position":0, "track":0}
                self.export_clips([clip], self.romaji.get(clip["label"], "clip"))
        except Exception as error:
            self.message.setText(str(error))

    def export_all(self):
        try:
            self.export_clips(self.recipe(), self.romaji.get(self.target_alias(),"splice"))
        except Exception as error:
            self.message.setText(str(error))

    def save_candidate(self):
        try:
            candidate = save_candidate(self.store, self.bank_id, self.target_alias(), self.recipe())
            if self.host.store and self.host.store.directory==self.store.directory and self.host.bank_id==self.bank_id:
                self.host.save_marks()
                self.host.refresh_targets()
            self.player.stop()
            self.message.setText(f"已加入 {self.store.display_alias(self.bank_id,candidate.alias)} 候选（{candidate.candidate_id[:8]}）。请回主工作台校对 oto、试听接受，再正式使用与同步。")
        except Exception as error:
            self.message.setText(str(error))

    def save_recipe(self):
        try:
            alias=self.target_alias()
            clips = validate_clips(self.store, self.recipe())
            directory = self.store.directory/"assemblies"
            directory.mkdir(exist_ok=True)
            path = directory/(self.romaji.get(alias,"splice").replace(" ","_")+"_"+uuid4().hex[:8]+".json")
            write_json(path, {"schema_version":1,"project_id":self.store.project_id,"alias":alias,"clips":clips})
            self.message.setText("拼接草稿已保存："+str(path))
        except Exception as error:
            self.message.setText(str(error))

    def load_recipe(self):
        filename, _ = QFileDialog.getOpenFileName(self,"打开拼接草稿",str(self.store.directory/"assemblies"),"JSON (*.json)")
        if not filename:
            return
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            if data["schema_version"] != 1 or data["project_id"] != self.store.project_id:
                raise ValueError("这个拼接草稿不属于当前工程")
            clips = validate_clips(self.store, data["clips"])
            self.alias_search.clear()
            self.refresh_aliases()
            index = self.alias.findData(data["alias"])
            if index < 0:
                raise ValueError("草稿目标音不属于当前版本")
            before = self.recipe()
            self.restore(clips)
            self.alias.setCurrentIndex(index)
            self.record(before)
            self.fit()
        except Exception as error:
            self.message.setText(str(error))

    def save_recovery(self, *args):
        # QDialog.finished covers the close button, Esc and programmatic reject.
        if self.recipe():
            directory = self.store.directory/"assemblies"
            directory.mkdir(exist_ok=True)
            write_json(directory/("autosave_"+uuid4().hex[:8]+".json"), {"schema_version":1,"project_id":self.store.project_id,"alias":self.last_target_alias,"clips":self.recipe()})
        self.player.stop()
