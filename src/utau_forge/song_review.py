"""Visual review of VAD-assisted lyric windows before committing a song."""
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)


class SongReviewDialog(QDialog):
    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result
        self.updating = False
        self.play_end = None
        self.setWindowTitle("检查歌声与歌词匹配")
        self.resize(1050, 700)
        layout = QVBoxLayout(self)
        title = QLabel("选择歌词行后拖动范围，或编辑秒数与歌词。绿色表示声音区间，虚线为平移后的歌词时间。取消行末勾选可跳过署名等内容；请逐行试听。")
        title.setWordWrap(True)
        layout.addWidget(title)
        offsets = QHBoxLayout()
        estimate = result.get("offset_estimate", {})
        offsets.addWidget(QLabel("歌词整体平移（负数表示干声删掉了前奏）"))
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-3600, 3600)
        self.offset.setDecimals(3)
        self.offset.setSingleStep(.1)
        self.offset.setSuffix(" 秒")
        self.offset.setValue(estimate.get("seconds", 0))
        offsets.addWidget(self.offset)
        rematch = QPushButton("按此偏移重新匹配")
        rematch.setEnabled("lyric_rows" in result)
        rematch.setToolTip("根据原始 LRC 重新计算所有行，会覆盖表格中尚未确认的边界与歌词修改。")
        rematch.clicked.connect(self.rematch)
        offsets.addWidget(rematch)
        offsets.addWidget(QLabel(estimate.get("confidence", "") + " · " + result.get("activity_method", "VAD")))
        layout.addLayout(offsets)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "音频时间", units="s")
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.region = pg.LinearRegionItem(values=(0, 1), bounds=(0, result["duration"]))
        self.plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self.region_changed)
        self.original_lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#eab660", style=Qt.PenStyle.DashLine)) for _ in range(2)]
        for line in self.original_lines:
            self.plot.addItem(line)
        for start, end in result.get("activity_spans", result.get("vad_spans", [])):
            vad_region = pg.LinearRegionItem((start, end), movable=False, brush=pg.mkBrush(74, 169, 109, 35), pen=pg.mkPen(None))
            vad_region.setZValue(-10)
            self.plot.addItem(vad_region)
        layout.addWidget(self.plot, 2)
        self.table = QTableWidget(len(result["rows"]), 5)
        self.table.setHorizontalHeaderLabels(["开始（秒）", "结束（秒）", "歌词", "匹配提示", "启用"])
        self.table.setColumnWidth(3, 210)
        self.table.setColumnWidth(4, 50)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.populate_table()
        self.table.itemSelectionChanged.connect(self.select_row)
        self.table.itemChanged.connect(self.table_changed)
        layout.addWidget(self.table, 3)
        self.player = QMediaPlayer(self)
        self.output = QAudioOutput(self)
        self.player.setAudioOutput(self.output)
        self.player.setSource(QUrl.fromLocalFile(result["preview_audio"]))
        self.player.positionChanged.connect(self.position_changed)
        bar = QHBoxLayout()
        play = QPushButton("试听当前片段")
        play.clicked.connect(self.play_row)
        bar.addWidget(play)
        stop = QPushButton("停止")
        stop.clicked.connect(self.player.stop)
        bar.addWidget(stop)
        bar.addStretch()
        layout.addLayout(bar)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认并开始对齐")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept_checked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.load_wave(Path(result["preview_audio"]))
        if self.table.rowCount():
            self.table.selectRow(next((i for i, row in enumerate(result["rows"]) if row.get("include", True)), 0))

    def populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.result["rows"]))
        for i, row in enumerate(self.result["rows"]):
            end = min(row["end"], math.floor(self.result["duration"] * 1000) / 1000)
            for col, value in enumerate((f'{row["start"]:.3f}', f'{end:.3f}', row["text"], row["match"])):
                item = QTableWidgetItem(value)
                if col == 3:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, col, item)
            enabled = QTableWidgetItem()
            enabled.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            enabled.setCheckState(Qt.CheckState.Checked if row.get("include", True) else Qt.CheckState.Unchecked)
            self.table.setItem(i, 4, enabled)
        self.table.blockSignals(False)

    def rematch(self):
        from .song_import import match_song_rows
        self.player.stop()
        self.result["rows"] = match_song_rows(self.result["lyric_rows"], self.result["activity_spans"], self.offset.value(), self.result["duration"])
        self.result["offset_seconds"] = self.offset.value()
        self.populate_table()
        self.table.selectRow(next((i for i, row in enumerate(self.result["rows"]) if row.get("include", True)), 0))
        self.select_row()

    def load_wave(self, path):
        with sf.SoundFile(path) as stream:
            stride = max(1, stream.frames // 100000)
            times, values = [], []
            offset = 0
            while True:
                block = stream.read(441000, dtype="float32")
                if not len(block):
                    break
                indices = np.arange(0, len(block), stride)
                times.append((indices + offset) / stream.samplerate)
                values.append(block[indices])
                offset += len(block)
        if times:
            self.plot.plot(np.concatenate(times), np.concatenate(values), pen=pg.mkPen("#7fc4df", width=1))

    def select_row(self):
        row = self.table.currentRow()
        if row < 0 or self.updating:
            return
        try:
            start, end = (float(self.table.item(row, col).text()) for col in (0, 1))
        except ValueError:
            return
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= self.result["duration"]):
            return
        self.updating = True
        original = self.result["rows"][row]
        self.original_lines[0].setValue(original.get("lyric_start", original["start"]))
        self.original_lines[1].setValue(original.get("lyric_end", original["end"]))
        self.region.setRegion((start, end))
        self.plot.setXRange(max(0, start - .5), min(self.result["duration"], end + .5), padding=0)
        self.updating = False

    def region_changed(self):
        row = self.table.currentRow()
        if row < 0 or self.updating:
            return
        self.updating = True
        for col, value in enumerate(self.region.getRegion()):
            self.table.item(row, col).setText(f"{value:.3f}")
        self.updating = False

    def table_changed(self, item):
        if item.row() == self.table.currentRow() and item.column() < 2:
            self.select_row()

    def play_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        try:
            start = float(self.table.item(row, 0).text())
            self.play_end = float(self.table.item(row, 1).text())
        except ValueError:
            return
        self.player.stop()
        self.player.setPosition(round(start * 1000))
        self.player.play()

    def position_changed(self, position):
        if self.play_end is not None and position >= self.play_end * 1000:
            self.player.stop()
            self.play_end = None

    def accept_checked(self):
        rows = []
        last_end = 0.0
        try:
            for i in range(self.table.rowCount()):
                if self.table.item(i, 4).checkState() != Qt.CheckState.Checked:
                    continue
                start, end = (float(self.table.item(i, col).text()) for col in (0, 1))
                text = self.table.item(i, 2).text().strip()
                if not text or not (last_end <= start < end <= self.result["duration"]):
                    raise ValueError(f"第 {i + 1} 行缺少歌词、时间无效、超出范围或与上一行重叠")
                rows.append({"start": start, "end": end, "text": text})
                last_end = end
            if not rows:
                raise ValueError("请至少启用一行歌词")
        except ValueError as error:
            QMessageBox.warning(self, "请检查歌词片段", str(error))
            return
        self.result["rows"] = rows
        self.player.stop()
        self.accept()

    def reject(self):
        self.player.stop()
        super().reject()
