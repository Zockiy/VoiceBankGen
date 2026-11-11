from types import SimpleNamespace

import pytest

from utau_forge import song_import
from utau_forge.models import Utterance


def test_lrc_boundaries_and_silence_are_not_assigned_as_lyrics():
    rows = song_import.parse_lrc("[ar:歌手]\n[00:01.20]第一句\n[00:03.00]\n[00:04.00]第二句\n[00:06.00]", 7)
    assert rows == [{"start": 1.2, "end": 3.0, "text": "第一句"},
                    {"start": 4.0, "end": 6.0, "text": "第二句"}]
    assert song_import.match_vad(rows, [(1.4, 2.8)])[1]["match"] == "未检出声音，保留歌词时间"


@pytest.mark.parametrize("lrc", ["歌词没有时间戳", "[00:01.00]一\n[00:01.00]二", "[offset:100]\n[00:01.00]一", "[00:11.00]超出"])
def test_invalid_lyrics_are_rejected(lrc):
    with pytest.raises(ValueError):
        song_import.parse_lrc(lrc, 10)


def test_reviewed_song_survives_retry_and_rejects_different_reimport(monkeypatch):
    saved = []
    monkeypatch.setattr(song_import, "read_jsonl", lambda path, model: list(saved))
    monkeypatch.setattr(song_import, "write_jsonl", lambda path, records: saved.__setitem__(slice(None), records))
    asset = SimpleNamespace(source_id="s", speaker_id="p", language_id="ja", canonical_samples=441000)
    rows = [{"start": 1, "end": 2, "text": "「あ」"}, {"start": 3, "end": 4, "text": "い"}]
    from pathlib import Path
    song_import.save_song_utterances(Path("unused"), asset, rows)
    assert all(isinstance(item, Utterance) and item.revised_text for item in saved)
    assert saved[0].revised_text == "あ"
    song_import.save_song_utterances(Path("unused"), asset, rows)
    assert len(saved) == 2
    with pytest.raises(ValueError):
        song_import.save_song_utterances(Path("unused"), asset, [{"start": 1, "end": 3, "text": "違う"}])


def test_review_edits_are_saved_from_waveform(monkeypatch):
    from PySide6.QtWidgets import QApplication, QDialog
    from utau_forge.song_review import SongReviewDialog
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(SongReviewDialog, "load_wave", lambda self, path: None)
    result = {"audio": "unused", "preview_audio": "unused.wav", "duration": 5.0,
              "rows": [{"start": 1.0, "end": 2.0, "text": "あ", "match": "需检查"}]}
    dialog = SongReviewDialog(result)
    dialog.table.selectRow(0)
    dialog.region.setRegion((1.2, 2.4))
    app.processEvents()
    dialog.table.item(0, 2).setText("い")
    dialog.accept_checked()
    assert dialog.result == {**result, "rows": [{"start": 1.2, "end": 2.4, "text": "い"}]}
    assert QDialog.result(dialog) == QDialog.DialogCode.Accepted


def test_trimmed_intro_is_matched_across_sections_and_credits_are_skipped():
    rows = song_import.parse_lrc("[00:00.000]作曲 : 作曲者\n[00:23.680]一\n[00:35.000]\n[01:23.680]二\n[01:35.000]\n[02:23.680]三", 140, allow_outside=True)
    activity = [(2.08, 13.0), (62.08, 73.0), (122.08, 135.0)]
    estimate = song_import.estimate_offset(rows, activity)
    assert estimate["seconds"] == pytest.approx(-21.6)
    assert estimate["supported"] == 3
    matched = song_import.match_song_rows(rows, activity, estimate["seconds"], 140)
    assert not matched[0]["include"]
    assert matched[-1]["start"] == pytest.approx(122.08)
    assert matched[-1]["end"] <= 140
