import numpy as np
import soundfile as sf

from utau_forge.audio import import_audio
from utau_forge.config import VadConfig
from utau_forge.models import SampleSpan
from utau_forge.project import initialize, load
from utau_forge.segment import segment_source


def test_48k_import_maps_two_seconds_to_canonical_samples(tmp_path):
    project = tmp_path / "project"
    initialize(project)
    input_path = tmp_path / "日语 clip 48k.wav"
    sf.write(input_path, np.zeros(96000, dtype="float32"), 48000, subtype="PCM_16")
    asset = import_audio(project, input_path, "speaker_001", "ja")
    assert asset.original_sample_rate == 48000
    assert asset.canonical_samples == 88200
    assert sf.info(project / asset.canonical_path).subtype == "FLOAT"


def test_long_manual_span_splits_at_low_energy(tmp_path):
    project = tmp_path / "project"
    initialize(project)
    input_path = tmp_path / "long.wav"
    audio = np.ones(6 * 44100, dtype="float32") * 0.05
    audio[round(2.9 * 44100):round(3.1 * 44100)] = 0
    sf.write(input_path, audio, 44100, subtype="PCM_16")
    asset = import_audio(project, input_path, "speaker_001", "ja")
    config = load(project)[1].model_copy(update={"vad": VadConfig(padding_ms=0, max_utterance_s=4)})
    utterances = segment_source(project, asset, config, SampleSpan(start=0, end=asset.canonical_samples))
    assert len(utterances) == 2
    assert all(utt.span.end - utt.span.start <= 4 * 44100 for utt in utterances)
    assert abs(utterances[0].span.end / 44100 - 3) < 0.2
    assert all("LONG_SPEECH_SPLIT_LOW_ENERGY" in utt.quality_flags for utt in utterances)
