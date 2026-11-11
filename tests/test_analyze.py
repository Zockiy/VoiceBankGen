import numpy as np

from utau_forge.analyze import _features, _longest_sustain
from utau_forge.config import Config
from utau_forge.models import Candidate, SampleSpan, Utterance


def _candidate(samples):
    return Candidate(speaker_id="speaker", source_id="source", utterance_id="utt", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="a", alias="あ", phones=["a"], raw_phones=["a"], span=SampleSpan(start=0, end=samples), phonetic_context={"vowel_span": {"start": 0, "end": samples}})


def _utterance(samples):
    span = SampleSpan(start=0, end=samples)
    return Utterance(source_id="source", speaker_id="speaker", span=span, vad_span=span, corpus_offset_samples=0, language_id="ja")


def test_silence_never_has_sustain_region():
    fs = 44100
    audio = np.zeros(fs // 2, dtype="float32")
    config = Config()
    marks, _ = _longest_sustain(_candidate(len(audio)), _features(audio, fs, config), _utterance(len(audio)), config)
    assert marks is None
