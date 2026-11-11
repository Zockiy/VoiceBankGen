import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from utau_forge.onnx_whisper import MODEL_FILES, OnnxWhisper, log_mel
from utau_forge.model_settings import apply_engine_model_paths, validate_model_paths
from utau_forge.config import Config


def model_directory(path):
    for name in MODEL_FILES:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}', encoding='utf-8')
    return path


def test_onnx_model_path_sets_backend_without_copying_weights(tmp_path):
    directory = model_directory(tmp_path / 'model')
    assert validate_model_paths({'asr': str(directory)})['asr'] == str(directory)
    config, lock = Config(), {'models': []}
    apply_engine_model_paths(config, lock, {'asr': str(directory)})
    assert config.asr.backend == 'onnx_whisper'
    assert lock['models'][0]['backend'] == 'onnx_whisper'
    assert lock['models'][0]['local_absolute_path'] == str(directory)
    (directory / MODEL_FILES[-1]).rename(directory / 'missing.weights')
    with pytest.raises(ValueError, match='decoder.weights'):
        validate_model_paths({'asr': str(directory)})


def test_explicit_directml_does_not_silently_run_on_cpu(tmp_path, monkeypatch):
    directory = model_directory(tmp_path / 'model')
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider']))
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: None)))
    with pytest.raises(RuntimeError, match='DirectML 环境'):
        OnnxWhisper(directory, 'directml')


def test_auto_retries_only_current_utterance_on_cpu():
    model = object.__new__(OnnxWhisper)
    model.requested, model.device = 'auto', 'directml'
    attempts, events = [], []
    def infer(*args):
        attempts.append(model.device)
        if model.device == 'directml':
            raise RuntimeError('GPU operator failed')
        return 'テスト', 0.
    model._transcribe = infer
    model._sessions = lambda device: setattr(model, 'device', device)
    model.emit = lambda **event: events.append(event)
    assert model.transcribe(np.zeros(160), 'ja', 16000) == ('テスト', 0.)
    assert attempts == ['directml', 'cpu']
    assert events[0]['device'] == 'cpu'
    model.device, model.requested = 'directml', 'directml'
    with pytest.raises(RuntimeError, match='GPU operator'):
        model.transcribe(np.zeros(160), 'ja', 16000)


def test_mel_shape_and_silence_are_finite():
    features = log_mel(np.zeros(1600, dtype=np.float32), 16000)
    assert features.shape == (1, 80, 3000)
    assert np.isfinite(features).all()
    assert np.allclose(features, -1.5)
    with pytest.raises(ValueError, match='单声道'):
        log_mel(np.array([np.nan]), 16000)


def test_beam_search_finishes_without_following_repetitive_live_beams():
    """Two valid endings must beat an unbounded, high-probability continuation."""
    model = object.__new__(OnnxWhisper)
    model.config = {'decoder_start_token_id': 10, 'eos_token_id': 9,
                    'decoder_attention_heads': 1, 'd_model': 1,
                    'max_target_positions': 32}
    model.generation = {'suppress_tokens': [0, 10, 11, 12, 13]}
    model.tokenizer = SimpleNamespace(
        token_to_id=lambda token: {'<|ja|>': 11, '<|transcribe|>': 12,
                                   '<|notimestamps|>': 13}.get(token),
        decode=lambda tokens, **_: ''.join(str(token) for token in tokens))
    inputs = [SimpleNamespace(name='input_ids'), SimpleNamespace(name='encoder_hidden_states')]
    calls = []

    def infer(_, feeds):
        calls.append(feeds['input_ids'].copy())
        assert len(calls) <= 2, 'Continued decoding after enough completed beams'
        ids = feeds['input_ids']
        logits = np.full((*ids.shape, 20), -100., dtype=np.float32)
        if len(calls) == 1:
            logits[0, -1, [1, 2]] = np.log([.8, .2])
        else:
            for row, token in enumerate(ids[:, -1]):
                logits[row, -1, [3, 9]] = np.log([.6, .4] if token == 1 else [.01, .99])
        return [logits]

    session = SimpleNamespace(get_inputs=lambda: inputs,
                              get_outputs=lambda: [SimpleNamespace(name='logits')], run=infer)
    model.decoder = model.decoder_next = session
    text, _, _ = model._decode(np.zeros((1, 2, 1), dtype=np.float32), 'ja', 2)
    assert text == '1'
    assert len(calls) == 2


def test_repetitive_and_low_confidence_text_are_retried_without_reencoding():
    model = object.__new__(OnnxWhisper)
    model.device = 'directml'
    model.emit = lambda **_: None
    attempts = []
    results = [('へ' * 100, .1, -.1), ('わからない', .1, -1.5), ('ここまで手が回っているとは', .1, -.2)]
    hidden = object()

    def decode(features, language, beam_size, temperature):
        assert features is hidden
        attempts.append(temperature)
        return results[len(attempts) - 1]

    model._decode = decode
    assert model._decode_with_fallback(hidden, 'ja', 5) == (results[-1][0], .1)
    assert attempts == [0., .2, .4]


def test_low_confidence_nonspeech_does_not_publish_hallucinated_text():
    model = object.__new__(OnnxWhisper)
    model._decode = lambda *args: ('見てくれてありがとう', .9, -2.)
    assert model._decode_with_fallback(None, 'ja', 5) == ('', .9)
