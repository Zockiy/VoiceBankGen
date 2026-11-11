"""Offline Whisper encoder/cached decoder inference through ONNX Runtime.

The model stays in the software resource directory. DirectML sessions use the
CPU EP for unsupported operators; an explicit GPU request never becomes a
CPU-only session silently. No CUDA-specific attention/export operators needed.
"""
from __future__ import annotations

import json
from pathlib import Path


class DecodeLimitError(RuntimeError):
    """A decode attempt exhausted the text context without any ending."""


MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "voicebankgen-onnx.json",
               "onnx/encoder_model_fp16.onnx", "onnx/decoder_model.onnx",
               "onnx/decoder_with_past_model.onnx", "onnx/decoder.weights")


def is_onnx_model(path: Path) -> bool:
    return (path / "onnx/decoder_model.onnx").is_file()


def missing_files(path: Path) -> list[str]:
    return [name for name in MODEL_FILES if not (path / name).is_file()]


def log_mel(audio, sample_rate=44100):
    import math
    import numpy as np
    from scipy.signal import resample_poly
    from librosa.filters import mel

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Whisper 输入必须是有限数值的非空单声道音频")
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
    if len(audio) > 480000:
        raise ValueError("单段 ONNX Whisper 输入不能超过 30 秒")
    audio = np.pad(audio, (0, 480000-len(audio)))
    audio = np.pad(audio, (200, 200), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160][:-1]
    window = np.hanning(401)[:-1].astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=-1)).astype(np.float32) ** 2
    spectrum = mel(sr=16000, n_fft=400, n_mels=80) @ power.T
    spectrum = np.log10(np.maximum(spectrum, 1e-10))
    return ((np.maximum(spectrum, spectrum.max()-8) + 4) / 4)[None].astype(np.float32)


class OnnxWhisper:
    def __init__(self, directory: Path, device: str, emit=lambda **_: None):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from .devices import normalize_device

        directory = Path(directory)
        missing = missing_files(directory)
        if missing:
            raise ValueError("ONNX Whisper 模型缺少：" + ", ".join(missing))
        self.emit = emit
        self.requested = normalize_device(device)
        self.directory = directory
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        marker = directory / "voicebankgen-onnx.json"
        self.precision = json.loads(marker.read_text(encoding="utf-8")).get("compute_type", "fp16") if marker.is_file() else "fp16"
        self.generation = json.loads((directory / "generation_config.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.ort = ort
        if self.requested == "directml" and "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Whisper 缺少 ONNX Runtime DirectML 环境，请安装 onnxruntime-directml 或选择 CPU")
        preferred = "directml" if self.requested != "cpu" and "DmlExecutionProvider" in ort.get_available_providers() else "cpu"
        try:
            self._sessions(preferred)
        except Exception as error:
            if self.requested != "auto" or preferred == "cpu":
                raise RuntimeError(f"Whisper {preferred} 模型加载失败：{error}") from error
            self.emit(stage="Whisper 转写", current=0, total=0, device="cpu", reason=f"DirectML 加载失败，回退 CPU：{error}")
            self._sessions("cpu")

    def _sessions(self, device):
        options = self.ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        # Extended FP16 transformer fusions in ORT 1.24 can produce invalid
        # LayerNorm cast references on CPU. Keep portable ONNX operators.
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        providers = ([('DmlExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
                     if device == "directml" else ['CPUExecutionProvider'])
        self.encoder = self.ort.InferenceSession(str(self.directory / "onnx/encoder_model_fp16.onnx"), sess_options=options, providers=providers)
        self.decoder = self.ort.InferenceSession(str(self.directory / "onnx/decoder_model.onnx"), sess_options=options, providers=providers)
        self.decoder_next = self.ort.InferenceSession(str(self.directory / "onnx/decoder_with_past_model.onnx"), sess_options=options, providers=providers)
        if device == "directml" and any("DmlExecutionProvider" not in s.get_providers() for s in (self.encoder, self.decoder, self.decoder_next)):
            raise RuntimeError("DirectML 会话创建后未启用 DmlExecutionProvider")
        self.device = device
        self.emit(stage="Whisper 转写", current=0, total=0, device=device,
                  precision=self.precision, reason="ONNX Runtime；不支持的算子由 CPU 执行")

    def transcribe(self, audio, language, sample_rate=44100, beam_size=5):
        if language not in {"ja", "zh"}:
            raise ValueError("ONNX Whisper 当前支持日语和普通话来源")
        try:
            return self._transcribe(audio, language, sample_rate, beam_size)
        except Exception as error:
            if self.requested != "auto" or self.device != "directml":
                raise
            self.emit(stage="Whisper 转写", current=0, total=0, device="cpu", reason=f"DirectML 推理失败，重试当前语句：{error}")
            self._sessions("cpu")
            return self._transcribe(audio, language, sample_rate, beam_size)

    def _transcribe(self, audio, language, sample_rate, beam_size):
        import numpy as np
        if not 1 <= beam_size <= 5:
            raise ValueError("Whisper beam_size 须在 1 到 5 之间")
        step = 30 * sample_rate
        texts, probabilities = [], []
        for start in range(0, len(audio), step):
            features = log_mel(audio[start:start+step], sample_rate)
            dtype = np.float16 if self.encoder.get_inputs()[0].type == "tensor(float16)" else np.float32
            hidden = self.encoder.run(None, {"input_features": features.astype(dtype)})[0]
            if not np.isfinite(hidden).all():
                raise RuntimeError("Whisper 编码器产生非有限数值")
            text, probability = self._decode_with_fallback(hidden, language, beam_size)
            texts.append(text)
            probabilities.append(probability)
        return "".join(texts).strip(), max(probabilities, default=0.)

    def _decode_with_fallback(self, hidden, language, beam_size):
        import zlib

        attempts, nonrepetitive = [], []
        # Match the previous faster-whisper defaults: beam search first, then
        # best-of sampling when repetition or low confidence requires a retry.
        for temperature in (0., .2, .4, .6, .8, 1.):
            try:
                text, no_speech, average = self._decode(hidden, language, beam_size, temperature)
            except DecodeLimitError:
                self.emit(stage="Whisper 重试", current=0, total=0, device=self.device,
                          reason=f"解码未结束，调整采样温度（当前 {temperature:g}）")
                continue
            payload = text.encode("utf-8")
            repetitive = len(payload) / len(zlib.compress(payload)) > 2.4
            result = (text, no_speech, average)
            attempts.append(result)
            if no_speech > .6 and average < -1.:
                return "", no_speech
            if not repetitive:
                nonrepetitive.append(result)
                if average >= -1.:
                    return text, no_speech
            self.emit(stage="Whisper 重试", current=0, total=0, device=self.device,
                      reason=f"{'文本重复' if repetitive else '识别置信度低'}，调整采样温度（当前 {temperature:g}）")
        if not attempts:
            raise DecodeLimitError("Whisper 多次解码均未结束；请缩短语句或校对文本")
        best = max(nonrepetitive or attempts, key=lambda result: result[2])
        self.emit(stage="Whisper 需校对", current=0, total=0, device=self.device,
                  reason="多次尝试后使用最佳识别结果，请校对该句文本")
        return best[0], best[1]

    def _decode(self, hidden, language, beam_size, temperature=0.):
        import numpy as np
        prompt = [self.config["decoder_start_token_id"], self.tokenizer.token_to_id(f"<|{language}|>"),
                  self.tokenizer.token_to_id("<|transcribe|>"), self.tokenizer.token_to_id("<|notimestamps|>")]
        if any(v is None for v in prompt):
            raise ValueError("Whisper tokenizer 缺少语言或任务标记")
        eos = self.config["eos_token_id"]
        suppressed = list(self.generation.get("suppress_tokens", []))
        timestamp = self.tokenizer.token_to_id("<|notimestamps|>") + 1
        heads = self.config["decoder_attention_heads"]
        dim = self.config["d_model"] // heads
        feeds = {"encoder_hidden_states": hidden, "use_cache_branch": np.array([False])}
        for item in self.decoder_next.get_inputs():
            if item.name.startswith("past_key_values."):
                dtype = np.float16 if item.type == "tensor(float16)" else np.float32
                feeds[item.name] = np.zeros((1, heads, 1, dim), dtype=dtype)
        histories, scores, finished = [[]], np.zeros(1, dtype=np.float32), {}
        ids = np.array([prompt], dtype=np.int64)
        # A fixed local seed makes a failed phrase reproducible for review.
        random = np.random.default_rng(0)
        no_speech = 0.
        max_tokens = min(448, self.config.get("max_target_positions", 448)) - len(prompt)
        for index in range(max_tokens):
            feeds["input_ids"] = ids
            session = self.decoder if index == 0 else self.decoder_next
            inputs = {v.name: feeds[v.name] for v in session.get_inputs()}
            outputs = dict(zip([v.name for v in session.get_outputs()], session.run(None, inputs)))
            logits = outputs["logits"][:, -1, :].astype(np.float32)
            if np.isnan(logits).any() or np.isposinf(logits).any() or not np.isfinite(logits).any(axis=1).all():
                raise RuntimeError(f"Whisper 解码产生非有限数值（步骤 {index}）")
            if index == 0:
                first = outputs["logits"][0, 0].astype(np.float32)
                prob = np.exp(first - first.max())
                token = self.tokenizer.token_to_id("<|nocaptions|>")
                if token is None:
                    token = self.tokenizer.token_to_id("<|nospeech|>")
                no_speech = float(prob[token]/prob.sum()) if token is not None else 0.
            logits[:, suppressed] = -np.inf
            logits[:, timestamp:] = -np.inf
            if index == 0:
                logits[:, self.generation.get("begin_suppress_tokens", [])] = -np.inf
            maxima = logits.max(axis=1, keepdims=True)
            log_probs = logits - maxima - np.log(np.exp(logits-maxima).sum(axis=1, keepdims=True))
            parents, tokens, next_histories, next_scores = [], [], [], []
            if temperature:
                scaled = logits / temperature
                distribution = np.exp(scaled - scaled.max(axis=1, keepdims=True)).astype(np.float64)
                distribution /= distribution.sum(axis=1, keepdims=True)
                sources = [0] * beam_size if index == 0 else range(len(histories))
                choices = [(parent, int(random.choice(logits.shape[1], p=distribution[parent])))
                           for parent in sources]
            else:
                candidates = (log_probs + scores[:, None]).reshape(-1)
                count = min(2*beam_size, candidates.size)
                best = np.argpartition(candidates, -count)[-count:]
                best = best[np.argsort(candidates[best])[::-1]]
                choices = [divmod(int(value), logits.shape[1]) for value in best]
            for parent, token in choices:
                score = float(scores[parent] + log_probs[parent, token])
                if token == eos:
                    if len(finished) < beam_size:
                        finished[tuple(histories[parent])] = score
                elif token != eos and len(parents) < beam_size:
                    parents.append(parent)
                    tokens.append(token)
                    next_histories.append([*histories[parent], token])
                    next_scores.append(score)
                    if len(parents) == beam_size:
                        break
            # Whisper's default patience is one: stop after beam_size completed
            # sequences. An average-score bound on the live beams can keep
            # searching hundreds of tokens and favor repetitive continuations.
            if (not temperature and len(finished) >= beam_size) or not parents:
                break
            for name in list(feeds):
                if not name.startswith("past_key_values."):
                    continue
                output = outputs.get(name.replace("past_key_values.", "present.", 1))
                value = output if output is not None and output.size else feeds[name]
                feeds[name] = value[parents]
            feeds["encoder_hidden_states"] = feeds["encoder_hidden_states"][parents]
            feeds["use_cache_branch"] = np.array([True])
            ids = np.array(tokens, dtype=np.int64)[:, None]
            histories, scores = next_histories, np.array(next_scores, dtype=np.float32)
        if not finished:
            raise DecodeLimitError("Whisper 达到解码上限仍未结束；请缩短语句或核对模型")
        best = max(finished, key=lambda sequence: finished[sequence] / max(1, len(sequence)))
        return (self.tokenizer.decode(list(best), skip_special_tokens=True).strip(),
                no_speech, finished[best] / (len(best) + 1))
