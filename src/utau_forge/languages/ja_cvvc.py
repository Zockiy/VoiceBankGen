"""Real adjacent-phone transitions for OpenUtau's JA CVVC phonemizer."""
# OpenUtau compatibility reference: Copyright (c) 2014 StAkira, MIT.
# See LICENSES/OpenUtau-MIT.txt and THIRD_PARTY_NOTICES.md.
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from .ja import JapaneseCVProfile
from ..models import Candidate, Landmarks, Phone, SampleSpan, TargetUnit
from ..vc_timing import crossfade_marks

VOWELS = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}


class JapaneseCVVCProfile(JapaneseCVProfile):
    profile_id = "ja-cvvc"
    version = "1"

    def required_units(self, inventory, scope):
        body = super().required_units(inventory, scope)
        onsets = sorted({seq[0] for seq in inventory["units"].values() if len(seq) == 2})
        return body + [TargetUnit(target_language="ja", inventory_version=inventory["version"], unit_type="VC", unit_key=v+"+"+c) for v in (*VOWELS, "n") for c in onsets] + [TargetUnit(target_language="ja", inventory_version=inventory["version"], unit_type="VV", unit_key=v+"+"+w) for v in (*VOWELS, "n") for w in VOWELS]

    def aliases(self, unit):
        if unit.unit_type == "VC":
            return [unit.unit_key.replace("+", " ")]
        if unit.unit_type == "VV":
            left, right = unit.unit_key.split("+")
            return [left + " " + VOWELS[right]]
        return super().aliases(unit)

    def filename_stem(self, unit):
        if unit.unit_type in {"VC", "VV"}:
            return unit.unit_type.lower() + "_" + unit.unit_key.replace("+", "_")
        return super().filename_stem(unit)


def transition_candidates(phones: list[Phone], utterances: dict, inventory: dict) -> list[Candidate]:
    """Never span silence, unknown phones, utterances or sources."""
    onsets = {seq[0] for seq in inventory["units"].values() if len(seq) == 2}
    result = []
    ordered = sorted(phones, key=lambda p: (p.source_id, p.utterance_id, p.span.start))
    for left, right in zip(ordered, ordered[1:]):
        if left.unknown or right.unknown or left.source_id != right.source_id or left.utterance_id != right.utterance_id or abs(right.span.start-left.span.end) > 1:
            continue
        v = left.phoneme_id.rsplit(":", 1)[-1] if left.phoneme_id else ""
        c = right.phoneme_id.rsplit(":", 1)[-1] if right.phoneme_id else ""
        v = "n" if v == "N" else v
        if v not in {*VOWELS, "n"} or c not in onsets | set(VOWELS):
            continue
        role = "VV" if c in VOWELS else "VC"
        original_onset=c
        # OpenUtau's primary table explicitly uses ky for き and hy for ひ.
        # Its separate fallback table can simplify missing VC aliases, but the
        # profile emits the complete primary identities whenever available.
        # Preserve the actual transition and short preceding vowel tail. VC
        # receives a boundary score, never a long-vowel sustain requirement.
        start = max(left.span.start, left.span.end - 4410)
        end = right.span.end
        if end - start < 882:
            continue
        boundary = right.span.start
        fixed = min(end-1, boundary + 882) if role == "VV" else end-1
        marks = Landmarks(start=start, overlap=start, preutter=boundary, fixed_end=fixed, end=end)
        if role == "VC":
            marks = crossfade_marks(marks)
        unit = TargetUnit(target_language="ja", inventory_version=inventory["version"], unit_type=role, unit_key=v+"+"+c)
        identifier = str(uuid5(NAMESPACE_URL, ":".join((left.source_id, left.alignment_revision, role, str(start), str(end), unit.unit_key))))
        utt = utterances[left.utterance_id]
        result.append(Candidate(candidate_id=identifier, speaker_id=utt.speaker_id, source_id=left.source_id, utterance_id=left.utterance_id, alignment_revision=left.alignment_revision, source_language="ja", unit_role=role, unit_key=unit.unit_key, alias=JapaneseCVVCProfile().aliases(unit)[0], phones=[v,original_onset], raw_phones=[left.raw_model_label,right.raw_model_label], span=SampleSpan(start=start,end=end), landmarks=marks, automatic_landmarks=marks, quality_score=min(1,(left.span.end-start)/4410)*.5+.5, phonetic_context={"phone_ids":[left.phone_id,right.phone_id],"transition_boundary":boundary,"profile_onset":c,"ranking_reason":"相邻真实音素；按过渡边界及前元音尾长排序"}))
    return result
