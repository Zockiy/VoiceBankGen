"""Extract only explicitly labelled breaths, never infer breaths from pauses.

TIFA configs/g2p.yaml merges AP with br; EP is deliberately not classified.
Raw labels also support projects aligned by older adapters that stored AP as sp.
"""
from uuid import NAMESPACE_URL, uuid5

from .models import Candidate, Landmarks


def is_breath(phone) -> bool:
    return phone.features.get("backend") == "tifa" and phone.raw_model_label.rsplit("/",1)[-1] in {"AP", "br"}


def breath_candidates(phones, utterances) -> list[Candidate]:
    result = []
    for phone in phones:
        if not is_breath(phone) or phone.utterance_id not in utterances:
            continue
        utt = utterances[phone.utterance_id]
        # Preserve the complete event as a fixed region, with no vowel looping.
        marks = Landmarks(start=phone.span.start, overlap=phone.span.start,
                          preutter=phone.span.start, fixed_end=phone.span.end-1, end=phone.span.end)
        result.append(Candidate(
            candidate_id=str(uuid5(NAMESPACE_URL, "breath:" + phone.phone_id)),
            speaker_id=utt.speaker_id, source_id=phone.source_id, utterance_id=phone.utterance_id,
            alignment_revision=phone.alignment_revision, source_language=phone.language_id,
            unit_role="breath", unit_key="br", alias="br", phones=["br"], raw_phones=[phone.raw_model_label],
            span=phone.span, landmarks=marks, automatic_landmarks=marks,
            quality_score=min(1, (phone.span.end-phone.span.start)/44100),
            phonetic_context={"phone_ids": [phone.phone_id], "ranking_reason": "TIFA 明确标记的呼吸音；请试听确认边界，保留原音高与音量"},
        ))
    return result
