import json
from unittest.mock import patch

import pytest

from utau_forge.bank_aliases import export_aliases, to_katakana
from utau_forge.desktop_store import ProjectStore, UNDO_LIMIT
from utau_forge.models import Candidate, Landmarks, SampleSpan


@pytest.fixture
def editable(tmp_path,monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR",str(tmp_path/"userdata"))
    store=ProjectStore.create(tmp_path/"project","History")
    bank=store.banks()[0]
    marks=Landmarks(start=0,overlap=100,preutter=500,fixed_end=2000,end=10000)
    candidate=Candidate(candidate_id="editable",speaker_id=store.meta("speaker_id"),source_id="source",utterance_id="utt",alignment_revision="r",source_language="ja",unit_role="sustain_vowel",unit_key="a",alias="あ",phones=["a"],raw_phones=[],span=SampleSpan(start=0,end=10000),landmarks=marks,automatic_landmarks=marks,phonetic_context={"audio_samples":44100})
    store.ingest("candidates",[candidate])
    return store,bank,candidate


def test_new_banks_default_to_hiragana_and_aliases_are_reversible(editable):
    store,bank,candidate=editable
    assert bank["preset"]["alias_format"]=="hiragana"
    assert store.display_alias(bank["id"],"あ_2")=="あ_2"
    assert export_aliases("ja-cv","し_2")==["し_2"]
    assert export_aliases("ja-cv","し_2","katakana")==["シ_2"]
    assert export_aliases("ja-cv","きゃ")==["きゃ"]
    assert export_aliases("ja-cv","きゃ","katakana")==["キャ"]
    assert export_aliases("ja-cvvc","a い_2")==["a い_2"]
    assert export_aliases("ja-cvvc","a い_2","katakana")==["a イ_2"]
    assert export_aliases("ja-cvvc","a k")==["a k"]
    assert export_aliases("ja-cv","し","romaji")==["shi"]
    assert export_aliases("ja-cv","し","hiragana")==["し"]
    assert to_katakana("あいうえお が っ br_1")=="アイウエオ ガ ッ br_1"
    with pytest.raises(ValueError,match="重复"):
        store.create_custom_unit(bank["id"],"ア")
    custom=store.create_custom_unit(bank["id"],"笑声","xiaosheng")
    assert export_aliases("ja-cv",custom["alias"],custom_units=[custom])==["笑声"]


def test_timing_history_has_eight_steps_and_survives_reopen(editable):
    store,bank,candidate=editable
    for number in range(1,11):
        marks=candidate.landmarks.model_copy(update={"preutter":500+number*10})
        store.edit_timing(bank["id"],candidate.candidate_id,marks,[])
    store=ProjectStore(store.directory)
    for number in range(9,1,-1):
        assert store.history(bank["id"])
        assert store.candidate(candidate.candidate_id,bank["id"]).landmarks.preutter==500+number*10
    assert not store.history(bank["id"])
    for number in range(3,11):
        assert store.history(bank["id"],True)
        assert store.candidate(candidate.candidate_id,bank["id"]).landmarks.preutter==500+number*10
    assert not store.history(bank["id"],True)
    with store.connection() as connection:
        assert connection.execute("SELECT count(*) FROM edit_events WHERE active=1 AND bank_id=?",(bank["id"],)).fetchone()[0]==UNDO_LIMIT


def test_noop_edit_does_not_consume_history_or_clear_redo(editable):
    store,bank,candidate=editable
    changed=candidate.landmarks.model_copy(update={"preutter":600})
    store.edit_timing(bank["id"],candidate.candidate_id,changed,[])
    assert store.history(bank["id"])
    revision=store.revision
    store.edit_timing(bank["id"],candidate.candidate_id,candidate.landmarks,[])
    store.review(bank["id"],candidate.candidate_id,candidate.review_status)
    assert store.revision==revision
    assert store.history(bank["id"],True)
    assert store.candidate(candidate.candidate_id,bank["id"]).landmarks==changed


def test_redo_is_scoped_to_bank_and_new_edit_branches_history(editable):
    store,bank,candidate=editable
    other=store.create_bank("ja-cv")
    unit=store.create_custom_unit(other["id"],"effect")
    assert store.history(other["id"])
    store.edit_timing(bank["id"],candidate.candidate_id,candidate.landmarks.model_copy(update={"preutter":600}),[])
    assert store.history(other["id"],True)
    assert store.custom_units(other["id"])==[unit]
    assert store.history(bank["id"])
    replacement=candidate.landmarks.model_copy(update={"preutter":700})
    store.edit_timing(bank["id"],candidate.candidate_id,replacement,[])
    assert not store.history(bank["id"],True)


def test_old_long_history_is_bounded_before_first_undo(editable):
    store,bank,candidate=editable
    with patch("utau_forge.desktop_store.UNDO_LIMIT",100):
        for number in range(1,11):
            store.edit_timing(bank["id"],candidate.candidate_id,candidate.landmarks.model_copy(update={"preutter":500+number*10}),[])
    assert sum(store.history(bank["id"]) for _ in range(12))==8
