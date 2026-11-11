from utau_forge.languages import language, profile, register_language
from utau_forge.models import Phone, SampleSpan


def test_inventory_and_nasal_are_separate():
    ja = language("ja")
    aliases = ja.inventory["units"]
    assert len(aliases) >= 100
    assert aliases["ん"] == ["N"]
    assert aliases["な"] == ["n", "a"]
    assert len(profile("ja-cv").required_units(ja.inventory, "core")) == len(aliases)


def test_japanese_cv_filenames_are_readable_and_unique():
    ja = language("ja")
    cv = profile("ja-cv")
    units = cv.required_units(ja.inventory, "core")
    stems = {cv.aliases(unit)[0]: cv.filename_stem(unit) for unit in units}
    assert len(set(stems.values())) == len(stems) == 101
    assert {alias: stems[alias] for alias in ("い", "し", "ち", "つ", "ふ", "じ", "きゃ", "しゃ", "ん")} == {"い": "i", "し": "shi", "ち": "chi", "つ": "tsu", "ふ": "fu", "じ": "ji", "きゃ": "kya", "しゃ": "sha", "ん": "n"}
    assert all(stem.isascii() and stem.isalpha() and stem.islower() for stem in stems.values())


def test_unknown_phone_is_not_coerced_to_silence():
    ja = language("ja")
    assert ja.normalize_phones(["not-a-locked-label"]) == [None]
    phone = Phone(utterance_id="u", source_id="s", language_id="ja", raw_model_label="?", span=SampleSpan(start=1, end=100), alignment_revision="r", unknown=True)
    assert ja.propose_units([phone], profile("ja-cv")) == []


def test_locked_mfa_ipa_map_preserves_palatal_and_devoiced_distinctions():
    ja = language("ja")
    assert ja.normalize_phones(["c", "ɨ", "ɕ", "i̥", "ɴ", "n"]) == ["ky", "u", "sh", "i", "N", "n"]
    assert ja.inventory["units"]["きゃ"] == ["ky", "a"]
    assert ja.inventory["units"]["しゅ"] == ["sh", "u"]
    assert ja.normalize_phones(["kː", "spn", "ɰ̃"]) == [None, None, "N"]


def test_unknown_geminate_onset_cannot_become_pure_vowel():
    ja = language("ja")
    phones = [
        Phone(utterance_id="u", source_id="s", language_id="ja", raw_model_label="kː", span=SampleSpan(start=0, end=100), alignment_revision="r", unknown=True),
        Phone(utterance_id="u", source_id="s", language_id="ja", raw_model_label="a", phoneme_id="ja@2:a", span=SampleSpan(start=100, end=300), alignment_revision="r"),
    ]
    assert ja.propose_units(phones, profile("ja-cv")) == []


def test_fake_language_registration_does_not_change_japanese_profile():
    class FakePack:
        language_id = "test-x"
        version = "1"

    register_language(FakePack())
    assert language("test-x").language_id == "test-x"
    assert profile("ja-cv").language_id == "ja"
    assert language("ja").inventory["language_id"] == "ja"
