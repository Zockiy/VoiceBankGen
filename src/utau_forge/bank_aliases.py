"""Canonical selection identities stay stable when oto notation changes."""
from __future__ import annotations

import re
from functools import lru_cache

from .languages import language, profile

BREATH_ALIAS = "br"
DEFAULT_ALIAS_FORMAT = "hiragana"
ALIAS_FORMATS = ("hiragana", "romaji", "katakana")


def to_katakana(text: str) -> str:
    return "".join(chr(ord(char)+0x60) if "\u3041" <= char <= "\u3096" else char for char in text)


def split_variant(alias: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+)(_([1-9][0-9]*))", alias)
    return (match[1], match[2]) if match else (alias, "")


@lru_cache(maxsize=16)
def units_by_alias(profile_id: str) -> dict:
    export = profile(profile_id)
    return {export.aliases(unit)[0]: unit for unit in export.required_units(language(export.language_id).inventory, "core")}


def roman_alias(export, unit) -> str:
    if unit.unit_type in {"VC", "VV"}:
        return unit.unit_key.replace("+", " ")
    return export.filename_stem(unit)


def export_aliases(profile_id: str, selection_alias: str, alias_format: str = DEFAULT_ALIAS_FORMAT, custom_units=()) -> list[str]:
    base, suffix = split_variant(selection_alias)
    custom = next((unit for unit in custom_units if unit["alias"] == base), None)
    if custom:
        return [custom["label"] + suffix]
    if base == BREATH_ALIAS:
        return [base + suffix]
    export = profile(profile_id)
    unit = units_by_alias(profile_id).get(base)
    if unit is None:
        raise ValueError("选音不属于当前音源格式")
    if alias_format not in ALIAS_FORMATS:
        raise ValueError("未知 oto 发音格式")
    aliases = export.aliases(unit) if export.language_id=="zh" else ([roman_alias(export, unit)] if alias_format == "romaji" else export.aliases(unit))
    if alias_format == "katakana" and export.language_id=="ja":
        aliases = [to_katakana(alias) for alias in aliases]
    return [alias + suffix for alias in aliases]


def filename_stem(profile_id: str, selection_alias: str, custom_units=()) -> str:
    base, suffix = split_variant(selection_alias)
    custom = next((unit for unit in custom_units if unit["alias"] == base), None)
    if custom:
        return custom["filename_stem"] + suffix
    if base == BREATH_ALIAS:
        return base + suffix
    return profile(profile_id).filename_stem(units_by_alias(profile_id)[base]) + suffix
