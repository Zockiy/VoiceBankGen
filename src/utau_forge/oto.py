"""UTAU oto serialization with an independent reader for round-trip validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from .models import Landmarks


def _ms(samples: int, sample_rate: int) -> Decimal:
    return Decimal(samples) * Decimal(1000) / Decimal(sample_rate)


def _format_ms(value: Decimal) -> str:
    q = value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return format(q.normalize(), "f")


def compile_oto(filename: str, alias: str, marks: Landmarks, total_samples: int, sample_rate: int = 44100) -> str:
    marks.validate_oto(total_samples)
    if "," in filename or "=" in filename or "\n" in filename or "," in alias or "\n" in alias:
        raise ValueError("oto filename/alias contains a delimiter")
    return "=".join((filename, ",".join((
        alias,
        _format_ms(_ms(marks.start, sample_rate)),
        _format_ms(_ms(marks.fixed_end - marks.start, sample_rate)),
        _format_ms(-_ms(marks.end - marks.start, sample_rate)),
        _format_ms(_ms(marks.preutter - marks.start, sample_rate)),
        _format_ms(_ms(marks.overlap - marks.start, sample_rate)),
    ))))


@dataclass(frozen=True)
class ParsedOto:
    filename: str
    alias: str
    offset_ms: Decimal
    consonant_ms: Decimal
    cutoff_ms: Decimal
    preutter_ms: Decimal
    overlap_ms: Decimal

    def absolute_ms(self, wav_duration_ms: Decimal) -> dict[str, Decimal]:
        end = self.offset_ms - self.cutoff_ms if self.cutoff_ms < 0 else wav_duration_ms - self.cutoff_ms
        return {
            "start": self.offset_ms,
            "overlap": self.offset_ms + self.overlap_ms,
            "preutter": self.offset_ms + self.preutter_ms,
            "fixed_end": self.offset_ms + self.consonant_ms,
            "end": end,
        }


def parse_oto(line: str) -> ParsedOto:
    filename, separator, values = line.strip().partition("=")
    fields = values.split(",")
    if not separator or len(fields) != 6 or not filename:
        raise ValueError("invalid oto line")
    try:
        numbers = [Decimal(value) for value in fields[1:]]
    except Exception as exc:
        raise ValueError("invalid oto number") from exc
    if any(not n.is_finite() for n in numbers):
        raise ValueError("non-finite oto number")
    return ParsedOto(filename, fields[0], *numbers)


def encode_oto(lines: list[str], encoding: str) -> bytes:
    return ("\r\n".join(lines) + "\r\n").encode(encoding, errors="strict")
