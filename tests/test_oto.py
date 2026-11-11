from decimal import Decimal
from itertools import permutations

import pytest

from utau_forge.models import Landmarks
from utau_forge.oto import compile_oto, encode_oto, parse_oto


def test_manual_regression_round_trip():
    marks = Landmarks(start=882, overlap=1323, preutter=4410, fixed_end=6174, end=19845)
    line = compile_oto("ka.wav", "か", marks, 22050)
    assert line == "ka.wav=か,20,120,-430,80,10"
    parsed = parse_oto(line)
    assert parsed.absolute_ms(Decimal(500))["end"] == Decimal(450)
    assert parsed.absolute_ms(Decimal(500))["fixed_end"] == Decimal(140)


def test_positive_cutoff_is_from_file_end():
    parsed = parse_oto("a.wav=あ,20,120,50,80,10")
    assert parsed.absolute_ms(Decimal(500))["end"] == Decimal(450)


def test_cp932_rejects_unencodable_alias():
    with pytest.raises(UnicodeEncodeError):
        encode_oto(["x.wav=😀,0,10,-100,0,0"], "cp932")


def test_cv_bounds_reject_unsupported_style():
    marks = Landmarks(start=0, overlap=100, preutter=50, fixed_end=200, end=300)
    with pytest.raises(ValueError, match="UNSUPPORTED_TIMING_STYLE"):
        marks.validate_cv(400)


@pytest.mark.parametrize("anchors",list(permutations((100,200,300))))
def test_manual_oto_anchor_order_is_independent(anchors):
    marks=Landmarks(start=50,overlap=anchors[0],preutter=anchors[1],fixed_end=anchors[2],end=400)
    line=compile_oto("x.wav","あ",marks,500,sample_rate=1000)
    absolute=parse_oto(line).absolute_ms(Decimal(500))
    assert {key:int(value) for key,value in absolute.items()}==marks.model_dump()


@pytest.mark.parametrize("changes",[{"overlap":49},{"preutter":401},{"fixed_end":450},{"start":400},{"end":501}])
def test_manual_oto_still_rejects_outside_usage(changes):
    marks=Landmarks(start=50,overlap=100,preutter=200,fixed_end=300,end=400).model_copy(update=changes)
    with pytest.raises(ValueError,match="使用区间"):
        compile_oto("x.wav","あ",marks,500)


def test_manual_oto_can_place_all_anchors_on_endpoints():
    marks=Landmarks(start=50,overlap=400,preutter=50,fixed_end=400,end=400)
    assert compile_oto("x.wav","あ",marks,500,sample_rate=1000)=="x.wav=あ,50,350,-350,0,350"
