"""Manual assembly recipes over immutable TIFA candidate preview audio."""
from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5, uuid4
import copy
import math

import numpy as np
import soundfile as sf

from .desktop_service import preview_file, safe_path
from .languages import language, profile
from .models import Candidate, Landmarks, SampleSpan
from .storage import json_hash, write_json
from .bank_aliases import split_variant, filename_stem

RATE = 44100
MAX_SECONDS = 120


def clip_from_candidate(store, identifier, bank_id, part="whole"):
    candidate = store.candidate(identifier, bank_id)
    result = preview_file(store, identifier)
    info = sf.info(result["path"])
    origin = result["start"]
    marks = candidate.landmarks
    start = marks.start if marks else candidate.span.start
    end = marks.end if marks else candidate.span.end
    vowel = candidate.phonetic_context.get("vowel_span")
    if part != "whole":
        if not vowel:
            raise ValueError("这个候选没有 TIFA 元音分界；请添加整段后手动裁剪")
        boundary = vowel["start"]
        if part == "consonant":
            end = min(end, boundary)
        elif part == "vowel":
            start = max(start, boundary)
        else:
            raise ValueError("未知片段类型")
    left, right = max(0, start-origin), min(info.frames, end-origin)
    if right <= left:
        raise ValueError("所选部分没有可用音频；可添加整段后手动裁剪")
    return {"candidate_id": identifier, "label": candidate.alias,
            "part": {"whole":"整音", "consonant":"辅音", "vowel":"元音"}[part],
            "path": Path(result["path"]).resolve().relative_to(store.directory).as_posix(),
            "source_start": left, "source_end": right, "position": 0,
            "track": 0, "fade_in": 0, "fade_out": 0}


def validate_clips(store, clips):
    if not 1 <= len(clips) <= 64:
        raise ValueError("时间线需要 1–64 个片段")
    checked = []
    for original in clips:
        clip = dict(original)
        gain = clip.get("gain_db",0)
        if not isinstance(gain,(int,float)) or not math.isfinite(gain) or not -24<=gain<=24:
            raise ValueError("片段音量必须在 -24 至 24 dB 之间")
        candidate = store.candidate(clip["candidate_id"])
        expected = preview_file(store, candidate.candidate_id)
        path = safe_path(store.directory, clip["path"])
        if path != Path(expected["path"]).resolve():
            raise ValueError("片段音频与候选不匹配")
        info = sf.info(path)
        if info.samplerate != RATE or info.channels != 1:
            raise ValueError("拼接要求 44.1kHz 单声道")
        for field in ("source_start", "source_end", "position", "track", "fade_in", "fade_out"):
            if type(clip[field]) is not int:
                raise ValueError("片段坐标必须是整数采样点")
        if not 0 <= clip["source_start"] < clip["source_end"] <= info.frames:
            raise ValueError("裁剪范围超出候选 WAV")
        duration = clip["source_end"]-clip["source_start"]
        if not 0 <= clip["position"] < MAX_SECONDS*RATE or clip["position"]+duration > MAX_SECONDS*RATE:
            raise ValueError("时间线最多 120 秒，位置不能为负")
        if not 0 <= clip["track"] <= 3 or not 0 <= clip["fade_in"] <= duration or not 0 <= clip["fade_out"] <= duration:
            raise ValueError("轨道或淡化时长无效")
        inherited=clip.get("inherited_fades",[])
        if not isinstance(inherited,list) or len(inherited)>64:
            raise ValueError("切分淡化信息无效")
        for fade in inherited:
            if not isinstance(fade,dict) or any(type(fade.get(key)) is not int for key in ("source_start","source_end","fade_in","fade_out")):
                raise ValueError("切分淡化坐标必须是整数采样点")
            if not 0<=fade["source_start"]<fade["source_end"]<=info.frames or any(not 0<=fade[key]<=fade["source_end"]-fade["source_start"] for key in ("fade_in","fade_out")):
                raise ValueError("切分淡化超出素材范围")
        family=clip.get("split_family")
        if family is not None and (not isinstance(family,str) or not 1<=len(family)<=64):
            raise ValueError("切分片段关联无效")
        checked.append(clip)
    return checked


def timeline_fades(clips, strict=True):
    """Contiguous split siblings retain the original logical crossfade boundaries."""
    groups=[]
    families={}
    for index,clip in sorted(enumerate(clips),key=lambda pair:(pair[1]["track"],pair[1]["position"])):
        key=(clip.get("split_family"),clip["track"],clip["candidate_id"],clip["path"],clip["position"]-clip["source_start"])
        prior=families.get(key) if clip.get("split_family") else None
        end=clip["position"]+clip["source_end"]-clip["source_start"]
        if prior and prior["end"]==clip["position"]:
            prior["end"]=end
            prior["members"].append(index)
        else:
            prior={"start":clip["position"],"end":end,"track":clip["track"],"members":[index]}
            groups.append(prior)
        if clip.get("split_family"):
            families[key]=prior
    result={}
    for track in range(4):
        ordered=sorted((g for g in groups if g["track"]==track),key=lambda g:g["start"])
        for index,group in enumerate(ordered):
            start,end=group["start"],group["end"]
            fade_in=fade_out=0
            if index:
                previous_end=ordered[index-1]["end"]
                overlap=max(0,previous_end-start)
                if strict and (overlap>end-start or end<=previous_end):
                    raise ValueError("同轨片段不能完全包含；请移动片段或换到另一轨")
                fade_in=overlap
                if strict and index>1 and previous_end>start and ordered[index-2]["end"]>start:
                    raise ValueError("同一轨道不能有三个片段同时重叠")
            if index+1<len(ordered):
                fade_out=max(0,end-ordered[index+1]["start"])
            for member in group["members"]:
                result[member]=(start,end,fade_in,fade_out)
    return result


def _ramp(indices,start,end,fade_in,fade_out):
    incoming=np.ones(len(indices));outgoing=incoming.copy()
    if fade_in:
        incoming=np.clip((indices-start)/max(1,fade_in-1),0,1)
    if fade_out>1:
        outgoing=np.clip((end-1-indices)/(fade_out-1),0,1)
    return incoming,outgoing


def clip_envelope(clip,group,indices=None):
    indices=np.arange(clip["source_start"],clip["source_end"]) if indices is None else indices
    start,end,incoming,outgoing=group
    time=indices-clip["source_start"]+clip["position"]
    fade_in,fade_out=_ramp(time,start,end,incoming,outgoing)
    records=[clip,*clip.get("inherited_fades",[])]
    for record in records:
        left,right=_ramp(indices,record["source_start"],record["source_end"],record["fade_in"],record["fade_out"])
        fade_in=np.minimum(fade_in,left)
        fade_out=np.minimum(fade_out,right)
    return fade_in*fade_out


def split_clip(clip,position):
    duration=clip["source_end"]-clip["source_start"]
    if type(position) is not int or not clip["position"]<position<clip["position"]+duration:
        raise ValueError("时间光标需位于所选片段内部，不能在边缘或片段之外")
    cut=clip["source_start"]+position-clip["position"]
    family=clip.get("split_family") or uuid4().hex
    records=copy.deepcopy(clip.get("inherited_fades",[]))
    if clip["fade_in"] or clip["fade_out"]:
        records.append({key:clip[key] for key in ("source_start","source_end","fade_in","fade_out")})
    left,right=copy.deepcopy(clip),copy.deepcopy(clip)
    for part in (left,right):
        part.update(split_family=family,inherited_fades=copy.deepcopy(records),fade_in=0,fade_out=0)
    left["source_end"]=cut
    right["source_start"]=cut
    right["position"]=position
    return left,right


def render_assembly(store, clips):
    clips=validate_clips(store,clips)
    groups=timeline_fades(clips)
    length=max(c["position"]+c["source_end"]-c["source_start"] for c in clips)
    output=np.zeros(length,dtype="float64")
    for index,clip in enumerate(clips):
        audio,_=sf.read(safe_path(store.directory,clip["path"]),start=clip["source_start"],stop=clip["source_end"],dtype="float64")
        audio*=10**(clip.get("gain_db",0)/20)
        start=clip["position"]
        output[start:start+len(audio)]+=audio*clip_envelope(clip,groups[index])
    if not np.isfinite(output).all() or np.max(np.abs(output)) > 1:
        raise ValueError("多轨混音产生削波；请减少重叠或使用同轨交叉淡化")
    return output.astype("float32")


def candidate_recipe(store, bank_id, identifier):
    """Reopen a manual assembly or reconstruct the two parts of automatic repair."""
    candidate = store.candidate(identifier,bank_id)
    context = candidate.phonetic_context
    if context.get("manual_assembly"):
        return validate_clips(store, context["manual_assembly"]["clips"])
    if candidate.provenance_kind.value not in {"cv_spliced","vowel_extended"} or len(candidate.parent_candidate_ids)!=2:
        return []
    if not all(key in context for key in ("source_range","donor_range","join")):
        return []
    source_id,donor_id = candidate.parent_candidate_ids
    source = clip_from_candidate(store,source_id,bank_id)
    donor = clip_from_candidate(store,donor_id,bank_id)
    for clip,key in ((source,"source_range"),(donor,"donor_range")):
        preview=preview_file(store,clip["candidate_id"])
        clip["source_start"],clip["source_end"]=[n-preview["start"] for n in context[key]]
    donor["source_start"] += context["join"]["donor_shift_samples"]
    donor["gain_db"] = context["join"]["gain_db"]
    donor["position"] = source["source_end"]-source["source_start"]-context["join"]["crossfade_samples"]
    source["part"],donor["part"] = "起始部分","元音供体"
    return validate_clips(store,[source,donor])


def assembly_preview(store, clips):
    audio = render_assembly(store, clips)
    key = json_hash({"algorithm": "manual-splice-1", "clips": clips})
    path = store.directory/"analyses"/"assemblies"/(key+".wav")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, RATE, subtype="PCM_16")
    return path


def export_audio(store, clips, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("目标文件已存在，请选择新的文件名")
    audio = render_assembly(store, clips)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a race after the file dialog.
    with destination.open("xb") as stream:
        sf.write(stream, audio, RATE, format="WAV", subtype="PCM_16")
    return destination


def save_candidate(store, bank_id, alias, clips):
    clips = validate_clips(store, clips)
    export_profile = profile(store.bank(bank_id)["profile_id"])
    units = export_profile.required_units(language(export_profile.language_id).inventory, "core")
    if alias not in dict(store.targets(bank_id)):
        raise ValueError("请选择当前音源格式的目标音")
    alias=split_variant(alias)[0]
    unit = next((u for u in units if export_profile.aliases(u)[0] == alias), None)
    custom_units=store.custom_units(bank_id)
    custom=any(item["alias"]==alias for item in custom_units)
    if not unit and not custom and alias!="br":
        raise ValueError("请选择当前音源格式的目标音")
    audio = render_assembly(store, clips)
    parents = [store.candidate(c["candidate_id"]) for c in clips]
    if len({(p.speaker_id,p.source_language) for p in parents}) != 1:
        raise ValueError("拼接只能使用同一说话人、同一语言的候选")
    recipe = {"algorithm": "manual-splice-1", "alias": alias, "clips": clips}
    identifier = str(uuid5(NAMESPACE_URL, json_hash(recipe)))
    stem = filename_stem(store.bank(bank_id)["profile_id"],alias,custom_units)
    path = store.directory/"derived"/(stem+"_manual_"+identifier+".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        sf.write(path, audio, RATE, subtype="FLOAT")
    first = parents[0]
    # Initial marks are editable estimates, never advertised as automatic sustain.
    fixed = min(len(audio)-1, max(0, max(c["position"] for c in clips)))
    marks = Landmarks(start=0, overlap=0, preutter=0, fixed_end=fixed, end=len(audio))
    marks.validate_cv(len(audio))
    candidate = Candidate(speaker_id=first.speaker_id, source_id=first.source_id,
        utterance_id=first.utterance_id, alignment_revision=first.alignment_revision,
        source_language=first.source_language, target_language=export_profile.language_id, candidate_id=identifier,
        alias=alias, unit_key=unit.unit_key if unit else alias,
        unit_role=(unit.unit_type if unit.unit_type in {"VC","VV"} else "sustain_vowel") if unit else ("custom" if custom else "breath"),
        phones=unit.unit_key.split("+") if unit else ([] if custom else ["br"]), raw_phones=[phone for p in parents for phone in p.raw_phones],
        span=SampleSpan(start=0, end=len(audio)), landmarks=marks, automatic_landmarks=marks,
        provenance_kind="cv_spliced", parent_candidate_ids=list(dict.fromkeys(p.candidate_id for p in parents)),
        quality_flags=["REPAIR_REVIEW_REQUIRED"],
        phonetic_context={"derived_audio_path": path.relative_to(store.directory).as_posix(), "manual_assembly": recipe, "audio_samples": len(audio),
                          "vowel_span": {"start":fixed,"end":len(audio)}})
    if first.source_language!=export_profile.language_id:
        candidate.quality_flags.append("CROSS_LANGUAGE_APPROXIMATE")
        candidate.phonetic_context["cross_language"]={"direction":first.source_language+"->"+export_profile.language_id,"grade":"approximate","losses":["手动跨语种配方，音素等价性未经自动验证"],"donors":[{"candidate_id":c["candidate_id"],"preview_path":c["path"],"source_start":c["source_start"],"source_end":c["source_end"]} for c in clips]}
    with store.connection() as connection:
        if not connection.execute("SELECT 1 FROM candidates WHERE id=?", (identifier,)).fetchone():
            connection.execute("INSERT INTO candidates VALUES(?,?,?,?)", (identifier, alias, store.unit_type(candidate), candidate.model_dump_json()))
            store._bump(connection, bank_id)
    write_json(path.with_suffix(".json"), recipe)
    return store.candidate(identifier, bank_id)
