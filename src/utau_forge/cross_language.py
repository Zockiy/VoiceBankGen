"""Directional Japanese/Mandarin recipes built from actual aligned phone spans."""
from __future__ import annotations
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5, uuid4
import numpy as np
import soundfile as sf
from .audio import read_region
from .languages import language, profile
from .models import Candidate, Phone, SourceAsset, SampleSpan, Landmarks
from .storage import json_hash, write_json

# target phone -> source phones, loss. Never equate ordinary pinyin b/d/g
# with Japanese voiced stops, or a complete Mandarin rime with a plain vowel.
JA_FROM_ZH={
 "a":(["a"],""),"i":(["i"],""),"o":(["o"],""),"u":(["u"],"圆唇元音近似日语不圆唇う"),
 "e":(["ei"],"中文 ei 的滑动尾音替代日语 e"),"N":(["n"],"日语拨音的发音部位与长度未还原"),
 "m":(["m"],""),"n":(["n"],""),"s":(["s"],""),"y":(["y"],""),"w":(["w"],""),
 "k":(["g"],"普通话不送气清塞音近似日语 k"),"t":(["d"],"普通话不送气清塞音近似日语 t"),"p":(["b"],"普通话不送气清塞音近似日语 p"),
 "g":(["g"],"缺少浊音，使用不送气清音"),"d":(["d"],"缺少浊音，使用不送气清音"),"b":(["b"],"缺少浊音，使用不送气清音"),
 "h":(["h"],"普通话 h 发音部位近似日语 h"),"f":(["f"],"普通话唇齿音替代日语双唇音"),
 "sh":(["x"],"普通话 x 发音部位近似日语 sh"),"ch":(["j"],"普通话 j 近似日语 ch"),"j":(["j"],"缺少日语浊塞擦音"),
 "z":(["z"],"普通话清塞擦音替代日语浊音"),"ts":(["z"],"普通话不送气塞擦音近似日语 ts"),"r":(["l"],"普通话边音替代日语闪音"),
}
# Palatal onsets require two real fragments; no renamed whole CV syllables.
for palatal,base in {"ky":"k","gy":"g","ny":"n","hy":"h","by":"b","py":"p","my":"m","ry":"r"}.items():
    seq,loss=JA_FROM_ZH[base]
    JA_FROM_ZH[palatal]=(seq+["i"],(loss+"；" if loss else "")+"用 i 片段近似拗音辅音过渡")
ZH_FROM_JA={
 "a":(["a"],""),"i":(["i"],""),"o":(["o"],""),"u":(["u"],"日语う缺少普通话圆唇"),
 "e":(["e"],"日语 e 替代普通话后元音 e"),"v":(["i"],"缺少 ü 圆唇，使用 i 近似"),
 "i0":(["u"],"缺少舌尖前元音，使用う近似"),"ir":(["u"],"缺少舌尖后元音，使用う近似"),
 "m":(["m"],""),"n":(["n"],""),"s":(["s"],""),"y":(["y"],""),"w":(["w"],""),
 "b":(["p"],"日语清音近似普通话不送气 b"),"d":(["t"],"日语清音近似普通话不送气 d"),"g":(["k"],"日语清音近似普通话不送气 g"),
 "p":(["p"],"缺少普通话送气对立"),"t":(["t"],"缺少普通话送气对立"),"k":(["k"],"缺少普通话送气对立"),
 "f":(["f"],"日语双唇音近似普通话唇齿音"),"h":(["h"],"日语 h 近似普通话 h"),"l":(["r"],"日语闪音替代普通话边音"),
 "j":(["ch"],"日语 ch 近似普通话不送气 j"),"q":(["ch"],"缺少普通话送气 q"),"x":(["sh"],"日语 sh 近似普通话 x"),
 "z":(["ts"],"日语 ts 近似普通话不送气 z"),"c":(["ts"],"缺少普通话送气 c"),
 "zh":(["ch"],"缺少普通话卷舌 zh"),"ch":(["ch"],"缺少普通话送气、卷舌 ch"),"sh":(["sh"],"缺少普通话卷舌 sh"),"r":(["r"],"日语闪音替代普通话卷舌近音"),
}
# Complete Mandarin finals are explicit compositions, with audible codas retained.
FINAL_PARTS={"ai":["a","i"],"ei":["e","i"],"ao":["a","o"],"ou":["o","u"],"an":["a","N"],"en":["e","N"],"ang":["a","N"],"eng":["e","N"],"ong":["o","N"],"ia":["i","a"],"ie":["i","e"],"iao":["i","a","o"],"iu":["i","o","u"],"ian":["i","a","N"],"in":["i","N"],"iang":["i","a","N"],"ing":["i","N"],"iong":["i","o","N"],"ua":["u","a"],"uo":["u","o"],"uai":["u","a","i"],"ui":["u","e","i"],"uan":["u","a","N"],"un":["u","e","N"],"uang":["u","a","N"],"ueng":["u","e","N"],"ve":["i","e"],"van":["i","a","N"],"vn":["i","N"]}
for final,seq in FINAL_PARTS.items():
    losses=["日语独立音素拼接普通话复合韵母，连贯音变需试听"]
    if "N" in seq: losses.append("日语拨音替代普通话鼻尾，发音部位未还原")
    if final.startswith("v"): losses.append("缺少 ü 圆唇")
    if "u" in seq: losses.append("日语う缺少普通话圆唇")
    ZH_FROM_JA[final]=(seq,"；".join(losses))


def recipe(source_language,target_language,phone,allow_approximate):
    rules=JA_FROM_ZH if (source_language,target_language)==("zh","ja") else ZH_FROM_JA if (source_language,target_language)==("ja","zh") else {}
    value=rules.get(phone)
    if value is None or (value[1] and not allow_approximate): return None
    return value


def blend(parts,fade_samples):
    output=parts[0].copy(); boundaries=[0]; joins=[]
    for part in parts[1:]:
        overlap=min(fade_samples,max(1,len(output)//4),max(1,len(part)//4))
        start=len(output)-overlap
        ramp=np.linspace(0,1,overlap,dtype=np.float32)
        output=np.concatenate((output[:-overlap],output[-overlap:]*(1-ramp)+part[:overlap]*ramp,part[overlap:]))
        boundaries.append(start); joins.append(overlap)
    fade=min(88,len(output)//8)
    if fade>1:
        output[:fade]*=np.linspace(0,1,fade); output[-fade:]*=np.linspace(1,0,fade)
    if not np.isfinite(output).all() or np.max(np.abs(output))>1: raise ValueError("拼接结果削波或含无效数据")
    return output,boundaries,joins


def generate(store,bank_id,generation,source_language,allow_approximate=False,crossfade_ms=15,emit=lambda **e:None):
    bank=store.bank(bank_id); prof=profile(bank["profile_id"]); target=prof.language_id
    if {source_language,target}!={"ja","zh"}: raise ValueError("跨语种拼接需要选择不同的日语和普通话来源/目标")
    if not 1<=crossfade_ms<=200: raise ValueError("重叠时长需在 1–200 ms 之间")
    if bank["generation"]!=generation: return {"stale":True}
    speaker=store.meta("speaker_id")
    assets={a.source_id:a for a in store.records("assets",SourceAsset) if a.speaker_id==speaker}
    phones=[p for p in store.records("alignments",Phone) if p.language_id==source_language and not p.unknown and p.phoneme_id and p.source_id in assets]
    pool={}
    for phone in phones:
        label=phone.phoneme_id.rsplit(":",1)[-1]
        if label in {"sil","sp","cl"} or phone.span.end-phone.span.start<441: continue
        pool.setdefault(label,[]).append(phone)
    if not pool: raise ValueError("所选来源语言没有可用的真实音素；请先按来源语言自动切片")
    for label,items in pool.items():
        # Prefer usable durations, retain complete aligned phonemes.
        items.sort(key=lambda p:abs((p.span.end-p.span.start)-6615))
    native={c.alias for c in store.candidates(bank_id=bank_id) if c.provenance_kind.value=="original" and c.landmarks and c.review_status!="rejected" and not c.quality_flags}
    targets=prof.required_units(language(target).inventory,"core")
    generated=[]; missing=[]; cache={}; failed=[]
    for index,unit in enumerate(targets):
        alias=prof.aliases(unit)[0]
        if alias in native: continue
        donor_phones=[]; losses=[]; target_groups=[]; reason=None
        for part_index,target_phone in enumerate(unit.unit_key.split("+")):
            if target=="ja" and part_index==0 and unit.unit_type in {"VC","VV"} and target_phone=="n": target_phone="N"
            value=recipe(source_language,target,target_phone,allow_approximate)
            if not value: reason="无兼容规则："+target_phone; break
            seq,loss=value
            if loss: losses.append(loss)
            target_groups.append(len(donor_phones))
            for label in seq:
                usable=None
                for phone in pool.get(label,[])[:40]:
                    if phone.phone_id not in cache:
                        audio=read_region(store.directory,assets[phone.source_id],phone.span.start,phone.span.end)
                        cache[phone.phone_id]=audio if len(audio)>0 and np.isfinite(audio).all() and float(np.sqrt(np.mean(audio**2)))>1e-4 else None
                    if cache[phone.phone_id] is not None: usable=phone; break
                if not usable: reason="缺少供体音素："+label; break
                donor_phones.append(usable)
            if reason: break
        if reason:
            missing.append({"alias":alias,"reason":reason}); continue
        donors=[{"phone_id":p.phone_id,"source_id":p.source_id,"utterance_id":p.utterance_id,"language":p.language_id,"source_sha256":assets[p.source_id].sha256,"raw_label":p.raw_model_label,"span":p.span.model_dump(),"alignment_revision":p.alignment_revision} for p in donor_phones]
        grade="approximate" if losses else ("composed" if len(donor_phones)>1 else "compatible")
        record={"algorithm":"cross-language-1","direction":source_language+"->"+target,"target_alias":alias,"target_phones":unit.unit_key.split("+"),"grade":grade,"losses":list(dict.fromkeys(losses)),"requested_crossfade_ms":crossfade_ms,"donors":donors}
        identifier=str(uuid5(NAMESPACE_URL,json_hash(record)))
        try:
            audio,boundaries,joins=blend([cache[p.phone_id] for p in donor_phones],round(crossfade_ms*44.1))
            record.update(actual_crossfade_samples=joins)
            path=store.directory/"derived"/(prof.filename_stem(unit)+"_cross_"+identifier+".wav")
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists(): sf.write(path,audio,44100,subtype="FLOAT")
            vowel=boundaries[target_groups[-1]] if unit.unit_type not in {"VC","VV"} else 0
            preutter=boundaries[target_groups[-1]] if unit.unit_type=="VC" else vowel
            fixed=len(audio)-1 if unit.unit_type=="VC" else min(len(audio)-1,max(preutter,vowel+882))
            marks=Landmarks(start=0,overlap=0,preutter=preutter,fixed_end=fixed,end=len(audio))
            first=donor_phones[0]
            candidate=Candidate(candidate_id=identifier,speaker_id=speaker,source_id=first.source_id,utterance_id=first.utterance_id,alignment_revision=first.alignment_revision,source_language=source_language,target_language=target,unit_role=unit.unit_type if unit.unit_type in {"VC","VV"} else "sustain_vowel",unit_key=unit.unit_key,alias=alias,phones=unit.unit_key.split("+"),raw_phones=[p.raw_model_label for p in donor_phones],span=SampleSpan(start=0,end=len(audio)),landmarks=marks,automatic_landmarks=marks,provenance_kind="cv_spliced",quality_score=.45 if losses else .65,quality_flags=["REPAIR_REVIEW_REQUIRED"]+(["CROSS_LANGUAGE_APPROXIMATE"] if losses else []),phonetic_context={"derived_audio_path":path.relative_to(store.directory).as_posix(),"audio_samples":len(audio),"vowel_span":{"start":vowel,"end":len(audio)},"phone_ids":[p.phone_id for p in donor_phones],"cross_language":record})
            generated.append(candidate)
            write_json(path.with_suffix(".json"),record)
        except (OSError,ValueError) as error: failed.append({"alias":alias,"reason":str(error)})
        if index%25==0: emit(stage="跨语种音素拼接",current=index+1,total=len(targets),device="cpu")
    with store.connection() as connection:
        if connection.execute("SELECT generation FROM banks WHERE id=?",(bank_id,)).fetchone()[0]!=generation: return {"stale":True}
        for item in generated:
            old=connection.execute("SELECT payload FROM candidates WHERE id=?",(item.candidate_id,)).fetchone()
            if old: continue # preserve reviews and manual timing on reruns
            connection.execute("INSERT INTO candidates VALUES(?,?,?,?)",(item.candidate_id,item.alias,store.unit_type(item),item.model_dump_json()))
        preset={**bank["preset"],"repair":True,"cross_source_language":source_language,"cross_allow_approximate":allow_approximate,"crossfade_ms":crossfade_ms}
        import json
        connection.execute("UPDATE banks SET preset=? WHERE id=?",(json.dumps(preset),bank_id))
        store._bump(connection,bank_id)
    store.recommend(bank_id)
    report={"source_language":source_language,"target_language":target,"generated":len(generated),"native_preserved":len(native),"grades":{g:sum(c.phonetic_context["cross_language"]["grade"]==g for c in generated) for g in ("compatible","composed","approximate")},"missing":missing,"failures":failed,"candidate_ids":[c.candidate_id for c in generated]}
    report_path=store.directory/"analyses"/("cross_language_"+uuid4().hex+".json")
    write_json(report_path,report)
    return {**report,"report_path":str(report_path)}
