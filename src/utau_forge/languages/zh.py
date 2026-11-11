"""Mandarin source phones and ASCII-pinyin UTAU CV/CVVC contracts."""
from importlib.resources import files
import json
from uuid import NAMESPACE_URL,uuid5
from ..models import Candidate,Landmarks,SampleSpan,TargetUnit
from ..oto import compile_oto


class MandarinPack:
    language_id="zh"
    version="1"

    def __init__(self):
        self.inventory=json.loads((files("utau_forge")/"data/zh_cv_inventory.json").read_text(encoding="utf-8"))
        self.sequence_alias={tuple(seq):alias for alias,seq in self.inventory["units"].items()}
        self.finals={seq[-1] for seq in self.inventory["units"].values()}

    def normalize_text(self,text,context):
        return text.strip()

    def asr_spec(self,model_lock):
        return {"language":"zh"}

    def alignment_spec(self,model_lock):
        return {"backend":"tifa","language":"zh"}

    def normalize_phones(self,raw_phones):
        known={p for seq in self.inventory["units"].values() for p in seq}
        return [p.removeprefix("zh/") if p.removeprefix("zh/") in known else None for p in raw_phones]

    def g2p(self,text,lexicon):
        raise NotImplementedError("普通话文本转音素由 TIFA 官方前端完成")

    def propose_units(self,phone_intervals,profile):
        if profile.language_id!=self.language_id:
            raise ValueError("profile language mismatch")
        ordered=sorted(phone_intervals,key=lambda p:p.span.start)
        result=[]; i=0
        while i<len(ordered):
            left=ordered[i]; matched=None
            for size in (2,1):
                batch=ordered[i:i+size]
                if len(batch)!=size or any(p.unknown or p.language_id!="zh" or p.utterance_id!=left.utterance_id or p.source_id!=left.source_id for p in batch): continue
                if any(abs(a.span.end-b.span.start)>1 for a,b in zip(batch,batch[1:])): continue
                seq=tuple(p.phoneme_id.rsplit(":",1)[-1] if p.phoneme_id else "" for p in batch)
                alias=self.sequence_alias.get(seq)
                if alias:
                    # Use the aligned reading when dictionary equivalents share phones.
                    annotated=left.features.get("syllable_alias")
                    if annotated and tuple(self.inventory["units"].get(annotated,()))==seq: alias=annotated
                    matched=(alias,seq,batch); break
            if not matched: i+=1; continue
            alias,seq,batch=matched
            identifier=str(uuid5(NAMESPACE_URL,f"zh:{left.source_id}:{left.alignment_revision}:{alias}:{left.span.start}:{batch[-1].span.end}"))
            result.append(Candidate(candidate_id=identifier,speaker_id="",source_id=left.source_id,utterance_id=left.utterance_id,alignment_revision=left.alignment_revision,source_language="zh",target_language="zh",unit_role="sustain_vowel",unit_key="+".join(seq),alias=alias,phones=list(seq),raw_phones=[p.raw_model_label for p in batch],span=SampleSpan(start=left.span.start,end=batch[-1].span.end),phonetic_context={"phone_ids":[p.phone_id for p in batch],"vowel_span":batch[-1].span.model_dump(),"lexical_tone":left.features.get("lexical_tone"),"tone_origin":"text-g2p" if left.features.get("lexical_tone") else "unknown","final":seq[-1]}))
            i+=len(batch)
        return result

    def timing_policy(self,unit): return {"unit_role":unit.unit_role,"cv_order":True}
    def phonotactic_constraints(self,target_sequence):
        keys={"+".join(seq) for seq in self.inventory["units"].values()}
        return ["UNSUPPORTED_UNIT" for unit in target_sequence if unit.unit_key not in keys]


class MandarinCVProfile:
    profile_id="zh-cv"
    language_id="zh"
    version="1"
    oto_encoding="utf-8"
    def required_units(self,inventory,scope):
        if scope!="core": raise ValueError("only core inventory is implemented")
        return [TargetUnit(target_language="zh",inventory_version=inventory["version"],unit_type="CV" if len(seq)>1 else "V_or_N",unit_key="+".join(seq),required_context={"pinyin":alias}) for alias,seq in inventory["units"].items()]
    def aliases(self,unit):
        if unit.unit_type in {"VC","VV"}: return [unit.unit_key.replace("+"," ")]
        return [self.pinyin(unit)]
    def filename_stem(self,unit):
        return unit.unit_type.lower()+"_"+unit.unit_key.replace("+","_") if unit.unit_type in {"VC","VV"} else self.pinyin(unit)
    def pinyin(self,unit):
        if unit.required_context.get("pinyin"): return unit.required_context["pinyin"]
        units=MandarinPack().inventory["units"]
        return next((alias for alias,seq in units.items() if "+".join(seq)==unit.unit_key),unit.unit_key.replace("+",""))
    def compile_timing(self,landmarks,total_samples,filename,alias): return compile_oto(filename,alias,landmarks,total_samples)
    def renderer_contract(self): return {"format":"UTAU","sample_rate":44100,"channels":1,"subtype":"PCM_16","oto_encoding":"utf-8"}


class MandarinCVVCProfile(MandarinCVProfile):
    profile_id="zh-cvvc"
    def required_units(self,inventory,scope):
        body=super().required_units(inventory,scope)
        finals=sorted({seq[-1] for seq in inventory["units"].values()})
        onsets=sorted({seq[0] for seq in inventory["units"].values() if len(seq)>1})
        return body+[TargetUnit(target_language="zh",inventory_version=inventory["version"],unit_type="VC",unit_key=v+"+"+c) for v in finals for c in onsets]


def transition_candidates(phones,utterances,inventory):
    finals={seq[-1] for seq in inventory["units"].values()}
    onsets={seq[0] for seq in inventory["units"].values() if len(seq)>1}
    result=[]
    ordered=sorted(phones,key=lambda p:(p.source_id,p.utterance_id,p.span.start))
    for left,right in zip(ordered,ordered[1:]):
        if left.language_id!="zh" or right.language_id!="zh" or left.unknown or right.unknown or left.source_id!=right.source_id or left.utterance_id!=right.utterance_id or abs(left.span.end-right.span.start)>1: continue
        v=left.phoneme_id.rsplit(":",1)[-1] if left.phoneme_id else ""
        c=right.phoneme_id.rsplit(":",1)[-1] if right.phoneme_id else ""
        if v not in finals or c not in onsets or right.span.end-left.span.start<882: continue
        start=max(left.span.start,left.span.end-4410); end=right.span.end
        if end-start<882: continue
        from ..vc_timing import crossfade_marks
        marks=crossfade_marks(Landmarks(start=start,overlap=start,preutter=right.span.start,fixed_end=end-1,end=end))
        utt=utterances[left.utterance_id]
        key=v+"+"+c
        result.append(Candidate(candidate_id=str(uuid5(NAMESPACE_URL,f"zh-vc:{left.source_id}:{left.alignment_revision}:{start}:{end}:{key}")),speaker_id=utt.speaker_id,source_id=left.source_id,utterance_id=left.utterance_id,alignment_revision=left.alignment_revision,source_language="zh",target_language="zh",unit_role="VC",unit_key=key,alias=key.replace("+"," "),phones=[v,c],raw_phones=[left.raw_model_label,right.raw_model_label],span=SampleSpan(start=start,end=end),landmarks=marks,automatic_landmarks=marks,quality_score=.8,phonetic_context={"phone_ids":[left.phone_id,right.phone_id],"transition_boundary":right.span.start,"lexical_tone":left.features.get("lexical_tone")}))
    return result


def presamp_text(inventory):
    vowels={}; consonants={}
    for alias,seq in inventory["units"].items():
        vowels.setdefault(seq[-1],[]).append(alias)
        if len(seq)>1: consonants.setdefault(seq[0],[]).append(alias)
    return "[VOWEL]\r\n"+"\r\n".join(f"{v}={v.upper()}={','.join(names)}=100" for v,names in sorted(vowels.items()))+"\r\n[CONSONANT]\r\n"+"\r\n".join(f"{c}={','.join(names)}=1" for c,names in sorted(consonants.items()))+"\r\n[REPLACE]\r\nü=v\r\n[ALIAS]\r\n[PRIORITY]\r\n"
