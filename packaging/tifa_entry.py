"""Official TIFA inference backend with explicit accelerator selection."""
from __future__ import annotations
import json
import sys
from pathlib import Path

def run(request_path):
    for stream in (sys.stdout,sys.stderr):
        if stream and hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="replace")
    request=json.loads(Path(request_path).read_text(encoding="utf-8"))
    upstream=Path(request["upstream"])
    sys.path.insert(0,str(upstream))
    import torch
    import lightning.pytorch as pl
    from inference.api import load_inference_model
    from inference.data import AudioTextDataset
    from inference.module import ForcedAlignmentInferenceModule
    from inference.callbacks import SaveTextGridCallback, StatisticsCallback
    from lib.config.schema import ConfigurationScope, G2PPipelineConfig
    device=request["device"]
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("TIFA PyTorch CUDA 不可用")
    model=Path(request["model"])
    backend,vocabulary,config=load_inference_model(model,scope=ConfigurationScope.FA)
    # Preserve the official model config. A Japanese-only front end prevents
    # unrelated Chinese/English resources from being loaded for Japanese input.
    front={"preprocessors":[{"id":"filter-punctuation"},{"id":"strip-whitespace"},{"id":"lowercase"}],"converters":[{"id":"japanese-mecab","kwargs":{"dict_path":"@dictionaries/japanese_dict_full.txt","unidic_dir":request["unidic"],"nbest":32,"double_written_sokuon":False}},{"id":"dictionary","language":"ja","kwargs":{"dict_path":"@dictionaries/japanese_dict_full.txt"}}]}
    lang=request.get("language","ja")
    if lang=="zh":
        front={"preprocessors":[{"id":"filter-punctuation"},{"id":"strip-whitespace"},{"id":"lowercase"}],"converters":[{"id":"chinese-pinyin","kwargs":{"dict_path":"@dictionaries/ds-zh-pinyin-lite.txt"}},{"id":"dictionary","language":"zh","kwargs":{"dict_path":"@dictionaries/ds-zh-pinyin-lite.txt"}}]}
    elif lang!="ja":
        raise ValueError("不支持的来源语言："+lang)
    g2p=G2PPipelineConfig.model_validate(front,scope=ConfigurationScope.FA)
    filemap={path.stem:path for path in Path(request["corpus"]).glob("*.wav")}
    dataset=AudioTextDataset(filemap=filemap,g2p_config=g2p,g2p_root=model.parent,vocabulary=vocabulary,audio_sample_rate=backend.sample_rate,language=[lang],oov_handling="discard")
    output=Path(request["output"])
    if lang=="zh":
        from g2p.converters.cpp_pinyin import PinyinEngine
        from g2p.converters.cpp_pinyin.constants import STYLE_TONE3
        engine=PinyinEngine(str(upstream/"g2p/converters/cpp_pinyin/dicts/mandarin"))
        readings={}
        for stem,path in filemap.items():
            text=path.with_suffix(".txt").read_text(encoding="utf-8")
            chars=[ch for ch in text if "\u4e00"<=ch<="\u9fa5"]
            values=engine.query_raw(engine.simplify(chars),style=STYLE_TONE3)
            readings[stem]=[{"alias":v[0].rstrip("012345").replace("ü","v"),"tone":int(v[0][-1]) if v[0][-1:].isdigit() else None} for v in values if v and v[0]]
        output.mkdir(parents=True,exist_ok=True)
        (output/"readings.json").write_text(json.dumps(readings,ensure_ascii=False),encoding="utf-8")
    callbacks=[SaveTextGridCallback(output_dir=output,language=lang,timestep=backend.timestep,skip_handling="omit"),StatisticsCallback(save_dir=output/"statistics")]
    class Progress(pl.Callback):
        def on_predict_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
            print(json.dumps({"stage":"TIFA 对齐","current":min(len(dataset),(batch_idx+1)*request.get("batch_size",1)),"total":len(dataset),"device":device,"precision":"32-true"},ensure_ascii=False),flush=True)
    callbacks.append(Progress())
    module=ForcedAlignmentInferenceModule(backend,score_unit="levenshtein",skip_penalty=.5)
    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=0))
    trainer=pl.Trainer(accelerator="gpu" if device=="cuda" else "cpu",devices=1,precision="32-true",logger=False,enable_checkpointing=False,enable_progress_bar=True,callbacks=callbacks)
    loader=torch.utils.data.DataLoader(dataset,batch_size=request.get("batch_size",1),num_workers=0,shuffle=False,collate_fn=dataset.collate)
    trainer.predict(module,loader)
    print(json.dumps({"kind":"tifa_complete","device":device,"torch":torch.__version__,"cuda":torch.version.cuda,"grids":len(list(output.glob("*.TextGrid")))}),flush=True)

if __name__=="__main__":
    run(sys.argv[1])
