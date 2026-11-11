"""Merge matching frozen dependency code/native files into one portable runtime.

No source environment dependency code is substituted for frozen dependencies.
GPU editions share CUDA PyTorch for both CUDA and CPU execution.
"""
from pathlib import Path
import argparse,json,marshal,os,shutil,sys,hashlib
from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.archive.writers import CArchiveWriter,ZlibArchiveWriter
ROOT=Path(__file__).resolve().parents[1]

def allowed(relative):
    parts=relative.parts
    if any(p in {'__pycache__','.git','.history','tests','test'} for p in parts):return False
    if relative.suffix.lower() in {'.pyc','.pyo','.lib','.exp','.pdb'}:return False
    if parts[:2] in {('torch','include'),('torch','share'),('torch','bin')}:return False
    if relative.suffix.lower()=='.md' and not any(v in relative.name.lower() for v in ('license','notice','copying','copyright')):return False
    return True

def write_exe(source,target,pyz,work):
    reader=CArchiveReader(str(source));last=[n for n,v in reader.toc.items() if v[4]=='s'][-1]
    script=marshal.dumps(compile((ROOT/'packaging/desktop_entry.py').read_text(encoding='utf-8'),'desktop_entry.py','exec'))
    class Writer(CArchiveWriter):
        def _write_entry(self,stream,item):
            name,_,compress,kind=item
            if kind=='o':return self._write_blob(stream,b'',name,kind)
            blob=pyz.read_bytes() if name=='PYZ.pyz' else script if name==last else reader.extract(name)
            return self._write_blob(stream,blob,name,kind,compress=compress)
    pkg=work/(target.stem+'.pkg')
    Writer(str(pkg),[(n,'',v[3],v[4]) for n,v in reader.toc.items()]+[(n,'',False,'o') for n in reader.options],'python311.dll')
    with source.open('rb') as original,target.open('xb') as output:
        output.write(original.read(reader._start_offset))
        with pkg.open('rb') as payload:shutil.copyfileobj(payload,output)

def prepare(base,output,gpu,resume=False,legal_source=None):
    legal_source=legal_source or base/'licenses'
    if gpu and not (legal_source/'GPU').is_dir():
        raise ValueError('Provide --legal-source with the audited GPU license directory')
    if output.exists() and not resume:raise FileExistsError(output)
    package=output/'VoiceBankGen';work=output/'records';package.mkdir(parents=True,exist_ok=resume);work.mkdir(exist_ok=resume)
    runtime=base/'runtime'/('tifa-gpu' if gpu else 'tifa-cpu')
    mapping={}
    # A runtime-owned package is taken whole from its matching frozen build.
    # In particular CUDA torch must never be combined with CPU torch 2.14.
    owned={p.name for p in (runtime/'_internal').iterdir() if p.is_dir()}
    for directory in (base/'_internal',runtime/'_internal'):
        for path in directory.rglob('*'):
            if not path.is_file():continue
            rel=path.relative_to(directory)
            if not allowed(rel) or rel.parts[:2]==('utau_forge','data'):continue
            if directory==base/'_internal' and rel.parts[0] in owned:continue
            if gpu and directory==base/'_internal' and rel.parts[0].startswith(('torch-','functorch-')) and rel.parts[0].endswith('.dist-info'):continue
            mapping[rel]=path
    for rel,path in mapping.items():
        dest=package/'_internal'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(path,dest)
    # The source CUDA environment supplies metadata missing in the historical
    # frozen GPU directory, not replacement Python code or native binaries.
    if gpu:
        for meta in (ROOT/'.tifa-env/Lib/site-packages').glob('torch-*.dist-info'):
            for path in meta.rglob('*'):
                if path.is_file() and allowed(path.relative_to(meta)):
                    dest=package/'_internal'/meta.name/path.relative_to(meta);dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():shutil.copy2(path,dest)
    for relbase,source in [('resources',base/'resources'),('runtime/ffmpeg',base/'runtime/ffmpeg'),('licenses',base/'licenses'),('_internal/utau_forge/data',ROOT/'src/utau_forge/data')]:
        for path in source.rglob('*'):
            if not path.is_file():continue
            rel=path.relative_to(source)
            if not allowed(rel):continue
            if relbase=='resources' and (rel.as_posix()=='unidic/model.bin' or rel.suffix in {'.zip','.7z'}):continue
            if relbase=='licenses' and not gpu and rel.parts[0]=='GPU':continue
            dest=package/relbase/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():os.link(path,dest)
    bridge=package/'_internal/utau_forge/data/runtime/tifa_entry.py'
    bridge.parent.mkdir(parents=True,exist_ok=True)
    # This may already be a linked historical data file; replace by a new inode.
    temp=work/'tifa_entry.py';shutil.copy2(ROOT/'packaging/tifa_entry.py',temp);os.replace(temp,bridge)
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md'):
        shutil.copy2(ROOT/name,package/name)
    for path in (ROOT/'LICENSES').iterdir():
        if path.is_file() and path.suffix.lower()!='.md':
            dest=package/'licenses'/path.name
            if not dest.exists():shutil.copy2(path,dest)
    for path in legal_source.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(legal_source)
        if rel.parts[0]=='GPU' and not gpu:continue
        if not any(t in path.name.lower() for t in ('license','notice','copying','copyright','eula','sla','terms','source_references')):continue
        dest=package/'licenses'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(path,dest)
    for source,name in [('PORTABLE_QUICK_START.txt','使用说明.txt'),('RELEASE_NOTICES.txt','第三方许可说明.txt')]:
        text=(ROOT/'packaging'/source).read_text(encoding='utf-8')
        if not gpu and name=='使用说明.txt':
            start=text.index('CPU / GPU 选择');end=text.index('空格试听',start)
            text=text[:start]+'CPU 版\n仅含 CPU 环境与完整模型，请选择 CPU。\n\n'+text[end:]
        (package/name).write_text(text,encoding='utf-8')
    entries={};code={}
    for executable in (base/'VoiceBankGen_UI_Fork.exe',runtime/'TIFAWorker.exe'):
        archive=CArchiveReader(str(executable)).open_embedded_archive('PYZ.pyz')
        for name,row in archive.toc.items():
            kind=row[0];filename=None if kind==3 else name.replace('.','/')+('/__init__.py' if kind==1 else '.py')
            entries[name]=(name,filename,'PYMODULE')
            if kind!=3:code[name]=archive.extract(name)
    for path in (ROOT/'src/utau_forge').rglob('*.py'):
        if any(p in {'__pycache__','.history'} for p in path.parts):continue
        rel=path.relative_to(ROOT/'src').with_suffix('');parts=rel.parts[:-1] if rel.name=='__init__' else rel.parts;name='.'.join(parts)
        entries[name]=(name,str(path),'PYMODULE');code[name]=compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    name='utau_forge._tifa_runtime';source=ROOT/'packaging/tifa_entry.py'
    entries[name]=(name,str(source),'PYMODULE');code[name]=compile(source.read_text(encoding='utf-8'),str(source),'exec')
    pyz=work/'shared.pyz';ZlibArchiveWriter(str(pyz),list(entries.values()),code_dict=code)
    for name,original in [('VoiceBankGen.exe','VoiceBankGen_UI_Fork.exe'),('VoiceBankGenWorker.exe','VoiceBankGenWorker.exe')]:
        if (package/name).exists():
            previous=work/(name+'.previous')
            if previous.exists():raise FileExistsError(previous)
            os.replace(package/name,previous)
        write_exe(base/original,package/name,pyz,work)
    (package/'shared-runtime.json').write_text(json.dumps({'schema':1,'cuda':gpu,'layout':'single-worker-v1'}),encoding='utf-8')
    # A compact installed-file catalog, no duplicate offline repair ZIPs.
    records=[]
    for path in package.rglob('*'):
        if path.is_file() and path.name!='components.json' and allowed(path.relative_to(package)):
            records.append({'path':path.relative_to(package).as_posix(),'size':path.stat().st_size,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()})
    catalog={'schema':1,'version':'2026.11.05.4','edition':'portable','components':[{'id':'app-core','name':'VoiceBankGen '+('CPU / CUDA' if gpu else 'CPU'),'version':'2026.11.05.4','required':True,'description':'共用一套推理依赖；不附重复安装缓存。','artifacts':[],'generated':[], 'probe_files':['VoiceBankGen.exe','VoiceBankGenWorker.exe']} ]}
    catalog['components'][0]['generated']=records
    (package/'components.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'profile':'CPU-GPU' if gpu else 'CPU','bytes':sum(r['size'] for r in records),'files':len(records),'runtime_source':str(runtime),'dependency_modules':len(entries)}
    (work/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--legal-source',type=Path);args=parser.parse_args()
    prepare(args.base.resolve(),args.output.resolve(),args.gpu,args.resume,args.legal_source)
