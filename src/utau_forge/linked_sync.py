"""Synchronize dirty installations that share the same CV foundation."""
from pathlib import Path
import json
from .shared_banks import related_ids


def sync_linked_versions(store, bank_id, emit=lambda **event:None, *, exclude_installation=None):
    from .desktop_service import compile_bank
    from .desktop_sync import sync_installation
    with store.connection() as connection:
        identifiers=related_ids(connection,bank_id)
        rows=[dict(row) for identifier in identifiers for row in connection.execute(
            "SELECT * FROM installations WHERE bank_id=? AND (state!='synced' OR desired_revision>applied_revision)",(identifier,))
              if row["id"]!=exclude_installation]
    results=[]
    prepared={}
    for index,binding in enumerate(rows):
        target=binding["bank_id"]
        emit(stage="同步共用 CV 的关联版本",current=index,total=len(rows),reason=store.bank(target)["name"])
        try:
            root=Path(binding["path"])
            marker=root/"uforge.project-link.json"
            if root.name.casefold()=="singers" or not root.is_dir() or not marker.is_file():
                results.append({"bank_id":target,"state":"skipped","reason":"关联目录无效或已移除，请手动选择独立音源目录重新安装"})
                continue
            identity=json.loads(marker.read_text(encoding="utf-8"))
            if identity.get("project_id")!=store.project_id or identity.get("bank_id")!=target:
                results.append({"bank_id":target,"state":"skipped","reason":"安装标记不匹配，保留目录不改动"})
                continue
            if target not in prepared:
                prepared[target]=compile_bank(store,target,store.bank(target)["generation"],emit)
            build=prepared[target]
            if build.get("stale"):
                results.append({"bank_id":target,"state":"pending","reason":"编辑已变化，下次同步继续"})
                continue
            result=sync_installation(store,binding["id"],force=False)
            results.append({"bank_id":target,"installation_id":binding["id"],"state":result["state"],"path":result.get("path")})
        except Exception as error:
            results.append({"bank_id":target,"installation_id":binding["id"],"state":"failed","error":str(error)})
    emit(stage="关联版本同步完成",current=len(rows),total=len(rows))
    return results
