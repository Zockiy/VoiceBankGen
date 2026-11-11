from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sqlite3

from utau_forge.source_removal import removal_plan


def test_shared_audio_and_transitive_splice_dependents(monkeypatch):
    first=SimpleNamespace(source_id="speech",source_kind="speech",original_path="sources//original/shared.wav",canonical_path="audio//canonical/shared.wav")
    second=SimpleNamespace(source_id="song",source_kind="singing",original_path="sources/original/shared.wav",canonical_path="audio/canonical/shared.wav")
    def candidate(identifier,source,parents=(),context=None):
        return SimpleNamespace(candidate_id=identifier,source_id=source,parent_candidate_ids=list(parents),phonetic_context=context or {})
    candidates=[candidate("original","speech"),candidate("splice","song",["original"]),candidate("tail","song",["splice"]),candidate("cross","song",context={"donors":[{"source_id":"speech"}]}),candidate("keep","song")]
    connection=sqlite3.connect(":memory:")
    connection.row_factory=sqlite3.Row
    connection.execute("CREATE TABLE bank_selections(candidate_id TEXT)")
    connection.execute("INSERT INTO bank_selections VALUES('tail')")
    @contextmanager
    def db():
        yield connection
    store=SimpleNamespace(directory=Path.cwd(),records=lambda table,model:[first,second] if table=="assets" else candidates,connection=db)
    monkeypatch.setattr(Path,"is_file",lambda path:True)
    plan=removal_plan(store,["speech"])
    assert set(plan["candidate_ids"])=={"original","splice","tail","cross"}
    assert plan["selections"]==1
    assert not plan["files"]
    connection.close()
