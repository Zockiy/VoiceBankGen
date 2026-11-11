from utau_forge.desktop_store import ProjectStore, forget_recent_project, recent_projects, visible_recent_projects


def test_forget_recent_keeps_project_and_installation_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR",str(tmp_path / "userdata"))
    store=ProjectStore.create(tmp_path / "project","Recent")
    before=store.database.read_bytes()
    forget_recent_project(store.project_id)
    assert store.project_id not in visible_recent_projects()
    assert recent_projects()[store.project_id]==str(store.directory)
    assert store.database.read_bytes()==before
    forget_recent_project(store.project_id)
    assert store.project_id not in visible_recent_projects()
    ProjectStore(store.directory)
    assert visible_recent_projects()[store.project_id]==str(store.directory)


def test_forget_one_recent_does_not_hide_other_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR",str(tmp_path / "userdata"))
    first=ProjectStore.create(tmp_path / "one","First")
    second=ProjectStore.create(tmp_path / "two","Second")
    forget_recent_project(first.project_id)
    forget_recent_project("absent")
    assert visible_recent_projects()=={second.project_id:str(second.directory)}
