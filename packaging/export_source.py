"""Export an allowlisted source snapshot, never local history or generated assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "packaging/source_release_manifest.json"
BLOCKED_PARTS = {".git", ".history", ".venv", ".tifa-env", ".mfa-env", "runtime",
                 "dist", "build", "verification_runs", "examples", "deliverables"}
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".toml", ".lock", ".yaml", ".yml",
                   ".json", ".ps1", ".svg", ".license"}
IMAGE_EXTENSIONS = {".png", ".ico"}
SECRET_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b|\bgithub_pat_[A-Za-z0-9_]{40,255}\b"),
    "huggingface-token": re.compile(r"\bhf_[A-Za-z0-9]{30,100}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "slack-token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,200}\b"),
    "credential-in-url": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    "local-user-path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b)[^\s\"']+", re.I),
}


def checked_files(root: Path) -> list[tuple[str, Path, bytes]]:
    root = root.resolve(strict=True)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    result, seen = [], set()
    for relative in manifest["files"]:
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative
                or any(p.lower() in BLOCKED_PARTS for p in path.parts)):
            raise ValueError(f"Unsafe source path: {relative}")
        if relative.casefold() in seen:
            raise ValueError(f"Duplicate source path: {relative}")
        seen.add(relative.casefold())
        source = root / relative
        if not source.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Source link escapes workspace: {relative}")
        # Never copy files through a symlink or junction, even inside the workspace.
        if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
               for p in (source, *source.parents) if p != root and p.is_relative_to(root)):
            raise ValueError(f"Linked source path: {relative}")
        allowed = (source.suffix.lower() in TEXT_EXTENSIONS | IMAGE_EXTENSIONS
                   or source.name in {"LICENSE", ".gitignore", ".gitattributes"})
        if not source.is_file() or not allowed or source.name.startswith(".env"):
            raise ValueError(f"Not a permitted source file: {relative}")
        if source.stat().st_size > 10 * 1024**2:
            raise ValueError(f"Source file unexpectedly large: {relative}")
        content = source.read_bytes()
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            valid = content.startswith(b"\x89PNG\r\n\x1a\n") if source.suffix == ".png" else content.startswith(b"\0\0\1\0")
            if not valid or not relative.startswith("src/utau_forge/data/ui/"):
                raise ValueError(f"Only application icons may be binary: {relative}")
        else:
            if b"\0" in content:
                raise ValueError(f"Binary content in source file: {relative}")
            text = content.decode("utf-8-sig")
            # Match the public repository's LF policy so the SHA-256 manifest
            # also remains valid after a Git checkout on another platform.
            text = text.replace("\r\n", "\n")
            content = text.encode("utf-8")
            # Existing proxy-validation test uses this exact invalid host and
            # dummy credentials. No file-wide or general credential exemption.
            scan_text = text.replace("http://user" + ":pass@host:1234", "<invalid-proxy-test-fixture>")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(scan_text):
                    # Do not print matched secret contents.
                    raise ValueError(f"Review required ({label}) in {relative}")
        result.append((relative, source, content))
    return result


def export_source(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    output = output.resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Output must be a new directory inside this workspace")
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise FileExistsError("Refusing to overwrite an existing source export")
    files = checked_files(root)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    report = {"schema": 1, "scope": "Source-only snapshot; no development Git history, recordings, models or third-party runtimes",
              "source_branch": manifest.get("source_branch"),
              "files": [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                        for name, _, content in files]}
    output.mkdir(parents=True, exist_ok=False)
    for name, _, content in files:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    metadata = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    (output / "SOURCE_MANIFEST.json").write_bytes(metadata)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, _, content in files:
            bundle.writestr("VoiceBankGen/" + name, content)
        bundle.writestr("VoiceBankGen/SOURCE_MANIFEST.json", metadata)
    return {"directory": str(output), "archive": str(archive), "files": len(files),
            "source_bytes": sum(len(content) for _, _, content in files),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output", type=Path, default=ROOT / "release" / ("VoiceBankGen-source-" + stamp))
    args = parser.parse_args()
    print(json.dumps(export_source(ROOT, args.output), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
