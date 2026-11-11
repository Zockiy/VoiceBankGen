"""Inventory locked dependencies from official PyPI metadata, without downloading code."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
import importlib.metadata
from pathlib import Path
import re
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {}
OFFLINE = False


def dependencies(root: Path) -> dict:
    result = {}
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    for item in lock["package"]:
        if item["name"] != "voicebankgen":
            result.setdefault((item["name"], item["version"]), set()).add("uv.lock")
    catalog = json.loads((root / "src/utau_forge/data/runtime-catalog.json").read_text(encoding="utf-8"))
    for component in catalog["components"]:
        for artifact in component.get("artifacts", []):
            name = artifact.get("cache", "").rsplit("/", 1)[-1]
            match = re.match(r"(.+?)-(\d[^-]*)-(?:py|cp)", name)
            if match:
                package, version = match.groups()
                result.setdefault((package.replace("_", "-").lower(), version.replace("%2B", "+")), set()).add("runtime-catalog")
    return result


def inspect(item):
    (name, version), scopes = item
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    result = {"name": name, "version": version, "declared_in": sorted(scopes),
              "metadata_url": url, "vendored_in_source_release": False}
    local = LOCAL.get(re.sub(r"[-_.]+", "-", name).lower())
    if local is not None and local.version == version:
        info = local.metadata
        result.update(license_expression=info.get("License-Expression"),
                      license_classifiers=[c for c in info.get_all("Classifier", []) if c.startswith("License ::")],
                      license_metadata_excerpt=(info.get("License") or "").strip()[:240],
                      project_urls=info.get_all("Project-URL", []),
                      metadata_status="local-dist-info-exact-version")
        return result
    if OFFLINE:
        result.update(metadata_status="unverified-not-installed", note="Consult the exact upstream distribution before binary redistribution.")
        return result
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "VoiceBankGen-license-audit"})
        with urllib.request.urlopen(request, timeout=18) as response:
            info = json.load(response)["info"]
        expression = info.get("license_expression")
        classifiers = [c for c in info.get("classifiers", []) if c.startswith("License ::")]
        legacy = (info.get("license") or "").strip()
        result.update(license_expression=expression, license_classifiers=classifiers,
                      license_metadata_excerpt=legacy[:240],
                      project_urls=info.get("project_urls") or {}, metadata_status="retrieved")
    except Exception as error:
        result.update(metadata_status="unverified", error=type(error).__name__)
        # CPU wheels use a PyTorch index rather than PyPI. Do not pretend
        # base-release PyPI metadata proves the contents of that native wheel.
        if name == "torch" and version.endswith("+cpu"):
            base = version.split("+", 1)[0]
            source_url = f"https://raw.githubusercontent.com/pytorch/pytorch/v{base}/LICENSE"
            try:
                with urllib.request.urlopen(source_url, timeout=18) as response:
                    license_text = response.read().decode("utf-8")
                if "Redistribution and use in source and binary forms" not in license_text:
                    raise ValueError("Unexpected upstream license")
                result.update(metadata_status="upstream-source-license-only",
                              source_license_url=source_url,
                              license_metadata_excerpt=license_text[:240],
                              note="Fixed base-release source license checked. The +cpu wheel and bundled native notices still require a binary-specific audit.")
                result.pop("error", None)
            except Exception:
                pass
    return result


def main():
    global OFFLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/dependency_inventory.json")
    parser.add_argument("--local", action="store_true", help="Use exact-version local dist-info without network access")
    parser.add_argument("--metadata-root", type=Path, action="append", default=[],
                        help="Also inspect exact-version dist-info in a prepared runtime (no code imported)")
    args = parser.parse_args()
    OFFLINE = args.local
    for directory in args.metadata_root:
        for dist in importlib.metadata.distributions(path=[str(directory)]):
            LOCAL[re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()] = dist
    if args.local:
        for dist in importlib.metadata.distributions():
            LOCAL[re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()] = dist
    with ThreadPoolExecutor(max_workers=8) as pool:
        packages = list(pool.map(inspect, sorted(dependencies(ROOT).items())))
    report = {"audit_date": date.today().isoformat(),
              "scope": "Declared locks; metadata is evidence, not a binary redistribution clearance. No dependency code or models are included.",
              "packages": packages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"packages": len(packages), "unverified": sum(p["metadata_status"].startswith("unverified") for p in packages)}))


if __name__ == "__main__":
    main()
