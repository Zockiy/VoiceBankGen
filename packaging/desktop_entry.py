import sys
from pathlib import Path

# Compact portable editions share one dependency tree with TIFA. This path is
# outside _internal so the normal filesystem importer takes precedence over
# dependency versions retained in the frozen PYZ. Standard editions are unchanged.
if getattr(sys, "frozen", False):
    shared = Path(sys.executable).parent / "runtime/tifa-cpu/Lib/site-packages"
    if (shared / "torch_directml/__init__.py").is_file():
        sys.path.insert(0, str(shared))

from utau_forge.startup import launch

if __name__=="__main__":
    launch()
