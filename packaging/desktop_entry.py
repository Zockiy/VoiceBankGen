import sys

if "--tifa-worker" in sys.argv:
    from utau_forge._tifa_runtime import run
    run(sys.argv[sys.argv.index("--tifa-worker") + 1])
    raise SystemExit(0)

from utau_forge.startup import launch

if __name__=="__main__":
    launch()
