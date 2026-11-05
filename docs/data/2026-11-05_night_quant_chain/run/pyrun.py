#!/usr/bin/env python3
# Runs a script with a given gguf-py first on its path, for `python3 -I` (isolated mode ignores PYTHONPATH and the
# script's own folder, so nothing beside a model file or in the current folder is imported):
#   python3 -I pyrun.py <gguf-py folder> <script.py> [args...]
# gguf-py imports PyYAML for its model-card reader only; where it is missing, an empty module stands in, since the
# GGUF reader and writer never call it.
import runpy
import sys
import types

sys.path.insert(0, sys.argv[1])
try:
    import yaml  # noqa: F401
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
