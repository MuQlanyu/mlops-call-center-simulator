"""Call-center simulator: personality-conditioned dialogue trainer."""

import os as _os

# Guard against Colab/Jupyter-only matplotlib backends leaking into
# subprocess context where they are not valid (e.g. when running
# `dvc repro` from a Colab notebook).
_mpl_backend = _os.environ.get("MPLBACKEND", "")
if _mpl_backend.startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"
