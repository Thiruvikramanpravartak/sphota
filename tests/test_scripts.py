import pytest

from sphota import unsupported_script
from sphota.api import _warn_unsupported


def test_unsupported_scripts_detected():
    assert unsupported_script("ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ") == "Ol Chiki (Santali)"
    assert unsupported_script("ꯃꯤꯇꯩꯂꯣꯟ") == "Meetei Mayek (Manipuri)"
    assert unsupported_script("मेरा ऑर्डर नहीं आया") is None
    assert unsupported_script("hello") is None
    assert unsupported_script("123 !!") is None


def test_warning_only_for_unsupported():
    with pytest.warns(UserWarning, match="Ol Chiki"):
        _warn_unsupported(["ok", "ᱥᱟᱱᱛᱟᱲᱤ"])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_unsupported(["ok", "ठीक है"])


def test_kaggle_notebook_never_imports_sphota_in_kernel():
    """The package is pip-installed after the kernel starts, so an in-kernel ``import sphota``
    fails (run 3's summary cell, D-054). Every use must go through a ``!python`` subprocess, and
    the results zip must be written before any summary step."""
    import json
    import re
    from pathlib import Path

    nb = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "kaggle_gpu_gate.ipynb").read_text("utf-8")
    )
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        for line in src.splitlines():
            if line.lstrip().startswith(("!", "#")):
                continue
            assert not re.search(r"\b(from|import)\s+sphota\b", line), line
    zip_cell = next(i for i, s in enumerate(code) if "zip " in s)
    summary_cell = next(i for i, s in enumerate(code) if "v3summary --scale medium" in s)
    assert zip_cell < summary_cell
