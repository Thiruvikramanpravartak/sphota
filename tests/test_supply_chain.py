import hashlib
import json
from pathlib import Path

import pytest

from sphota import encoders
from sphota.encoders import EncoderError, pinned_revision, verify_file


def test_anchor_model_is_pinned_with_hashes():
    pins = json.loads(
        (Path(encoders.__file__).parent / "data" / "models.json").read_text(encoding="utf-8")
    )
    e5 = pins["onnx"]["intfloat/multilingual-e5-small"]
    # the commit the frozen s1 release was built from
    assert e5["revision"] == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert set(e5["sha256"]) >= {"onnx/model.onnx", "tokenizer.json"}
    assert all(len(h) == 64 for p in pins["onnx"].values() for h in p["sha256"].values())
    assert pinned_revision("sentence-transformers/LaBSE", "torch")
    assert pinned_revision("someone/unknown-model") is None


def test_verify_file_accepts_and_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("SPHOTA_CACHE_DIR", str(tmp_path / "cache"))
    f = tmp_path / "model.onnx"
    f.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    verify_file(f, good, "test/model")
    assert (tmp_path / "cache" / "verified.json").exists()
    verify_file(f, good, "test/model")  # cached path
    f.write_bytes(b"tampered!")
    with pytest.raises(EncoderError, match="refusing to load"):
        verify_file(f, good, "test/model")


def test_bundle_rejects_paths_outside(tmp_path):
    from sphota.bundle import verify_bundle

    (tmp_path / "bundle.json").write_text(
        json.dumps({"files": {"../outside.txt": "0" * 64}}), encoding="utf-8"
    )
    assert verify_bundle(tmp_path) == ["path outside the bundle ../outside.txt"]
