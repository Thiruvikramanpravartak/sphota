import json
import shutil

import numpy as np
import pytest

from sphota.migrate import apply_table, build_table, migrate_code, table_from_vectors
from sphota.release import load_release, sha256_file

GOLD = json.loads(open("tests/vectors/golden.json", encoding="utf-8").read())


@pytest.fixture
def perturbed_release(tmp_path):
    """A copy of s1 with slightly moved L2 centroids, standing in for a future s2."""
    src = load_release().root
    dst = tmp_path / "s2"
    shutil.copytree(src, dst)
    with np.load(dst / "codebooks.npz") as z:
        cb = {k: z[k] for k in z.files}
    cb["level2"] = cb["level2"] + 0.5 * np.random.default_rng(0).standard_normal(
        cb["level2"].shape
    ).astype(np.float32)
    np.savez(dst / "codebooks.npz", **cb)
    m = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    m["files"]["codebooks.npz"]["sha256"] = sha256_file(dst / "codebooks.npz")
    (dst / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return src, dst


def test_migration_table_roundtrip(perturbed_release, tmp_path):
    src, dst = perturbed_release
    X = np.array([g["vector"] for g in GOLD["items"]], dtype=np.float32)
    same = table_from_vectors(src, src, X)
    assert same["mean_top1"] == pytest.approx(1.0)
    res = build_table(src, dst, X=X, out=tmp_path / "t.json")
    assert res["n_prefixes"] > 0 and 0 < res["mean_top1"] <= 1
    code = GOLD["items"][0]["code"]
    new, p = migrate_code(code, res)
    assert len(new.split("-")) == 2 and 0 < p <= 1
    csv_in = tmp_path / "codes.csv"
    csv_in.write_text(
        "id,code\n" + "\n".join(f"{i},{g['code']}" for i, g in enumerate(GOLD["items"])),
        encoding="utf-8",
    )
    n = apply_table(csv_in, tmp_path / "t.json", tmp_path / "out.csv")
    assert n == len(GOLD["items"])
    assert "code_s1" in (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()[0]
    with pytest.raises(ValueError):
        build_table(src, dst, out=tmp_path / "x.json")


def test_checksum_verification_detects_tampering(perturbed_release):
    from sphota.release import ReleaseError

    _, dst = perturbed_release
    (dst / "labels.json").write_text("{}", encoding="utf-8")
    load_release.cache_clear()
    with pytest.raises(ReleaseError, match="checksum"):
        load_release(str(dst))
    load_release.cache_clear()


def test_adapter_refuses_non_npz_and_bad_checksum(tmp_path):
    from sphota.adapters import Adapter, AdapterError, load_adapter

    p = tmp_path / "evil.pkl"
    p.write_bytes(b"x")
    with pytest.raises(AdapterError, match="pickles"):
        load_adapter(p)
    ad = Adapter("m", np.eye(4, 256, dtype=np.float32), np.zeros(256, np.float32), {"prefix": ""})
    f = ad.save(tmp_path / "a.npz")
    assert load_adapter(f).in_dim == 4
    with pytest.raises(AdapterError, match="checksum"):
        load_adapter(f, sha256="0" * 64)
    with pytest.raises(AdapterError, match="expects 4-d"):
        ad(np.ones((1, 5)))


def test_bundle_verify(tmp_path, monkeypatch):
    from sphota import bundle

    monkeypatch.setattr(bundle, "ENCODER_FILES", {})
    out = bundle.make_bundle(tmp_path / "b")
    assert bundle.verify_bundle(out) == []
    (out / "release" / "s1" / "labels.json").write_text("{}", encoding="utf-8")
    assert any("labels.json" in x for x in bundle.verify_bundle(out))
