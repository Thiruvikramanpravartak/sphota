import json

import pytest
from typer.testing import CliRunner

from sphota.cli import app
from sphota.codes import from_indices

runner = CliRunner()
CODE = str(from_indices([12, 3, 4, 5]))


def run(*args, input=None):
    return runner.invoke(app, list(args), input=input)


def test_validate():
    assert run("validate", CODE).exit_code == 0
    r = run("validate", "7F3-AC2")
    assert r.exit_code == 1 and "XXXX-XX-XX-XX" in r.output


def test_explain_neighbors_path_similarity():
    assert CODE in run("explain", CODE).output
    assert run("neighbors", CODE[:4], "-k", "3").exit_code == 0
    assert run("path", CODE, str(from_indices([99]))).exit_code == 0
    assert float(run("similarity", CODE, CODE).output.strip()) == pytest.approx(1.0)


def test_info_and_adapters():
    r = run("info")
    assert r.exit_code == 0 and "s1" in r.output
    assert run("adapters").exit_code == 0


def test_encode_cli(fake_encoder):
    r = run("encode", "The CEO resigned", "-a", "--top-k", "2")
    assert r.exit_code == 0 and "L1" in r.output
    r = run("encode", "--json", "The CEO resigned")
    d = json.loads(r.output.strip().splitlines()[0])
    assert d["code"] and len(d["confidence"]) == 4
    r = run("encode", "-", "--json", input="The CEO resigned\nI love you\n")
    assert len(r.output.strip().splitlines()) == 2


def test_fingerprint_compare_diff_cli(fake_encoder, tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("The CEO resigned. Quantum computers use qubits.")
    b = tmp_path / "b.txt"
    b.write_text("The CEO resigned. Add two cups of flour and stir until smooth.")
    assert run("fingerprint", str(a)).exit_code == 0
    assert run("fingerprint", str(a), "--json").exit_code == 0
    assert run("compare", str(a), str(b)).exit_code == 0
    assert run("diff", str(a), str(b)).exit_code == 0
