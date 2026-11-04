"""Runs the SQL helpers against a real temporary PostgreSQL cluster (skipped if unavailable)."""

import glob
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
BIN = sorted(glob.glob("/usr/lib/postgresql/*/bin"))
ROOT = Path(__file__).resolve().parents[1]


def _run(cmd):
    if os.geteuid() == 0:
        cmd = ["su", "postgres", "-c", " ".join(cmd)]
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def conn():
    if not BIN:
        pytest.skip("PostgreSQL binaries not found")
    b = BIN[-1]
    d = tempfile.mkdtemp(prefix="sphota-pg-")
    os.chmod(d, 0o777)
    if os.geteuid() == 0:
        shutil.chown(d, "postgres")
    data = f"{d}/data"
    try:
        _run([f"{b}/initdb", "-D", data, "-A", "trust", "-U", "postgres"])
        _run(
            [
                f"{b}/pg_ctl",
                "-D",
                data,
                "-o",
                f"'-k {d} -p 55433 -c listen_addresses='''''",
                "-l",
                f"{d}/log",
                "start",
                "-w",
            ]
        )
    except (subprocess.CalledProcessError, KeyError) as e:
        pytest.skip(f"could not start PostgreSQL: {e}")
    time.sleep(0.5)
    c = psycopg.connect(host=d, port=55433, user="postgres", dbname="postgres")
    yield c
    c.close()
    _run([f"{b}/pg_ctl", "-D", data, "stop", "-m", "fast"])
    shutil.rmtree(d, ignore_errors=True)


def test_sql_helpers(conn):
    from sphota.integrations.postgres import install

    install(conn, k=8)
    with conn.cursor() as cur:
        cur.execute((ROOT / "integrations/postgres/demo.sql").read_text())
        conn.commit()
        cur.execute(
            "SELECT sphota_depth('JYZA-7M-X0-A8'), sphota_prefix('JYZA-7M-X0-A8', 2), sphota_level('JYZA-7M-X0-A8', 3)"
        )
        assert cur.fetchone() == (4, "JYZA-7M", "X0")
        cur.execute(
            "SELECT sphota_is_prefix('JYZA', 'JYZA-7M'), sphota_is_prefix('JYZ', 'JYZA-7M'), sphota_common_prefix('JYZA-7M-X0-A8','JYZA-7M-0Q-BS')"
        )
        assert cur.fetchone() == (True, False, 2)
        cur.execute(
            "SELECT sphota_is_code('JYZA-7M'), sphota_is_code('JYZ-7M'), sphota_is_code('JYZA-7M-X0-A8-11')"
        )
        assert cur.fetchone() == (True, False, False)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute("INSERT INTO docs (body, code) VALUES ('x', 'not-a-code')")
        conn.rollback()
        cur.execute("SELECT count(*) FROM docs WHERE code LIKE sphota_like('JYZA-7M')")
        assert cur.fetchone()[0] >= 3  # en, es, de, ja CEO sentences
        cur.execute(
            "SELECT sphota_similarity('JYZA-7M-X0-A8','JYZA-7M-X0-A8'), sphota_similarity('JYZA-7M-X0-A8','JYZA-7M-0Q-BS')"
        )
        same, part = cur.fetchone()
        assert same == pytest.approx(1.0) and 0.7 < part < 1.0
        cur.execute("SELECT count(*) FROM sphota_nearby_l1('JYZA-7M', 4)")
        assert cur.fetchone()[0] == 5
        cur.execute("SET enable_seqscan = off")
        cur.execute("EXPLAIN SELECT * FROM docs WHERE code LIKE 'JYZA-7M%'")
        plan = " ".join(r[0] for r in cur.fetchall())
        assert "docs_code_prefix" in plan
