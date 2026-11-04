# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL helpers: install the pure-SQL functions and the L1 neighbour table.

    from sphota.integrations.postgres import install
    install(conn)                        # psycopg 3 connection

or from the shell::

    psql "$DATABASE_URL" -f integrations/postgres/sphota.sql
    python -m sphota.integrations.postgres --neighbors > neighbors.csv
    psql "$DATABASE_URL" -c "\\copy sphota_l1_neighbors FROM 'neighbors.csv' CSV"
"""

from __future__ import annotations

import io
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from sphota.codes import from_indices
from sphota.release import load_release

SQL_PATH = Path(__file__).resolve().parents[3] / "integrations" / "postgres" / "sphota.sql"


def schema_sql() -> str:
    """The SQL helper library (bundled copy, falls back to the repo file)."""
    try:
        return (
            resources.files("sphota.integrations")
            .joinpath("sphota.sql")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return SQL_PATH.read_text(encoding="utf-8")


def neighbors_rows(k: int = 16) -> list[tuple[str, str, int, float, float]]:
    rel = load_release()
    C = rel.codebooks[0].astype(np.float64)
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    nb, dist = rel.l1_neighbors, rel.l1_neighbor_dist
    assert nb is not None and dist is not None
    rows = []
    for i in range(rel.sizes[0]):
        a = str(from_indices([i]))
        for r, (j, d) in enumerate(zip(nb[i][:k], dist[i][:k]), start=1):
            rows.append(
                (a, str(from_indices([int(j)])), r, float(d), float(max(0.0, Cn[i] @ Cn[int(j)])))
            )
    return rows


def neighbors_copy(k: int = 16) -> str:
    buf = io.StringIO()
    for row in neighbors_rows(k):
        buf.write(",".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in row) + "\n")
    return buf.getvalue()


def install(conn: Any, k: int = 16) -> None:
    """Create functions/tables and load the neighbour table (psycopg 3 connection)."""
    with conn.cursor() as cur:
        cur.execute(schema_sql())
        cur.execute("TRUNCATE sphota_l1_neighbors")
        with cur.copy("COPY sphota_l1_neighbors FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(neighbors_copy(k))
    conn.commit()


if __name__ == "__main__":  # pragma: no cover
    if "--neighbors" in sys.argv:
        sys.stdout.write(neighbors_copy())
    else:
        sys.stdout.write(schema_sql())
