"""Run the investigation SQL over IEEE-CIS CSVs with DuckDB.

    python scripts/run_investigation.py                 # uses data/ieee/ or a mock
    python scripts/run_investigation.py --dir data/ieee

DuckDB queries the CSVs directly (no database server, no load step), which is how an
analyst would explore the raw competition files. If the real files are absent, a
schema-faithful mock is written to a temp directory so the queries still run.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import duckdb

from fraud import ieee_data

SQL_FILE = Path(__file__).resolve().parent.parent / "sql" / "investigation.sql"


def parse_queries(text: str):
    """Split the .sql file into (title, statement) pairs on ';' boundaries."""
    out = []
    for chunk in text.split(";"):
        stmt = chunk.strip()
        if not stmt:
            continue
        title = "query"
        for line in stmt.splitlines():
            s = line.strip()
            if s.lower().startswith("-- name:"):
                title = s.split(":", 1)[1].strip()
                break
        out.append((title, stmt))
    return out


def resolve_dir(cli_dir: str | None) -> Path:
    if cli_dir:
        return Path(cli_dir)
    default = Path("data/ieee")
    if (default / "train_transaction.csv").exists():
        return default
    tmp = Path(tempfile.mkdtemp(prefix="ieee_mock_"))
    print(f"Real IEEE-CIS not found; writing a mock to {tmp}\n")
    ieee_data.write_mock_ieee(tmp, seed=7)
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="folder with train_transaction.csv / train_identity.csv")
    ap.add_argument("--rows", type=int, default=10, help="rows to print per query")
    args = ap.parse_args()

    data_dir = resolve_dir(args.dir)
    tx = (data_dir / "train_transaction.csv").as_posix()
    idn = (data_dir / "train_identity.csv").as_posix()

    con = duckdb.connect()
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_csv_auto('{tx}')")
    if Path(idn).exists():
        con.execute(f"CREATE VIEW idn AS SELECT * FROM read_csv_auto('{idn}')")
        con.execute("CREATE VIEW joined AS SELECT tx.*, idn.DeviceInfo, idn.DeviceType "
                    "FROM tx LEFT JOIN idn USING (TransactionID)")
    else:
        con.execute("CREATE VIEW idn AS SELECT NULL AS TransactionID WHERE 1=0")
        con.execute("CREATE VIEW joined AS SELECT tx.*, CAST(NULL AS VARCHAR) AS DeviceInfo, "
                    "CAST(NULL AS VARCHAR) AS DeviceType FROM tx")

    for title, stmt in parse_queries(SQL_FILE.read_text()):
        print(f"\n=== {title} ===")
        print(con.execute(stmt).df().head(args.rows).to_string(index=False))


if __name__ == "__main__":
    main()
