# Sourcing data

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    import pandas as pd  # type-checking only: pandas is not a runtime dependency of the package

_READERS: dict[str, str] = {
    ".parquet": "read_parquet",
    ".csv": "read_csv",  # auto-detects dialect and types
    ".tsv": "read_csv",
    ".json": "read_json_auto",
    ".jsonl": "read_json_auto",
    ".ndjson": "read_json_auto",
}


def quote_ident(name: str) -> str:
    """Quote a column or view name. Column names come from untrusted data."""
    return '"' + name.replace('"', '""') + '"'


def scalar_int(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Run a query that returns a single integer, e.g. SELECT count(*) ..."""
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no rows: {sql}")
    return int(row[0])


@dataclass
class DataSource:
    """A DuckDB connection plus the name of the view that all profiling queries run against."""

    con: duckdb.DuckDBPyConnection
    view: str = "data"
    origin: str = ""

    @classmethod
    def from_path(cls, path: str | Path, view: str = "data") -> DataSource:
        """Load data from a filepath"""
        path = Path(path)
        reader = _READERS.get(path.suffix.lower())
        if reader is None:
            raise ValueError(
                f"Unsupported file type {path.suffix}. Must be one of {_READERS.keys()}."
            )
        con = duckdb.connect()
        escaped = path.as_posix().replace("'", "''")  # views can't take bound parameters
        con.execute(f"CREATE VIEW {quote_ident(view)} AS SELECT * FROM {reader}('{escaped}')")
        return cls(con, view, origin=str(path))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, view: str = "data") -> DataSource:
        """Load data from a dataframe"""
        con = duckdb.connect()
        con.register(view, df)  # zero-copy view over the DataFrame
        return cls(con, view, origin="dataframe")

    @classmethod
    def from_table(
        cls, con: duckdb.DuckDBPyConnection, table: str, *, view: str = "data"
    ) -> DataSource:
        """Wrap an existing table in a connection.
        `table` is interpolated as-is, so it must be trusted."""
        con.execute(f"CREATE OR REPLACE VIEW {quote_ident(view)} AS SELECT * FROM {table}")
        return cls(con, view, origin=f"table:{table}")

    # helpers used by profiler, executor and the rule validator
    def columns(self) -> dict[str, str]:
        rows = self.con.execute(f"DESCRIBE {quote_ident(self.view)}").fetchall()
        return {name: dtype for name, dtype, *_ in rows}

    def row_count(self) -> int:
        return scalar_int(self.con, f"SELECT count(*) FROM {quote_ident(self.view)}")

    def sampled(self, n: int = 100_000, seed: int = 42) -> DataSource:
        """Cheap profiling on a big table: a reproducible reservoir sample as a second view."""
        v = f"{self.view}_sample"
        self.con.execute(
            f"CREATE OR REPLACE VIEW {quote_ident(v)} AS "
            f"SELECT * FROM {quote_ident(self.view)} "
            f"USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})"
        )
        return DataSource(self.con, v, origin=f"{self.origin} (sample {n})")
