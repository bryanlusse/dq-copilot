from decimal import Decimal
from typing import Any

import duckdb
from pydantic import BaseModel

from dq_copilot.core.source import DataSource, quote_ident, scalar_int


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    distinct: int
    uniqueness: float | None
    min: float | str | None = None
    max: float | str | None = None
    mean: float | None = None
    std: float | None = None
    empty_count: int | None = None  # strings
    negative_count: int | None = None  # numerics
    zero_count: int | None = None  # numerics
    future_count: int | None = None  # dates
    top_values: list[tuple[str, int]] = []
    patterns: list[tuple[str, int]] = []
    flags: list[str] = []


class Profile(BaseModel):
    row_count: int
    duplicate_rows: int
    columns: dict[str, ColumnProfile]


def _fetch_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row, strict=False)) for row in cur.fetchall()]


def _kind(dtype: str) -> str:
    d = dtype.upper()

    # nested types first: "INTEGER[]" and "STRUCT(a INTEGER)" would otherwise match numeric prefixes
    if "[" in d or d.startswith(("STRUCT", "MAP", "UNION", "LIST", "ENUM")):
        return "other"

    if d == "VARCHAR":
        return "string"
    if d == "BOOLEAN":
        return "boolean"
    if d.startswith(("DATE", "TIMESTAMP")):
        return "date"
    if d.startswith(
        (
            "DECIMAL",
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "FLOAT",
            "DOUBLE",
        )
    ):
        return "numeric"
    return "other"


def _num(x: object) -> float | None:
    """SUMMARIZE returns most stats as text, so convert safely."""
    if isinstance(x, int | float | str | Decimal):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def profile(src: DataSource) -> Profile:
    v = quote_ident(src.view)

    # table-level
    row_count = scalar_int(src.con, f"SELECT count(*) FROM {v}")
    distinct_rows = scalar_int(src.con, f"SELECT count(*) FROM (SELECT DISTINCT * FROM {v})")
    duplicate_rows = row_count - distinct_rows

    # SUMMARIZE: base stats for every column, keyed by column name
    summary = {r["column_name"]: r for r in _fetch_dicts(src.con, f"SUMMARIZE {v}")}

    columns: dict[str, ColumnProfile] = {}
    for name, dtype in src.columns().items():
        c = quote_ident(name)
        kind = _kind(dtype)
        base = summary[name]

        # exact extras: one combined query per column
        parts = [
            f"count({c}) AS non_null_count",
            f"count(*) - count({c}) AS null_count",
            f"count(DISTINCT {c}) AS distinct_count",
        ]
        if kind == "string":
            parts.append(f"count(*) FILTER (WHERE trim({c}) = '') AS empty_count")
        elif kind == "numeric":
            parts.append(f"count(*) FILTER (WHERE {c} < 0) AS negative_count")
            parts.append(f"count(*) FILTER (WHERE {c} = 0) AS zero_count")
        elif kind == "date":
            parts.append(f"count(*) FILTER (WHERE {c} > current_date) AS future_count")

        extras = _fetch_dicts(src.con, f"SELECT {', '.join(parts)} FROM {v}")[0]

        non_null = extras["non_null_count"]
        numeric = kind == "numeric"

        columns[name] = ColumnProfile(
            name=name,
            dtype=dtype,
            null_count=extras["null_count"],
            null_pct=extras["null_count"] / row_count if row_count else 0.0,
            distinct=extras["distinct_count"],
            uniqueness=extras["distinct_count"] / non_null if non_null else None,
            min=_num(base["min"]) if numeric else base["min"],
            max=_num(base["max"]) if numeric else base["max"],
            mean=_num(base["avg"]) if numeric else None,
            std=_num(base["std"]) if numeric else None,
            empty_count=extras.get("empty_count"),
            negative_count=extras.get("negative_count"),
            zero_count=extras.get("zero_count"),
            future_count=extras.get("future_count"),
        )

    return Profile(row_count=row_count, duplicate_rows=duplicate_rows, columns=columns)
