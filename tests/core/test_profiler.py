import pandas as pd
import pytest
from dq_copilot.core.profiler import Profile, _kind, _num, profile
from dq_copilot.core.source import DataSource
from tests.conftest import inject_duplicates, inject_nulls, inject_outliers

MAX_CLEAN_FARE = 3 + 2.5 * 40  # clean_df clips distance at 40, so no clean fare exceeds 103


def profile_of(df: pd.DataFrame) -> Profile:
    return profile(DataSource.from_dataframe(df))


# ======================= tiny_df: exact hand-computed values =======================


def test_table_level_stats(tiny_df):
    p = profile_of(tiny_df)
    assert isinstance(p, Profile)
    assert p.row_count == 5
    assert p.duplicate_rows == 0
    assert list(p.columns) == ["id", "amount", "category"]


def test_duplicate_rows_counted(tiny_df):
    df = pd.concat([tiny_df, tiny_df.iloc[[0]]], ignore_index=True)
    p = profile_of(df)
    assert p.row_count == 6
    assert p.duplicate_rows == 1


def test_duplicate_rows_treats_nulls_as_equal(tiny_df):
    df = pd.concat([tiny_df, tiny_df.iloc[[2]]], ignore_index=True)  # row 2 has a null amount
    assert profile_of(df).duplicate_rows == 1


def test_id_column(tiny_df):
    c = profile_of(tiny_df).columns["id"]
    assert c.dtype == "BIGINT"
    assert c.null_count == 0
    assert c.null_pct == 0.0
    assert c.distinct == 5
    assert c.uniqueness == pytest.approx(1.0)
    assert c.min == 1
    assert c.max == 5
    assert c.mean == pytest.approx(3.0)


def test_amount_column_with_null(tiny_df):
    c = profile_of(tiny_df).columns["amount"]
    assert c.dtype == "DOUBLE"
    assert c.null_count == 1
    assert c.null_pct == pytest.approx(0.2)
    assert c.distinct == 4
    assert c.uniqueness == pytest.approx(1.0)  # distinct / non-null rows
    assert c.min == pytest.approx(10.0)
    assert c.max == pytest.approx(50.0)
    assert c.mean == pytest.approx(30.0)  # 120 / 4, nulls ignored


def test_category_column(tiny_df):
    c = profile_of(tiny_df).columns["category"]
    assert c.dtype == "VARCHAR"
    assert c.distinct == 2
    assert c.uniqueness == pytest.approx(0.4)  # 2 / 5
    assert c.min == "a"
    assert c.max == "b"
    assert c.mean is None  # stats only for numerics
    assert c.std is None


def test_awkward_column_name(tiny_df):
    df = tiny_df.rename(columns={"amount": 'we"ird name'})
    assert profile_of(df).columns['we"ird name'].null_count == 1


def test_parquet_matches_dataframe(tiny_df, tmp_path):
    path = tmp_path / "tiny.parquet"
    tiny_df.to_parquet(path, index=False)
    from_file = profile(DataSource.from_path(path))
    assert from_file.model_dump() == profile_of(tiny_df).model_dump()


# ======================= edge_df: type-specific extras =======================


def test_string_extras(edge_df):
    c = profile_of(edge_df).columns["name"]
    assert c.null_count == 1
    assert c.empty_count == 2  # '' and ' ' (trimmed)
    assert c.negative_count is None
    assert c.future_count is None


def test_numeric_extras(edge_df):
    c = profile_of(edge_df).columns["delta"]
    assert c.null_count == 1
    assert c.negative_count == 1
    assert c.zero_count == 2
    assert c.empty_count is None
    assert c.future_count is None


def test_date_extras(edge_df):
    c = profile_of(edge_df).columns["when"]
    assert c.null_count == 1
    assert c.future_count == 1  # 2200-01-01
    assert c.empty_count is None
    assert c.negative_count is None


def test_extras_not_computed_for_other_types(tiny_df):
    c = profile_of(tiny_df).columns["id"]
    assert c.empty_count is None
    assert c.future_count is None


# ======================= clean_df: baseline of a healthy table =======================


def test_clean_table_baseline(clean_df):
    p = profile_of(clean_df)
    assert p.row_count == 500
    assert p.duplicate_rows == 0
    assert set(p.columns) == set(clean_df.columns)
    assert all(c.null_count == 0 for c in p.columns.values())


def test_clean_key_column(clean_df):
    c = profile_of(clean_df).columns["trip_id"]
    assert c.distinct == 500
    assert c.uniqueness == pytest.approx(1.0)
    assert c.min == 0
    assert c.max == 499


def test_clean_column_types(clean_df):
    cols = profile_of(clean_df).columns
    assert cols["pickup_at"].dtype.startswith("TIMESTAMP")
    assert cols["payment_type"].dtype == "VARCHAR"
    assert cols["fare_amount"].dtype == "DOUBLE"
    assert _kind(cols["pickup_at"].dtype) == "date"


def test_clean_categoricals(clean_df):
    cols = profile_of(clean_df).columns
    assert cols["payment_type"].distinct == 3
    assert cols["vendor_id"].distinct == 2


def test_clean_numeric_ranges(clean_df):
    cols = profile_of(clean_df).columns
    assert 1 <= cols["passenger_count"].min <= cols["passenger_count"].max <= 4
    assert cols["fare_amount"].min >= 3.25 - 1e-9  # 3 + 2.5 * 0.1 (clip floor)
    assert cols["fare_amount"].max <= MAX_CLEAN_FARE
    assert cols["fare_amount"].negative_count == 0
    assert cols["fare_amount"].zero_count == 0
    assert cols["pickup_at"].future_count == 0


def test_clean_parquet_matches_dataframe(clean_df, tmp_path):
    path = tmp_path / "clean.parquet"
    clean_df.to_parquet(path, index=False)
    from_file = profile(DataSource.from_path(path))
    from_df = profile_of(clean_df)
    # compare counts only: timestamp type names can differ between the two paths
    assert from_file.row_count == from_df.row_count
    for name, col in from_df.columns.items():
        assert from_file.columns[name].null_count == col.null_count
        assert from_file.columns[name].distinct == col.distinct


# ======================= injected defects: ground truth from the helpers =======================


@pytest.mark.parametrize("col", ["fare_amount", "payment_type"])
def test_injected_nulls_are_counted(clean_df, col):
    bad, truth = inject_nulls(clean_df, col, frac=0.1, seed=1)
    p = profile_of(bad)

    assert p.columns[col].null_count == len(truth["rows"]) == 50
    assert p.columns[col].null_pct == pytest.approx(0.1)
    for other in p.columns:
        if other != col:
            assert p.columns[other].null_count == 0


def test_injected_outliers_shift_numeric_stats(clean_df):
    bad, truth = inject_outliers(clean_df, "fare_amount", factor=100, n=5, seed=1)
    clean_col = profile_of(clean_df).columns["fare_amount"]
    bad_col = profile_of(bad).columns["fare_amount"]

    assert len(truth["rows"]) == 5
    assert bad_col.max > MAX_CLEAN_FARE  # smallest possible outlier is 3.25 * 100
    assert bad_col.mean > clean_col.mean
    assert bad_col.std > clean_col.std
    assert bad_col.null_count == 0  # outliers are not nulls


def test_injected_duplicates_are_counted(clean_df):
    bad, truth = inject_duplicates(clean_df, "trip_id", n=5, seed=1)
    p = profile_of(bad)

    assert p.row_count == 505
    assert p.duplicate_rows == len(truth["rows"]) == 5
    key = p.columns["trip_id"]
    assert key.distinct == 500
    assert key.uniqueness == pytest.approx(500 / 505)  # no longer a clean key


# ======================= edge cases =======================


def test_all_null_column():
    df = pd.DataFrame({"x": pd.Series([None, None, None], dtype="float64")})
    c = profile_of(df).columns["x"]
    assert c.null_count == 3
    assert c.null_pct == pytest.approx(1.0)
    assert c.distinct == 0
    assert c.uniqueness is None  # no non-null rows: must not divide by zero


def test_empty_table():
    df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    p = profile_of(df)
    assert p.row_count == 0
    assert p.duplicate_rows == 0
    assert p.columns["a"].null_pct == 0.0  # must not divide by zero
    assert p.columns["a"].uniqueness is None


# ======================= helpers =======================


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("VARCHAR", "string"),
        ("BOOLEAN", "boolean"),
        ("BIGINT", "numeric"),
        ("INTEGER", "numeric"),
        ("DOUBLE", "numeric"),
        ("DECIMAL(18,3)", "numeric"),
        ("UBIGINT", "numeric"),
        ("DATE", "date"),
        ("TIMESTAMP", "date"),
        ("TIMESTAMP_NS", "date"),
        ("TIMESTAMP WITH TIME ZONE", "date"),
        ("BLOB", "other"),
        ("INTEGER[]", "other"),
        ("STRUCT(a INTEGER)", "other"),
        ("MAP(VARCHAR, INTEGER)", "other"),
    ],
)
def test_kind(dtype, expected):
    assert _kind(dtype) == expected


@pytest.mark.parametrize("raw, expected", [("12.5", 12.5), (3, 3.0), (None, None), ("abc", None)])
def test_num(raw, expected):
    assert _num(raw) == expected
