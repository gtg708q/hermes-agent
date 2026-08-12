import argparse

import pytest

from hermes_cli.main import (
    _apply_maintenance_chunk_rows,
    _maintenance_chunk_rows_arg,
)


@pytest.mark.parametrize("value", ["500", "50000", "100000"])
def test_maintenance_chunk_rows_accepts_bounded_values(value):
    assert _maintenance_chunk_rows_arg(value) == int(value)


@pytest.mark.parametrize("value", ["0", "499", "100001", "not-a-number"])
def test_maintenance_chunk_rows_rejects_unsafe_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _maintenance_chunk_rows_arg(value)


def test_apply_maintenance_chunk_rows_changes_only_requested_database():
    class FakeDB:
        _FTS_REBUILD_CHUNK_ROWS = 500

    first = FakeDB()
    second = FakeDB()

    _apply_maintenance_chunk_rows(first, 50_000)
    _apply_maintenance_chunk_rows(second, None)

    assert first._FTS_REBUILD_CHUNK_ROWS == 50_000
    assert second._FTS_REBUILD_CHUNK_ROWS == 500
    assert FakeDB._FTS_REBUILD_CHUNK_ROWS == 500
