import json
import logging
from pathlib import Path

from bot import backtest_store


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_upsert_run_index_creates_file(tmp_path: Path) -> None:
    index_path = tmp_path / "runs.jsonl"
    record = {"run_id": "run-1", "param_hash": "aaa", "data_fingerprint": "bbb"}

    backtest_store.upsert_run_index_record(index_path, record)

    lines = _read_jsonl(index_path)
    assert lines == [record]


def test_upsert_run_index_appends_new_run_id(tmp_path: Path) -> None:
    index_path = tmp_path / "runs.jsonl"
    record_a = {"run_id": "run-1", "param_hash": "aaa", "data_fingerprint": "bbb"}
    record_b = {"run_id": "run-2", "param_hash": "ccc", "data_fingerprint": "ddd"}

    backtest_store.upsert_run_index_record(index_path, record_a)
    backtest_store.upsert_run_index_record(index_path, record_b)

    lines = _read_jsonl(index_path)
    assert lines == [record_a, record_b]


def test_upsert_run_index_replaces_existing_run_id(tmp_path: Path) -> None:
    index_path = tmp_path / "runs.jsonl"
    record_a = {"run_id": "run-1", "param_hash": "aaa", "data_fingerprint": "bbb"}
    record_b = {"run_id": "run-1", "param_hash": "ccc", "data_fingerprint": "ddd"}

    backtest_store.upsert_run_index_record(index_path, record_a)
    backtest_store.upsert_run_index_record(index_path, record_b)

    lines = _read_jsonl(index_path)
    assert lines == [record_b]


def test_upsert_run_index_ignores_bad_lines(tmp_path: Path, caplog) -> None:
    index_path = tmp_path / "runs.jsonl"
    index_path.write_text('{"run_id":"old"}\n{bad json\n\n', encoding="utf-8")
    record = {"run_id": "run-2", "param_hash": "ccc", "data_fingerprint": "ddd"}

    with caplog.at_level(logging.WARNING):
        backtest_store.upsert_run_index_record(index_path, record)

    lines = _read_jsonl(index_path)
    assert lines == [{"run_id": "old"}, record]
    assert "Ignored 1 malformed runs.jsonl lines" in caplog.text
