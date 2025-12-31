import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bot import backtest_store
from bot import config
from bot import param_sweep


def _read_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["run_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_param_sweep_parallel_writes_runs_index_in_main(tmp_path, monkeypatch) -> None:
    results_dir = tmp_path / "backtest_result"
    monkeypatch.setattr(config, "BACKTEST_RESULT_DIR", str(results_dir))
    monkeypatch.setattr(param_sweep, "ProcessPoolExecutor", ThreadPoolExecutor)

    calls: list[tuple[str, str]] = []
    original_upsert = backtest_store.upsert_run_index_record

    def _recording_upsert(index_path: Path, record: dict) -> None:
        calls.append((threading.current_thread().name, record.get("run_id")))
        original_upsert(index_path, record)

    monkeypatch.setattr(backtest_store, "upsert_run_index_record", _recording_upsert)

    param_path = Path("tests/fixtures/params_sweep_smoke.json")
    results = param_sweep.run_param_sweep(
        param_path,
        "2024-12-01",
        "2025-12-20",
        "BTCTEST",
        overwrite=True,
        workers=2,
    )

    run_ids = {result.get("run_id") for result in results if result.get("run_id")}
    runs_index_path = results_dir / "runs.jsonl"

    assert run_ids
    assert _read_run_ids(runs_index_path) == run_ids
    assert len(calls) == len(run_ids)
    assert all(thread_name == "MainThread" for thread_name, _ in calls)
