"""
Parameter sweep runner for backtests with Cartesian product support.

Reads a JSON file with parameter combinations and runs backtests for each.
Supports both explicit format (base + sweep) and implicit format (lists in nested dicts).
"""

import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Tuple

from bot import backtest
from bot import backtest_store
from bot.backtest_params import BacktestParams
from bot import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("param_sweep")


def expand_sweep_params(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand sweep configuration into all parameter combinations.
    
    Supports two formats:
    1. Explicit format with 'sweep' and 'base':
       {
         "base": {"trend_existence": {"window": 20}, ...},
         "sweep": {
           "trend_existence.window": [20, 30, 40],
           "trend_quality.window": [10, 15]
         }
       }
    
    2. Implicit format (lists in nested dicts are treated as sweep values):
       {
         "trend_existence": {"window": [20, 30, 40], "ma_type": "ema"},
         "trend_quality": {"window": [10, 15]}
       }
    """
    # Check for explicit format
    if "sweep" in sweep_config and "base" in sweep_config:
        base = sweep_config["base"]
        sweep = sweep_config["sweep"]
        
        # Extract sweep keys and values
        sweep_keys = []
        sweep_values = []
        for key_path, values in sweep.items():
            if not isinstance(values, list):
                values = [values]  # Single value -> list
            sweep_keys.append(key_path)
            sweep_values.append(values)
        
        # Generate Cartesian product
        combinations = []
        for combo in itertools.product(*sweep_values):
            param_dict = _deep_copy_dict(base)
            for key_path, value in zip(sweep_keys, combo):
                _set_nested_value(param_dict, key_path, value)
            combinations.append(param_dict)
        
        return combinations
    
    # Implicit format: find all list values and generate combinations
    flat_sweep = _flatten_dict_with_lists(sweep_config)
    
    if not flat_sweep:
        # No lists found, return as single param set
        return [sweep_config]
    
    # Extract keys and values for product
    sweep_keys = list(flat_sweep.keys())
    sweep_values = [v if isinstance(v, list) else [v] for v in flat_sweep.values()]
    
    # Generate Cartesian product
    combinations = []
    for combo in itertools.product(*sweep_values):
        param_dict = _deep_copy_dict(sweep_config)
        for key_path, value in zip(sweep_keys, combo):
            _set_nested_value(param_dict, key_path, value)
        combinations.append(param_dict)
    
    return combinations


def _flatten_dict_with_lists(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict, collecting paths that have list values."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict_with_lists(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = v
        # else: scalar value, skip (will use from base/default)
    return items


def _set_nested_value(d: Dict[str, Any], key_path: str, value: Any, sep: str = ".") -> None:
    """Set a nested value using dot-notation path."""
    keys = key_path.split(sep)
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy a dictionary."""
    return json.loads(json.dumps(d))


def _has_list_values(d: Dict[str, Any]) -> bool:
    """Check if dict contains any list values (recursively)."""
    for v in d.values():
        if isinstance(v, list):
            return True
        if isinstance(v, dict):
            if _has_list_values(v):
                return True
    return False


def load_param_sweep_config(config_path: Path) -> List[Dict[str, Any]]:
    """
    Load parameter sweep configuration from JSON file.
    
    Supports multiple formats:
    1. List of param dicts: [{"trend_existence": {...}}, ...]
    2. Single param dict: {"trend_existence": {...}}
    3. Sweep config with base+sweep: {"base": {...}, "sweep": {...}}
    4. Implicit sweep (lists in nested dicts): {"trend_existence": {"window": [20, 30]}}
    """
    data = json.loads(config_path.read_text(encoding="utf-8"))
    
    if isinstance(data, list):
        # List of param sets (no sweep, just multiple explicit sets)
        return data
    elif isinstance(data, dict):
        # Check if it's a sweep config or single param set
        if "sweep" in data or _has_list_values(data):
            # Sweep mode: expand combinations
            return expand_sweep_params(data)
        else:
            # Single param set
            return [data]
    else:
        raise ValueError(f"Invalid param.json format: expected dict, list of dicts, or sweep config")


def _run_single_backtest(args_tuple: Tuple[int, Dict[str, Any], List[str], str, str, bool]) -> Dict:
    """
    Worker function for parallel execution.
    
    Args:
        args_tuple: (idx, param_dict, symbols, start, end, overwrite)
    
    Returns:
        Backtest result dictionary
    """
    idx, param_dict, symbols, start, end, overwrite = args_tuple
    
    try:
        # Run backtest (workers never write runs.jsonl)
        result = backtest.run_backtest(
            symbols,
            start,
            end,
            param_dict,
            run_id=None,  # Use deterministic run_id based on param_hash
            overwrite=overwrite,
            write_run_index=False,
        )
        return {
            "ok": True,
            "run_id": result.get("run_id"),
            "run_index_record": result.get("run_index_record"),
            "error": None,
            "result": result,
            "param_index": idx,
        }
    except Exception as exc:
        return {
            "ok": False,
            "run_id": None,
            "run_index_record": None,
            "error": str(exc),
            "result": None,
            "param_index": idx,
        }


def run_param_sweep(
    param_json_path: str | Path,
    start: str,
    end: str,
    symbols: str | List[str],
    overwrite: bool = False,
    workers: int = 1,
) -> List[Dict]:
    """
    Run backtest for each parameter combination in param.json.
    
    Args:
        param_json_path: Path to parameter JSON file
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        symbols: Comma-separated symbols or list of symbols
        overwrite: Whether to overwrite existing runs
        workers: Number of parallel workers (1 = sequential, >1 = parallel)
    
    Returns:
        List of backtest result dictionaries
    """
    param_path = Path(param_json_path)
    if not param_path.exists():
        raise FileNotFoundError(f"Param file not found: {param_path}")
    
    param_list = load_param_sweep_config(param_path)
    log.info("Loaded %d parameter combinations from %s", len(param_list), param_path)
    
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    runs_jsonl_path = Path(config.BACKTEST_RUNS_JSONL)
    results_dict: Dict[int, Dict[str, Any]] = {}
    completed_count = 0

    def _log_params(idx: int, param_dict: Dict[str, Any]) -> None:
        complete_params, unapplied = BacktestParams.validate_and_materialize(param_dict)
        if unapplied:
            raise ValueError(f"Unapplied params detected: {unapplied}")
        log.info("=" * 80)
        log.info("Running parameter set %d/%d", idx + 1, len(param_list))

        default_params = BacktestParams.default_params_dict()
        diff_params = {}
        for k, v in complete_params.items():
            if k == "schema_version":
                continue
            default_v = default_params.get(k)
            if v != default_v:
                if isinstance(v, dict) and isinstance(default_v, dict):
                    if json.dumps(v, sort_keys=True) != json.dumps(default_v, sort_keys=True):
                        diff_params[k] = v
                else:
                    diff_params[k] = v

        if diff_params:
            log.info("Params (non-defaults): %s", json.dumps(diff_params, indent=2, ensure_ascii=False))

    def _handle_worker_result(worker_result: Dict[str, Any], idx: int) -> None:
        nonlocal completed_count
        completed_count += 1
        if worker_result.get("ok"):
            result = worker_result.get("result") or {}
            record = worker_result.get("run_index_record")
            if record:
                backtest_store.upsert_run_index_record(runs_jsonl_path, record)
            elif result.get("status") == "multi":
                for run in result.get("runs", []):
                    run_record = run.get("run_index_record")
                    if run_record:
                        backtest_store.upsert_run_index_record(runs_jsonl_path, run_record)
            results_dict[idx] = result
            if result.get("status") == "skipped":
                log.info("[%d/%d] Run skipped (already exists): %s",
                         completed_count, len(param_list), result.get("run_id"))
            else:
                log.info("[%d/%d] Run completed: %s",
                         completed_count, len(param_list), result.get("run_id"))
        else:
            log.error(
                "[%d/%d] Run failed with exception: %s",
                idx + 1,
                len(param_list),
                worker_result.get("error"),
            )
            results_dict[idx] = {
                "status": "failed",
                "error": worker_result.get("error"),
                "param_index": idx,
            }

    if workers == 1:
        for idx, param_dict in enumerate(param_list):
            _log_params(idx, param_dict)
            worker_result = _run_single_backtest(
                (idx, param_dict, symbols, start, end, overwrite)
            )
            _handle_worker_result(worker_result, idx)
        return [results_dict[i] for i in range(len(param_list))]

    log.info("Running with %d parallel workers", workers)

    tasks = [
        (idx, param_dict, symbols, start, end, overwrite)
        for idx, param_dict in enumerate(param_list)
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_run_single_backtest, task): idx
            for idx, task in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            worker_result = future.result()
            _handle_worker_result(worker_result, idx)

    return [results_dict[i] for i in range(len(param_list))]


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Run parameter sweep from param.json with Cartesian product support"
    )
    ap.add_argument("--params", required=True, help="Path to param.json file")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--symbols", default="BTC", help="Comma-separated symbols")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing runs")
    ap.add_argument("--workers", type=int, default=1, 
                    help="Number of parallel workers (default: 1, sequential)")
    args = ap.parse_args()
    
    results = run_param_sweep(
        args.params,
        args.start,
        args.end,
        args.symbols,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    
    log.info("=" * 80)
    log.info("Parameter sweep completed: %d runs", len(results))
    for result in results:
        log.info("  - %s: %s", result.get("run_id"), result.get("status"))


if __name__ == "__main__":
    main()
