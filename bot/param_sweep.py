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


def merge_with_defaults(param_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Merge param_dict with config defaults to create complete BacktestParams."""
    # Build default params from config
    default_params = {
        "schema_version": 1,
        "timeframes": dict(config.TIMEFRAMES),
        "trend_existence": dict(config.TREND_EXISTENCE),
        "trend_quality": dict(config.TREND_QUALITY),
        "execution": dict(config.EXECUTION),
        "angle_sizing_enabled": config.ANGLE_SIZING_ENABLED,
        "angle_sizing_a": config.ANGLE_SIZING_A,
        "angle_sizing_q": config.ANGLE_SIZING_Q,
        "vol_window_div": config.VOL_WINDOW_DIV,
        "vol_window_min": config.VOL_WINDOW_MIN,
        "vol_window_max": config.VOL_WINDOW_MAX,
        "vol_eps": config.VOL_EPS,
        "direction_mode": config.DIRECTION_MODE,
        "max_long_frac": 1.0,
        "max_short_frac": 0.25,
        "starting_cash_usdc_per_symbol": config.STARTING_CASH_USDC_PER_SYMBOL,
        "taker_fee_bps": config.TAKER_FEE_BPS,
        "min_trade_notional_pct": config.MIN_TRADE_NOTIONAL_PCT,
    }
    
    # Deep merge: param_dict overrides defaults
    def deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
        return result
    
    return deep_merge(default_params, param_dict)


def _run_single_backtest(args_tuple: Tuple[int, Dict[str, Any], List[str], str, str, bool]) -> Dict:
    """
    Worker function for parallel execution.
    
    Args:
        args_tuple: (idx, param_dict, symbols, start, end, overwrite)
    
    Returns:
        Backtest result dictionary
    """
    idx, param_dict, symbols, start, end, overwrite = args_tuple
    
    # Merge with defaults to ensure complete params
    complete_params = merge_with_defaults(param_dict)
    params = BacktestParams.from_dict(complete_params)
    
    # Run backtest
    result = backtest.run_backtest(
        symbols,
        start,
        end,
        params,
        run_id=None,  # Use deterministic run_id based on param_hash
        overwrite=overwrite,
    )
    
    return result


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
    
    if workers == 1:
        # Sequential execution (original behavior)
        results = []
        for idx, param_dict in enumerate(param_list):
            # Merge with defaults to ensure complete params
            complete_params = merge_with_defaults(param_dict)
            
            log.info("=" * 80)
            log.info("Running parameter set %d/%d", idx + 1, len(param_list))
            
            # Show only non-default values for clarity
            default_params = merge_with_defaults({})
            diff_params = {}
            for k, v in complete_params.items():
                if k == "schema_version":
                    continue
                default_v = default_params.get(k)
                if v != default_v:
                    if isinstance(v, dict) and isinstance(default_v, dict):
                        # Compare nested dicts
                        if json.dumps(v, sort_keys=True) != json.dumps(default_v, sort_keys=True):
                            diff_params[k] = v
                    else:
                        diff_params[k] = v
            
            if diff_params:
                log.info("Params (non-defaults): %s", json.dumps(diff_params, indent=2, ensure_ascii=False))
            
            params = BacktestParams.from_dict(complete_params)
            result = backtest.run_backtest(
                symbols,
                start,
                end,
                params,
                run_id=None,  # Use deterministic run_id based on param_hash
                overwrite=overwrite,
            )
            results.append(result)
            
            if result.get("status") == "skipped":
                log.info("Run skipped (already exists): %s", result.get("run_id"))
            else:
                log.info("Run completed: %s", result.get("run_id"))
        
        return results
    else:
        # Parallel execution
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        
        log.info("Running with %d parallel workers", workers)
        
        # Prepare tasks
        tasks = [
            (idx, param_dict, symbols, start, end, overwrite)
            for idx, param_dict in enumerate(param_list)
        ]
        
        # Execute in parallel
        results_dict = {}
        completed_count = 0
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_run_single_backtest, task): idx
                for idx, task in enumerate(tasks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results_dict[idx] = result
                    completed_count += 1
                    
                    if result.get("status") == "skipped":
                        log.info("[%d/%d] Run skipped (already exists): %s", 
                                completed_count, len(param_list), result.get("run_id"))
                    else:
                        log.info("[%d/%d] Run completed: %s", 
                                completed_count, len(param_list), result.get("run_id"))
                except Exception as exc:
                    log.error("[%d/%d] Run failed with exception: %s", idx + 1, len(param_list), exc)
                    results_dict[idx] = {
                        "status": "failed",
                        "error": str(exc),
                        "param_index": idx,
                    }
        
        # Return results in original order
        results = [results_dict[i] for i in range(len(param_list))]
        return results


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

