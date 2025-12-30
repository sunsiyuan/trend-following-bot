"""
backtest.py

Backtest runner that is strictly "strategy-driven":
- loads market data (cached jsonl; downloads if missing)
- computes indicators (via strategy.prepare_features_*)
- iterates execution bars (default 4h)
- calls strategy.decide(...) to get target position fraction
- simulates rebalancing with a simple cash + position model
- writes results:
    data/backtest_result/{run_id}/{symbol}/summary.json
    data/backtest_result/{run_id}/{symbol}/equity_by_day.csv
    data/backtest_result/{run_id}/{symbol}/trades.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from bot import config
from bot import data_client
from bot import metrics
from bot import backtest_store
from bot import execution_policy
from bot.backtest_params import BacktestParams, calc_param_hash
from bot.quarterly_stats import generate_quarterly_stats
from bot import strategy as strat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("backtest")


def parse_date_to_ms(s: str) -> int:
    """
    Parse 'YYYY-MM-DD' (treated as UTC 00:00) into epoch ms.
    """
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def ms_to_ymd(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

def normalize_date_range(start: str | int, end: str | int) -> Tuple[int, int, str, str]:
    if isinstance(start, str):
        start_ms = parse_date_to_ms(start)
        start_label = start
    else:
        start_ms = int(start)
        start_label = ms_to_ymd(start_ms)

    if isinstance(end, str):
        end_ms = parse_date_to_ms(end) + 24 * 60 * 60 * 1000
        end_label = end
    else:
        end_ms = int(end)
        end_label = ms_to_ymd(end_ms - 1)

    if end_ms <= start_ms:
        raise ValueError("end must be greater than start")
    return start_ms, end_ms, start_label, end_label

def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def append_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")

def compute_trade_decision_counts(trades: List[Dict]) -> Dict:
    # Count decision/action outcomes from trade records.
    decision_count = len(trades)
    rebalance_count = 0
    hold_count = 0
    noop_small_delta_count = 0
    rebalance_by_reason: Dict[str, int] = {}
    hold_by_reason: Dict[str, int] = {}

    for trade in trades:
        action = trade.get("action")
        reason = trade.get("reason", "unknown")
        if trade.get("trade_intent") == "NOOP_SMALL_DELTA":
            noop_small_delta_count += 1
        if action == "REBALANCE":
            rebalance_count += 1
            rebalance_by_reason[reason] = rebalance_by_reason.get(reason, 0) + 1
        elif action == "HOLD":
            hold_count += 1
            hold_by_reason[reason] = hold_by_reason.get(reason, 0) + 1

    return {
        "decision_count": decision_count,
        "rebalance_count": rebalance_count,
        "hold_count": hold_count,
        "trade_count": rebalance_count,
        "noop_small_delta_count": noop_small_delta_count,
        "rebalance_by_reason": rebalance_by_reason,
        "hold_by_reason": hold_by_reason,
    }

def compute_diagnostic_counts(
    trades_jsonl_path: Path,
    equity_by_day_csv_path: Path,
    direction_mode: str,
) -> Tuple[Dict, List[str]]:
    warnings: List[str] = []
    trades_by_day: Dict[str, Dict] = {}

    def warn_once(msg: str) -> None:
        if msg not in warnings:
            warnings.append(msg)

    def parse_ts_to_day(ts_val: int | float | str) -> str | None:
        try:
            ts_num = float(ts_val)
        except (TypeError, ValueError):
            return None
        if ts_num > 1e12:
            ts_ms = int(ts_num)
        else:
            ts_ms = int(ts_num * 1000)
        return ms_to_ymd(ts_ms)

    # Latest trade per day is used for daily diagnostics.
    if trades_jsonl_path.exists():
        with trades_jsonl_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    warn_once("Failed to parse trades.jsonl; diagnostics may be incomplete.")
                    continue
                day = record.get("date_utc")
                if not day:
                    ts_guess = record.get("ts_ms") or record.get("ts_utc")
                    day = parse_ts_to_day(ts_guess) if ts_guess is not None else None
                if not day:
                    warn_once("Missing date_utc/ts_ms in trades.jsonl; skipping some diagnostics.")
                    continue
                ts_sort = record.get("ts_ms") or record.get("ts_utc") or idx
                prev = trades_by_day.get(day)
                if prev is None or ts_sort >= prev.get("_ts_sort", -1):
                    record["_ts_sort"] = ts_sort
                    trades_by_day[day] = record
    else:
        warn_once("Missing trades.jsonl; diagnostics may be incomplete.")

    days_total = 0
    flat_by_day: Dict[str, bool] = {}
    dates: List[str] = []
    # Use equity_by_day for coverage + net exposure; fallback to trades if missing.
    if equity_by_day_csv_path.exists():
        df_day = pd.read_csv(equity_by_day_csv_path)
        if "date_utc" in df_day.columns:
            dates = df_day["date_utc"].dropna().astype(str).tolist()
            days_total = len(pd.unique(pd.Series(dates)))
        else:
            warn_once("Missing date_utc in equity_by_day.csv; falling back to trades.")
        if "net_exposure" in df_day.columns and "date_utc" in df_day.columns:
            net_exposure = pd.to_numeric(df_day["net_exposure"], errors="coerce").fillna(0.0)
            for day, exposure in zip(df_day["date_utc"].astype(str), net_exposure, strict=False):
                flat_by_day[day] = abs(float(exposure)) <= 1e-12
        else:
            warn_once("Missing net_exposure in equity_by_day.csv; inferring flat days from trades.")
    else:
        warn_once("Missing equity_by_day.csv; diagnostics may be incomplete.")

    if not dates:
        dates = sorted(trades_by_day.keys())
        days_total = len(dates)

    raw_dir_long_days = 0
    raw_dir_short_days = 0
    flat_days_total = 0
    flat_due_to_long_only_days = 0
    flat_due_to_execution_gate_days = 0
    flat_due_to_other_days = 0
    target_frac_days_0 = 0
    target_frac_days_0_5 = 0
    target_frac_days_1 = 0
    target_frac_days_other = 0
    days_align_ge_0_3 = 0
    days_align_ge_0_7 = 0
    abs_target_sum = 0.0
    abs_target_days = 0

    eps = 1e-6
    missing_trade_days = 0
    has_equity_flat = bool(flat_by_day)
    if has_equity_flat:
        flat_days_total = int(sum(1 for value in flat_by_day.values() if value))

    for day in dates:
        record = trades_by_day.get(day)
        if record is None:
            missing_trade_days += 1
            target_frac_days_other += 1
            if flat_by_day.get(day):
                flat_due_to_other_days += 1
            continue

        raw_dir = record.get("raw_dir")
        raw_dir = str(raw_dir) if raw_dir is not None else None
        if raw_dir == "LONG":
            raw_dir_long_days += 1
        elif raw_dir == "SHORT":
            raw_dir_short_days += 1
        elif raw_dir is None:
            warn_once("Missing raw_dir in trades.jsonl; raw_dir counts may be low.")

        align_val = record.get("align")
        if align_val is None:
            warn_once("Missing align in trades.jsonl; align diagnostics may be low.")
        else:
            try:
                align_val = float(align_val)
            except (TypeError, ValueError):
                align_val = None
                warn_once("Invalid align in trades.jsonl; align diagnostics may be low.")
        if align_val is not None:
            if align_val >= 0.3:
                days_align_ge_0_3 += 1
            if align_val >= 0.7:
                days_align_ge_0_7 += 1

        target_frac = record.get("target_pos_frac")
        if target_frac is None:
            target_frac_days_other += 1
            warn_once("Missing target_pos_frac in trades.jsonl; target histogram may be incomplete.")
        else:
            try:
                target_frac = float(target_frac)
            except (TypeError, ValueError):
                target_frac = None
                target_frac_days_other += 1
                warn_once("Invalid target_pos_frac in trades.jsonl; target histogram may be incomplete.")
        if target_frac is None:
            pass
        elif abs(target_frac) <= eps:
            target_frac_days_0 += 1
        elif abs(target_frac - 0.5) <= eps:
            target_frac_days_0_5 += 1
        elif abs(target_frac - 1.0) <= eps:
            target_frac_days_1 += 1
        else:
            target_frac_days_other += 1
        if target_frac is not None:
            abs_target_sum += abs(float(target_frac))
            abs_target_days += 1

        flat_flag = flat_by_day.get(day)
        if flat_flag is None:
            pos_frac = record.get("position_frac_after")
            pos_qty = record.get("position_qty_after")
            if pos_frac is not None:
                flat_flag = abs(float(pos_frac)) <= eps
            elif pos_qty is not None:
                flat_flag = abs(float(pos_qty)) <= eps
            else:
                flat_flag = False
                warn_once("Missing net_exposure and position fields; flat day counts may be low.")

        if not flat_flag:
            continue
        if not has_equity_flat:
            flat_days_total += 1

        reason = str(record.get("reason", "")).lower()
        market_state = record.get("market_state")
        market_state = str(market_state) if market_state is not None else ""
        record_direction_mode = record.get("direction_mode", direction_mode)
        target_non_zero = target_frac is not None and abs(target_frac) > eps

        if record_direction_mode == "long_only" and market_state == "SHORT":
            flat_due_to_long_only_days += 1
        elif target_non_zero and ("execution_gate" in reason or "gate_blocked" in reason):
            flat_due_to_execution_gate_days += 1
        else:
            flat_due_to_other_days += 1

    raw_dir_days_covered = raw_dir_long_days + raw_dir_short_days
    target_frac_days_covered = (
        target_frac_days_0
        + target_frac_days_0_5
        + target_frac_days_1
        + target_frac_days_other
    )
    if days_total and raw_dir_days_covered < 0.8 * days_total:
        warn_once(
            "Diagnostics coverage issue: raw_dir coverage below 80% of days_total."
        )

    return {
        "raw_dir_long_days": raw_dir_long_days,
        "raw_dir_short_days": raw_dir_short_days,
        "flat_days_total": flat_days_total,
        "flat_due_to_long_only_days": flat_due_to_long_only_days,
        "flat_due_to_execution_gate_days": flat_due_to_execution_gate_days,
        "flat_due_to_other_days": flat_due_to_other_days,
        "target_frac_days_0": target_frac_days_0,
        "target_frac_days_0_5": target_frac_days_0_5,
        "target_frac_days_1": target_frac_days_1,
        "target_frac_days_other": target_frac_days_other,
        "days_align_ge_0_3": days_align_ge_0_3,
        "days_align_ge_0_7": days_align_ge_0_7,
        "avg_abs_target": (abs_target_sum / abs_target_days) if abs_target_days else 0.0,
        "diagnostics_sanity": {
            "raw_dir_days_covered": raw_dir_days_covered,
            "target_frac_days_covered": target_frac_days_covered,
        },
    }, warnings

def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)

def update_avg_and_realized(q0: float, avg0: float, dq: float, price: float) -> Tuple[float, float]:
    """
    Maintain average entry price and compute realized pnl for the closed portion.
    q0: existing position qty (signed)
    avg0: avg entry price for existing position (positive)
    dq: delta qty (signed)
    price: trade price
    Returns (new_avg, realized_pnl)
    """
    q1 = q0 + dq
    if abs(q0) < 1e-12:
        # opening a new position
        return (abs(price), 0.0) if abs(q1) > 1e-12 else (0.0, 0.0)

    # same direction?
    if q0 * q1 > 0:
        if abs(q1) > abs(q0):
            # adding
            add_qty = abs(q1) - abs(q0)
            # weighted avg by absolute qty
            new_avg = (abs(q0) * avg0 + add_qty * price) / abs(q1)
            return new_avg, 0.0
        else:
            # reducing
            closed_qty = abs(q0) - abs(q1)
            realized = closed_qty * (price - avg0) * sign(q0)
            return avg0, realized

    # direction flip
    realized_close = abs(q0) * (price - avg0) * sign(q0)
    if abs(q1) < 1e-12:
        return 0.0, realized_close
    # new position opened in opposite direction at this price
    return price, realized_close

def build_params_from_config() -> BacktestParams:
    return BacktestParams(
        timeframes=dict(config.TIMEFRAMES),
        trend_existence=dict(config.TREND_EXISTENCE),
        trend_quality=dict(config.TREND_QUALITY),
        execution=dict(config.EXECUTION),
        angle_sizing_enabled=bool(config.ANGLE_SIZING_ENABLED),
        angle_sizing_a=float(config.ANGLE_SIZING_A),
        angle_sizing_q=float(config.ANGLE_SIZING_Q),
        vol_window_div=float(config.VOL_WINDOW_DIV),
        vol_window_min=int(config.VOL_WINDOW_MIN),
        vol_window_max=int(config.VOL_WINDOW_MAX),
        vol_eps=float(config.VOL_EPS),
        direction_mode=config.DIRECTION_MODE,
        max_long_frac=float(config.MAX_LONG_FRAC),
        max_short_frac=float(config.MAX_SHORT_FRAC),
        starting_cash_usdc_per_symbol=float(config.STARTING_CASH_USDC_PER_SYMBOL),
        taker_fee_bps=float(config.TAKER_FEE_BPS),
        min_trade_notional_pct=float(config.MIN_TRADE_NOTIONAL_PCT),
    )


def compute_default_run_id(
    symbol_label: str,
    start_label: str,
    end_label: str,
    param_hash: str,
    data_fingerprint: str,
) -> str:
    return (
        f"{symbol_label}__{start_label}__{end_label}__"
        f"{param_hash[:8]}__{data_fingerprint[:8]}"
    )


def run_backtest(
    symbol: str | List[str],
    start: str | int,
    end: str | int,
    params: BacktestParams | Dict,
    run_id: str | None = None,
    runs_jsonl_path: Path | None = None,
    overwrite: bool = False,
) -> Dict:
    """
    Callable backtest entrypoint for reproducible runs.
    start/end use start-inclusive, end-exclusive semantics.
    """
    symbols = [symbol] if isinstance(symbol, str) else list(symbol)
    if not symbols:
        raise ValueError("symbol is required")
    if len(symbols) > 1:
        if run_id:
            raise ValueError("run_id must be None when running multiple symbols")
        runs = [
            run_backtest(
                sym,
                start,
                end,
                params,
                run_id=None,
                runs_jsonl_path=runs_jsonl_path,
                overwrite=overwrite,
            )
            for sym in symbols
        ]
        return {"runs": runs, "status": "multi"}

    start_ms, end_ms, start_label, end_label = normalize_date_range(start, end)

    params_obj = params if isinstance(params, BacktestParams) else BacktestParams.from_dict(params)
    params_hashable = params_obj.to_hashable_dict()
    param_hash = calc_param_hash(params_hashable)

    per_symbol_summaries: List[Dict] = []
    per_symbol_manifests: Dict[str, List[Dict]] = {}
    data_by_symbol: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    end_fetch_ms = end_ms - 1
    ms_trend = data_client.interval_to_ms(params_obj.timeframes["trend"])
    ms_exec = data_client.interval_to_ms(params_obj.timeframes["execution"])
    warmup_1d = max(params_obj.trend_existence["window"], params_obj.trend_quality["window"]) + 10
    warmup_exec_steps = max(
        params_obj.execution["build_min_step_bars"],
        params_obj.execution["reduce_min_step_bars"],
    )
    warmup_exec = params_obj.execution["window"] + warmup_exec_steps + 10

    sym = symbols[0]
    log.info("Backtesting %s from %s to %s ...", sym, start_label, end_label)
    fetch_start = min(start_ms - warmup_1d * ms_trend, start_ms - warmup_exec * ms_exec)
    fetch_start = max(0, fetch_start)

    df_trend = data_client.ensure_market_data(
        sym,
        params_obj.timeframes["trend"],
        fetch_start,
        end_fetch_ms,
    )
    df_exec = data_client.ensure_market_data(
        sym,
        params_obj.timeframes["execution"],
        fetch_start,
        end_fetch_ms,
    )

    sliced_trend = backtest_store.slice_bars(df_trend, start_ms, end_ms)
    sliced_exec = backtest_store.slice_bars(df_exec, start_ms, end_ms)

    manifest_trend = backtest_store.build_data_manifest(
        params_obj.timeframes["trend"],
        start_ms,
        end_ms,
        sliced_trend,
    )
    manifest_exec = backtest_store.build_data_manifest(
        params_obj.timeframes["execution"],
        start_ms,
        end_ms,
        sliced_exec,
    )

    per_symbol_manifests[sym] = [manifest_trend, manifest_exec]
    data_by_symbol[sym] = (df_trend, df_exec)

    manifest_flat: List[Dict] = []
    for sym in sorted(per_symbol_manifests.keys()):
        manifest_flat.extend(per_symbol_manifests[sym])
    manifest_flat = sorted(
        manifest_flat,
        key=lambda item: (
            item.get("tf"),
            item.get("requested_start_ts"),
            item.get("requested_end_ts"),
            item.get("actual_first_ts") or -1,
            item.get("actual_last_ts") or -1,
            item.get("row_count"),
            item.get("expected_row_count"),
        ),
    )
    data_fingerprint = backtest_store.calc_data_fingerprint(manifest_flat)

    symbol_label = symbols[0]
    resolved_run_id = run_id or compute_default_run_id(
        symbol_label,
        start_label,
        end_label,
        param_hash,
        data_fingerprint,
    )
    run_dir = Path(config.BACKTEST_RESULT_DIR) / resolved_run_id
    run_record_path = run_dir / "run_record.json"
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            if run_record_path.exists():
                existing_record = json.loads(run_record_path.read_text(encoding="utf-8"))
                existing_param_hash = existing_record.get("param_hash")
                existing_data_fingerprint = existing_record.get("data_fingerprint")
                if existing_param_hash == param_hash and existing_data_fingerprint == data_fingerprint:
                    log.info("Run %s exists, skipped.", resolved_run_id)
                    return {
                        "run_id": resolved_run_id,
                        "start": start_label,
                        "end": end_label,
                        "symbols": symbols,
                        "param_hash": param_hash,
                        "data_fingerprint": data_fingerprint,
                        "data_manifest_by_symbol": per_symbol_manifests,
                        "per_symbol": [],
                        "status": "skipped",
                    }
                raise RuntimeError(
                    "Run ID conflict: existing run_record.json has "
                    f"param_hash={existing_param_hash}, data_fingerprint={existing_data_fingerprint} "
                    f"but requested param_hash={param_hash}, data_fingerprint={data_fingerprint}."
                )
            raise RuntimeError(
                f"Run directory {run_dir} already exists without run_record.json. "
                "Use overwrite to rerun."
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    df_trend, df_exec = data_by_symbol[sym]
    summary = run_backtest_for_symbol(
        sym,
        start_ms,
        end_ms,
        run_dir,
        params_obj,
        df_trend,
        df_exec,
        run_id=resolved_run_id,
        param_hash=param_hash,
        data_fingerprint=data_fingerprint,
        strategy_version=params_hashable.get("strategy_version"),
    )
    per_symbol_summaries.append(summary)

    if runs_jsonl_path is None:
        runs_jsonl_path = Path(config.BACKTEST_RESULT_DIR) / "runs.jsonl"

    record = {
        "run_id": resolved_run_id,
        "symbol_label": symbol_label,
        "symbols": symbols,
        "start": start_label,
        "end": end_label,
        "param_hash": param_hash,
        "data_fingerprint": data_fingerprint,
        "param_schema_version": params_obj.schema_version,
        "data_schema_version": backtest_store.data_schema_version,
        "strategy_version": params_hashable.get("strategy_version"),
        "params_hashable": params_hashable,
        "data_manifest_by_symbol": per_symbol_manifests,
        "data_manifest_by_tf": manifest_flat,
        "metrics": per_symbol_summaries,
    }
    write_json(run_record_path, {
        "run_id": resolved_run_id,
        "param_hash": param_hash,
        "data_fingerprint": data_fingerprint,
        "param_schema_version": params_obj.schema_version,
        "data_schema_version": backtest_store.data_schema_version,
        "strategy_version": params_hashable.get("strategy_version"),
        "params_hashable": params_hashable,
        "data_manifest_by_tf": manifest_flat,
        "data_manifest_by_symbol": per_symbol_manifests,
    })
    write_json(run_dir / "summary_all.json", record)

    existing_runs = backtest_store.read_jsonl(runs_jsonl_path)
    existing_same = [
        r for r in existing_runs
        if r.get("run_id") == resolved_run_id
    ]
    if existing_same:
        fingerprints_match = all(
            r.get("param_hash") == param_hash and r.get("data_fingerprint") == data_fingerprint
            for r in existing_same
        )
        if not fingerprints_match:
            if not overwrite:
                raise RuntimeError(
                    "runs.jsonl conflict: existing run_id has mismatched fingerprints."
                )
            existing_runs = [r for r in existing_runs if r.get("run_id") != resolved_run_id]
        else:
            log.info("runs.jsonl already contains run_id %s, skipping append.", resolved_run_id)
            return {
                "run_id": resolved_run_id,
                "start": start_label,
                "end": end_label,
                "symbols": symbols,
                "param_hash": param_hash,
                "data_fingerprint": data_fingerprint,
                "data_manifest_by_symbol": per_symbol_manifests,
                "per_symbol": per_symbol_summaries,
                "status": "completed",
            }

    backtest_store.write_jsonl(runs_jsonl_path, [*existing_runs, record])

    return {
        "run_id": resolved_run_id,
        "start": start_label,
        "end": end_label,
        "symbols": symbols,
        "param_hash": param_hash,
        "data_fingerprint": data_fingerprint,
        "data_manifest_by_symbol": per_symbol_manifests,
        "per_symbol": per_symbol_summaries,
        "status": "completed",
    }


def run_backtest_for_symbol(
    symbol: str,
    start_ms: int,
    end_ms: int,
    run_dir: Path,
    params: BacktestParams,
    df_1d: pd.DataFrame,
    df_ex: pd.DataFrame,
    run_id: str | None = None,
    param_hash: str | None = None,
    data_fingerprint: str | None = None,
    strategy_version: str | None = None,
) -> Dict:
    """
    One-symbol backtest (simple and debuggable). Multi-symbol aggregation is handled by caller.
    """
    if df_1d.empty or df_ex.empty:
        raise RuntimeError(f"Insufficient market data for {symbol}")

    df_1d_feat = strat.prepare_features_1d(df_1d, params=params)
    df_ex_feat = strat.prepare_features_exec(df_ex)

    # Backtest state
    cash = float(params.starting_cash_usdc_per_symbol)
    qty = 0.0
    avg_entry = 0.0
    realized_pnl_cum = 0.0
    state = strat.StrategyState()

    fee_rate = config.fee_rate_from_bps(params.taker_fee_bps)

    trades: List[Dict] = []
    last_day: str = ""
    last_mark_price: float | None = None
    equity_by_day: List[Dict] = []

    def round8(x: float) -> float:
        return float(round(x, 8))

    def position_side_from_qty(position_qty: float) -> str:
        if abs(position_qty) < 1e-12:
            return "FLAT"
        return "LONG" if position_qty > 0 else "SHORT"

    def build_day_row(day: str, mark_price: float) -> Dict:
        # Daily equity snapshot uses mark-to-market at last execution close.
        equity = cash + qty * mark_price
        position_value = abs(qty) * mark_price
        net_exposure = position_value / equity if abs(equity) > 1e-12 else 0.0
        unrealized_pnl = qty * (mark_price - avg_entry) if abs(qty) > 1e-12 else 0.0
        return {
            "date_utc": day,
            "equity": round8(equity),
            "cash_usdc": round8(cash),
            "position_side": position_side_from_qty(qty),
            "position_qty": round8(qty),
            "avg_entry_price": round8(avg_entry),
            "mark_price": round8(mark_price),
            "position_value_usdc": round8(position_value),
            "net_exposure": round8(net_exposure),
            "unrealized_pnl_usdc": round8(unrealized_pnl),
            "realized_pnl_usdc_cum": round8(realized_pnl_cum),
        }

    def compute_exposure_diagnostics(df_day_in: pd.DataFrame) -> Dict:
        days_total = int(len(df_day_in))
        if days_total == 0:
            return {
                "avg_net_exposure": 0.0,
                "days_total": 0,
                "days_in_position": 0,
                "days_long": 0,
                "days_short": 0,
                "days_exposure_ge_0_3": 0,
                "days_exposure_ge_0_7": 0,
                "pct_in_position": 0.0,
                "pct_long": 0.0,
                "pct_short": 0.0,
                "pct_exposure_ge_0_3": 0.0,
                "pct_exposure_ge_0_7": 0.0,
                "exposure_min": 0.0,
                "exposure_p50": 0.0,
                "exposure_p90": 0.0,
                "exposure_max": 0.0,
            }

        if "net_exposure" in df_day_in.columns:
            net_exposure = pd.to_numeric(df_day_in["net_exposure"], errors="coerce").fillna(0.0)
        else:
            net_exposure = pd.Series([0.0] * days_total)

        if "position_side" in df_day_in.columns:
            position_side = df_day_in["position_side"].fillna("FLAT").astype(str)
        else:
            position_side = pd.Series(["FLAT"] * days_total)

        if "position_qty" in df_day_in.columns:
            position_qty = pd.to_numeric(df_day_in["position_qty"], errors="coerce").fillna(0.0)
        else:
            position_qty = pd.Series([0.0] * days_total)

        in_position = (position_side != "FLAT") | (position_qty.abs() > 0)
        is_long = (position_side == "LONG") | (position_qty > 0)
        is_short = (position_side == "SHORT") | (position_qty < 0)

        days_in_position = int(in_position.sum())
        days_long = int(is_long.sum())
        days_short = int(is_short.sum())
        days_exposure_ge_0_3 = int((net_exposure >= 0.3).sum())
        days_exposure_ge_0_7 = int((net_exposure >= 0.7).sum())

        def pct(value: int) -> float:
            return float(value / days_total) if days_total > 0 else 0.0

        exposure_min = float(net_exposure.min())
        exposure_p50 = float(net_exposure.quantile(0.5))
        exposure_p90 = float(net_exposure.quantile(0.9))
        exposure_max = float(net_exposure.max())

        return {
            "avg_net_exposure": round8(float(net_exposure.mean())),
            "days_total": days_total,
            "days_in_position": days_in_position,
            "days_long": days_long,
            "days_short": days_short,
            "days_exposure_ge_0_3": days_exposure_ge_0_3,
            "days_exposure_ge_0_7": days_exposure_ge_0_7,
            "pct_in_position": round8(pct(days_in_position)),
            "pct_long": round8(pct(days_long)),
            "pct_short": round8(pct(days_short)),
            "pct_exposure_ge_0_3": round8(pct(days_exposure_ge_0_3)),
            "pct_exposure_ge_0_7": round8(pct(days_exposure_ge_0_7)),
            "exposure_min": round8(exposure_min),
            "exposure_p50": round8(exposure_p50),
            "exposure_p90": round8(exposure_p90),
            "exposure_max": round8(exposure_max),
        }

    # Iterate execution bars inside evaluation window
    ex_items = list(df_ex_feat.loc[(df_ex_feat.index >= start_ms) & (df_ex_feat.index < end_ms)].iterrows())
    for bar_idx, (ts, row_ex) in enumerate(ex_items):
        price = float(row_ex["close"])

        day = ms_to_ymd(int(ts))
        if last_day and day != last_day and last_mark_price is not None:
            equity_by_day.append(build_day_row(last_day, last_mark_price))
        last_day = day
        last_mark_price = price

        equity = cash + qty * price

        decision = strat.decide(
            ts_ms=int(ts),
            exec_bar_idx=bar_idx,
            df_1d_feat=df_1d_feat,
            df_exec_feat=df_ex_feat,
            state=state,
            params=params,
        )

        missing = [key for key in strat.DECISION_KEYS if key not in decision]
        assert not missing, f"Decision missing keys: {missing}"

        target_frac = float(decision.get("target_pos_frac") or 0.0)
        current_notional = qty * price
        current_pos_frac = current_notional / equity if abs(equity) > 1e-12 else 0.0
        # Use current equity pre-trade to translate fraction -> notional.
        target_notional = target_frac * equity
        next_pos_frac = target_frac
        delta_pos_frac = next_pos_frac - current_pos_frac

        must_trade = False
        if abs(target_notional) <= 1e-12 and abs(current_notional) > 1e-12:
            must_trade = True

        policy_result = execution_policy.compute_trade_intent(
            equity=float(equity),
            current_notional=float(current_notional),
            target_notional=float(target_notional),
            min_trade_notional_pct=float(params.min_trade_notional_pct),
            must_trade=must_trade,
        )
        trade_intent = policy_result["trade_intent"]
        delta_notional = float(policy_result["delta_notional"])
        threshold_notional = float(policy_result["threshold_notional"])

        if trade_intent == "NOOP_SMALL_DELTA":
            ts_utc = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).isoformat()
            record: Dict[str, object] = {}
            record.update(decision)
            record.update({
                "ts_ms": int(ts),
                "ts_utc": ts_utc,
                "date_utc": ms_to_ymd(int(ts)),
                "symbol": symbol,
                "bar_interval": params.timeframes["execution"],
                "action": "HOLD",
                "trade_intent": trade_intent,
                "min_trade_notional_pct": float(params.min_trade_notional_pct),
                "threshold_notional_usdc": float(threshold_notional),
                "decision_reason": decision.get("reason"),
                "update_last_exec": False,
                "close_px": price,
                "current_pos_frac": float(current_pos_frac),
                "next_pos_frac": float(current_pos_frac),
                "delta_pos_frac": 0.0,
                "delta_notional_usdc": 0.0,
                "delta_qty": 0.0,
                "fee_usdc": 0.0,
                "equity_before": float(equity),
                "equity_after": float(equity),
                "position_qty_after": float(qty),
                "position_frac_after": float(current_pos_frac),
                "avg_entry_after": float(avg_entry),
                "realized_pnl_usdc": 0.0,
                "reason": "noop_small_delta",
            })
            trades.append(record)
            continue

        reason = decision.get("reason") or "already_at_target"

        dq = delta_notional / price
        equity_before = equity

        # Realized pnl bookkeeping (avg entry) before cash/qty update.
        new_avg_entry, realized_pnl = update_avg_and_realized(qty, avg_entry, dq, price)
        realized_pnl_cum += realized_pnl

        # Apply trade to cash + position; fee is applied to absolute notional.
        cash -= delta_notional
        fee = abs(delta_notional) * fee_rate
        cash -= fee
        qty += dq
        avg_entry = new_avg_entry

        equity_after = cash + qty * price

        # Update state position frac based on post-trade equity.
        if abs(equity_after) > 1e-9:
            state.position.frac = (qty * price) / equity_after
        else:
            state.position.frac = 0.0

        if decision.get("update_last_exec"):
            state.last_exec_bar_idx = bar_idx

        ts_utc = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).isoformat()
        record = {}
        record.update(decision)
        record.update({
            "ts_ms": int(ts),
            "ts_utc": ts_utc,
            "date_utc": ms_to_ymd(int(ts)),
            "symbol": symbol,
            "bar_interval": params.timeframes["execution"],
            "action": decision.get("action", "REBALANCE"),
            "trade_intent": trade_intent,
            "min_trade_notional_pct": float(params.min_trade_notional_pct),
            "threshold_notional_usdc": float(threshold_notional),
            "close_px": price,
            "current_pos_frac": float(current_pos_frac),
            "next_pos_frac": float(next_pos_frac),
            "delta_pos_frac": float(delta_pos_frac),
            "delta_notional_usdc": float(delta_notional),
            "delta_qty": float(dq),
            "fee_usdc": float(fee),
            "equity_before": float(equity_before),
            "equity_after": float(equity_after),
            "position_qty_after": float(qty),
            "position_frac_after": float(state.position.frac),
            "avg_entry_after": float(avg_entry),
            "realized_pnl_usdc": float(realized_pnl),
            "reason": reason,
        })
        trades.append(record)

    # record final day
    if last_day and last_mark_price is not None:
        equity_by_day.append(build_day_row(last_day, last_mark_price))

    # Build daily equity dataframe
    df_day = pd.DataFrame(equity_by_day)
    # Ensure unique dates (keep last)
    df_day = df_day.drop_duplicates(subset=["date_utc"], keep="last")
    df_day["equity_usdc"] = pd.to_numeric(df_day["equity"], errors="coerce")
    df_day["close_px"] = pd.to_numeric(df_day["mark_price"], errors="coerce")

    # Metrics
    equity_series = df_day["equity_usdc"]
    strategy_metrics = metrics.compute_equity_metrics(
        equity_series,
        starting_cash=float(params.starting_cash_usdc_per_symbol),
    )
    dates = pd.Index(df_day["date_utc"], name="date_utc")
    close_px = pd.Series(df_day["close_px"].to_numpy(), index=dates, name="close_px")
    if close_px.isna().any():
        log.warning("Missing close_px values detected; forward-filling for buy & hold benchmark.")
        close_px = close_px.ffill()

    bh_equity = metrics.build_buy_hold_curve(
        dates=dates,
        close_px=close_px,
        starting_cash=float(params.starting_cash_usdc_per_symbol),
    )
    bh_metrics = metrics.compute_equity_metrics(
        bh_equity,
        starting_cash=float(params.starting_cash_usdc_per_symbol),
    )

    # Write outputs
    run_dir.mkdir(parents=True, exist_ok=True)
    equity_by_day_path = run_dir / "equity_by_day.csv"
    df_day.to_csv(equity_by_day_path, index=False)
    equity_by_day_bh_path = run_dir / "equity_by_day_bh.csv"
    pd.DataFrame({
        "date_utc": dates,
        "close_px": close_px.values,
        "bh_equity": bh_equity.values,
    }).to_csv(equity_by_day_bh_path, index=False)
    if "net_exposure" in df_day.columns:
        pd.DataFrame({
            "date_utc": dates,
            "strategy_equity": equity_series.values,
            "bh_equity": bh_equity.values,
            "net_exposure": df_day["net_exposure"].values,
        }).to_csv(run_dir / "equity_by_day_with_benchmark.csv", index=False)

    # overwrite trades file for determinism
    trades_path = run_dir / "trades.jsonl"
    if trades_path.exists():
        trades_path.unlink()
    append_jsonl(trades_path, trades)

    exposure_diagnostics = compute_exposure_diagnostics(df_day)
    decision_counts = compute_trade_decision_counts(trades)
    diagnostic_counts, diagnostic_warnings = compute_diagnostic_counts(
        trades_path,
        equity_by_day_path,
        params.direction_mode,
    )
    summary = {
        "run_id": run_id,
        "param_hash": param_hash,
        "data_fingerprint": data_fingerprint,
        "strategy_version": strategy_version,
        "symbol": symbol,
        "start_date_utc": ms_to_ymd(start_ms),
        "end_date_utc": ms_to_ymd(end_ms - 1),
        "starting_cash_usdc": float(params.starting_cash_usdc_per_symbol),
        **decision_counts,
        "fee_bps": float(params.taker_fee_bps),
        "min_trade_notional_pct": float(params.min_trade_notional_pct),
        "layers": {
            "trend_existence": dict(params.trend_existence),
            "trend_quality": dict(params.trend_quality),
            "execution": dict(params.execution),
            "direction_mode": params.direction_mode,
        },
        "risk_sizing": {
            "angle_sizing_enabled": bool(params.angle_sizing_enabled),
            "angle_sizing_a": float(params.angle_sizing_a),
            "angle_sizing_q": float(params.angle_sizing_q),
            "vol_window_div": float(params.vol_window_div),
            "vol_window_min": int(params.vol_window_min),
            "vol_window_max": int(params.vol_window_max),
            "vol_eps": float(params.vol_eps),
        },
        **exposure_diagnostics,
        **diagnostic_counts,
    }
    summary.update(strategy_metrics)
    summary.update({
        "buy_hold_ending_equity_usdc": bh_metrics["ending_equity_usdc"],
        "buy_hold_total_return": bh_metrics["total_return"],
        "buy_hold_max_drawdown": bh_metrics["max_drawdown"],
        "buy_hold_sharpe_ratio": bh_metrics["sharpe_ratio"],
        "buy_hold_ulcer_index": bh_metrics["ulcer_index"],
        "buy_hold_ulcer_performance_index": bh_metrics["ulcer_performance_index"],
        "alpha_vs_buy_hold": summary.get("total_return", 0.0) - bh_metrics["total_return"],
    })
    if "ulcer_performance_index" in summary:
        summary["ulcer_performance_index"] = metrics.safe_float_for_json(
            summary["ulcer_performance_index"],
        )
    summary["buy_hold_ulcer_performance_index"] = metrics.safe_float_for_json(
        summary["buy_hold_ulcer_performance_index"],
    )
    if diagnostic_warnings:
        summary["diagnostics_warning"] = "; ".join(diagnostic_warnings)

    write_json(run_dir / "summary.json", summary)

    strategy_label = (
        summary.get("layers", {}).get("direction_mode")
        or getattr(config, "DIRECTION_MODE", None)
        or "long_only"
    )
    generate_quarterly_stats(
        equity_by_day_path=equity_by_day_path,
        equity_by_day_bh_path=equity_by_day_bh_path,
        trades_path=trades_path,
        output_path=run_dir / "quarterly_stats.csv",
        strategy_label=strategy_label,
    )

    return summary

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--symbols", default=",".join(config.SYMBOLS), help="Comma-separated symbols (coins) e.g. BTC,ETH,SOL")
    ap.add_argument("--run_id", default="", help="Optional run id; default is deterministic")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing run_id outputs")
    args = ap.parse_args()

    params = build_params_from_config()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_result = run_backtest(
        symbols,
        args.start,
        args.end,
        params,
        run_id=args.run_id.strip() or None,
        overwrite=args.overwrite,
    )
    if run_result.get("status") == "multi":
        run_results = run_result["runs"]
    else:
        run_results = [run_result]

    for result in run_results:
        if result.get("status") == "skipped":
            log.info("Run %s exists, skipped.", result["run_id"])
            continue
        run_id = result["run_id"]
        run_dir = Path(config.BACKTEST_RESULT_DIR) / run_id
        run_symbols = result["symbols"]

        # config snapshot
        write_json(run_dir / "config_snapshot.json", {
            "symbols": run_symbols,
            "start": result["start"],
            "end": result["end"],
            "param_hash": result["param_hash"],
            "data_fingerprint": result["data_fingerprint"],
            "params_hashable": params.to_hashable_dict(),
            "QUOTE_ASSET": config.QUOTE_ASSET,
            "MARKET_TYPE": config.MARKET_TYPE,
            "TIMEFRAMES": dict(config.TIMEFRAMES),
            "TREND_EXISTENCE": dict(config.TREND_EXISTENCE),
            "TREND_QUALITY": dict(config.TREND_QUALITY),
            "EXECUTION": dict(config.EXECUTION),
            "DIRECTION_MODE": config.DIRECTION_MODE,
            "STARTING_CASH_USDC_PER_SYMBOL": config.STARTING_CASH_USDC_PER_SYMBOL,
            "TAKER_FEE_BPS": config.TAKER_FEE_BPS,
            "MIN_TRADE_NOTIONAL_PCT": config.MIN_TRADE_NOTIONAL_PCT,
            "HL_INFO_URL": config.HL_INFO_URL,
        })

        log.info("Done. Results in %s", run_dir.as_posix())

if __name__ == "__main__":
    main()
