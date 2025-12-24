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
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from bot import config
from bot import data_client
from bot import metrics
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

def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def append_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")

def compute_trade_decision_counts(trades: List[Dict]) -> Dict:
    decision_count = len(trades)
    rebalance_count = 0
    hold_count = 0
    rebalance_by_reason: Dict[str, int] = {}
    hold_by_reason: Dict[str, int] = {}

    for trade in trades:
        action = trade.get("action")
        reason = trade.get("reason", "unknown")
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
    if equity_by_day_csv_path.exists():
        df_day = pd.read_csv(equity_by_day_csv_path)
        if "date_utc" in df_day.columns:
            dates = df_day["date_utc"].dropna().astype(str).tolist()
            days_total = len(pd.unique(dates))
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

    def get_raw_dir(record: Dict) -> str | None:
        raw_dir = record.get("raw_dir") or record.get("trend_dir")
        if raw_dir:
            return str(raw_dir)
        trend = record.get("trend")
        if trend in {"LONG", "SHORT"}:
            warn_once("raw_dir missing; using trend as fallback for some days.")
            return trend
        return None

    def get_risk_mode(record: Dict) -> str | None:
        risk = record.get("risk_mode") or record.get("risk")
        return str(risk) if risk is not None else None

    def get_target_frac(record: Dict) -> float | None:
        for key in ("target_pos_frac", "target_frac", "target_fraction"):
            if key in record:
                try:
                    return float(record[key])
                except (TypeError, ValueError):
                    return None
        return None

    raw_dir_long_days = 0
    raw_dir_short_days = 0
    risk_on_days = 0
    risk_neutral_days = 0
    risk_off_days = 0
    flat_days_total = 0
    flat_due_to_long_only_days = 0
    flat_due_to_risk_off_days = 0
    flat_due_to_range_days = 0
    flat_due_to_execution_gate_days = 0
    flat_due_to_other_days = 0
    target_frac_days_0 = 0
    target_frac_days_0_5 = 0
    target_frac_days_1 = 0
    target_frac_days_other = 0

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

        raw_dir = get_raw_dir(record)
        if raw_dir == "LONG":
            raw_dir_long_days += 1
        elif raw_dir == "SHORT":
            raw_dir_short_days += 1
        elif raw_dir is None:
            warn_once("Missing raw_dir/trend_dir in trades.jsonl; raw_dir counts may be low.")

        risk_mode = get_risk_mode(record)
        if risk_mode == "RISK_ON":
            risk_on_days += 1
        elif risk_mode == "RISK_NEUTRAL":
            risk_neutral_days += 1
        elif risk_mode == "RISK_OFF":
            risk_off_days += 1
        elif risk_mode is None:
            warn_once("Missing risk_mode/risk in trades.jsonl; risk counts may be low.")

        target_frac = get_target_frac(record)
        if target_frac is None:
            target_frac_days_other += 1
            warn_once("Missing target_pos_frac in trades.jsonl; target histogram may be incomplete.")
        elif abs(target_frac) <= eps:
            target_frac_days_0 += 1
        elif abs(target_frac - 0.5) <= eps:
            target_frac_days_0_5 += 1
        elif abs(target_frac - 1.0) <= eps:
            target_frac_days_1 += 1
        else:
            target_frac_days_other += 1

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
        market_state = record.get("market_state") or record.get("trend")
        market_state = str(market_state) if market_state is not None else ""
        record_direction_mode = record.get("direction_mode", direction_mode)
        target_non_zero = target_frac is not None and abs(target_frac) > eps

        if record_direction_mode == "long_only" and raw_dir == "SHORT":
            flat_due_to_long_only_days += 1
        elif raw_dir == "LONG" and risk_mode == "RISK_OFF":
            flat_due_to_risk_off_days += 1
        elif market_state == "RANGE" or reason.startswith("range_"):
            flat_due_to_range_days += 1
        elif target_non_zero and ("execution_gate" in reason or "range_exit_blocked" in reason or "gate_blocked" in reason):
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
        "risk_on_days": risk_on_days,
        "risk_neutral_days": risk_neutral_days,
        "risk_off_days": risk_off_days,
        "flat_days_total": flat_days_total,
        "flat_due_to_long_only_days": flat_due_to_long_only_days,
        "flat_due_to_risk_off_days": flat_due_to_risk_off_days,
        "flat_due_to_range_days": flat_due_to_range_days,
        "flat_due_to_execution_gate_days": flat_due_to_execution_gate_days,
        "flat_due_to_other_days": flat_due_to_other_days,
        "target_frac_days_0": target_frac_days_0,
        "target_frac_days_0_5": target_frac_days_0_5,
        "target_frac_days_1": target_frac_days_1,
        "target_frac_days_other": target_frac_days_other,
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

def run_backtest_for_symbol(
    symbol: str,
    start_ms: int,
    end_ms: int,
    run_dir: Path,
) -> Dict:
    """
    One-symbol backtest (simple and debuggable). Multi-symbol aggregation is handled by caller.
    """
    # Warmup: ensure enough data before start for indicator windows
    ms_1d = data_client.interval_to_ms(config.TIMEFRAMES["trend"])
    ms_ex = data_client.interval_to_ms(config.TIMEFRAMES["execution"])
    warmup_1d = max(config.TREND_EXISTENCE["window"], config.TREND_QUALITY["window"]) + 10
    warmup_exec_steps = max(config.EXECUTION["build_min_step_bars"], config.EXECUTION["reduce_min_step_bars"])
    warmup_ex = config.EXECUTION["window"] + warmup_exec_steps + 10
    fetch_start = min(start_ms - warmup_1d * ms_1d, start_ms - warmup_ex * ms_ex)
    fetch_start = max(0, fetch_start)

    df_1d = data_client.ensure_market_data(symbol, config.TIMEFRAMES["trend"], fetch_start, end_ms)
    df_ex = data_client.ensure_market_data(symbol, config.TIMEFRAMES["execution"], fetch_start, end_ms)

    if df_1d.empty or df_ex.empty:
        raise RuntimeError(f"Insufficient market data for {symbol}")

    df_1d_feat = strat.prepare_features_1d(df_1d)
    df_ex_feat = strat.prepare_features_exec(df_ex)

    # Backtest state
    cash = float(config.STARTING_CASH_USDC_PER_SYMBOL)
    qty = 0.0
    avg_entry = 0.0
    realized_pnl_cum = 0.0
    state = strat.StrategyState()

    fee_rate = config.fee_rate_from_bps(config.TAKER_FEE_BPS)

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
    ex_items = list(df_ex_feat.loc[(df_ex_feat.index >= start_ms) & (df_ex_feat.index <= end_ms)].iterrows())
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
        )

        target_value = decision.get("target_frac")
        if target_value is None:
            target_value = decision.get("target_pos_frac")
        target_frac = float(target_value)
        market_state = decision.get("market_state") or decision.get("trend")
        raw_dir = decision.get("raw_dir") or decision.get("trend_dir")
        if raw_dir is None and decision.get("trend") in {"LONG", "SHORT"}:
            raw_dir = decision.get("trend")
        risk_mode = decision.get("risk_mode") or decision.get("risk")
        reason = decision.get("reason") or "already_at_target"
        current_notional = qty * price
        # Use current equity pre-trade to translate fraction -> notional
        target_notional = target_frac * equity

        delta_notional = target_notional - current_notional
        if abs(delta_notional) < 1e-8:
            # no trade
            position_frac = (qty * price) / equity if abs(equity) > 1e-12 else 0.0
            trades.append({
                "ts_ms": int(ts),
                "date_utc": ms_to_ymd(int(ts)),
                "symbol": symbol,
                "price": price,
                "delta_notional": 0.0,
                "delta_qty": 0.0,
                "fee": 0.0,
                "equity_before": float(equity),
                "equity_after": float(equity),
                "position_qty_after": float(qty),
                "position_frac_after": float(position_frac),
                "avg_entry_after": float(avg_entry),
                "realized_pnl": 0.0,
                "trend": decision.get("trend"),
                "trend_dir": decision.get("trend_dir"),
                "raw_dir": raw_dir,
                "market_state": market_state,
                "risk": decision.get("risk"),
                "risk_mode": risk_mode,
                "regime": decision.get("regime", "TREND"),
                "direction_mode": config.DIRECTION_MODE,
                "target_pos_frac": float(target_frac),
                "action": "HOLD",
                "reason": reason,
            })
            if decision.get("update_last_exec"):
                state.last_exec_bar_idx = bar_idx
            continue

        dq = delta_notional / price
        equity_before = equity

        # Realized pnl bookkeeping (avg entry)
        new_avg_entry, realized_pnl = update_avg_and_realized(qty, avg_entry, dq, price)
        realized_pnl_cum += realized_pnl

        # Apply trade to cash + position
        cash -= delta_notional
        fee = abs(delta_notional) * fee_rate
        cash -= fee
        qty += dq
        avg_entry = new_avg_entry

        equity_after = cash + qty * price

        # Update state position frac based on post-trade equity
        if abs(equity_after) > 1e-9:
            state.position.frac = (qty * price) / equity_after
        else:
            state.position.frac = 0.0

        if decision.get("update_last_exec"):
            state.last_exec_bar_idx = bar_idx

        trades.append({
            "ts_ms": int(ts),
            "date_utc": ms_to_ymd(int(ts)),
            "symbol": symbol,
            "price": price,
            "delta_notional": float(delta_notional),
            "delta_qty": float(dq),
            "fee": float(fee),
            "equity_before": float(equity_before),
            "equity_after": float(equity_after),
            "position_qty_after": float(qty),
            "position_frac_after": float(state.position.frac),
            "avg_entry_after": float(avg_entry),
            "realized_pnl": float(realized_pnl),
            "trend": decision.get("trend"),
            "trend_dir": decision.get("trend_dir"),
            "raw_dir": raw_dir,
            "market_state": market_state,
            "risk": decision.get("risk"),
            "risk_mode": risk_mode,
            "regime": decision.get("regime", "TREND"),
            "direction_mode": config.DIRECTION_MODE,
            "target_pos_frac": float(target_frac),
            "action": decision.get("action", "REBALANCE"),
            "reason": reason,
        })

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
        starting_cash=float(config.STARTING_CASH_USDC_PER_SYMBOL),
    )
    dates = pd.Index(df_day["date_utc"], name="date_utc")
    close_px = pd.Series(df_day["close_px"].to_numpy(), index=dates, name="close_px")
    if close_px.isna().any():
        log.warning("Missing close_px values detected; forward-filling for buy & hold benchmark.")
        close_px = close_px.ffill()

    bh_equity = metrics.build_buy_hold_curve(
        dates=dates,
        close_px=close_px,
        starting_cash=float(config.STARTING_CASH_USDC_PER_SYMBOL),
    )
    bh_metrics = metrics.compute_equity_metrics(
        bh_equity,
        starting_cash=float(config.STARTING_CASH_USDC_PER_SYMBOL),
    )

    # Write outputs
    sym_dir = run_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    equity_by_day_path = sym_dir / "equity_by_day.csv"
    df_day.to_csv(equity_by_day_path, index=False)
    pd.DataFrame({
        "date_utc": dates,
        "close_px": close_px.values,
        "bh_equity": bh_equity.values,
    }).to_csv(sym_dir / "equity_by_day_bh.csv", index=False)
    if "net_exposure" in df_day.columns:
        pd.DataFrame({
            "date_utc": dates,
            "strategy_equity": equity_series.values,
            "bh_equity": bh_equity.values,
            "net_exposure": df_day["net_exposure"].values,
        }).to_csv(sym_dir / "equity_by_day_with_benchmark.csv", index=False)

    # overwrite trades file for determinism
    trades_path = sym_dir / "trades.jsonl"
    if trades_path.exists():
        trades_path.unlink()
    append_jsonl(trades_path, trades)

    exposure_diagnostics = compute_exposure_diagnostics(df_day)
    decision_counts = compute_trade_decision_counts(trades)
    diagnostic_counts, diagnostic_warnings = compute_diagnostic_counts(
        trades_path,
        equity_by_day_path,
        config.DIRECTION_MODE,
    )
    summary = {
        "symbol": symbol,
        "start_date_utc": ms_to_ymd(start_ms),
        "end_date_utc": ms_to_ymd(end_ms),
        "starting_cash_usdc": float(config.STARTING_CASH_USDC_PER_SYMBOL),
        **decision_counts,
        "win_rate": metrics.trade_win_rate(trades),
        "fee_bps": float(config.TAKER_FEE_BPS),
        "layers": {
            "trend_existence": dict(config.TREND_EXISTENCE),
            "trend_quality": dict(config.TREND_QUALITY),
            "execution": dict(config.EXECUTION),
            "max_position_frac": dict(config.MAX_POSITION_FRAC),
            "direction_mode": config.DIRECTION_MODE,
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
        "alpha_vs_buy_hold": summary.get("total_return", 0.0) - bh_metrics["total_return"],
    })
    if diagnostic_warnings:
        summary["diagnostics_warning"] = "; ".join(diagnostic_warnings)

    write_json(sym_dir / "summary.json", summary)

    return summary

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--symbols", default=",".join(config.SYMBOLS), help="Comma-separated symbols (coins) e.g. BTC,ETH,SOL")
    ap.add_argument("--run_id", default="", help="Optional run id; default is utc timestamp")
    args = ap.parse_args()

    start_ms = parse_date_to_ms(args.start)
    # end is inclusive; add almost a day to catch candles
    end_ms = parse_date_to_ms(args.end) + 24 * 60 * 60 * 1000 - 1

    run_id = args.run_id.strip() or utc_now_compact()
    run_dir = Path(config.BACKTEST_RESULT_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # config snapshot
    write_json(run_dir / "config_snapshot.json", {
        "symbols": args.symbols.split(","),
        "QUOTE_ASSET": config.QUOTE_ASSET,
        "MARKET_TYPE": config.MARKET_TYPE,
        "TIMEFRAMES": dict(config.TIMEFRAMES),
        "TREND_EXISTENCE": dict(config.TREND_EXISTENCE),
        "TREND_QUALITY": dict(config.TREND_QUALITY),
        "EXECUTION": dict(config.EXECUTION),
        "RANGE": dict(config.RANGE),
        "MAX_POSITION_FRAC": dict(config.MAX_POSITION_FRAC),
        "DIRECTION_MODE": config.DIRECTION_MODE,
        "STARTING_CASH_USDC_PER_SYMBOL": config.STARTING_CASH_USDC_PER_SYMBOL,
        "TAKER_FEE_BPS": config.TAKER_FEE_BPS,
        "HL_INFO_URL": config.HL_INFO_URL,
    })

    summaries: List[Dict] = []
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        log.info("Backtesting %s from %s to %s ...", sym, args.start, args.end)
        summ = run_backtest_for_symbol(sym, start_ms, end_ms, run_dir)
        summaries.append(summ)

    # Aggregate summary
    write_json(run_dir / "summary_all.json", {
        "run_id": run_id,
        "start_date_utc": args.start,
        "end_date_utc": args.end,
        "symbols": symbols,
        "per_symbol": summaries,
    })

    log.info("Done. Results in %s", run_dir.as_posix())

if __name__ == "__main__":
    main()
