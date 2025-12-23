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

        target_frac = float(decision["target_frac"])
        current_notional = qty * price
        # Use current equity pre-trade to translate fraction -> notional
        target_notional = target_frac * equity

        delta_notional = target_notional - current_notional
        if abs(delta_notional) < 1e-8:
            # no trade
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
            "trend": decision["trend"],
            "risk": decision["risk"],
            "regime": decision.get("regime", "TREND"),
            "action": decision["action"],
            "reason": decision["reason"],
        })

    # record final day
    if last_day and last_mark_price is not None:
        equity_by_day.append(build_day_row(last_day, last_mark_price))

    # Build daily equity dataframe
    df_day = pd.DataFrame(equity_by_day)
    # Ensure unique dates (keep last)
    df_day = df_day.drop_duplicates(subset=["date_utc"], keep="last")

    # Metrics
    equity_series = df_day["equity"]
    exposure_diagnostics = compute_exposure_diagnostics(df_day)
    decision_counts = compute_trade_decision_counts(trades)
    summary = {
        "symbol": symbol,
        "start_date_utc": ms_to_ymd(start_ms),
        "end_date_utc": ms_to_ymd(end_ms),
        "starting_cash_usdc": float(config.STARTING_CASH_USDC_PER_SYMBOL),
        "ending_equity_usdc": float(equity_series.iloc[-1]) if not equity_series.empty else float(config.STARTING_CASH_USDC_PER_SYMBOL),
        "total_return": metrics.total_return(equity_series) if len(equity_series) >= 2 else 0.0,
        "max_drawdown": metrics.max_drawdown(equity_series) if len(equity_series) >= 2 else 0.0,
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
    }

    # Write outputs
    sym_dir = run_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    write_json(sym_dir / "summary.json", summary)
    df_day.to_csv(sym_dir / "equity_by_day.csv", index=False)

    # overwrite trades file for determinism
    trades_path = sym_dir / "trades.jsonl"
    if trades_path.exists():
        trades_path.unlink()
    append_jsonl(trades_path, trades)

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
