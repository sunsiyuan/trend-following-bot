"""
quarterly_stats.py

Build per-quarter performance statistics for strategy and buy-hold curves.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

pd_spec = importlib.util.find_spec("pandas")
if pd_spec is not None:  # pragma: no cover - import guarded for minimal envs
    import pandas as pd
else:  # pragma: no cover - pandas-free fallback
    pd = None

from bot import metrics


@dataclass
class EquityRow:
    date: datetime
    date_str: str
    equity: float
    net_exposure: Optional[float]


def _parse_date(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _read_equity_rows(
    path: Path,
    equity_columns: Iterable[str],
    include_net_exposure: bool = False,
) -> List[EquityRow]:
    if not path.exists():
        return []
    if pd is not None:
        df = pd.read_csv(path)
        if "date_utc" not in df.columns:
            return []
        df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
        df = df.dropna(subset=["date_utc"]).sort_values("date_utc")
        equity_col = next((c for c in equity_columns if c in df.columns), None)
        if equity_col is None:
            return []
        equity_values = pd.to_numeric(df[equity_col], errors="coerce")
        net_exposure = None
        if include_net_exposure and "net_exposure" in df.columns:
            net_exposure = pd.to_numeric(df["net_exposure"], errors="coerce")
        rows: List[EquityRow] = []
        for idx, dt_value in df["date_utc"].items():
            equity = equity_values.loc[idx]
            if pd.isna(equity):
                continue
            exposure_value = None
            if include_net_exposure and net_exposure is not None:
                exposure_value = net_exposure.loc[idx]
                if pd.isna(exposure_value):
                    exposure_value = None
            date_str = dt_value.strftime("%Y-%m-%d")
            rows.append(EquityRow(dt_value, date_str, float(equity), exposure_value))
        return rows

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            dt_value = _parse_date(raw.get("date_utc", ""))
            if dt_value is None:
                continue
            equity_val = None
            for col in equity_columns:
                if col in raw and raw[col] not in (None, ""):
                    equity_val = raw[col]
                    break
            if equity_val is None:
                continue
            try:
                equity = float(equity_val)
            except ValueError:
                continue
            exposure_value = None
            if include_net_exposure:
                raw_exposure = raw.get("net_exposure")
                if raw_exposure not in (None, ""):
                    try:
                        exposure_value = float(raw_exposure)
                    except ValueError:
                        exposure_value = None
            date_str = dt_value.strftime("%Y-%m-%d")
            rows.append(EquityRow(dt_value, date_str, equity, exposure_value))
        rows.sort(key=lambda r: r.date)
        return rows


def _quarter_label(dt_value: datetime) -> str:
    quarter = (dt_value.month - 1) // 3 + 1
    return f"{dt_value.year}Q{quarter}"


def _ordered_periods(rows: List[EquityRow]) -> List[str]:
    periods: List[str] = []
    seen = set()
    for row in rows:
        label = _quarter_label(row.date)
        if label not in seen:
            periods.append(label)
            seen.add(label)
    return periods


def _mean(values: List[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def _compute_trade_metrics(trades_path: Path) -> Optional[dict]:
    if not trades_path.exists():
        return None
    by_period = {}
    total_count = 0
    total_fees = 0.0
    try:
        with trades_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                date_value = record.get("date_utc")
                action = record.get("action")
                reason = record.get("reason")
                fee = record.get("fee_usdc")
                if date_value is None or action is None or reason is None:
                    return None
                dt_value = _parse_date(str(date_value))
                if dt_value is None:
                    return None
                if action != "REBALANCE" or reason != "gate_passed":
                    continue
                if fee is None:
                    return None
                try:
                    fee_val = float(fee)
                except (TypeError, ValueError):
                    return None
                label = _quarter_label(dt_value)
                count, fee_sum = by_period.get(label, (0, 0.0))
                by_period[label] = (count + 1, fee_sum + fee_val)
                total_count += 1
                total_fees += fee_val
    except json.JSONDecodeError:
        return None
    by_period["__all__"] = (total_count, total_fees)
    return by_period


def _build_row(
    period: str,
    strategy: str,
    rows: List[EquityRow],
    net_exposure_rows: bool,
    trade_metrics: Optional[dict],
    use_trades: bool,
) -> dict:
    equity = [row.equity for row in rows]
    days = len(equity)
    start_equity = equity[0] if equity else math.nan
    end_equity = equity[-1] if equity else math.nan
    pnl = math.nan
    if equity and start_equity != 0:
        pnl = end_equity / start_equity - 1.0

    returns = metrics.equity_returns_with_first_zero(equity)
    sharpe = metrics.sharpe_annualized_from_returns(returns)

    mdd, ui = metrics.mdd_and_ulcer_index(equity)
    annualized_return = metrics.annualized_return(start_equity, end_equity, days)
    upi = math.nan
    if ui and not math.isnan(ui) and ui != 0:
        upi = annualized_return / ui

    avg_exposure = math.nan
    pct_days_in_position = math.nan
    pct_days_long = math.nan
    pct_days_short = math.nan
    turnover_proxy = math.nan
    if net_exposure_rows:
        exposures = [row.net_exposure for row in rows]
        if exposures and all(value is not None for value in exposures) and days > 0:
            exposure_values = [float(value) for value in exposures]
            avg_exposure = _mean(exposure_values)
            eps = 1e-6
            pct_days_in_position = sum(abs(v) > eps for v in exposure_values) / days
            pct_days_long = sum(v > eps for v in exposure_values) / days
            pct_days_short = sum(v < -eps for v in exposure_values) / days
            turnover_proxy = 0.0
            for idx in range(1, len(exposure_values)):
                turnover_proxy += abs(exposure_values[idx] - exposure_values[idx - 1])

    trade_count = math.nan
    fees_paid = math.nan
    if use_trades and trade_metrics is not None:
        if period.startswith("All "):
            count, fees = trade_metrics.get("__all__", (0, 0.0))
        else:
            count, fees = trade_metrics.get(period, (0, 0.0))
        trade_count = count
        fees_paid = fees

    start_date = rows[0].date_str if rows else ""
    end_date = rows[-1].date_str if rows else ""

    return {
        "period": period,
        "strategy": strategy,
        "pnl": pnl,
        "sharpe_annualized": sharpe,
        "upi_annualized": upi,
        "mdd": mdd,
        "ui": ui,
        "avg_net_exposure": avg_exposure,
        "pct_days_in_position": pct_days_in_position,
        "pct_days_long": pct_days_long,
        "pct_days_short": pct_days_short,
        "trade_count": trade_count,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "fees_paid": fees_paid,
        "turnover_proxy": turnover_proxy,
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
    }


def generate_quarterly_stats(
    equity_by_day_path: Path,
    equity_by_day_bh_path: Path,
    trades_path: Path,
    output_path: Path,
    strategy_label: str,
) -> None:
    strategy_rows = _read_equity_rows(
        equity_by_day_path,
        ["equity_usdc", "equity"],
        include_net_exposure=True,
    )
    bh_rows = _read_equity_rows(
        equity_by_day_bh_path,
        ["bh_equity"],
        include_net_exposure=False,
    )

    primary_rows = strategy_rows if strategy_rows else bh_rows
    if not primary_rows:
        return

    periods = _ordered_periods(primary_rows)
    all_label = f"All {primary_rows[0].date_str}-{primary_rows[-1].date_str}"

    trade_metrics = _compute_trade_metrics(trades_path)
    # fees_paid aligns with trade_count to avoid counting other trade types.

    rows: List[dict] = []
    for period in periods:
        period_rows = [row for row in strategy_rows if _quarter_label(row.date) == period]
        rows.append(_build_row(period, strategy_label, period_rows, True, trade_metrics, True))
        if bh_rows:
            bh_period_rows = [row for row in bh_rows if _quarter_label(row.date) == period]
            rows.append(_build_row(period, "buy_hold", bh_period_rows, False, None, False))

    all_strategy_rows = strategy_rows
    rows.append(_build_row(all_label, strategy_label, all_strategy_rows, True, trade_metrics, True))
    if bh_rows:
        rows.append(_build_row(all_label, "buy_hold", bh_rows, False, None, False))

    columns = [
        "period",
        "strategy",
        "pnl",
        "sharpe_annualized",
        "upi_annualized",
        "mdd",
        "ui",
        "avg_net_exposure",
        "pct_days_in_position",
        "pct_days_long",
        "pct_days_short",
        "trade_count",
        "start_equity",
        "end_equity",
        "fees_paid",
        "turnover_proxy",
        "start_date",
        "end_date",
        "days",
    ]

    if pd is not None:
        pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
        return

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
