from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot import config
from bot import metrics
from bot.backtest_params import stable_json


ROLLING_WINDOW_DAYS = 180
ROLLING_STEP_DAYS = 60

UI_FLOOR = 0.05
MDD_GUARD = -0.30
GAMMA = 0.5

PRIMARY_COLS = [
    "run_id",
    "symbol",
    "symbols",
    "start",
    "end",
    "param_hash",
    "data_fingerprint",
    "strategy_version",
    "final",
    "E",
    "base",
    "ui",
    "mdd",
    "mdd_score",
    "mdd_pass",
    "R_total",
    "R_bh_total",
    "worst_final",
    "p25_final",
    "median_final",
    "p75_final",
    "best_final",
    "p0_E",
    "p25_E",
    "p50_E",
    "p75_E",
    "p100_E",
    "hit_rate",
    "window_count",
    "pnl",
    "return_annualized",
    "sharpe_annualized",
    "upi_annualized",
    "avg_net_exposure",
    "pct_days_in_position",
    "pct_days_long",
    "pct_days_short",
    "trade_count",
    "start_equity",
    "end_equity",
    "fees_paid",
    "turnover_proxy",
    "max_consecutive_days_in_position",
    "max_flat_streak_days",
    "avg_holding_period_days",
    "record_line_no",
    "invalid_reasons",
]


@dataclass
class RunRecord:
    run_id: str
    record: Dict[str, Any]
    line_no: int


def _parse_date(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_runs_jsonl(path: Path) -> List[RunRecord]:
    if not path.exists():
        return []
    seen: Dict[str, RunRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = record.get("run_id")
            if not run_id:
                continue
            seen[run_id] = RunRecord(run_id=run_id, record=record, line_no=idx)
    return sorted(seen.values(), key=lambda item: item.line_no)


def _canonical_hash(payload: Dict[str, Any]) -> str:
    raw = stable_json(payload).encode("utf-8")
    return sha256(raw).hexdigest()


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _split_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_spec(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_filters(args: argparse.Namespace, spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if spec is not None:
        return spec
    return {
        "symbols": args.symbols,
        "requested_start": args.requested_start,
        "requested_end": args.requested_end,
        "strategy_version": args.strategy_version,
        "param_hash_prefix": args.param_hash_prefix,
        "data_fingerprint_prefix": args.data_fingerprint_prefix,
        "run_ids": args.run_ids,
        "limit": args.limit,
    }


def _filter_runs(records: List[RunRecord], filters: Dict[str, Any]) -> List[RunRecord]:
    symbols = filters.get("symbols") or []
    requested_start = filters.get("requested_start")
    requested_end = filters.get("requested_end")
    strategy_version = filters.get("strategy_version")
    param_hash_prefix = filters.get("param_hash_prefix")
    data_fingerprint_prefix = filters.get("data_fingerprint_prefix")
    run_ids = set(filters.get("run_ids") or [])

    selected: List[RunRecord] = []
    for item in records:
        record = item.record
        if run_ids and item.run_id not in run_ids:
            continue
        record_symbols = set(record.get("symbols") or [])
        symbol_label = record.get("symbol_label")
        if symbols:
            if symbol_label and symbol_label in symbols:
                pass
            elif record_symbols.intersection(symbols):
                pass
            else:
                continue
        if requested_start and record.get("start") != requested_start:
            continue
        if requested_end and record.get("end") != requested_end:
            continue
        if strategy_version and record.get("strategy_version") != strategy_version:
            continue
        if param_hash_prefix:
            param_hash = record.get("param_hash", "")
            if not isinstance(param_hash, str) or not param_hash.startswith(param_hash_prefix):
                continue
        if data_fingerprint_prefix:
            fingerprint = record.get("data_fingerprint", "")
            if not isinstance(fingerprint, str) or not fingerprint.startswith(data_fingerprint_prefix):
                continue
        selected.append(item)

    limit = filters.get("limit")
    if isinstance(limit, int) and limit > 0:
        return selected[:limit]
    return selected


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_quarterly_stats(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "period" not in df.columns:
        return None
    df = df[df["period"].astype(str).str.startswith("All ")]
    if df.empty:
        return None
    if "strategy" in df.columns:
        strategy_rows = df[df["strategy"] != "buy_hold"]
        if not strategy_rows.empty:
            return strategy_rows.iloc[0].to_dict()
    return df.iloc[0].to_dict()


def _load_equity_frames(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    equity_path = run_dir / "equity_by_day.csv"
    bh_path = run_dir / "equity_by_day_bh.csv"
    if not equity_path.exists() or not bh_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    df_equity = pd.read_csv(equity_path)
    df_bh = pd.read_csv(bh_path)
    if "date_utc" not in df_equity.columns or "date_utc" not in df_bh.columns:
        return pd.DataFrame(), pd.DataFrame()
    df_equity["date_utc"] = pd.to_datetime(df_equity["date_utc"], errors="coerce")
    df_bh["date_utc"] = pd.to_datetime(df_bh["date_utc"], errors="coerce")
    df_equity = df_equity.dropna(subset=["date_utc"]).sort_values("date_utc")
    df_bh = df_bh.dropna(subset=["date_utc"]).sort_values("date_utc")
    return df_equity, df_bh


def _compute_full_period_metrics(
    df_equity: pd.DataFrame,
    df_bh: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if df_equity.empty or df_bh.empty:
        return None, None, None, None
    equity = pd.to_numeric(df_equity.get("equity_usdc"), errors="coerce").dropna()
    bh_equity = pd.to_numeric(df_bh.get("bh_equity"), errors="coerce").dropna()
    if equity.empty or bh_equity.empty:
        return None, None, None, None
    total_ret = metrics.total_return_from_equity(equity)
    bh_ret = metrics.total_return_from_equity(bh_equity)
    mdd, ui = metrics.mdd_and_ulcer_index(equity)
    return total_ret, bh_ret, mdd, ui


def _compute_score(
    total_ret: float,
    bh_ret: float,
    mdd: float,
    ui: float,
) -> Tuple[float, float, bool, float, float]:
    e_value = total_ret - bh_ret
    ui_eff = max(ui, UI_FLOOR)
    mdd_pass = mdd > MDD_GUARD
    mdd_score = max(0.0, min(1.0, 1.0 - abs(mdd) / abs(MDD_GUARD)))
    base = e_value / ui_eff
    final = base * (mdd_score ** GAMMA)
    return final, e_value, mdd_pass, base, mdd_score


def _rolling_windows(
    df_equity: pd.DataFrame,
    df_bh: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[str]]:
    invalid_reasons: List[str] = []
    if df_equity.empty or df_bh.empty:
        invalid_reasons.append("missing_equity_series")
        return {
            "worst_final": None,
            "p25_final": None,
            "median_final": None,
            "p75_final": None,
            "best_final": None,
            "p0_E": None,
            "p25_E": None,
            "p50_E": None,
            "p75_E": None,
            "p100_E": None,
            "hit_rate": None,
            "window_count": 0,
        }, invalid_reasons

    df_merge = pd.merge(
        df_equity,
        df_bh,
        on="date_utc",
        how="inner",
        suffixes=("", "_bh"),
    )
    df_merge = df_merge.dropna(subset=["date_utc"])
    if df_merge.empty:
        invalid_reasons.append("missing_equity_alignment")
        return {
            "worst_final": None,
            "p25_final": None,
            "median_final": None,
            "p75_final": None,
            "best_final": None,
            "p0_E": None,
            "p25_E": None,
            "p50_E": None,
            "p75_E": None,
            "p100_E": None,
            "hit_rate": None,
            "window_count": 0,
        }, invalid_reasons

    df_merge = df_merge.sort_values("date_utc")
    start_date = df_merge["date_utc"].iloc[0].date()
    last_date = df_merge["date_utc"].iloc[-1].date()
    data_end_exclusive = last_date + timedelta(days=1)

    final_values: List[float] = []
    e_values: List[float] = []

    window_start = start_date
    while window_start + timedelta(days=ROLLING_WINDOW_DAYS) <= data_end_exclusive:
        window_end = window_start + timedelta(days=ROLLING_WINDOW_DAYS)
        mask = (df_merge["date_utc"].dt.date >= window_start) & (
            df_merge["date_utc"].dt.date < window_end
        )
        window = df_merge.loc[mask]
        if len(window) >= 2:
            equity = pd.to_numeric(window.get("equity_usdc"), errors="coerce").dropna()
            bh_equity = pd.to_numeric(window.get("bh_equity"), errors="coerce").dropna()
            if len(equity) >= 2 and len(bh_equity) >= 2:
                total_ret = metrics.total_return_from_equity(equity)
                bh_ret = metrics.total_return_from_equity(bh_equity)
                mdd, ui = metrics.mdd_and_ulcer_index(equity)
                final, e_value, _, _, _ = _compute_score(total_ret, bh_ret, mdd, ui)
                final_values.append(final)
                e_values.append(e_value)
        window_start = window_start + timedelta(days=ROLLING_STEP_DAYS)

    window_count = len(final_values)
    if window_count == 0:
        invalid_reasons.append("rolling_window_count_zero")
        return {
            "worst_final": None,
            "p25_final": None,
            "median_final": None,
            "p75_final": None,
            "best_final": None,
            "p0_E": None,
            "p25_E": None,
            "p50_E": None,
            "p75_E": None,
            "p100_E": None,
            "hit_rate": None,
            "window_count": 0,
        }, invalid_reasons

    final_arr = np.array(final_values, dtype="float64")
    e_arr = np.array(e_values, dtype="float64")
    hit_rate = float(np.sum(e_arr > 0) / window_count) if window_count > 0 else None
    return {
        "worst_final": float(np.percentile(final_arr, 0)),
        "p25_final": float(np.percentile(final_arr, 25)),
        "median_final": float(np.percentile(final_arr, 50)),
        "p75_final": float(np.percentile(final_arr, 75)),
        "best_final": float(np.percentile(final_arr, 100)),
        "p0_E": float(np.percentile(e_arr, 0)),
        "p25_E": float(np.percentile(e_arr, 25)),
        "p50_E": float(np.percentile(e_arr, 50)),
        "p75_E": float(np.percentile(e_arr, 75)),
        "p100_E": float(np.percentile(e_arr, 100)),
        "hit_rate": hit_rate,
        "window_count": window_count,
    }, invalid_reasons


def _compute_style_metrics(
    run_dir: Path,
    equity_dates: Iterable[datetime],
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    invalid_reasons: List[str] = []
    trades_path = run_dir / "trades.jsonl"
    if not trades_path.exists():
        invalid_reasons.append("missing_trades_jsonl")
        return {
            "max_consecutive_days_in_position": None,
            "max_flat_streak_days": None,
            "avg_holding_period_days": None,
        }, invalid_reasons

    trades: List[Tuple[datetime, float]] = []
    with trades_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_reasons.append("invalid_trades_jsonl")
                return {
                    "max_consecutive_days_in_position": None,
                    "max_flat_streak_days": None,
                    "avg_holding_period_days": None,
                }, invalid_reasons
            pos_frac = record.get("position_frac_after")
            if pos_frac is None:
                invalid_reasons.append("missing_position_frac_after")
                return {
                    "max_consecutive_days_in_position": None,
                    "max_flat_streak_days": None,
                    "avg_holding_period_days": None,
                }, invalid_reasons
            ts_ms = record.get("ts_ms")
            date_value = None
            if ts_ms is not None:
                try:
                    date_value = datetime.utcfromtimestamp(int(ts_ms) / 1000).date()
                except (TypeError, ValueError, OSError):
                    date_value = None
            if date_value is None:
                date_str = record.get("date_utc")
                parsed = _parse_date(str(date_str)) if date_str else None
                if parsed is not None:
                    date_value = parsed.date()
            if date_value is None:
                invalid_reasons.append("missing_trade_date")
                return {
                    "max_consecutive_days_in_position": None,
                    "max_flat_streak_days": None,
                    "avg_holding_period_days": None,
                }, invalid_reasons
            trades.append((datetime.combine(date_value, datetime.min.time()), float(pos_frac)))

    if not trades:
        invalid_reasons.append("empty_trades_jsonl")
        return {
            "max_consecutive_days_in_position": None,
            "max_flat_streak_days": None,
            "avg_holding_period_days": None,
        }, invalid_reasons

    trades.sort(key=lambda item: item[0])
    trades_by_day: Dict[datetime.date, float] = {}
    for trade_date, pos_frac in trades:
        trades_by_day[trade_date.date()] = float(pos_frac)

    daily_flags: List[bool] = []
    current_frac = 0.0
    for dt_value in equity_dates:
        if dt_value.date() in trades_by_day:
            current_frac = trades_by_day[dt_value.date()]
        daily_flags.append(abs(current_frac) > 0)

    if not daily_flags:
        invalid_reasons.append("missing_equity_calendar")
        return {
            "max_consecutive_days_in_position": None,
            "max_flat_streak_days": None,
            "avg_holding_period_days": None,
        }, invalid_reasons

    max_true = 0
    max_false = 0
    true_runs: List[int] = []
    current_state = daily_flags[0]
    current_len = 0
    for flag in daily_flags:
        if flag == current_state:
            current_len += 1
        else:
            if current_state:
                true_runs.append(current_len)
                max_true = max(max_true, current_len)
            else:
                max_false = max(max_false, current_len)
            current_state = flag
            current_len = 1
    if current_state:
        true_runs.append(current_len)
        max_true = max(max_true, current_len)
    else:
        max_false = max(max_false, current_len)

    avg_holding = float(sum(true_runs) / len(true_runs)) if true_runs else 0.0

    return {
        "max_consecutive_days_in_position": float(max_true),
        "max_flat_streak_days": float(max_false),
        "avg_holding_period_days": avg_holding,
    }, invalid_reasons


def _compute_profile_metrics(
    summary: Dict[str, Any],
    quarterly_row: Optional[Dict[str, Any]],
    df_equity: pd.DataFrame,
    trades_path: Path,
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    invalid_reasons: List[str] = []
    fields = {
        "pnl": None,
        "return_annualized": None,
        "sharpe_annualized": None,
        "upi_annualized": None,
        "mdd": None,
        "ui": None,
        "avg_net_exposure": None,
        "pct_days_in_position": None,
        "pct_days_long": None,
        "pct_days_short": None,
        "trade_count": None,
        "start_equity": None,
        "end_equity": None,
        "fees_paid": None,
        "turnover_proxy": None,
    }

    if quarterly_row:
        for key in fields.keys():
            if key in quarterly_row:
                fields[key] = _safe_float(quarterly_row.get(key))
        return fields, invalid_reasons

    if df_equity.empty:
        invalid_reasons.append("missing_equity_series_for_profile")
        return fields, invalid_reasons

    equity = pd.to_numeric(df_equity.get("equity_usdc"), errors="coerce").dropna()
    if equity.empty:
        invalid_reasons.append("missing_equity_series_for_profile")
        return fields, invalid_reasons

    start_equity = float(equity.iloc[0])
    end_equity = float(equity.iloc[-1])
    fields["start_equity"] = start_equity
    fields["end_equity"] = end_equity
    fields["pnl"] = metrics.total_return_from_equity(equity)

    days = int((df_equity["date_utc"].iloc[-1] - df_equity["date_utc"].iloc[0]).days)
    if days > 0:
        fields["return_annualized"] = metrics.annualized_return(start_equity, end_equity, days)

    returns = metrics.equity_returns_with_first_zero(equity)
    fields["sharpe_annualized"] = metrics.sharpe_annualized_from_returns(returns)
    mdd, ui = metrics.mdd_and_ulcer_index(equity)
    fields["mdd"] = mdd
    fields["ui"] = ui
    if ui and not math.isnan(ui) and ui != 0:
        fields["upi_annualized"] = (
            fields["return_annualized"] / ui
            if fields["return_annualized"] is not None
            else None
        )

    fields["avg_net_exposure"] = _safe_float(summary.get("avg_net_exposure"))
    fields["pct_days_in_position"] = _safe_float(summary.get("pct_in_position"))
    fields["pct_days_long"] = _safe_float(summary.get("pct_long"))
    fields["pct_days_short"] = _safe_float(summary.get("pct_short"))
    fields["trade_count"] = _safe_float(summary.get("trade_count"))

    if trades_path.exists():
        fees_paid = 0.0
        try:
            with trades_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("action") != "REBALANCE" or record.get("reason") != "gate_passed":
                        continue
                    fee = _safe_float(record.get("fee_usdc"))
                    if fee is None:
                        continue
                    fees_paid += fee
            fields["fees_paid"] = float(fees_paid)
        except json.JSONDecodeError:
            invalid_reasons.append("invalid_trades_jsonl_for_fees")

    if "net_exposure" in df_equity.columns:
        exposures = pd.to_numeric(df_equity["net_exposure"], errors="coerce").dropna()
        if len(exposures) >= 2:
            turnover = 0.0
            for idx in range(1, len(exposures)):
                turnover += abs(exposures.iloc[idx] - exposures.iloc[idx - 1])
            fields["turnover_proxy"] = float(turnover)

    return fields, invalid_reasons


def _rank_runs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(item: Dict[str, Any]) -> Tuple[int, float]:
        mdd_pass = item.get("mdd_pass")
        final = item.get("final")
        mdd_rank = 1 if mdd_pass else 0
        final_value = final if isinstance(final, (int, float)) else float("-inf")
        return (mdd_rank, final_value)

    return sorted(records, key=sort_key, reverse=True)


def _serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _build_rank_columns(rows: List[Dict[str, Any]]) -> List[str]:
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    remaining = sorted(key for key in all_keys if key not in PRIMARY_COLS)
    return PRIMARY_COLS + remaining


def write_rank_results_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _serialize_csv_value(row.get(col)) for col in columns})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-jsonl", default="", help="Path to runs.jsonl (default: data/backtest_result/runs.jsonl)")
    ap.add_argument("--output-root", default="data/backtest_rank", help="Output root directory")
    ap.add_argument("--symbols", default="", help="Comma-separated symbols")
    ap.add_argument("--requested-start", default="", help="Requested start (YYYY-MM-DD)")
    ap.add_argument("--requested-end", default="", help="Requested end (YYYY-MM-DD)")
    ap.add_argument("--strategy-version", default="", help="Strategy version filter")
    ap.add_argument("--param-hash-prefix", default="", help="Param hash prefix filter")
    ap.add_argument("--data-fingerprint-prefix", default="", help="Data fingerprint prefix filter")
    ap.add_argument("--run-ids", default="", help="Comma-separated run ids")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of runs")
    ap.add_argument("--spec", default="", help="Path to rank_spec.json")
    args = ap.parse_args()

    spec = _load_spec(Path(args.spec)) if args.spec else None
    if spec is None:
        args.symbols = _split_csv_arg(args.symbols)
        args.run_ids = _split_csv_arg(args.run_ids)
        args.requested_start = args.requested_start.strip() or None
        args.requested_end = args.requested_end.strip() or None
        args.strategy_version = args.strategy_version.strip() or None
        args.param_hash_prefix = args.param_hash_prefix.strip() or None
        args.data_fingerprint_prefix = args.data_fingerprint_prefix.strip() or None

    filters = _collect_filters(args, spec)
    rank_id = _canonical_hash(filters)
    output_root = Path(args.output_root)
    created_at_utc = _utc_now_compact()
    rank_dir_name = f"{created_at_utc}__{rank_id}"
    output_dir = output_root / rank_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if spec is not None:
        (output_dir / "rank_spec.json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    runs_jsonl = Path(args.runs_jsonl) if args.runs_jsonl else Path(config.BACKTEST_RESULT_DIR) / "runs.jsonl"
    run_records = _read_runs_jsonl(runs_jsonl)
    selected_runs = _filter_runs(run_records, filters)

    results: List[Dict[str, Any]] = []
    invalid_runs: List[Dict[str, Any]] = []

    for run in selected_runs:
        record = run.record
        run_id = run.run_id
        run_dir = Path(config.BACKTEST_RESULT_DIR) / run_id
        invalid_reasons: List[str] = []

        summary = _load_summary(run_dir / "summary.json")
        quarterly_row = _read_quarterly_stats(run_dir / "quarterly_stats.csv")
        df_equity, df_bh = _load_equity_frames(run_dir)
        trades_path = run_dir / "trades.jsonl"

        total_ret, bh_ret, mdd, ui = _compute_full_period_metrics(df_equity, df_bh)
        if total_ret is None or bh_ret is None or mdd is None or ui is None:
            invalid_reasons.append("missing_equity_metrics")

        final = None
        e_value = None
        mdd_pass = None
        base = None
        mdd_score = None
        if total_ret is not None and bh_ret is not None and mdd is not None and ui is not None:
            final, e_value, mdd_pass, base, mdd_score = _compute_score(total_ret, bh_ret, mdd, ui)

        rolling_metrics, rolling_invalids = _rolling_windows(df_equity, df_bh)
        invalid_reasons.extend(rolling_invalids)

        style_metrics, style_invalids = _compute_style_metrics(
            run_dir,
            df_equity["date_utc"] if not df_equity.empty else [],
        )
        invalid_reasons.extend(style_invalids)

        profile_metrics, profile_invalids = _compute_profile_metrics(
            summary,
            quarterly_row,
            df_equity,
            trades_path,
        )
        invalid_reasons.extend(profile_invalids)

        result = {
            "run_id": run_id,
            "symbol": record.get("symbol_label"),
            "symbols": record.get("symbols"),
            "start": record.get("start"),
            "end": record.get("end"),
            "param_hash": record.get("param_hash"),
            "data_fingerprint": record.get("data_fingerprint"),
            "strategy_version": record.get("strategy_version"),
            "record_line_no": run.line_no,
            "final": final,
            "E": e_value,
            "base": base,
            "mdd_score": mdd_score,
            "mdd_pass": mdd_pass,
            "mdd": mdd,
            "ui": ui,
            "R_total": total_ret,
            "R_bh_total": bh_ret,
            **rolling_metrics,
            **profile_metrics,
            **style_metrics,
            "invalid_reasons": sorted(set(invalid_reasons)),
        }
        if invalid_reasons:
            invalid_runs.append(result)
        results.append(result)

    ranked = _rank_runs(results)
    rank_output = {
        "rank_id": rank_id,
        "rank_dir_name": rank_dir_name,
        "created_at_utc": created_at_utc,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "filters": filters,
        "run_count": len(ranked),
        "results": ranked,
    }
    (output_dir / "rank_results.json").write_text(
        json.dumps(rank_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    columns = _build_rank_columns(ranked)
    write_rank_results_csv(output_dir / "rank_results.csv", ranked, columns)


if __name__ == "__main__":
    main()
