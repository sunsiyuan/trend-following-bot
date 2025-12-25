import json

import pandas as pd
import pytest

from bot.quarterly_stats import generate_quarterly_stats


def _write_equity_by_day(path, rows):
    data = []
    for row in rows:
        data.append(
            {
                "date_utc": row["date_utc"],
                "equity": row["equity"],
                "cash_usdc": 0.0,
                "position_side": "long",
                "position_qty": 0.0,
                "avg_entry_price": 0.0,
                "mark_price": row["equity"],
                "position_value_usdc": 0.0,
                "net_exposure": row["net_exposure"],
                "unrealized_pnl_usdc": 0.0,
                "realized_pnl_usdc_cum": 0.0,
                "equity_usdc": row["equity"],
                "close_px": row["equity"],
            }
        )
    pd.DataFrame(data).to_csv(path, index=False)


def _write_equity_by_day_bh(path, rows):
    data = []
    for row in rows:
        data.append(
            {
                "date_utc": row["date_utc"],
                "close_px": row["close_px"],
                "bh_equity": row["bh_equity"],
            }
        )
    pd.DataFrame(data).to_csv(path, index=False)


def _mdd_ui(equity):
    peak = equity[0]
    dd_values = []
    dd_depth = []
    for value in equity:
        peak = max(peak, value)
        dd_values.append(value / peak - 1.0)
        dd_depth.append(1 - value / peak)
    mdd = min(dd_values)
    ui = (sum(depth**2 for depth in dd_depth) / len(dd_depth)) ** 0.5
    return mdd, ui


def test_generate_quarterly_stats(tmp_path):
    equity_rows = [
        {"date_utc": "2024-01-01", "equity": 100.0, "net_exposure": 0.5},
        {"date_utc": "2024-01-02", "equity": 110.0, "net_exposure": 0.6},
        {"date_utc": "2024-01-03", "equity": 105.0, "net_exposure": 0.4},
        {"date_utc": "2024-04-01", "equity": 120.0, "net_exposure": 0.7},
        {"date_utc": "2024-04-02", "equity": 90.0, "net_exposure": -0.2},
    ]
    bh_rows = [
        {"date_utc": "2024-01-01", "close_px": 10.0, "bh_equity": 100.0},
        {"date_utc": "2024-01-02", "close_px": 11.0, "bh_equity": 110.0},
        {"date_utc": "2024-01-03", "close_px": 10.5, "bh_equity": 105.0},
        {"date_utc": "2024-04-01", "close_px": 12.0, "bh_equity": 120.0},
        {"date_utc": "2024-04-02", "close_px": 9.0, "bh_equity": 90.0},
    ]

    equity_path = tmp_path / "equity_by_day.csv"
    bh_path = tmp_path / "equity_by_day_bh.csv"
    trades_path = tmp_path / "trades.jsonl"
    output_path = tmp_path / "quarterly_stats.csv"

    _write_equity_by_day(equity_path, equity_rows)
    _write_equity_by_day_bh(bh_path, bh_rows)

    trades = [
        {"date_utc": "2024-01-02", "action": "REBALANCE", "reason": "gate_passed", "fee_usdc": 1.0},
        {"date_utc": "2024-04-01", "action": "REBALANCE", "reason": "gate_passed", "fee_usdc": 2.0},
        {"date_utc": "2024-04-01", "action": "HOLD", "reason": "other", "fee_usdc": 3.0},
    ]
    with trades_path.open("w") as handle:
        for trade in trades:
            handle.write(json.dumps(trade) + "\n")

    generate_quarterly_stats(
        equity_by_day_path=equity_path,
        equity_by_day_bh_path=bh_path,
        trades_path=trades_path,
        output_path=output_path,
        strategy_label="long_only",
    )

    df = pd.read_csv(output_path)
    expected_order = [
        ("2024Q1", "long_only"),
        ("2024Q1", "buy_hold"),
        ("2024Q2", "long_only"),
        ("2024Q2", "buy_hold"),
        ("All 2024-01-01-2024-04-02", "long_only"),
        ("All 2024-01-01-2024-04-02", "buy_hold"),
    ]
    assert list(zip(df["period"], df["strategy"])) == expected_order

    q1_row = df[(df["period"] == "2024Q1") & (df["strategy"] == "long_only")].iloc[0]
    q1_equity = [100.0, 110.0, 105.0]
    expected_mdd_q1, expected_ui_q1 = _mdd_ui(q1_equity)
    assert q1_row["pnl"] == pytest.approx(0.05)
    assert q1_row["mdd"] == pytest.approx(expected_mdd_q1)
    assert q1_row["ui"] == pytest.approx(expected_ui_q1)
    assert q1_row["trade_count"] == 1

    q2_row = df[(df["period"] == "2024Q2") & (df["strategy"] == "long_only")].iloc[0]
    assert q2_row["pnl"] == pytest.approx(-0.25)
    assert q2_row["trade_count"] == 1

    all_row = df[(df["period"] == "All 2024-01-01-2024-04-02") & (df["strategy"] == "long_only")].iloc[0]
    all_equity = [100.0, 110.0, 105.0, 120.0, 90.0]
    expected_mdd_all, expected_ui_all = _mdd_ui(all_equity)
    assert all_row["mdd"] == pytest.approx(expected_mdd_all)
    assert all_row["ui"] == pytest.approx(expected_ui_all)
    assert all_row["trade_count"] == 2
