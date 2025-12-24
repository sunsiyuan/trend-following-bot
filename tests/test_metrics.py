from bot import metrics


def test_trade_win_rate_prefers_realized_pnl_usdc():
    trades = [
        {"realized_pnl_usdc": 10.0},
        {"realized_pnl_usdc": -2.0},
        {"realized_pnl_usdc": 0.0},
    ]
    assert metrics.trade_win_rate(trades) == 1 / 3


def test_trade_win_rate_falls_back_to_realized_pnl():
    trades = [
        {"realized_pnl": 5.0},
        {"realized_pnl": -1.0},
    ]
    assert metrics.trade_win_rate(trades) == 0.5
