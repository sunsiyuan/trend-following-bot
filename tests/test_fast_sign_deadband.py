import numpy as np
import pandas as pd

from bot.strategy import StrategyState, _apply_fast_sign_deadband, decide


def test_deadband_disabled_matches_raw():
    fast_state = pd.Series([0.001, -0.001, 0.0])
    fast_sign_raw = pd.Series([1.0, -1.0, 0.0])
    eff, active = _apply_fast_sign_deadband(fast_state, fast_sign_raw, 0.0)
    assert eff.equals(fast_sign_raw)
    assert active.eq(False).all()


def test_deadband_sticky_behavior():
    fast_state = pd.Series([0.0005, -0.0004, 0.002, -0.0002])
    fast_sign_raw = pd.Series([1.0, -1.0, 1.0, -1.0])
    eff, active = _apply_fast_sign_deadband(fast_state, fast_sign_raw, 0.001)
    expected = pd.Series([1.0, 1.0, 1.0, 1.0])
    assert eff.equals(expected)
    assert active.tolist() == [True, True, False, True]


def test_deadband_nan_does_not_pollute_prev():
    fast_state = pd.Series([0.002, np.nan, -0.002, 0.0002])
    fast_sign_raw = pd.Series([1.0, np.nan, -1.0, 1.0])
    eff, active = _apply_fast_sign_deadband(fast_state, fast_sign_raw, 0.001)
    expected = pd.Series([1.0, np.nan, -1.0, -1.0])
    assert eff.equals(expected)
    assert active.tolist() == [False, False, False, True]


def test_decide_uses_fast_sign_eff_for_short():
    df_1d_feat = pd.DataFrame(
        {
            "close": [90.0],
            "hlc3": [90.0],
            "fast_sign_raw": [-1.0],
            "fast_sign_eff": [-1.0],
            "slow_sign": [1.0],
            "align": [1.0],
            "fast_deadband_active": [False],
        },
        index=[0],
    )
    df_exec_feat = pd.DataFrame(
        {"close": [90.0], "exec_ma": [100.0]},
        index=[0],
    )
    state = StrategyState()
    decision = decide(0, 0, df_1d_feat, df_exec_feat, state)
    assert decision["market_state"] == "SHORT"
    assert decision["desired_pos_frac"] < 0
