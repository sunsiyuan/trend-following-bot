import math

import numpy as np
import pandas as pd

from bot import config
from bot.strategy import compute_deadband_conf, compute_desired_target_frac


def test_deadband_conf_disabled_is_one():
    fast_state = pd.Series([0.0, 0.1, -0.2, math.nan])
    conf, active = compute_deadband_conf(fast_state, deadband_pct=0.0)

    assert conf.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert active.tolist() == [False, False, False, False]


def test_deadband_conf_scales_with_eps():
    deadband_pct = 0.01
    eps = np.log1p(deadband_pct)
    fast_state = pd.Series([0.0, eps / 2.0, eps, eps * 2.0])
    conf, active = compute_deadband_conf(fast_state, deadband_pct=deadband_pct)

    np.testing.assert_allclose(conf.to_numpy(), [0.0, 0.5, 1.0, 1.0])
    assert active.tolist() == [True, True, False, False]


def test_deadband_conf_nan_is_safe_for_target():
    fast_state = pd.Series([math.nan])
    conf, active = compute_deadband_conf(fast_state, deadband_pct=0.05)

    assert conf.iloc[0] == 1.0
    assert active.iloc[0] == False
    desired = compute_desired_target_frac(
        fast_sign=-1.0,
        align=1.0,
        direction_mode="both_side",
        deadband_conf=math.nan,
    )
    assert not math.isnan(desired)
    assert desired == -config.MAX_SHORT_FRAC


def test_deadband_scales_short_side_in_both_mode():
    desired = compute_desired_target_frac(
        fast_sign=-1.0,
        align=1.0,
        direction_mode="both_side",
        deadband_conf=0.5,
    )
    assert desired == -0.5 * config.MAX_SHORT_FRAC
