import math

from bot import config
from bot.strategy import compute_desired_target_frac


def test_both_side_scales_long_and_short():
    assert compute_desired_target_frac(1.0, 1.0, "both_side", deadband_conf=1.0) == config.MAX_LONG_FRAC
    assert compute_desired_target_frac(-1.0, 1.0, "both_side", deadband_conf=1.0) == -config.MAX_SHORT_FRAC


def test_long_only_ignores_short_and_scales_long():
    assert compute_desired_target_frac(1.0, 1.0, "long_only", deadband_conf=1.0) == config.MAX_LONG_FRAC
    assert compute_desired_target_frac(-1.0, 1.0, "long_only", deadband_conf=1.0) == 0.0


def test_short_only_ignores_long_and_scales_short():
    assert compute_desired_target_frac(-1.0, 1.0, "short_only", deadband_conf=1.0) == -config.MAX_SHORT_FRAC
    assert compute_desired_target_frac(1.0, 1.0, "short_only", deadband_conf=1.0) == 0.0


def test_desired_target_handles_zero_or_nan():
    assert compute_desired_target_frac(0.0, 1.0, "both_side", deadband_conf=1.0) == 0.0
    assert compute_desired_target_frac(math.nan, 1.0, "both_side", deadband_conf=1.0) == 0.0
