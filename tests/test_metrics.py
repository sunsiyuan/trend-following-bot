import math

import numpy as np
import pandas as pd
import pytest

from bot import metrics


def test_ulcer_index_monotonic_increase():
    equity = [100.0, 110.0, 120.0, 130.0]
    assert metrics.ulcer_index(equity) == 0.0
    assert math.isinf(metrics.ulcer_performance_index(equity))


def test_ulcer_index_flat_equity():
    equity = pd.Series([100.0, 100.0, 100.0])
    assert metrics.ulcer_index(equity) == 0.0
    assert metrics.ulcer_performance_index(equity) == 0.0


def test_ulcer_index_drawdown_recovery():
    equity = [100.0, 80.0, 90.0, 110.0]
    ui = metrics.ulcer_index(equity)
    upi = metrics.ulcer_performance_index(equity)
    assert ui > 0.0
    assert upi > 0.0


def test_ulcer_index_known_example():
    equity = np.array([100.0, 120.0, 90.0, 110.0])
    ui = metrics.ulcer_index(equity)
    upi = metrics.ulcer_performance_index(equity)
    assert ui == pytest.approx(0.131761, rel=1e-6, abs=1e-6)
    assert upi == pytest.approx(0.7587, rel=1e-3, abs=1e-3)
