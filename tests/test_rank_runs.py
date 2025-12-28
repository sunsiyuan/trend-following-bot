import pytest

from bot import rank_runs


def test_compute_mdd_score_guards():
    assert rank_runs.compute_mdd_score(-0.20, -0.30, -0.60) == 1.0
    assert rank_runs.compute_mdd_score(-0.30, -0.30, -0.60) == 1.0
    assert rank_runs.compute_mdd_score(-0.45, -0.30, -0.60) == pytest.approx(0.5)
    assert rank_runs.compute_mdd_score(-0.60, -0.30, -0.60) == 0.0
    assert rank_runs.compute_mdd_score(-0.80, -0.30, -0.60) == 0.0


def test_final_monotone_with_mdd_score():
    total_ret = 0.2
    bh_ret = 0.0
    ui = 0.1
    mdds = [-0.20, -0.30, -0.45, -0.60, -0.80]
    finals = [
        rank_runs._compute_score(total_ret, bh_ret, mdd, ui)[0]  # pylint: disable=protected-access
        for mdd in mdds
    ]
    assert all(prev >= nxt for prev, nxt in zip(finals, finals[1:]))
