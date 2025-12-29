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


def test_final_uses_ui_floor_and_new_formula():
    total_ret = 0.2
    bh_ret = 0.1
    ui = 0.01
    mdd = -0.20
    final, e_value, _, base, mdd_score = rank_runs._compute_score(  # pylint: disable=protected-access
        total_ret, bh_ret, mdd, ui
    )
    expected = e_value * ((mdd_score / rank_runs.UI_FLOOR) ** rank_runs.GAMMA)
    assert base == pytest.approx(e_value / rank_runs.UI_FLOOR)
    assert final == pytest.approx(expected)


def test_final_allows_negative_e_value():
    total_ret = -0.1
    bh_ret = 0.0
    ui = 0.2
    mdd = -0.20
    final, _, _, _, _ = rank_runs._compute_score(  # pylint: disable=protected-access
        total_ret, bh_ret, mdd, ui
    )
    assert final < 0


def test_final_zero_when_mdd_score_zero():
    total_ret = 0.3
    bh_ret = 0.0
    ui = 0.2
    mdd = -0.80
    final, _, _, _, _ = rank_runs._compute_score(  # pylint: disable=protected-access
        total_ret, bh_ret, mdd, ui
    )
    assert final == pytest.approx(0.0)
