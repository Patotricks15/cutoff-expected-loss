import numpy as np
import pytest

from cutoff_econ.econ import best_cutoff, breakeven_pd, cutoff_profit, expected_loss, sensitivity_grid


def test_breakeven_formula():
    assert breakeven_pd(0.12, 0.8) == pytest.approx(0.12 / 0.92)


def test_single_loan_is_zero_profit_at_breakeven():
    m, lgd = 0.12, 0.8
    p = breakeven_pd(m, lgd)
    assert (1 - p) * m - p * lgd == pytest.approx(0.0)


def test_breakeven_moves_with_margin_and_lgd():
    assert breakeven_pd(0.16, 0.8) > breakeven_pd(0.12, 0.8)
    assert breakeven_pd(0.12, 0.6) > breakeven_pd(0.12, 0.8)


def test_breakeven_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        breakeven_pd(0.1, 0.0)
    with pytest.raises(ValueError):
        breakeven_pd(-0.1, 0.5)


def test_expected_loss_formula():
    assert expected_loss(0.2, 0.5, 1000.0) == pytest.approx(100.0)


def test_best_cutoff_picks_max_row():
    table = cutoff_profit(
        pd_=[0.05, 0.1, 0.3],
        defaulted=[0, 0, 1],
        ead=[100.0, 100.0, 100.0],
        margin=0.12,
        lgd=0.8,
        cutoffs=[0.06, 0.2, 0.5],
    )
    best = best_cutoff(table)
    assert best["expected_profit"] == table["expected_profit"].max()


def test_optimal_cutoff_tracks_breakeven_on_calibrated_pd():
    rng = np.random.default_rng(42)
    pd_ = rng.uniform(0.01, 0.5, size=50_000)
    defaulted = rng.binomial(1, pd_).astype(float)
    ead = np.full_like(pd_, 1000.0)
    cutoffs = np.arange(0.01, 0.601, 0.005)
    margin, lgd = 0.12, 0.8
    table = cutoff_profit(pd_, defaulted, ead, margin=margin, lgd=lgd, cutoffs=cutoffs)
    assert abs(best_cutoff(table)["cutoff_pd"] - breakeven_pd(margin, lgd)) <= 0.02


def test_sensitivity_grid_shape_and_breakeven_column():
    rng = np.random.default_rng(0)
    pd_ = rng.uniform(0.01, 0.5, size=2_000)
    defaulted = rng.binomial(1, pd_).astype(float)
    ead = np.full_like(pd_, 1000.0)
    grid = sensitivity_grid(pd_, defaulted, ead, np.arange(0.01, 0.6, 0.01), [0.6, 0.8], [0.08, 0.16])
    assert len(grid) == 4
    assert np.allclose(grid["breakeven_pd"], [breakeven_pd(m, l) for l, m in zip(grid["lgd"], grid["margin"])])
