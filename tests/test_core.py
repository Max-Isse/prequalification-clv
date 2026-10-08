"""Checks against known answers. Run:  python tests/test_core.py   (or pytest, if installed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import clv, funnel, simulate  # noqa: E402


def _customers(n, seed, max_followup=24):
    rng = np.random.default_rng(seed)
    amount = np.full(n, 10_000.0)
    h_def, churn = 0.01, 0.04
    p = h_def + (1 - h_def) * churn
    exit_month = rng.geometric(p, n)
    defaulted = rng.random(n) < h_def / p
    followup = rng.integers(13, max_followup + 1, n)
    expected = (simulate.MARGIN * 10_000 - h_def * simulate.LGD * 10_000) * (1 - (1 - p) ** 24) / p
    return amount, exit_month, defaulted, followup, expected


def test_newcombe_matches_published_example():
    # Newcombe (1998), method 10: 56/70 vs 48/80 -> 0.2000 (0.0524, 0.3339)
    d, lo, hi = funnel.newcombe(56, 70, 48, 80)
    assert abs(d - 0.2) < 1e-12 and abs(lo - 0.0524) < 1e-4 and abs(hi - 0.3339) < 1e-4


def test_partitioned_equals_sample_mean_without_censoring():
    amount, exit_month, defaulted, _, _ = _customers(5_000, 0)
    mats = clv.monthly_matrices(amount, exit_month, defaulted, np.full(5_000, 24))
    assert abs(clv.partitioned_mean(*mats) - mats[2].sum(1).mean()) < 1e-6


def test_partitioned_recovers_closed_form_value_under_censoring():
    amount, exit_month, defaulted, followup, expected = _customers(400_000, 1)
    mats = clv.monthly_matrices(amount, exit_month, defaulted, followup)
    est = clv.partitioned_mean(*mats)
    naive = mats[2].sum(1).mean()
    assert abs(est / expected - 1) < 0.01  # unbiased
    assert naive / expected < 0.9  # value to date is biased low


def test_bootstrap_weights_match_duplicated_rows():
    amount, exit_month, defaulted, followup, _ = _customers(300, 2)
    w = np.random.default_rng(3).integers(0, 4, 300)
    weighted = clv.partitioned_mean(*clv.monthly_matrices(amount, exit_month, defaulted, followup), W=w[None, :].astype(float))[0]
    rep = np.repeat(np.arange(300), w)
    duplicated = clv.partitioned_mean(*clv.monthly_matrices(amount[rep], exit_month[rep], defaulted[rep], followup[rep]))
    assert abs(weighted - duplicated) < 1e-9


def test_shown_amount_is_calibrated_quantile():
    pop = simulate.population(400_000, np.random.default_rng(4))
    for sigma, q in ((0.35, 0.5), (0.25, 0.2), (0.45, 0.8)):
        share_below = (pop.offer < simulate.shown_amount(pop, sigma, q)).mean()
        assert abs(share_below - q) < 0.005


def test_model_without_information_ignores_residual():
    pop = simulate.population(1_000, np.random.default_rng(5))
    shown = simulate.shown_amount(pop, simulate.RESID_SD, 0.5)
    assert np.allclose(np.log(shown), pop.base)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
