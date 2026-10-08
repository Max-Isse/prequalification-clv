"""Simulate a pre-qualification experiment for SME lending.

Real lending data is private, so this repo uses a synthetic population whose true
behaviour is known. That lets every estimate be checked against ground truth.

Journey: visit -> apply -> approved -> draw down -> monthly margin until the customer
leaves or defaults, over a 24-month CLtV horizon.

Treatment: before applying, the customer sees an indicative amount ("you could
borrow up to £X") from a pre-qualification model. Underwriting's final offer is
log(offer) = base + eps, where base depends on turnover and risk and eps has sd
RESID_SD. The model sees a noisy signal of eps, so its predictive distribution for
log(offer) is normal with sd `sigma` (the model error). The amount shown is the
q-quantile of that distribution, so the final offer meets or beats it for a share
1 - q of customers. Holding eps fixed and changing only what the model knows keeps
the comparison between model versions fair (common random numbers).

All behavioural parameters below are my assumptions, not estimates.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

H = 24  # CLtV horizon, months
MARGIN = 0.015  # monthly net margin on the amount drawn
LGD = 0.8  # loss given default, share of the amount drawn
CHURN = 0.04  # monthly chance an active customer leaves
RESID_SD = 0.45  # sd of log(offer) not explained by turnover and risk
SIGMA_NOW = 0.35  # current pre-qualification model error (log scale)
Q_NOW = 0.5  # current choice: show the model's median amount
ENROL_MONTHS = 12  # visitors arrive over 12 months
ANALYSIS_MONTH = 24  # analysis date, months after the first visitor
BANDS = ("low", "medium", "high")


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Population:
    risk: np.ndarray  # standardised credit risk score, higher = riskier
    band: np.ndarray  # 0, 1, 2 = low, medium, high risk (population thirds)
    need: np.ndarray  # amount the business wants, £
    offer: np.ndarray  # what underwriting would offer, £
    base: np.ndarray  # log-offer explained by turnover and risk
    eps: np.ndarray  # log-offer residual underwriting will see
    e_std: np.ndarray  # standard normal noise in the model's signal
    p_approve: np.ndarray
    h_default: np.ndarray  # monthly default hazard

    def __len__(self):
        return len(self.risk)


def population(n, rng):
    risk = rng.normal(0, 1, n)
    turnover = np.exp(rng.normal(np.log(30_000), 0.7, n))  # monthly turnover, £
    need = turnover * np.exp(rng.normal(np.log(0.6), 0.5, n))
    base = np.log(turnover) + np.log(0.8) - 0.25 * risk
    eps = rng.normal(0, RESID_SD, n)
    pd12 = expit(-3.2 + 1.1 * risk)
    return Population(
        risk=risk,
        band=np.digitize(risk, norm.ppf([1 / 3, 2 / 3])),
        need=need,
        offer=np.exp(base + eps),
        base=base,
        eps=eps,
        e_std=rng.normal(0, 1, n),
        p_approve=expit(1.2 - 1.0 * risk),
        h_default=1 - (1 - pd12) ** (1 / 12),
    )


def shown_amount(pop, sigma=SIGMA_NOW, q=Q_NOW):
    """q-quantile of the model's predictive distribution for the final offer."""
    if sigma >= RESID_SD:
        known = np.zeros(len(pop))
    elif sigma <= 0:
        known = pop.eps
    else:
        nu = sigma * RESID_SD / np.sqrt(RESID_SD**2 - sigma**2)  # signal noise sd
        k = RESID_SD**2 / (RESID_SD**2 + nu**2)
        known = k * (pop.eps + nu * pop.e_std)  # posterior mean; posterior sd = sigma
    return np.exp(pop.base + known + sigma * norm.ppf(q))


def p_apply(pop, shown=None):
    x = -0.9 - 0.15 * pop.risk
    if shown is not None:
        # reassurance helps most where eligibility is least certain; a low amount puts people off
        x = x + 0.35 + 0.25 * pop.risk + 0.6 * np.clip(np.log(shown / pop.need), -1.5, 0.3)
    return expit(x)


def p_draw(pop, shown=None):
    x = 1.0 + 0.8 * np.clip(np.log(pop.offer / pop.need), -1.5, 0.5)
    if shown is not None:
        # disappointment: an offer below the amount shown loses customers
        x = x - 2.0 * np.clip(np.log(shown / pop.offer), 0, 1)
    return expit(x)


def amount_drawn(pop):
    return np.minimum(pop.offer, 1.2 * pop.need)


def exit_hazard(pop):
    return pop.h_default + (1 - pop.h_default) * CHURN


def expected_value(pop, horizon=H):
    """Expected CLtV of a customer who draws down, over `horizon` months (closed form)."""
    amt = amount_drawn(pop)
    p = exit_hazard(pop)
    per_month = MARGIN * amt - pop.h_default * LGD * amt
    return per_month * (1 - (1 - p) ** horizon) / p


def expected_outcomes(pop, shown=None):
    """Per-visitor probability of drawing down and expected CLtV (no sampling noise)."""
    conv = p_apply(pop, shown) * pop.p_approve * p_draw(pop, shown)
    return conv, conv * expected_value(pop)


def simulate_experiment(seed=7, n_per_arm=50_000, sigma=SIGMA_NOW, q=Q_NOW):
    """Randomised experiment: half the visitors see a pre-qualified amount.
    Value is observed month by month until the analysis date (administrative censoring)."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm
    pop = population(n, rng)
    treat = np.zeros(n, bool)
    treat[rng.permutation(n)[:n_per_arm]] = True
    shown_all = shown_amount(pop, sigma, q)
    shown = np.where(treat, shown_all, np.nan)
    pa = np.where(treat, p_apply(pop, shown_all), p_apply(pop))
    pdw = np.where(treat, p_draw(pop, shown_all), p_draw(pop))
    applied = rng.random(n) < pa
    approved = applied & (rng.random(n) < pop.p_approve)
    drew = approved & (rng.random(n) < pdw)
    p_exit = exit_hazard(pop)
    exit_month = rng.geometric(p_exit)  # first month that ends in default or leaving
    defaulted = rng.random(n) < pop.h_default / p_exit
    enrol = rng.integers(0, ENROL_MONTHS, n)
    amt = amount_drawn(pop)
    return pd.DataFrame(
        {
            "treated": treat,
            "risk": pop.risk,
            "band": pop.band,
            "need": pop.need,
            "shown": shown,
            "offer": pop.offer,
            "applied": applied,
            "approved": approved,
            "drew": drew,
            "amount": np.where(drew, amt, 0.0),
            "exit_month": np.where(drew, exit_month, 0),
            "defaulted": drew & defaulted,
            "enrol_month": enrol,
            "followup": np.minimum(H, ANALYSIS_MONTH - enrol),
        }
    )


def truth(n=2_000_000, seed=99, sigma=SIGMA_NOW, q=Q_NOW):
    """Population-level true effects from expected values over a large sample."""
    pop = population(n, np.random.default_rng(seed))
    c_conv, c_val = expected_outcomes(pop)
    t_conv, t_val = expected_outcomes(pop, shown_amount(pop, sigma, q))
    out = {
        "conv_control": c_conv.mean(),
        "conv_treated": t_conv.mean(),
        "clv_visitor_control": c_val.mean(),
        "clv_visitor_treated": t_val.mean(),
        "clv_customer_control": c_val.sum() / c_conv.sum(),
        "clv_customer_treated": t_val.sum() / t_conv.sum(),
        "bands": {},
    }
    for b, name in enumerate(BANDS):
        m = pop.band == b
        out["bands"][name] = {
            "conv_lift": float(t_conv[m].mean() - c_conv[m].mean()),
            "clv_lift": float(t_val[m].mean() - c_val[m].mean()),
            "clv_customer_control": float(c_val[m].sum() / c_conv[m].sum()),
        }
    out["conv_lift"] = out["conv_treated"] - out["conv_control"]
    out["clv_lift"] = out["clv_visitor_treated"] - out["clv_visitor_control"]
    return {k: (float(v) if not isinstance(v, dict) else v) for k, v in out.items()}
