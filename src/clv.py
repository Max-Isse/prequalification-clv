"""Customer lifetime value (CLtV) from censored follow-up.

Each customer who draws down earns a monthly margin while active and may default
(a one-off loss) or leave. At the analysis date, customers who arrived later have
been observed for fewer months, so their 24-month value is not yet known.

    naive_to_date   value observed so far, per visitor (biased low)
    complete_cohort only visitors whose full 24 months have been observed (unbiased, noisy)
    partitioned     month-by-month estimator: P(still active at month t), from a
                    Kaplan-Meier curve, times the mean value earned in month t by
                    customers observed to be active then, summed over months
                    (Lin et al., 1997). Unbiased when follow-up length is unrelated
                    to outcomes, and it uses every customer.

partitioned_mean takes optional bootstrap weights W (B x n): B weighted copies of the
estimate are computed at once by matrix products (a Poisson bootstrap).
"""
import numpy as np

from .simulate import H, LGD, MARGIN


def monthly_matrices(amount, exit_month, defaulted, followup, horizon=H):
    """(n, horizon) matrices for customers who drew down: observed and active at the
    start of month t, observed exit in month t, and value earned in month t."""
    t = np.arange(1, horizon + 1)[None, :]
    observed = t <= followup[:, None]
    at_risk = observed & (t <= exit_month[:, None])
    exits = observed & (t == exit_month[:, None])
    value = at_risk * (MARGIN * amount)[:, None] - (exits & defaulted[:, None]) * (LGD * amount)[:, None]
    return at_risk.astype(float), exits.astype(float), value


def _sum(M, W):
    return M.sum(0) if W is None else W @ M


def km_active(at_risk, exits):
    """Kaplan-Meier share of customers still active at the end of each month."""
    R, D = at_risk.sum(0), exits.sum(0)
    return np.cumprod(1 - np.where(R > 0, D / np.maximum(R, 1), 0.0))


def partitioned_mean(at_risk, exits, value, W=None):
    """Mean CLtV per customer who drew down (or B bootstrap copies if W is given)."""
    R, D, V = _sum(at_risk, W), _sum(exits, W), _sum(value, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        hazard = np.where(R > 0, D / R, 0.0)
        mean_value = np.where(R > 0, V / R, 0.0)
    surv = np.cumprod(1 - hazard, axis=-1)
    surv_before = np.concatenate([np.ones(surv.shape[:-1] + (1,)), surv[..., :-1]], axis=-1)
    return (surv_before * mean_value).sum(-1)


def per_visitor(arm, method="partitioned"):
    """CLtV per visitor for one arm (a DataFrame of visitors)."""
    d = arm[arm["drew"]]
    mats = monthly_matrices(d["amount"].values, d["exit_month"].values, d["defaulted"].values, d["followup"].values)
    if method == "partitioned":
        return len(d) / len(arm) * partitioned_mean(*mats)
    if method == "naive":
        return mats[2].sum() / len(arm)
    raise ValueError(method)


def complete_cohort(arm):
    """Per-visitor CLtV using only visitors with the full horizon observed."""
    full = arm[arm["followup"] >= H]
    d = full[full["drew"]]
    mats = monthly_matrices(d["amount"].values, d["exit_month"].values, d["defaulted"].values, d["followup"].values)
    return mats[2].sum() / len(full)


def per_customer(arm):
    """Partitioned CLtV per customer who drew down (a descriptive, not causal, number)."""
    d = arm[arm["drew"]]
    return float(partitioned_mean(*monthly_matrices(d["amount"].values, d["exit_month"].values, d["defaulted"].values, d["followup"].values)))


def bootstrap_arm(arm, B, rng):
    """Poisson bootstrap draws for one arm: CLtV per visitor, conversion rate and
    CLtV per customer who drew down."""
    d = arm[arm["drew"]]
    W = rng.poisson(1.0, (B, len(d))).astype(float)
    W_others = rng.poisson(len(arm) - len(d), B).astype(float)
    mats = monthly_matrices(d["amount"].values, d["exit_month"].values, d["defaulted"].values, d["followup"].values)
    per_cust = partitioned_mean(*mats, W=W)
    conv = W.sum(1) / (W.sum(1) + W_others)
    return {"per_visitor": conv * per_cust, "conv": conv, "per_customer": per_cust}


def bootstrap_lift(df, B=300, seed=0, mask=None):
    """Poisson bootstrap of the treated-minus-control CLtV per visitor (partitioned)
    and of the conversion lift. Returns (clv_lift_draws, conv_lift_draws)."""
    rng = np.random.default_rng(seed)
    if mask is not None:
        df = df[mask]
    t = bootstrap_arm(df[df["treated"]], B, rng)
    c = bootstrap_arm(df[~df["treated"]], B, rng)
    return t["per_visitor"] - c["per_visitor"], t["conv"] - c["conv"]


def lift(df, method="partitioned", mask=None):
    if mask is not None:
        df = df[mask]
    t, c = df[df["treated"]], df[~df["treated"]]
    if method == "complete_cohort":
        return complete_cohort(t) - complete_cohort(c)
    return float(per_visitor(t, method=method) - per_visitor(c, method=method))
