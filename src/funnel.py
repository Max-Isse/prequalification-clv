"""Conversion by arm with honest uncertainty (Wilson intervals, Newcombe differences)."""
import numpy as np
import pandas as pd

STAGES = ["applied", "approved", "drew"]


def wilson(k, n, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def newcombe(k1, n1, k0, n0, z: float = 1.96):
    """Difference p1 - p0 with Newcombe's hybrid score interval (method 10)."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1, z)
    l0, u0 = wilson(k0, n0, z)
    d = p1 - p0
    lo = d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return float(d), float(lo), float(hi)


def arm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of all visitors reaching each stage, by arm, and the treated-minus-control lift."""
    rows = []
    t, c = df[df["treated"]], df[~df["treated"]]
    for s in STAGES:
        k1, n1, k0, n0 = t[s].sum(), len(t), c[s].sum(), len(c)
        d, lo, hi = newcombe(k1, n1, k0, n0)
        rows.append(
            {
                "stage": s,
                "control": k0 / n0,
                "treated": k1 / n1,
                "lift_pp": d * 100,
                "lift_ci_low_pp": lo * 100,
                "lift_ci_high_pp": hi * 100,
                "relative_lift": (k1 / n1) / (k0 / n0) - 1,
            }
        )
    return pd.DataFrame(rows)
