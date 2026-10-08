"""Reproduce every number and figure in the README:  python run_all.py"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import clv, funnel, simulate, style

N_WORLDS = 200
B = 300
METHODS = {"naive": "Value to date (naive)", "complete_cohort": "Complete cohort only", "partitioned": "Partitioned (KM-weighted)"}


def ci(draws, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def single_world(tr):
    df = simulate.simulate_experiment(seed=7)
    at = funnel.arm_table(df)
    at.to_csv("results/arm_table.csv", index=False)
    rng = np.random.default_rng(1)
    t, c = df[df["treated"]], df[~df["treated"]]
    bt, bc = clv.bootstrap_arm(t, B, rng), clv.bootstrap_arm(c, B, rng)
    est = {
        "per_visitor": {"treated": float(clv.per_visitor(t)), "control": float(clv.per_visitor(c))},
        "per_customer": {"treated": clv.per_customer(t), "control": clv.per_customer(c)},
        "per_visitor_ci": {"treated": ci(bt["per_visitor"]), "control": ci(bc["per_visitor"])},
        "per_customer_ci": {"treated": ci(bt["per_customer"]), "control": ci(bc["per_customer"])},
        "clv_lift": {m: float(clv.lift(df, m)) for m in METHODS},
        "clv_lift_ci": ci(bt["per_visitor"] - bc["per_visitor"]),
        "per_customer_diff_ci": ci(bt["per_customer"] - bc["per_customer"]),
    }
    approved_t = t[t["approved"]]
    est["share_offer_at_least_shown"] = float((approved_t["offer"] >= approved_t["shown"]).mean())

    rows = []
    for b, name in enumerate(simulate.BANDS):
        m = df["band"].values == b
        lift_draws, conv_draws = clv.bootstrap_lift(df, B=B, seed=10 + b, mask=m)
        sub = df[m]
        d, lo, hi = funnel.newcombe(sub.loc[sub.treated, "drew"].sum(), sub.treated.sum(), sub.loc[~sub.treated, "drew"].sum(), (~sub.treated).sum())
        rows.append(
            {
                "band": name,
                "visitors": int(m.sum()),
                "conv_lift_pp": d * 100,
                "conv_ci_low_pp": lo * 100,
                "conv_ci_high_pp": hi * 100,
                "true_conv_lift_pp": tr["bands"][name]["conv_lift"] * 100,
                "clv_lift": clv.lift(df, mask=m),
                "clv_ci_low": ci(lift_draws)[0],
                "clv_ci_high": ci(lift_draws)[1],
                "true_clv_lift": tr["bands"][name]["clv_lift"],
                "true_clv_customer_control": tr["bands"][name]["clv_customer_control"],
            }
        )
    seg = pd.DataFrame(rows)
    seg.to_csv("results/segment_table.csv", index=False)
    return df, at, est, seg


def monte_carlo(tr, n_worlds=N_WORLDS):
    est = {m: [] for m in METHODS}
    cover_clv, cover_conv, width = [], [], []
    for s in range(1000, 1000 + n_worlds):
        df = simulate.simulate_experiment(seed=s)
        for m in METHODS:
            est[m].append(clv.lift(df, m))
        lo, hi = ci(clv.bootstrap_lift(df, B=B, seed=s)[0])
        cover_clv.append(lo <= tr["clv_lift"] <= hi)
        width.append(hi - lo)
        t, c = df[df["treated"]], df[~df["treated"]]
        _, clo, chi = funnel.newcombe(t["drew"].sum(), len(t), c["drew"].sum(), len(c))
        cover_conv.append(clo <= tr["conv_lift"] <= chi)
    out = {"n_worlds": n_worlds, "true_clv_lift": tr["clv_lift"], "estimators": {}}
    for m, v in est.items():
        v = np.array(v)
        out["estimators"][m] = {
            "mean": float(v.mean()),
            "bias": float(v.mean() - tr["clv_lift"]),
            "rmse": float(np.sqrt(((v - tr["clv_lift"]) ** 2).mean())),
            "sd": float(v.std()),
        }
    out["coverage95_clv_partitioned_bootstrap"] = float(np.mean(cover_clv))
    out["mean_ci_width_clv"] = float(np.mean(width))
    out["coverage95_conversion_newcombe"] = float(np.mean(cover_conv))
    return out, {m: np.array(v) for m, v in est.items()}


def model_development(n=1_000_000, seed=5):
    """Expected lift per visitor for each model accuracy (sigma) and shown quantile (q),
    using the simulation's known behaviour. Same visitors for every combination."""
    pop = simulate.population(n, np.random.default_rng(seed))
    c_conv, c_val = simulate.expected_outcomes(pop)
    sigmas = [0.45, 0.35, 0.25, 0.15]
    qs = np.round(np.arange(0.05, 0.951, 0.05), 2)
    rows = []
    for sg in sigmas:
        for q in qs:
            t_conv, t_val = simulate.expected_outcomes(pop, simulate.shown_amount(pop, sg, q))
            rows.append({"sigma": sg, "q": q, "conv_lift_pp": (t_conv.mean() - c_conv.mean()) * 100, "clv_lift": t_val.mean() - c_val.mean()})
    grid = pd.DataFrame(rows)
    grid.to_csv("results/model_development_grid.csv", index=False)
    best = grid.loc[grid.groupby("sigma")["clv_lift"].idxmax()].sort_values("sigma", ascending=False)
    return grid, best


def figures(df, at, est, seg, tr, mc_draws, grid, best):
    style.apply()
    # 1. experiment overview
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(at))
    a1.barh(y - 0.18, at["control"] * 100, height=0.34, color=style.MUTED, label="Control")
    a1.barh(y + 0.18, at["treated"] * 100, height=0.34, color=style.BLUE, label="Shown a pre-qualified amount")
    for i, r in at.iterrows():
        a1.text(r["treated"] * 100 + 0.6, i + 0.18, f"{r['treated']:.1%} ({r['lift_pp']:+.1f}pp)", va="center", fontsize=8.5, color=style.INK2)
        a1.text(r["control"] * 100 + 0.6, i - 0.18, f"{r['control']:.1%}", va="center", fontsize=8.5, color=style.INK2)
    a1.set_yticks(y)
    a1.set_yticklabels(["Applied", "Approved", "Drew down"])
    a1.invert_yaxis()
    a1.set_xlim(0, 52)
    a1.set_title("Conversion by arm (% of all visitors)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5)
    a1.grid(axis="y", visible=False)

    labels = ["Per visitor", "Per customer\nwho drew down"]
    keys = ["per_visitor", "per_customer"]
    x = np.arange(2)
    for j, (arm, col, off) in enumerate((("control", style.MUTED, -0.18), ("treated", style.BLUE, 0.18))):
        vals = [est[k][arm] for k in keys]
        rel = [v / est[k]["control"] for v, k in zip(vals, keys)]
        a2.bar(x + off, rel, width=0.34, color=col)
        for i, k in enumerate(keys):
            lo, hi = est[k + "_ci"][arm]
            a2.vlines(i + off, lo / est[k]["control"], hi / est[k]["control"], color=style.INK2, linewidth=1.2)
            a2.text(i + off, hi / est[k]["control"] + 0.02, f"£{vals[i]:,.0f}", ha="center", fontsize=8.5, color=style.INK2)
    a2.axhline(1, color=style.AXIS, linewidth=1)
    a2.set_xticks(x)
    a2.set_xticklabels(labels)
    a2.set_ylim(0.8, 1.22)
    a2.set_ylabel("Relative to control")
    a2.set_title("24-month CLtV (partitioned estimator, 95% CI)")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("figures/experiment.png")
    plt.close(fig)

    # 2. censoring: KM curves and estimator comparison
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    for treated, col, lab in ((False, style.MUTED, "Control"), (True, style.BLUE, "Treated")):
        d = df[(df["treated"] == treated) & df["drew"]]
        ar, ex, _ = clv.monthly_matrices(d["amount"].values, d["exit_month"].values, d["defaulted"].values, d["followup"].values)
        s = clv.km_active(ar, ex)
        a1.step(np.arange(0, simulate.H + 1), np.r_[1, s], where="post", color=col, label=lab)
    a1.axvspan(13, 24, color=style.GRID, alpha=0.5, linewidth=0)
    a1.text(13.4, 0.97, "only earlier visitors\nobserved this long", fontsize=8, color=style.INK2, va="top")
    a1.set_ylim(0, 1.02)
    a1.set_xlim(0, 24)
    a1.set_xticks([0, 6, 12, 18, 24])
    a1.set_xlabel("Months since drawdown")
    a1.set_ylabel("Share still active")
    a1.set_title("Customers still active (Kaplan-Meier)")
    a1.legend(loc="lower left")

    names = list(METHODS)
    for i, m in enumerate(names):
        v = mc_draws[m]
        jitter = np.random.default_rng(i).uniform(-0.17, 0.17, len(v))
        a2.plot(v, i + jitter, "o", color=style.BLUE if m == "partitioned" else style.ORANGE, alpha=0.35, markersize=3.5, markeredgewidth=0)
        a2.plot(v.mean(), i, "|", color=style.INK, markersize=18, markeredgewidth=2)
    a2.axvline(tr["clv_lift"], color=style.INK, linestyle="--", linewidth=1)
    a2.text(tr["clv_lift"] + 2, len(names) - 0.45, f"truth £{tr['clv_lift']:.0f}", fontsize=8.5, color=style.INK)
    a2.set_yticks(range(len(names)))
    a2.set_yticklabels([METHODS[m] for m in names], fontsize=8.5)
    a2.set_ylim(-0.6, len(names) - 0.3)
    a2.invert_yaxis()
    a2.set_xlabel("Estimated CLtV lift per visitor (£)")
    a2.set_title(f"CLtV lift: {len(mc_draws['naive'])} simulated experiments")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig("figures/censoring.png")
    plt.close(fig)

    # 3. segments
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    yb = np.arange(len(seg))
    for ax, col, lo, hi, true, ttl, fmt in (
        (a1, "conv_lift_pp", "conv_ci_low_pp", "conv_ci_high_pp", "true_conv_lift_pp", "Conversion lift (pp)", lambda v: f"{v:+.1f}pp"),
        (a2, "clv_lift", "clv_ci_low", "clv_ci_high", "true_clv_lift", "CLtV lift per visitor (£)", lambda v: f"{'+' if v >= 0 else '−'}£{abs(v):.0f}"),
    ):
        ax.axvline(0, color=style.AXIS, linewidth=1)
        ax.hlines(yb, seg[lo], seg[hi], color=style.BLUE, linewidth=2)
        ax.plot(seg[col], yb, "o", color=style.BLUE, markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.5, label="Estimate, 95% CI")
        ax.plot(seg[true], yb, "D", color=style.ORANGE, markersize=5, label="Truth")
        for i, r in seg.iterrows():
            ax.text(r[hi], i, "  " + fmt(r[col]), fontsize=8.5, color=style.INK2, va="center")
        ax.set_title(ttl)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.18)
    a1.set_yticks(yb)
    a1.set_yticklabels([f"{b} risk" for b in seg["band"]])
    a1.invert_yaxis()
    a2.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig("figures/segments.png")
    plt.close(fig)

    # 4. model development
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = {0.45: style.MUTED, 0.35: style.BLUE, 0.25: style.AQUA, 0.15: style.ORANGE}
    labels = {0.45: "σ = 0.45 (turnover and risk only)", 0.35: "σ = 0.35 (current model)", 0.25: "σ = 0.25", 0.15: "σ = 0.15"}
    for sg, g in grid.groupby("sigma", sort=False):
        ax.plot(g["q"], g["clv_lift"], color=cols[sg], label=labels[sg])
    for _, r in best.iterrows():
        ax.plot(r["q"], r["clv_lift"], "o", color=cols[r["sigma"]], markersize=6, markeredgecolor=style.SURFACE, markeredgewidth=1.2)
    cur = grid[(grid["sigma"] == simulate.SIGMA_NOW) & (np.isclose(grid["q"], simulate.Q_NOW))].iloc[0]
    ax.plot(cur["q"], cur["clv_lift"], "s", color=style.INK, markersize=6)
    ax.annotate("tested in the experiment", (cur["q"], cur["clv_lift"]), xytext=(0.58, 30), fontsize=8.5, color=style.INK2, arrowprops={"arrowstyle": "-", "color": style.MUTED, "linewidth": 0.8})
    ax.axhline(0, color=style.AXIS, linewidth=1)
    ax.set_xlabel("Amount shown = q-quantile of the model's prediction (offer ≥ amount shown for 1 − q)")
    ax.set_ylabel("CLtV lift per visitor vs no pre-qual (£)")
    ax.set_title("Model accuracy (σ) vs how cautious the amount shown is (q)")
    ax.set_ylim(-150, 150)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig("figures/model_development.png")
    plt.close(fig)


def main():
    tr = simulate.truth()
    print("\nTRUTH\n", json.dumps(tr, indent=1))
    df, at, est, seg = single_world(tr)
    print("\nARM TABLE\n", at.round(4).to_string())
    print("\nSINGLE WORLD\n", json.dumps(est, indent=1))
    print("\nSEGMENTS\n", seg.round(2).to_string())
    mc, mc_draws = monte_carlo(tr)
    print("\nMONTE CARLO\n", json.dumps(mc, indent=1))
    grid, best = model_development()
    print("\nMODEL DEVELOPMENT: best q per sigma\n", best.round(3).to_string())
    print("\nMODEL DEVELOPMENT: q = 0.5\n", grid[np.isclose(grid["q"], 0.5)].round(3).to_string())
    figures(df, at, est, seg, tr, mc_draws, grid, best)
    with open("results/results.json", "w") as f:
        json.dump(
            {
                "truth": tr,
                "single_world": est,
                "arm_table": at.to_dict("records"),
                "segments": seg.to_dict("records"),
                "monte_carlo": mc,
                "model_development_best": best.to_dict("records"),
                "model_development_q50": grid[np.isclose(grid["q"], 0.5)].to_dict("records"),
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
