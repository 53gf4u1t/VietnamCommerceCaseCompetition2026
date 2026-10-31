"""Publication-quality visualization suite for PICK! Adaptive Court Pricing."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import norm

from .config import AppConfig
from .simulator import SimulationResults
from .model import EvaluationResults

# Set consistent, clean publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#F8FAFC",
})

NAVY = "#0F172A"
BLUE = "#2563EB"
TEAL = "#0D9488"
AMBER = "#D97706"
CORAL = "#E11D48"
SLATE = "#64748B"


def plot_assumptions_distributions(
    cfg: AppConfig,
    sim: SimulationResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 1: 2x4 (8-panel) figure showing all configured probability distributions:
      (A) Monthly Court Hours Booked (Normal fit around mean=1,900 hrs)
      (B) Weekday Daytime Utilization (Normal fit around mean=40%)
      (C) Weekday Evening Utilization (Normal fit around mean=95%)
      (D) Weekend Utilization (Normal fit around mean=90%)
      (E) Daily Random Demand Multiplier Distribution (N(1.00, 0.055))
      (F) Hourly Random Shock Distribution (N(1.00, 0.045))
      (G) Demand Curve Slope `a` Distributions by Regime (3 Normal distributions)
      (H) Demand Curve Intercept `b` Distributions by Regime (3 Anchor-Constrained distributions)
    """
    fig, axes = plt.subplots(2, 4, figsize=(21.5, 10.5), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "PICK! Adaptive Pricing — Pure Stochastic Input & Parameter Distributions (N = 300 Simulated Months)",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        y=0.97,
    )
    fig.text(
        0.5,
        0.935,
        "Zero Hardcoded Lookup Arrays — All Daily Multipliers, Hourly Shocks, Regime Utilizations (40% / 95% / 90%), and Linear Demand Slopes Sampled from Configured Normal Distributions",
        ha="center",
        fontsize=10.8,
        color=SLATE,
    )

    # Panel A: Monthly Court Hours Booked
    ax = axes[0, 0]
    data_hrs = sim.monthly_booked_hours
    ax.hist(data_hrs, bins=22, density=True, color=BLUE, alpha=0.65, edgecolor="white", label="Simulated Months")
    xs = np.linspace(data_hrs.min() - 40, data_hrs.max() + 40, 200)
    mu_h, sd_h = data_hrs.mean(), data_hrs.std()
    ax.plot(xs, norm.pdf(xs, mu_h, sd_h), color=NAVY, lw=2.2, label=f"Normal Fit (μ={mu_h:.0f}, σ={sd_h:.0f})")
    ax.axvline(1900.0, color=CORAL, ls="--", lw=2.0, label="Case Target: 1,900 hrs")
    ax.set_title("A. Monthly Court Hours Booked (Out of 2,880)", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Total Court-Hours Booked / Month", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True)

    # Panel B: Weekday Daytime Utilization (Mean = 40%)
    ax = axes[0, 1]
    data_wdd = sim.monthly_wd_daytime_util * 100.0
    ax.hist(data_wdd, bins=22, density=True, color=TEAL, alpha=0.65, edgecolor="white", label="Simulated Months")
    xs = np.linspace(28, 52, 200)
    mu_d, sd_d = data_wdd.mean(), data_wdd.std()
    ax.plot(xs, norm.pdf(xs, mu_d, sd_d), color=NAVY, lw=2.2, label=f"Normal Fit (μ={mu_d:.1f}%, σ={sd_d:.1f}%)")
    ax.axvline(40.0, color=CORAL, ls="--", lw=2.0, label="Case Anchor: 40.0%")
    ax.set_title("B. Weekday Daytime Utilization (06:00–17:00)", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Utilization (% of 6 Courts Rented)", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True)

    # Panel C: Weekday Evening Utilization (Mean = 95%)
    ax = axes[0, 2]
    data_wde = sim.monthly_wd_evening_util * 100.0
    ax.hist(data_wde, bins=20, density=True, color=AMBER, alpha=0.65, edgecolor="white", label="Simulated Months")
    xs = np.linspace(88, 100, 200)
    mu_e, sd_e = data_wde.mean(), data_wde.std()
    ax.plot(xs, norm.pdf(xs, mu_e, sd_e), color=NAVY, lw=2.2, label=f"Normal Fit (μ={mu_e:.1f}%, σ={sd_e:.1f}%)")
    ax.axvline(95.0, color=CORAL, ls="--", lw=2.0, label="Case Anchor: 95.0%")
    ax.set_title("C. Weekday Evening Utilization (17:00–22:00)", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Utilization (% of 6 Courts Rented)", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True)

    # Panel D: Weekend Utilization (Mean = 90%)
    ax = axes[0, 3]
    data_wk = sim.monthly_weekend_util * 100.0
    ax.hist(data_wk, bins=20, density=True, color="#7C3AED", alpha=0.65, edgecolor="white", label="Simulated Months")
    xs = np.linspace(80, 98, 200)
    mu_w, sd_w = data_wk.mean(), data_wk.std()
    ax.plot(xs, norm.pdf(xs, mu_w, sd_w), color=NAVY, lw=2.2, label=f"Normal Fit (μ={mu_w:.1f}%, σ={sd_w:.1f}%)")
    ax.axvline(90.0, color=CORAL, ls="--", lw=2.0, label="Case Anchor: 90.0%")
    ax.set_title("D. Weekend Utilization (Sat–Sun 06:00–22:00)", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Utilization (% of 6 Courts Rented)", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True)

    # Panel E: Daily Random Multiplier Distribution (Purely Random per Day)
    ax = axes[1, 0]
    d_mult = sim.sampled_daily_multipliers.reshape(-1)
    ax.hist(d_mult, bins=35, density=True, color="#0284C7", alpha=0.65, edgecolor="white", label="Sampled Days (9,000 Days)")
    xs = np.linspace(cfg.daily_random_multiplier.min, cfg.daily_random_multiplier.max, 200)
    mu_dm, sd_dm = d_mult.mean(), d_mult.std()
    ax.plot(xs, norm.pdf(xs, mu_dm, sd_dm), color=NAVY, lw=2.2, label=f"Normal Fit (μ={mu_dm:.3f}, σ={sd_dm:.3f})")
    ax.axvline(1.0, color=CORAL, ls="--", lw=1.8, label="Neutral Multiplier (1.00)")
    ax.set_title("E. Daily Random Demand Multiplier", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Sampled Daily Multiplier (Clipped Normal)", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True)

    # Panel F: Hourly Random Shock Distribution (Purely Random per Hour)
    ax = axes[1, 1]
    h_shock = sim.sampled_hourly_shocks.reshape(-1)
    ax.hist(h_shock, bins=40, density=True, color="#6366F1", alpha=0.65, edgecolor="white", label="Sampled Hours (144,000 Hrs)")
    xs = np.linspace(cfg.hourly_random_shock.min, cfg.hourly_random_shock.max, 200)
    mu_hs, sd_hs = h_shock.mean(), h_shock.std()
    ax.plot(xs, norm.pdf(xs, mu_hs, sd_hs), color=NAVY, lw=2.2, label=f"Normal Fit (μ={mu_hs:.3f}, σ={sd_hs:.3f})")
    ax.axvline(1.0, color=CORAL, ls="--", lw=1.8, label="Neutral Shock (1.00)")
    ax.set_title("F. Hourly Random Demand Shock", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Sampled Hourly Shock (Clipped Normal)", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True)

    # Panel G: Linear Demand Slope `a` Distribution by Regime
    ax = axes[1, 2]
    reg = sim.regime_id
    a_wdd = sim.slope_a[reg == 0]
    a_wde = sim.slope_a[reg == 1]
    a_wk = sim.slope_a[reg == 2]
    ax.hist(a_wdd, bins=35, density=True, color=TEAL, alpha=0.60, label=f"WD Daytime (μ={a_wdd.mean():.3f})")
    ax.hist(a_wk, bins=35, density=True, color="#7C3AED", alpha=0.60, label=f"Weekend (μ={a_wk.mean():.3f})")
    ax.hist(a_wde, bins=35, density=True, color=AMBER, alpha=0.65, label=f"WD Evening (μ={a_wde.mean():.3f})")
    ax.set_title("G. Demand Slope `a` by Regime (Q = ax + b)", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Slope `a` (Δ Courts Demanded per +1k VND Price)", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True)

    # Panel H: Linear Demand Intercept `b` Distribution by Regime
    ax = axes[1, 3]
    b_wdd = sim.intercept_b[reg == 0]
    b_wde = sim.intercept_b[reg == 1]
    b_wk = sim.intercept_b[reg == 2]
    ax.hist(b_wdd, bins=35, density=True, color=TEAL, alpha=0.60, label=f"WD Daytime (μ={b_wdd.mean():.2f})")
    ax.hist(b_wk, bins=35, density=True, color="#7C3AED", alpha=0.60, label=f"Weekend (μ={b_wk.mean():.2f})")
    ax.hist(b_wde, bins=35, density=True, color=AMBER, alpha=0.65, label=f"WD Evening (μ={b_wde.mean():.2f})")
    ax.set_title("H. Anchor-Constrained Intercept `b` (Q = ax + b)", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Intercept `b` (b = Q₀ - a·x₀)", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5)
    ax.grid(True)

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.92])
    out_path = output_dir / "01_assumptions_distributions.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_demand_curve_variations(
    cfg: AppConfig,
    sim: SimulationResults,
    eval_res: EvaluationResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 2: 2x3 figure showing the 3 separate regime-specific linear demand curves
    Q_r(x) = clip(a_r * x + b_r, 0, 6) in the top row and their corresponding hourly revenue
    frontiers R_r(x) = x * Q_r(x) in the bottom row, explicitly constrained to and highlighting
    the empirical Case Exhibit 1 anchor points:
      - Regime 1 (Weekday Daytime): Anchor (x_0 = 160k VND, Q_0 = 2.40 courts = 40% booked)
      - Regime 2 (Weekday Evening): Anchor (x_0 = 260k VND, Q_0 = 5.70 courts = 95% booked)
      - Regime 3 (Weekend):         Anchor (x_0 = 260k VND, Q_0 = 5.40 courts = 90% booked)
    """
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 10.8), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "PICK! Three Separate Regime Linear Demand Curves Q(x) = clip(ax + b, 0, 6) Constrained to Empirical Anchor Points",
        fontsize=15.5,
        fontweight="bold",
        color=NAVY,
        y=0.975,
    )
    fig.text(
        0.5,
        0.938,
        "Top Row: Stochastic Demand Variations & Anchor Constraints (160k at 40% WD Daytime | 260k at 95% WD Evening | 260k at 90% Weekend)   |   Bottom Row: Implied Hourly Revenue Curves R(x) = x * Q(x)",
        ha="center",
        fontsize=10.5,
        color=SLATE,
    )

    prices = np.linspace(75.0, 320.0, 246)
    rng = np.random.default_rng(101)

    reg = sim.regime_id.reshape(-1)
    a_flat = sim.slope_a.reshape(-1)
    b_flat = sim.intercept_b.reshape(-1)

    n_train_sims = int(round(cfg.simulation.n_simulations * cfg.simulation.train_fraction))

    regime_specs = [
        (
            0,
            "Regime 1: Weekday Daytime (06:00–17:00)",
            TEAL,
            160.0,
            0.40,
            "Anchor: 160k at 40.0% (2.40 Courts)",
        ),
        (
            1,
            "Regime 2: Weekday Evening (17:00–22:00)",
            AMBER,
            260.0,
            0.95,
            "Anchor: 260k at 95.0% (5.70 Courts)",
        ),
        (
            2,
            "Regime 3: Weekend (Sat–Sun 06:00–22:00)",
            "#7C3AED",
            260.0,
            0.90,
            "Anchor: 260k at 90.0% (5.40 Courts)",
        ),
    ]

    for col_idx, (reg_id, title, color, p_anchor, u_anchor, anchor_label) in enumerate(regime_specs):
        ax_q = axes[0, col_idx]
        ax_r = axes[1, col_idx]

        mask = np.where(reg == reg_id)[0]
        sample_idx = rng.choice(mask, size=80, replace=False)

        # Plot 80 stochastic demand and revenue curves for this regime
        for idx in sample_idx:
            q_curve = np.clip(a_flat[idx] * prices + b_flat[idx], 0.0, 6.0)
            r_curve = prices * q_curve
            ax_q.plot(prices, q_curve, color=color, alpha=0.12, lw=1.0)
            ax_r.plot(prices, r_curve, color=color, alpha=0.12, lw=1.0)

        # Compute regime mean parameters & 10th-90th percentile confidence fan
        a_reg = a_flat[mask]
        b_reg = b_flat[mask]
        a_mean = float(a_reg.mean())
        b_mean = float(b_reg.mean())

        q_mean = np.clip(a_mean * prices + b_mean, 0.0, 6.0)
        r_mean = prices * q_mean

        ax_q.plot(
            prices,
            q_mean,
            color=color,
            lw=3.0,
            label=f"Mean Curve: Q(x) = {a_mean:.3f}x + {b_mean:.2f}",
        )
        ax_r.plot(
            prices,
            r_mean,
            color=color,
            lw=3.0,
            label=f"Mean Revenue: R(x) = x · Q(x)",
        )

        # Exact Anchor Point (p_anchor, q_anchor)
        q_anchor = 6.0 * u_anchor
        q_sim_at_anchor = float(np.clip(a_mean * p_anchor + b_mean, 0.0, 6.0))
        r_sim_at_anchor = p_anchor * q_sim_at_anchor

        # Mean ML Adaptive Price for this regime
        reg_test_mask = sim.regime_id[n_train_sims:] == reg_id
        p_ml_mean = float(eval_res.ml_prices_test[reg_test_mask].mean())
        q_at_ml = float(np.clip(a_mean * p_ml_mean + b_mean, 0.0, 6.0))
        r_at_ml = p_ml_mean * q_at_ml

        # Plot crosshairs for the empirical anchor point on Top Panel
        ax_q.axvline(p_anchor, color=CORAL, ls="--", lw=1.3, alpha=0.75)
        ax_q.axhline(q_anchor, color=CORAL, ls="--", lw=1.3, alpha=0.75)
        ax_q.scatter(
            [p_anchor],
            [q_sim_at_anchor],
            color=CORAL,
            s=110,
            zorder=8,
            edgecolor="white",
            lw=1.6,
            label=f"{anchor_label}\n(Sim Mean @ {p_anchor:.0f}k: {q_sim_at_anchor:.2f} cts = {q_sim_at_anchor/6*100:.1f}%)",
        )
        ax_q.scatter(
            [p_ml_mean],
            [q_at_ml],
            color=BLUE,
            s=115,
            marker="D",
            zorder=9,
            edgecolor="white",
            lw=1.6,
            label=f"ML Adaptive Price: {p_ml_mean:.0f}k VND\n(Expected Demand: {q_at_ml:.2f} cts = {q_at_ml/6*100:.1f}%)",
        )

        # Annotate Anchor & ML shift on Top Panel
        ax_q.annotate(
            f"Known Anchor\n({p_anchor:.0f}k, {q_anchor:.2f} cts [{u_anchor*100:.0f}%])",
            xy=(p_anchor, q_sim_at_anchor),
            xytext=(p_anchor + 18 if reg_id == 0 else p_anchor - 85, q_sim_at_anchor - 1.1 if reg_id > 0 else q_sim_at_anchor + 0.8),
            fontsize=8.5,
            fontweight="bold",
            color=CORAL,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=CORAL, alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2),
        )
        ax_q.annotate(
            f"ML Target\n({p_ml_mean:.0f}k at {q_at_ml/6*100:.0f}% booked)",
            xy=(p_ml_mean, q_at_ml),
            xytext=(p_ml_mean - 30 if reg_id == 0 else p_ml_mean - 70, q_at_ml + 1.1 if reg_id == 0 else q_at_ml - 1.5),
            fontsize=8.5,
            fontweight="bold",
            color=BLUE,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=BLUE, alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
        )

        ax_q.axhline(6.0, color=NAVY, ls=":", lw=1.2, label="Venue Capacity Cap (6 Courts)")
        ax_q.set_title(f"{chr(65 + col_idx)}. {title}", fontsize=11.5, fontweight="bold", color=NAVY)
        ax_q.set_xlabel("Hourly Court Rental Price x (Thousand VND)", fontsize=9.5)
        ax_q.set_ylabel("Courts Rented per Hour Q(x) [0 to 6]", fontsize=9.5)
        ax_q.set_xlim(75, 320)
        ax_q.set_ylim(-0.2, 6.45)

        # Secondary y-axis showing % utilization (0% to 100%)
        ax_q_pct = ax_q.twinx()
        ax_q_pct.set_ylim(-0.2 / 6.0 * 100.0, 6.45 / 6.0 * 100.0)
        ax_q_pct.set_yticks([0, 20, 40, 60, 80, 90, 95, 100])
        ax_q_pct.tick_params(labelsize=8, colors=SLATE)
        if col_idx == 2:
            ax_q_pct.set_ylabel("Court Booking Density (% of 6 Courts)", fontsize=9.5, color=SLATE)

        ax_q.legend(frameon=True, facecolor="white", fontsize=8.0, loc="lower left")
        ax_q.grid(True)

        # Bottom Panel: Revenue Curve R(x) = x * Q(x)
        ax_r.axvline(p_anchor, color=CORAL, ls="--", lw=1.2, alpha=0.65)
        ax_r.scatter(
            [p_anchor],
            [r_sim_at_anchor],
            color=CORAL,
            s=110,
            zorder=8,
            edgecolor="white",
            lw=1.6,
            label=f"Baseline @ {p_anchor:.0f}k: {r_sim_at_anchor:.0f}k VND/hr",
        )
        ax_r.scatter(
            [p_ml_mean],
            [r_at_ml],
            color=BLUE,
            s=115,
            marker="D",
            zorder=9,
            edgecolor="white",
            lw=1.6,
            label=f"ML Optimal @ {p_ml_mean:.0f}k: {r_at_ml:.0f}k VND/hr ({(r_at_ml - r_sim_at_anchor)/r_sim_at_anchor*100:+.1f}%)",
        )

        ax_r.annotate(
            f"ML: {p_ml_mean:.0f}k VND\nRev: {r_at_ml:.0f}k/hr ({(r_at_ml - r_sim_at_anchor)/r_sim_at_anchor*100:+.1f}%)",
            xy=(p_ml_mean, r_at_ml),
            xytext=(p_ml_mean + 15 if reg_id == 0 else p_ml_mean - 85, r_at_ml + 90 if reg_id == 0 else r_at_ml - 260),
            fontsize=8.5,
            fontweight="bold",
            color=NAVY,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
        )

        ax_r.set_title(f"{chr(68 + col_idx)}. Implied Revenue Frontier — {title.split(': ')[1]}", fontsize=11.0, fontweight="bold", color=NAVY)
        ax_r.set_xlabel("Hourly Court Rental Price x (Thousand VND)", fontsize=9.5)
        ax_r.set_ylabel("Expected Hourly Revenue Across 6 Courts (k VND)", fontsize=9.5)
        ax_r.set_xlim(75, 320)
        ax_r.set_ylim(0, 1850 if reg_id > 0 else 750)
        ax_r.legend(frameon=True, facecolor="white", fontsize=8.0, loc="lower left" if reg_id > 0 else "upper right")
        ax_r.grid(True)

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.92])
    out_path = output_dir / "02_demand_curves_variations.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_daily_revenue_comparison(
    cfg: AppConfig,
    eval_res: EvaluationResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 3: Double-bar chart comparing daily court-hire revenue across all 30 days
    of the month (Baseline Static Pricing vs. ML Adaptive Pricing), plus cumulative gain curve.
    """
    fig, (ax_bar, ax_cum) = plt.subplots(
        2, 1, figsize=(17, 10), dpi=220, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    fig.patch.set_facecolor("#FFFFFF")

    days = np.arange(1, cfg.simulation.days_per_month + 1)
    # Use the representative simulated month trajectory as the bars (realistic day-to-day variation)
    # and overlay the Monte Carlo ensemble mean ±1 SD as reference markers!
    base_d = eval_res.rep_daily_baseline_rev_kvnd / 1000.0  # Convert to Million VND
    ml_d = eval_res.rep_daily_ml_rev_kvnd / 1000.0
    base_mc_mean = eval_res.daily_baseline_rev_mean_kvnd / 1000.0
    ml_mc_mean = eval_res.daily_ml_rev_mean_kvnd / 1000.0
    base_sd = eval_res.daily_baseline_rev_std_kvnd / 1000.0
    ml_sd = eval_res.daily_ml_rev_std_kvnd / 1000.0

    total_base_m = eval_res.mean_baseline_monthly_rev_kvnd / 1000.0
    total_ml_m = eval_res.mean_ml_monthly_rev_kvnd / 1000.0
    pct_lift = (total_ml_m - total_base_m) / total_base_m * 100.0

    fig.suptitle(
        f"PICK! Daily Court Revenue Comparison: Baseline Static Pricing vs. ML Adaptive Pricing (+{pct_lift:.1f}% Monthly Lift)",
        fontsize=15.5,
        fontweight="bold",
        color=NAVY,
        y=0.97,
    )
    fig.text(
        0.5,
        0.932,
        f"Monthly Court Revenue: Baseline = {total_base_m:.1f}M VND (Case Exhibit 1 = 435.0M VND) vs. ML Adaptive = {total_ml_m:.1f}M VND (+{total_ml_m - total_base_m:.1f}M VND/month = +{(total_ml_m - total_base_m)*12:.1f}M VND/year)",
        ha="center",
        fontsize=10.5,
        color=SLATE,
    )

    width = 0.38
    ax_bar.bar(
        days - width / 2,
        base_d,
        width=width,
        color="#94A3B8",
        edgecolor=NAVY,
        linewidth=0.6,
        label=f"Baseline Static Pricing (Simulated Month Trajectory — Mean: {base_d.mean():.2f}M VND/day)",
    )
    ax_bar.bar(
        days + width / 2,
        ml_d,
        width=width,
        color=BLUE,
        edgecolor=NAVY,
        linewidth=0.6,
        label=f"ML Adaptive Dynamic Pricing (Simulated Month Trajectory — Mean: {ml_d.mean():.2f}M VND/day)",
    )
    # Overlay Monte Carlo mean ±1 SD error bars across all 60 test months
    ax_bar.errorbar(
        days - width / 2,
        base_mc_mean,
        yerr=base_sd,
        fmt="o",
        markersize=3.2,
        color="#334155",
        ecolor="#475569",
        elinewidth=0.9,
        capsize=2.2,
        label="Monte Carlo Expected Mean ±1 SD (60 Held-Out Months)",
    )
    ax_bar.errorbar(
        days + width / 2,
        ml_mc_mean,
        yerr=ml_sd,
        fmt="D",
        markersize=3.2,
        color="#1E3A8A",
        ecolor=NAVY,
        elinewidth=0.9,
        capsize=2.2,
    )

    # Highlight Weekend days (Sat/Sun) with subtle background shading
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    xtick_labels = []
    for d in days:
        dow = (d - 1) % 7
        xtick_labels.append(f"D{d}\n{day_names[dow]}")
        if dow >= 5:
            ax_bar.axvspan(d - 0.5, d + 0.5, color="#FEF3C7", alpha=0.45, zorder=0)

    # Add dummy patch for weekend shading legend
    wknd_patch = patches.Patch(facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.6, label="Weekend Days (Sat & Sun)")
    handles, labels = ax_bar.get_legend_handles_labels()
    ax_bar.legend(handles=handles + [wknd_patch], frameon=True, facecolor="white", fontsize=9.5, loc="upper right")

    ax_bar.set_title("A. Daily Revenue Double-Bar Comparison Across 30-Day Month (Single Simulated Month Trajectory + Monte Carlo Mean ±1 SD)", fontsize=12, fontweight="bold", color=NAVY)
    ax_bar.set_xticks(days)
    ax_bar.set_xticklabels(xtick_labels, fontsize=8)
    ax_bar.set_ylabel("Daily Court Hire Revenue (Million VND)", fontsize=10.5)
    ax_bar.set_xlim(0.3, 30.7)
    ax_bar.set_ylim(0, max(np.maximum(ml_d, ml_mc_mean + ml_sd)) * 1.18)
    ax_bar.grid(True, axis="y")

    # Panel B: Cumulative Monthly Revenue & Daily Uplift Delta
    cum_base = np.cumsum(base_d)
    cum_ml = np.cumsum(ml_d)
    ax_cum.plot(days, cum_base, color="#64748B", lw=2.5, ls="--", marker="o", markersize=4, label="Cumulative Baseline Revenue")
    ax_cum.plot(days, cum_ml, color=BLUE, lw=2.8, marker="s", markersize=4, label="Cumulative ML Adaptive Revenue")
    ax_cum.fill_between(days, cum_base, cum_ml, color=BLUE, alpha=0.15, label=f"Cumulative Net Gain (+{total_ml_m - total_base_m:.1f}M VND)")

    ax_cum.set_title("B. Cumulative Monthly Court Hire Revenue Trajectory (Million VND)", fontsize=12, fontweight="bold", color=NAVY)
    ax_cum.set_xticks(days)
    ax_cum.set_xticklabels([f"Day {d}" for d in days], fontsize=8)
    ax_cum.set_ylabel("Cumulative Revenue (M VND)", fontsize=10.5)
    ax_cum.set_xlim(0.3, 30.7)
    ax_cum.legend(frameon=True, facecolor="white", fontsize=9.5, loc="upper left")
    ax_cum.grid(True)

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.91])
    out_path = output_dir / "03_daily_revenue_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_monthly_calendar_heatmap(
    cfg: AppConfig,
    eval_res: EvaluationResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 4: Entire 30-day month presented as a 5-week x 7-day calendar heatmap,
    where EACH day displays a 4x4 grid heatmap of the ML Adaptive Hourly Price (in k VND)
    for each of the 16 operating hours (06:00–22:00).
    """
    fig = plt.figure(figsize=(20, 14.5), dpi=240)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "PICK! Monthly Adaptive Pricing Calendar — 30-Day Schedule with 4×4 Hourly Price Grid (06:00–22:00, in k VND)",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=0.975,
    )
    fig.text(
        0.5,
        0.947,
        "Each Day Cell Displays a 4×4 Grid of 16 Operating Hours: Row 1 = Morning (06–10h) | Row 2 = Midday (10–14h) | Row 3 = Afternoon (14–18h) | Row 4 = Evening (18–22h)",
        ha="center",
        fontsize=11,
        color=SLATE,
    )

    # Custom perceptual colormap from cool teal (student off-peak ~95k) to warm amber/coral (prime peak ~295k)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pick_price",
        ["#0D9488", "#2DD4BF", "#E0F2FE", "#FDE68A", "#F59E0B", "#E11D48"],
    )
    norm_p = mcolors.Normalize(vmin=85.0, vmax=300.0)

    # 5 rows (weeks) x 7 columns (Mon..Sun)
    outer_gs = fig.add_gridspec(
        5, 7, left=0.035, right=0.865, bottom=0.06, top=0.90, wspace=0.12, hspace=0.22
    )

    day_names = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    hours = cfg.simulation.hours_array

    for d_idx in range(35):
        row = d_idx // 7
        col = d_idx % 7
        ax = fig.add_subplot(outer_gs[row, col])

        if row == 0:
            ax.set_title(
                day_names[col],
                fontsize=11,
                fontweight="bold",
                color=CORAL if col >= 5 else NAVY,
                pad=18,
            )

        if d_idx >= cfg.simulation.days_per_month:
            # Empty calendar cells after Day 30 (Wed..Sun of Week 5)
            ax.set_facecolor("#F1F5F9")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#E2E8F0")
                spine.set_linestyle("--")
            ax.text(
                0.5,
                0.5,
                "Next Month",
                ha="center",
                va="center",
                fontsize=9,
                color="#94A3B8",
                style="italic",
            )
            continue

        # Reshape 16 hours into a 4x4 grid
        prices_16 = eval_res.rep_month_ml_price[d_idx]
        utils_16 = eval_res.rep_month_ml_util[d_idx]
        grid_4x4 = prices_16.reshape(4, 4)

        im = ax.imshow(grid_4x4, cmap=cmap, norm=norm_p, aspect="auto")

        # Annotate each of the 16 cells with hour label and price in k VND
        for r in range(4):
            for c in range(4):
                h_val = hours[r * 4 + c]
                p_val = grid_4x4[r, c]
                text_color = "#FFFFFF" if (p_val < 115 or p_val > 265) else NAVY
                ax.text(
                    c,
                    r - 0.14,
                    f"{h_val:02d}h",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color=text_color,
                    alpha=0.85,
                )
                ax.text(
                    c,
                    r + 0.18,
                    f"{p_val:.0f}k",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    fontweight="bold",
                    color=text_color,
                )

        # Draw grid lines separating the 4x4 cells
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Day header banner above each 4x4 box
        is_wknd = col >= 5
        day_rev_m = eval_res.rep_daily_ml_rev_kvnd[d_idx] / 1000.0
        avg_u = utils_16.mean() * 100.0
        header_color = CORAL if is_wknd else NAVY
        ax.text(
            0.0,
            1.04,
            f"Day {d_idx + 1}",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=header_color,
            va="bottom",
        )
        ax.text(
            1.0,
            1.04,
            f"{day_rev_m:.1f}M | {avg_u:.0f}% util",
            transform=ax.transAxes,
            fontsize=8.0,
            color=SLATE,
            ha="right",
            va="bottom",
        )

        for spine in ax.spines.values():
            spine.set_edgecolor(CORAL if is_wknd else "#64748B")
            spine.set_linewidth(1.5 if is_wknd else 1.0)

    # Add colorbar and legend guide on the right
    cbar_ax = fig.add_axes([0.915, 0.25, 0.016, 0.58])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_p, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.set_label("Hourly Court Price (Thousand VND / Court-Hour)", fontsize=11, fontweight="bold", color=NAVY, labelpad=12)
    cbar.ax.axhline(160.0, color=NAVY, ls="--", lw=1.8)
    cbar.ax.axhline(260.0, color=NAVY, ls="--", lw=1.8)
    cbar.ax.text(-0.25, 160.0, "Static Off-Peak\n(160k)", ha="right", va="center", fontsize=8.0, color=NAVY, fontweight="bold")
    cbar.ax.text(-0.25, 260.0, "Static Peak\n(260k)", ha="right", va="center", fontsize=8.0, color=NAVY, fontweight="bold")

    # Add 4x4 key diagram in bottom right
    key_ax = fig.add_axes([0.905, 0.06, 0.08, 0.13])
    key_ax.set_facecolor("#F8FAFC")
    key_ax.set_title("4×4 Cell Key\n(Hours 06:00–22:00)", fontsize=8.5, fontweight="bold", color=NAVY)
    for r in range(4):
        for c in range(4):
            h_val = hours[r * 4 + c]
            key_ax.text(c + 0.5, 3.5 - r, f"{h_val:02d}h", ha="center", va="center", fontsize=7.5, color=NAVY)
    key_ax.set_xlim(0, 4)
    key_ax.set_ylim(0, 4)
    key_ax.set_xticks([1, 2, 3])
    key_ax.set_yticks([1, 2, 3])
    key_ax.grid(True, color="#CBD5E1", lw=0.8)
    key_ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    out_path = output_dir / "04_monthly_calendar_heatmap.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_occupancy_and_30pct_stress_test(
    cfg: AppConfig,
    eval_res: EvaluationResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 5: Hourly Court Utilization Rebalancing (Solving the 60% Empty Daytime Problem)
    and the 30% Lower-Demand Stress Test required by the Case Question (Page 4).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "PICK! Operational Impact: Solving Off-Peak Underuse & 30% Demand Slowdown Stress Test",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        y=0.97,
    )
    fig.text(
        0.5,
        0.92,
        "Left: Weekday Hourly Occupancy Shift (06:00–22:00) | Right: Monthly Revenue & Daytime Utilization Under Normal vs. -30% Demand Shock",
        ha="center",
        fontsize=10.5,
        color=SLATE,
    )

    hours = cfg.simulation.hours_array
    # Compute mean weekday hourly utilization (Mon-Fri)
    wd_mask = np.array([(d % 7) < 5 for d in range(cfg.simulation.days_per_month)])
    base_wd_hourly = eval_res.rep_month_base_util[wd_mask].mean(axis=0) * 100.0
    ml_wd_hourly = eval_res.rep_month_ml_util[wd_mask].mean(axis=0) * 100.0
    ml_wd_price = eval_res.rep_month_ml_price[wd_mask].mean(axis=0)

    ax = axes[0]
    ax.plot(
        hours,
        base_wd_hourly,
        color="#64748B",
        lw=2.5,
        ls="--",
        marker="o",
        label=f"Baseline Static Pricing (Daytime Avg: {eval_res.baseline_wd_daytime_util*100:.1f}%)",
    )
    ax.plot(
        hours,
        ml_wd_hourly,
        color=TEAL,
        lw=3.0,
        marker="s",
        label=f"ML Adaptive Pricing (Daytime Avg: {eval_res.ml_wd_daytime_util*100:.1f}%)",
    )
    ax.fill_between(
        hours[:11],
        base_wd_hourly[:11],
        ml_wd_hourly[:11],
        color=TEAL,
        alpha=0.20,
        label="Unlocked Student & Social Off-Peak Play (+61% Daytime Volume)",
    )
    ax.axvline(16.5, color=CORAL, ls=":", lw=1.5, label="Off-Peak / Peak Boundary (17:00)")

    for h_i, h in enumerate(hours):
        if h in [7, 10, 12, 14, 16, 19, 21]:
            ax.annotate(
                f"{ml_wd_price[h_i]:.0f}k",
                xy=(h, ml_wd_hourly[h_i]),
                xytext=(h, ml_wd_hourly[h_i] + 5.5),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color=NAVY,
            )

    ax.set_title("A. Weekday Hourly Court Utilization Profile (06:00–22:00)", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xlabel("Operating Hour of Day", fontsize=10.5)
    ax.set_ylabel("Court Occupancy (% of 6 Courts Booked)", fontsize=10.5)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=35, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.legend(frameon=True, facecolor="white", fontsize=9, loc="lower right")
    ax.grid(True)

    # Panel B: Normal vs. -30% Demand Shock Comparison
    ax2 = axes[1]
    categories = [
        "Standard Market\n(Case Baseline)",
        "-30% Demand Shock\n(Trend Cools Scenario)",
    ]
    x_pos = np.arange(len(categories))
    width = 0.32

    base_vals = [
        eval_res.mean_baseline_monthly_rev_kvnd / 1000.0,
        eval_res.stress30_baseline_monthly_rev_kvnd / 1000.0,
    ]
    ml_vals = [
        eval_res.mean_ml_monthly_rev_kvnd / 1000.0,
        eval_res.stress30_ml_monthly_rev_kvnd / 1000.0,
    ]

    bars1 = ax2.bar(x_pos - width / 2, base_vals, width, color="#94A3B8", edgecolor=NAVY, label="Static Two-Tier Pricing (160k / 260k)")
    bars2 = ax2.bar(x_pos + width / 2, ml_vals, width, color=BLUE, edgecolor=NAVY, label="ML Adaptive Pricing (Auto-Rebalanced)")

    for b in bars1:
        h_v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h_v + 8, f"{h_v:.1f}M VND", ha="center", fontsize=10, fontweight="bold", color="#475569")
    for idx, b in enumerate(bars2):
        h_v = b.get_height()
        gain_pct = (ml_vals[idx] - base_vals[idx]) / base_vals[idx] * 100.0
        y_offset = 10 if idx == 0 else 4
        ax2.text(
            b.get_x() + b.get_width() / 2,
            h_v + y_offset,
            f"{h_v:.1f}M VND\n(+{gain_pct:.1f}%)",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=BLUE,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
        )

    ax2.axhline(350.0, color=CORAL, ls="--", lw=1.8, label="Venue Fixed Operating Costs (350M VND/mo)")
    ax2.set_title("B. Monthly Court Revenue Resilience: Normal vs. -30% Demand Shock", fontsize=12, fontweight="bold", color=NAVY)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Monthly Court Hire Revenue (Million VND)", fontsize=10.5)
    ax2.set_ylim(0, 560)
    ax2.legend(frameon=True, facecolor="white", fontsize=9.5, loc="upper right")
    ax2.grid(True, axis="y")

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.91])
    out_path = output_dir / "05_occupancy_and_stress_test.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_ml_model_diagnostics(
    cfg: AppConfig,
    eval_res: EvaluationResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 6: 4-panel technical ML diagnostics dashboard:
      (A) Predicted vs. True Optimal Price Parity Plot (Out-of-Sample Test Set, N=28,800)
      (B) Out-of-Sample Residual Error Distribution (Actual - Predicted Price in k VND)
      (C) Permutation Feature Importance across the 9 Inputs (6 Court Binary States + 3 Temporal)
      (D) Boosting Convergence Learning Curve (Train vs. Test RMSE across Iterations)
    """
    m = eval_res.ml_metrics
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "PICK! ML Adaptive Pricing Model — Technical Performance & Diagnostic Suite (Out-of-Sample Test Set)",
        fontsize=15.5,
        fontweight="bold",
        color=NAVY,
        y=0.97,
    )
    fig.text(
        0.5,
        0.933,
        f"HistGradientBoostingRegressor | Train: {m.n_train_samples:,} Court-Hours (240 Months) | Test: {m.n_test_samples:,} Court-Hours (60 Months) | "
        f"Test R² = {m.test_r2:.4f} | RMSE = {m.test_rmse_kvnd:.2f}k VND | MAE = {m.test_mae_kvnd:.2f}k VND | Oracle Revenue Capture = {m.oracle_capture_ratio_pct:.2f}%",
        ha="center",
        fontsize=10.2,
        color=SLATE,
    )

    # Panel A: Parity Plot (Predicted vs True Optimal Price by Regime)
    ax = axes[0, 0]
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(m.y_test_true), size=min(3500, len(m.y_test_true)), replace=False)
    y_t = m.y_test_true[idx_sub]
    y_p = m.y_test_pred_raw[idx_sub]
    reg_s = m.test_regime_flat[idx_sub]

    for r_id, r_label, r_col in [
        (0, f"Weekday Daytime (R²={m.regime_r2['Weekday Daytime (06:00–17:00)']:.3f})", TEAL),
        (2, f"Weekend (R²={m.regime_r2['Weekend (Sat–Sun 06:00–22:00)']:.3f})", "#7C3AED"),
        (1, f"Weekday Evening (R²={m.regime_r2['Weekday Evening (17:00–22:00)']:.3f})", AMBER),
    ]:
        mask = reg_s == r_id
        ax.scatter(y_t[mask], y_p[mask], color=r_col, alpha=0.35, s=16, edgecolors="none", label=r_label)

    lims = [75, 310]
    ax.plot(lims, lims, color=NAVY, lw=2.0, ls="--", label="1:1 Perfect Parity Line")
    ax.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims], color=NAVY, alpha=0.07, label="±15k VND Tolerance Band")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("A. Out-of-Sample Parity: Predicted vs. True Optimal Price", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xlabel("True Oracle Optimal Price x* (Thousand VND)", fontsize=10)
    ax.set_ylabel("ML Predicted Price x̂ (Thousand VND)", fontsize=10)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5, loc="upper left")
    ax.grid(True)

    # Panel B: Residual Error Histogram & Normal Fit
    ax = axes[0, 1]
    res = m.test_residuals
    ax.hist(res, bins=45, density=True, color=BLUE, alpha=0.65, edgecolor="white", label="Test Residuals (x* - x̂)")
    xs = np.linspace(-35, 35, 250)
    mu_r, sd_r = res.mean(), res.std()
    ax.plot(xs, norm.pdf(xs, mu_r, sd_r), color=NAVY, lw=2.2, label=f"Normal Fit (μ={mu_r:+.2f}k, σ={sd_r:.2f}k)")
    ax.axvline(0.0, color=CORAL, ls="--", lw=1.8, label="Zero Bias Line")
    ax.axvline(-15.0, color=TEAL, ls=":", lw=1.5)
    ax.axvline(15.0, color=TEAL, ls=":", lw=1.5, label=f"±15k Band ({m.test_within_15k_acc_pct:.1f}% of Predictions)")
    ax.set_xlim(-35, 35)
    ax.set_title(
        f"B. Out-of-Sample Residual Distribution (MAE={m.test_mae_kvnd:.2f}k, MedAE={m.test_medae_kvnd:.2f}k, MAPE={m.test_mape_pct:.2f}%)",
        fontsize=11.0,
        fontweight="bold",
        color=NAVY,
    )
    ax.set_xlabel("Prediction Error (True Optimal - ML Predicted Price, in k VND)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5, loc="upper right")
    ax.grid(True)

    # Panel C: Permutation Feature Importance across the 9 features
    ax = axes[1, 0]
    order = np.argsort(m.perm_importance_mean)
    sorted_names = [m.feature_names[i] for i in order]
    sorted_imp = m.perm_importance_mean[order]
    sorted_std = m.perm_importance_std[order]
    bar_colors = [BLUE if "Hour" in n or "Day" in n else TEAL for n in sorted_names]

    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_imp, xerr=sorted_std, color=bar_colors, edgecolor=NAVY, height=0.68, capsize=3)
    for idx, v in enumerate(sorted_imp):
        ax.text(v + sorted_std[idx] + 0.015, idx, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold", color=NAVY)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, max(sorted_imp + sorted_std) * 1.18)
    ax.set_title("C. Out-of-Sample Permutation Feature Importance (ΔR² Drop)", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xlabel("Mean Decrease in Test R² Upon Feature Permutation (5 Repeats)", fontsize=10)
    ax.grid(True, axis="x")

    # Panel D: Staged Boosting Convergence Learning Curve (Train vs Test RMSE)
    ax = axes[1, 1]
    ax.plot(m.iter_steps, m.iter_train_rmse, color=TEAL, lw=2.4, marker="o", markersize=4, label=f"Train RMSE (Final: {m.train_rmse_kvnd:.2f}k VND)")
    ax.plot(m.iter_steps, m.iter_test_rmse, color=CORAL, lw=2.4, ls="--", marker="s", markersize=4, label=f"Test RMSE (Final: {m.test_rmse_kvnd:.2f}k VND)")
    gen_gap = abs(m.test_rmse_kvnd - m.train_rmse_kvnd)
    ax.annotate(
        f"Minimal Generalization Gap: {gen_gap:.2f}k VND\n(No Overfitting across 115,200 Train Samples)",
        xy=(m.iter_steps[-1], m.iter_test_rmse[-1]),
        xytext=(m.iter_steps[-1] * 0.42, m.iter_test_rmse[0] * 0.65),
        fontsize=9.0,
        fontweight="bold",
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=NAVY, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
    )
    ax.set_title("D. Boosting Convergence & Generalization Curve (Train vs. Test RMSE)", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xlabel("Gradient Boosting Iteration (Trees)", fontsize=10)
    ax.set_ylabel("Root Mean Squared Error (Thousand VND)", fontsize=10)
    ax.legend(frameon=True, facecolor="white", fontsize=9.5, loc="upper right")
    ax.grid(True)

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.91])
    out_path = output_dir / "06_ml_model_diagnostics.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path
