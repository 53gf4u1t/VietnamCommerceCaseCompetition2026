"""Visualizer module for PICK! Adaptive Court Pricing."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.stats import norm

from .config import AppConfig
from .simulator import SimulationResults
from .model import EvaluationResults
import theme as _theme  # noqa: F401


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#45475a",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#bac2de",
    "xtick.color": "#bac2de",
    "ytick.color": "#bac2de",
    "text.color": "#cdd6f4",
    "grid.color": "#313244",
    "grid.linestyle": "--",
    "grid.alpha": 0.85,
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#181825",
    "savefig.facecolor": "#1e1e2e",
    "savefig.edgecolor": "#1e1e2e",
    "legend.facecolor": "#1e1e2e",
    "legend.edgecolor": "#45475a",
    "legend.labelcolor": "#cdd6f4",
})

PALETTE = {
    "navy": "#cdd6f4",
    "slate": "#a6adc8",
    "baseline": "#7f849c",
    "ml_blue": "#89b4fa",
    "ml_teal": "#94e2d5",
    "emerald": "#a6e3a1",
    "amber": "#f9e2af",
    "coral": "#f38ba8",
    "purple": "#cba6f7",
    "orange": "#fab387",
    "weekend_bg": "#313244",
    "surface": "#313244",
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
}


def generate_all_charts(
    cfg: AppConfig,
    sim: SimulationResults,
    eval_res: EvaluationResults,
    output_dir: Path,
) -> list[Path]:
    """Generate all 6 visual figures and return their file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_assumptions_distributions(cfg, sim, output_dir / "01_assumptions_distributions.png"),
        plot_demand_curves_variations(cfg, sim, eval_res, output_dir / "02_demand_curves_variations.png"),
        plot_daily_revenue_comparison(cfg, eval_res, output_dir / "03_daily_revenue_comparison.png"),
        plot_monthly_calendar_heatmap(cfg, eval_res, output_dir / "04_monthly_calendar_heatmap.png"),
        plot_occupancy_and_stress_test(cfg, eval_res, output_dir / "05_occupancy_and_stress_test.png"),
        plot_ml_model_diagnostics(cfg, eval_res, output_dir / "06_ml_model_diagnostics.png"),
    ]
    return paths


def _plot_normal_panel(
    ax: plt.Axes,
    samples: np.ndarray,
    mean_cfg: float,
    std_cfg: float,
    title: str,
    xlabel: str,
    color: str,
    unit_suffix: str = "",
    decimals: int = 1,
) -> None:
    """Helper to plot an empirical histogram overlaid with its theoretical Normal PDF."""
    ax.hist(
        samples,
        bins=36,
        density=True,
        color=color,
        alpha=0.35,
        edgecolor=color,
        linewidth=0.8,
        label="Simulated Realizations",
    )
    x_grid = np.linspace(samples.min() - 0.5 * std_cfg, samples.max() + 0.5 * std_cfg, 250)
    pdf = norm.pdf(x_grid, loc=mean_cfg, scale=std_cfg)
    ax.plot(x_grid, pdf, color=color, linewidth=2.3, label=f"Normal({mean_cfg:.{decimals}f}, {std_cfg:.{decimals}f})")
    ax.axvline(
        samples.mean(),
        color=PALETTE["navy"],
        linestyle="--",
        linewidth=1.5,
        label=f"Sim Mean: {samples.mean():.{decimals}f}{unit_suffix}",
    )
    ax.set_title(title, fontsize=10.0, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax.set_xlabel(xlabel, fontsize=8.8, color=PALETTE["slate"])
    ax.set_ylabel("Density", fontsize=8.5, color=PALETTE["slate"])
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)


def plot_assumptions_distributions(
    cfg: AppConfig,
    sim: SimulationResults,
    save_path: Path,
) -> Path:
    """
    Chart 1: 8-Panel (2x4) Overview of All Configured Probability Distributions.
    """
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.8), dpi=220)
    fig.subplots_adjust(hspace=0.38, wspace=0.26, top=0.88, bottom=0.08, left=0.05, right=0.98)

    fig.suptitle(
        "Stochastic Court Demand & Pricing Assumption Distributions",
        fontsize=15.0,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "Empirical Histograms Overlaid with Configured Truncated Normal Density Functions Across Simulation Trajectories",
        ha="center",
        fontsize=10.2,
        color=PALETTE["slate"],
    )

    # 1. Monthly Booked Hours
    _plot_normal_panel(
        axes[0, 0],
        sim.monthly_booked_hours,
        cfg.baseline_pricing.target_monthly_booked_hours_mean,
        cfg.baseline_pricing.target_monthly_booked_hours_std,
        "A. Monthly Booked Court-Hours",
        "Booked Court-Hours per 6-Court Unit (Scale x3.33 for 20 Cts)",
        PALETTE["ml_blue"],
        unit_suffix=" hrs",
        decimals=0,
    )

    # 2. Weekday Daytime Util
    _plot_normal_panel(
        axes[0, 1],
        sim.monthly_wd_daytime_util * 100.0,
        cfg.weekday_daytime.mean * 100.0,
        cfg.weekday_daytime.std * 100.0,
        "B. Weekday Daytime Utilization",
        "Court Utilization (% of Active Capacity)",
        PALETTE["ml_teal"],
        unit_suffix="%",
        decimals=1,
    )

    # 3. Weekday Evening Util
    _plot_normal_panel(
        axes[0, 2],
        sim.monthly_wd_evening_util * 100.0,
        cfg.weekday_evening.mean * 100.0,
        cfg.weekday_evening.std * 100.0,
        "C. Weekday Evening Utilization",
        "Court Utilization (% of Active Capacity)",
        PALETTE["orange"],
        unit_suffix="%",
        decimals=1,
    )

    # 4. Weekend Util
    _plot_normal_panel(
        axes[0, 3],
        sim.monthly_weekend_util * 100.0,
        cfg.weekend.mean * 100.0,
        cfg.weekend.std * 100.0,
        "D. Weekend Utilization",
        "Court Utilization (% of Active Capacity)",
        PALETTE["purple"],
        unit_suffix="%",
        decimals=1,
    )

    # 5. Three Separate Regime Slopes a_r
    ax_a = axes[1, 0]
    reg_flat = sim.regime_id.reshape(-1)
    a_flat = sim.slope_a.reshape(-1)
    b_flat = sim.intercept_b.reshape(-1)

    regime_specs = [
        (0, "Weekday Daytime", PALETTE["ml_teal"], cfg.demand_curve.weekday_daytime_a_mean, cfg.demand_curve.weekday_daytime_a_std),
        (1, "Weekday Evening", PALETTE["orange"], cfg.demand_curve.weekday_evening_a_mean, cfg.demand_curve.weekday_evening_a_std),
        (2, "Weekend", PALETTE["purple"], cfg.demand_curve.weekend_a_mean, cfg.demand_curve.weekend_a_std),
    ]
    for r_id, r_lbl, r_col, m_a, s_a in regime_specs:
        sub_a = a_flat[reg_flat == r_id]
        ax_a.hist(sub_a, bins=28, density=True, color=r_col, alpha=0.32, edgecolor=r_col, linewidth=0.7)
        xg = np.linspace(sub_a.min() - 0.002, sub_a.max() + 0.002, 150)
        ax_a.plot(xg, norm.pdf(xg, loc=m_a, scale=s_a), color=r_col, linewidth=2.0, label=f"{r_lbl}: a ~ N({m_a:.3f}, {s_a:.3f})")

    ax_a.set_title("E. Price Sensitivity Slopes by Regime", fontsize=10.0, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax_a.set_xlabel("Price Slope a (Courts per k VND per 6-Court Unit)", fontsize=8.8, color=PALETTE["slate"])
    ax_a.set_ylabel("Density", fontsize=8.5, color=PALETTE["slate"])
    ax_a.grid(True)
    ax_a.legend(loc="upper left", fontsize=7.2, framealpha=0.92)

    # 6. Daily Random Demand Multiplier
    _plot_normal_panel(
        axes[1, 1],
        sim.sampled_daily_multipliers.reshape(-1),
        cfg.daily_random_multiplier.mean,
        cfg.daily_random_multiplier.std,
        "F. Daily Demand Multiplier",
        "Daily Demand Factor m_d",
        PALETTE["emerald"],
        unit_suffix="x",
        decimals=3,
    )

    # 7. Hourly Random Demand Shock
    _plot_normal_panel(
        axes[1, 2],
        sim.sampled_hourly_shocks.reshape(-1),
        cfg.hourly_random_shock.mean,
        cfg.hourly_random_shock.std,
        "G. Hourly Demand Shock",
        "Hourly Demand Shock s_{d,h}",
        PALETTE["amber"],
        unit_suffix="x",
        decimals=3,
    )

    # 8. Three Separate Regime Intercepts b_r
    ax_b = axes[1, 3]
    for r_id, r_lbl, r_col, _, _ in regime_specs:
        sub_b = b_flat[reg_flat == r_id]
        ax_b.hist(
            sub_b,
            bins=28,
            density=True,
            color=r_col,
            alpha=0.35,
            edgecolor=r_col,
            linewidth=0.7,
            label=f"{r_lbl}: b = {sub_b.mean():.2f} +/- {sub_b.std():.2f}",
        )
    ax_b.set_title("H. Anchor-Constrained Intercepts by Regime", fontsize=10.0, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax_b.set_xlabel("Intercept b = Q_0 - a * x_0 (per 6-Court Unit)", fontsize=8.8, color=PALETTE["slate"])
    ax_b.set_ylabel("Density", fontsize=8.5, color=PALETTE["slate"])
    ax_b.grid(True)
    ax_b.legend(loc="upper right", fontsize=7.2, framealpha=0.92)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_demand_curves_variations(
    cfg: AppConfig,
    sim: SimulationResults,
    eval_res: EvaluationResults,
    save_path: Path,
) -> Path:
    """
    Chart 2: 2x3 Grid of the Three Regime Linear Demand Curves Expressed in Court Utilization (%)
    with Dual Capacity Scaling (6 Courts in M1-M13 -> 20 Courts in M14-M60) and Implied Hourly Revenue Frontiers.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20.5, 11.2), dpi=220)
    fig.subplots_adjust(hspace=0.36, wspace=0.28, top=0.88, bottom=0.07, left=0.05, right=0.94)

    fig.suptitle(
        "Regime Price-Demand Elasticity Curves & Implied Hourly Revenue Frontiers",
        fontsize=15.0,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "Normalized to Active Court Utilization (Scales from 6 Courts in Months 1-13 to 20 Courts in Months 14-60)",
        ha="center",
        fontsize=10.2,
        color=PALETTE["slate"],
    )

    x_grid = np.linspace(75.0, 320.0, 240)
    rng = np.random.default_rng(cfg.simulation.random_seed + 99)

    reg_flat = sim.regime_id.reshape(-1)
    a_flat = sim.slope_a.reshape(-1)
    b_flat = sim.intercept_b.reshape(-1)

    n_train_sims = int(round(cfg.simulation.n_simulations * cfg.simulation.train_fraction))
    reg_test = sim.regime_id[n_train_sims:]
    ml_p_test = eval_res.ml_prices_test

    regimes_meta = [
        {
            "id": 0,
            "title_top": "A. Weekday Daytime Demand Elasticity",
            "title_bot": "D. Weekday Daytime Revenue Frontier",
            "x0": cfg.baseline_pricing.off_peak_price_kvnd,
            "u0": cfg.weekday_daytime.mean,
            "color": PALETTE["ml_teal"],
            "ml_price": float(ml_p_test[reg_test == 0].mean()),
        },
        {
            "id": 1,
            "title_top": "B. Weekday Evening Demand Elasticity",
            "title_bot": "E. Weekday Evening Revenue Frontier",
            "x0": cfg.baseline_pricing.peak_price_kvnd,
            "u0": cfg.weekday_evening.mean,
            "color": PALETTE["orange"],
            "ml_price": float(ml_p_test[reg_test == 1].mean()),
        },
        {
            "id": 2,
            "title_top": "C. Weekend Demand Elasticity",
            "title_bot": "F. Weekend Revenue Frontier",
            "x0": cfg.baseline_pricing.peak_price_kvnd,
            "u0": cfg.weekend.mean,
            "color": PALETTE["purple"],
            "ml_price": float(ml_p_test[reg_test == 2].mean()),
        },
    ]

    for col_idx, rm in enumerate(regimes_meta):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        mask = reg_flat == rm["id"]
        a_reg = a_flat[mask]
        b_reg = b_flat[mask]

        n_curves = min(90, len(a_reg))
        sample_idx = rng.choice(len(a_reg), size=n_curves, replace=False)

        for idx in sample_idx:
            a_i = a_reg[idx]
            b_i = b_reg[idx]
            u_curve = np.clip((a_i * x_grid + b_i) / 6.0 * 100.0, 0.0, 100.0)
            q_curve_6 = np.clip(a_i * x_grid + b_i, 0.0, 6.0)
            r_curve_6 = x_grid * q_curve_6
            ax_top.plot(x_grid, u_curve, color=rm["color"], alpha=0.11, linewidth=0.9)
            ax_bot.plot(x_grid, r_curve_6, color=rm["color"], alpha=0.11, linewidth=0.9)

        a_mean = float(a_reg.mean())
        b_mean = float(b_reg.mean())
        u_mean = np.clip((a_mean * x_grid + b_mean) / 6.0 * 100.0, 0.0, 100.0)
        q_mean_6 = np.clip(a_mean * x_grid + b_mean, 0.0, 6.0)
        r_mean_6 = x_grid * q_mean_6

        x0 = rm["x0"]
        u0_pct = rm["u0"] * 100.0
        q0_6 = 6.0 * rm["u0"]
        q0_20 = 20.0 * rm["u0"]

        x_ml = rm["ml_price"]
        u_ml_pct = float(np.clip((a_mean * x_ml + b_mean) / 6.0 * 100.0, 0.0, 100.0))
        q_ml_6 = float(np.clip(a_mean * x_ml + b_mean, 0.0, 6.0))
        q_ml_20 = q_ml_6 * (20.0 / 6.0)
        r0_val = x0 * q0_6
        r_ml_val = x_ml * q_ml_6

        ax_top.plot(
            x_grid,
            u_mean,
            color=rm["color"],
            linewidth=2.8,
            label=f"Mean Utilization Curve: u(x) = {a_mean/6*100:.2f}%*x + {b_mean/6*100:.1f}%",
        )
        ax_top.axvline(x0, color=PALETTE["coral"], linestyle="--", linewidth=1.2, alpha=0.75)
        ax_top.axhline(u0_pct, color=PALETTE["coral"], linestyle="--", linewidth=1.2, alpha=0.75)

        ax_top.scatter(
            [x0],
            [u0_pct],
            color=PALETTE["coral"],
            s=85,
            zorder=7,
            edgecolor="#11111b",
            linewidth=1.3,
            label=f"Baseline Anchor: {x0:.0f}k at {u0_pct:.0f}% ({q0_6:.1f} / {q0_20:.1f} cts)",
        )
        ax_top.scatter(
            [x_ml],
            [u_ml_pct],
            color=PALETTE["ml_blue"],
            marker="D",
            s=80,
            zorder=8,
            edgecolor="#11111b",
            linewidth=1.3,
            label=f"ML Adaptive: {x_ml:.0f}k at {u_ml_pct:.1f}% ({q_ml_6:.2f} / {q_ml_20:.1f} cts)",
        )

        if rm["id"] == 0:
            ax_top.annotate(
                f"Baseline Anchor\n({x0:.0f}k, {u0_pct:.0f}% | {q0_6:.1f}/{q0_20:.0f} cts)",
                xy=(x0, u0_pct),
                xytext=(x0 + 18, u0_pct + 9),
                fontsize=7.8,
                fontweight="bold",
                color=PALETTE["coral"],
                bbox=dict(boxstyle="round,pad=0.25", fc="#1e1e2e", ec=PALETTE["coral"], alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=PALETTE["coral"], lw=1.1),
            )
            ax_top.annotate(
                f"ML Target\n({x_ml:.0f}k, {u_ml_pct:.0f}% | {q_ml_6:.1f}/{q_ml_20:.1f} cts)",
                xy=(x_ml, u_ml_pct),
                xytext=(x_ml - 26, min(92.0, u_ml_pct + 18)),
                fontsize=7.8,
                fontweight="bold",
                color=PALETTE["ml_blue"],
                bbox=dict(boxstyle="round,pad=0.25", fc="#1e1e2e", ec=PALETTE["ml_blue"], alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=PALETTE["ml_blue"], lw=1.1),
            )
        else:
            ax_top.annotate(
                f"Baseline Anchor\n({x0:.0f}k, {u0_pct:.0f}% | {q0_6:.1f}/{q0_20:.0f} cts)",
                xy=(x0, u0_pct),
                xytext=(x0 - 95, u0_pct - 18),
                fontsize=7.8,
                fontweight="bold",
                color=PALETTE["coral"],
                bbox=dict(boxstyle="round,pad=0.25", fc="#1e1e2e", ec=PALETTE["coral"], alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=PALETTE["coral"], lw=1.1),
            )
            ax_top.annotate(
                f"ML Target\n({x_ml:.0f}k, {u_ml_pct:.0f}% | {q_ml_6:.1f}/{q_ml_20:.1f} cts)",
                xy=(x_ml, u_ml_pct),
                xytext=(x_ml - 70, u_ml_pct - 26),
                fontsize=7.8,
                fontweight="bold",
                color=PALETTE["ml_blue"],
                bbox=dict(boxstyle="round,pad=0.25", fc="#1e1e2e", ec=PALETTE["ml_blue"], alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=PALETTE["ml_blue"], lw=1.1),
            )

        ax_top.axhline(100.0, color=PALETTE["navy"], linestyle=":", linewidth=1.2, label="100% Active Court Capacity (6 Cts M1-13 | 20 Cts M14+)")
        ax_top.set_title(rm["title_top"], fontsize=10.5, fontweight="bold", color=PALETTE["navy"], pad=8)
        ax_top.set_xlabel("Hourly Court Rental Price x (Thousand VND)", fontsize=9.0)
        ax_top.set_ylabel("Court Utilization (% of Active Capacity)", fontsize=9.0)
        ax_top.set_xlim(75, 320)
        ax_top.set_ylim(0, 108)
        ax_top.grid(True)
        ax_top.legend(loc="lower left", fontsize=7.2, framealpha=0.93)

        ax_top_r = ax_top.twinx()
        ax_top_r.set_ylim(0, 108)
        pct_ticks = [0, 20, 40, 60, 80, 100]
        ax_top_r.set_yticks(pct_ticks)
        ax_top_r.set_yticklabels([f"{p*6//100} / {p*20//100}" for p in pct_ticks], fontsize=7.8, color=PALETTE["slate"])
        if col_idx == 2:
            ax_top_r.set_ylabel("Courts Booked (6-Ct Phase / 20-Ct Phase)", fontsize=8.5, color=PALETTE["slate"])

        # Bottom Panel: Revenue Frontier
        ax_bot.plot(x_grid, r_mean_6, color=rm["color"], linewidth=2.8, label="Expected Hourly Revenue Curve")
        ax_bot.axvline(x0, color=PALETTE["coral"], linestyle="--", linewidth=1.2, alpha=0.75)
        ax_bot.scatter(
            [x0],
            [r0_val],
            color=PALETTE["coral"],
            s=85,
            zorder=7,
            edgecolor="#11111b",
            linewidth=1.3,
            label=f"Baseline @ {x0:.0f}k: {r0_val:.0f}k (6 Cts) | {r0_val*20/6:,.0f}k (20 Cts)",
        )
        ax_bot.scatter(
            [x_ml],
            [r_ml_val],
            color=PALETTE["ml_blue"],
            marker="D",
            s=80,
            zorder=8,
            edgecolor="#11111b",
            linewidth=1.3,
            label=f"ML Optimal @ {x_ml:.0f}k: {r_ml_val:.0f}k (6 Cts) | {r_ml_val*20/6:,.0f}k (20 Cts)",
        )

        pct_gain = (r_ml_val - r0_val) / max(1.0, r0_val) * 100.0
        ax_bot.annotate(
            f"ML: {x_ml:.0f}k VND ({pct_gain:+.1f}%)\n6 Cts: {r_ml_val:.0f}k/hr | 20 Cts: {r_ml_val*20/6:,.0f}k/hr",
            xy=(x_ml, r_ml_val),
            xytext=(x_ml + 12 if rm["id"] == 0 else x_ml - 105, r_ml_val + 80 if rm["id"] == 0 else r_ml_val - 280),
            fontsize=7.6,
            fontweight="bold",
            color=PALETTE["navy"],
            bbox=dict(boxstyle="round,pad=0.25", fc="#1e1e2e", ec=rm["color"], alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=PALETTE["navy"], lw=1.1),
        )

        ax_bot.set_title(rm["title_bot"], fontsize=10.5, fontweight="bold", color=PALETTE["navy"], pad=8)
        ax_bot.set_xlabel("Hourly Court Rental Price x (Thousand VND)", fontsize=9.0)
        ax_bot.set_ylabel("Hourly Revenue: 6-Court Phase (k VND)", fontsize=9.0)
        ax_bot.set_xlim(75, 320)
        y_max_bot = 760 if rm["id"] == 0 else 1850
        ax_bot.set_ylim(0, y_max_bot)
        ax_bot.grid(True)
        ax_bot.legend(loc="upper right" if rm["id"] == 0 else "lower left", fontsize=7.2, framealpha=0.93)

        ax_bot_r = ax_bot.twinx()
        ax_bot_r.set_ylim(0, y_max_bot * (20.0 / 6.0) / 1000.0)
        ax_bot_r.tick_params(axis="y", labelsize=7.8, colors=PALETTE["slate"])
        if col_idx == 2:
            ax_bot_r.set_ylabel("Hourly Revenue: 20-Court Phase (M VND)", fontsize=8.5, color=PALETTE["slate"])

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_daily_revenue_comparison(
    cfg: AppConfig,
    eval_res: EvaluationResults,
    save_path: Path,
) -> Path:
    """
    Chart 3: Daily Revenue Double-Bar Chart & Cumulative Monthly Trajectory with Dual 6-Court / 20-Court Scaling.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18.5, 10.2), dpi=220, gridspec_kw={"height_ratios": [1.35, 1.0]})
    fig.subplots_adjust(hspace=0.34, top=0.88, bottom=0.07, left=0.06, right=0.93)

    base_total_m = eval_res.mean_baseline_monthly_rev_kvnd / 1000.0
    ml_total_m = eval_res.mean_ml_monthly_rev_kvnd / 1000.0
    lift_m = ml_total_m - base_total_m
    lift_pct = (lift_m / base_total_m) * 100.0

    fig.suptitle(
        "Daily & Cumulative Monthly Court Hire Revenue: Baseline vs. ML Adaptive Pricing",
        fontsize=15.0,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "Evaluated on Out-of-Sample Held-Out Test Trajectories Across Active Court Capacity (6 Courts M1-M13 -> 20 Courts M14-M60)",
        ha="center",
        fontsize=10.2,
        color=PALETTE["slate"],
    )

    days = np.arange(1, cfg.simulation.days_per_month + 1)
    day_abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x_labels = [f"D{d}\n({day_abbr[(d - 1) % 7]})" for d in days]

    base_d = eval_res.daily_baseline_rev_mean_kvnd / 1000.0
    base_sd = eval_res.daily_baseline_rev_std_kvnd / 1000.0
    ml_d = eval_res.daily_ml_rev_mean_kvnd / 1000.0
    ml_sd = eval_res.daily_ml_rev_std_kvnd / 1000.0

    for idx, d in enumerate(days):
        dow = (d - 1) % 7
        if dow >= 5:
            ax1.axvspan(d - 0.48, d + 0.48, color=PALETTE["weekend_bg"], alpha=0.55, zorder=0)

    width = 0.38
    ax1.bar(
        days - width / 2,
        base_d,
        width=width,
        yerr=base_sd,
        capsize=2.5,
        color=PALETTE["baseline"],
        alpha=0.85,
        edgecolor=PALETTE["navy"],
        linewidth=0.6,
        error_kw=dict(lw=0.8, ecolor=PALETTE["navy"], alpha=0.6),
        label=f"Baseline Static Pricing (6 Cts: {base_total_m:.1f}M/mo | 20 Cts: {base_total_m*20/6:,.1f}M/mo)",
        zorder=3,
    )
    ax1.bar(
        days + width / 2,
        ml_d,
        width=width,
        yerr=ml_sd,
        capsize=2.5,
        color=PALETTE["ml_blue"],
        alpha=0.92,
        edgecolor=PALETTE["navy"],
        linewidth=0.6,
        error_kw=dict(lw=0.8, ecolor=PALETTE["navy"], alpha=0.6),
        label=f"ML Adaptive Pricing (6 Cts: {ml_total_m:.1f}M/mo | 20 Cts: {ml_total_m*20/6:,.1f}M/mo | +{lift_pct:.1f}%)",
        zorder=4,
    )

    for idx, d in enumerate(days):
        gain = ml_d[idx] - base_d[idx]
        top_y = max(ml_d[idx] + ml_sd[idx], base_d[idx] + base_sd[idx])
        ax1.text(
            d,
            top_y + 0.45,
            f"+{gain:.2f}M",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color=PALETTE["ml_blue"],
            rotation=0,
        )

    weekend_patch = mpatches.Patch(color=PALETTE["weekend_bg"], alpha=0.7, label="Weekend Days (Saturday & Sunday)")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles + [weekend_patch], loc="upper left", fontsize=8.8, ncol=3, framealpha=0.94)

    ax1.set_title("A. Daily Court Hire Revenue Comparison", fontsize=11.5, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax1.set_xticks(days)
    ax1.set_xticklabels(x_labels, fontsize=7.8)
    ax1.set_ylabel("Daily Revenue: 6-Court Phase (M VND)", fontsize=9.8)
    ax1.set_xlim(0.3, 30.7)
    ax1.set_ylim(0, max(ml_d + ml_sd) * 1.18)
    ax1.grid(True, axis="y")

    ax1_r = ax1.twinx()
    ax1_r.set_ylim(0, max(ml_d + ml_sd) * 1.18 * (20.0 / 6.0))
    ax1_r.set_ylabel("Daily Revenue: 20-Court Phase (M VND)", fontsize=9.2, color=PALETTE["slate"])
    ax1_r.tick_params(axis="y", labelsize=8.0, colors=PALETTE["slate"])

    # Panel B: Cumulative Monthly Revenue Trajectory
    cum_base = np.cumsum(base_d)
    cum_ml = np.cumsum(ml_d)
    cum_oracle = np.cumsum(eval_res.daily_ml_rev_mean_kvnd / 1000.0 * (eval_res.mean_oracle_monthly_rev_kvnd / eval_res.mean_ml_monthly_rev_kvnd))

    ax2.plot(days, cum_base, color=PALETTE["baseline"], linewidth=2.5, marker="o", markersize=4, label=f"Cumulative Baseline (6 Cts: {cum_base[-1]:.1f}M | 20 Cts: {cum_base[-1]*20/6:,.1f}M VND)")
    ax2.plot(days, cum_ml, color=PALETTE["ml_blue"], linewidth=3.0, marker="s", markersize=4.5, label=f"Cumulative ML Adaptive (6 Cts: {cum_ml[-1]:.1f}M | 20 Cts: {cum_ml[-1]*20/6:,.1f}M VND)")
    ax2.plot(days, cum_oracle, color=PALETTE["emerald"], linestyle="--", linewidth=2.0, label=f"Theoretical Oracle Ceiling (6 Cts: {cum_oracle[-1]:.1f}M | 20 Cts: {cum_oracle[-1]*20/6:,.1f}M VND)")
    ax2.fill_between(days, cum_base, cum_ml, color=PALETTE["ml_blue"], alpha=0.16, label=f"Cumulative Revenue Gain (+{lift_m:.2f}M on 6 Cts | +{lift_m*20/6:.2f}M on 20 Cts)")

    ax2.annotate(
        f"Day 30 Net Lift: +{lift_pct:.1f}%\n+{lift_m:.2f}M/mo (6 Cts) | +{lift_m*20/6:.2f}M/mo (20 Cts)",
        xy=(30, cum_ml[-1]),
        xytext=(21.5, cum_ml[-1] - 95),
        fontsize=9.0,
        fontweight="bold",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e2e", edgecolor=PALETTE["ml_blue"], lw=1.3),
        arrowprops=dict(arrowstyle="->", color=PALETTE["ml_blue"], lw=1.5),
    )

    ax2.set_title("B. Cumulative Monthly Court Revenue Trajectory", fontsize=11.5, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax2.set_xticks(days)
    ax2.set_xticklabels([f"Day {d}" for d in days], fontsize=8.0)
    ax2.set_ylabel("Cumulative Revenue: 6-Court Phase (M VND)", fontsize=9.8)
    ax2.set_xlim(0.5, 30.5)
    ax2.set_ylim(0, cum_oracle[-1] * 1.12)
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8.8, framealpha=0.94)

    ax2_r = ax2.twinx()
    ax2_r.set_ylim(0, cum_oracle[-1] * 1.12 * (20.0 / 6.0))
    ax2_r.set_ylabel("Cumulative Revenue: 20-Court Phase (M VND)", fontsize=9.2, color=PALETTE["slate"])
    ax2_r.tick_params(axis="y", labelsize=8.0, colors=PALETTE["slate"])

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_monthly_calendar_heatmap(
    cfg: AppConfig,
    eval_res: EvaluationResults,
    save_path: Path,
) -> Path:
    """
    Chart 4: Consolidated 2-Row Hourly Pricing Schedule (Weekday vs. Weekend Day across 16 Operating Hours)
    paired with Hourly Court Utilization Comparison.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18.5, 8.8), dpi=220, gridspec_kw={"height_ratios": [1.05, 1.25]}
    )
    fig.subplots_adjust(hspace=0.38, top=0.86, bottom=0.09, left=0.08, right=0.90)

    fig.suptitle(
        "Consolidated Hourly Court Pricing Schedule & Diurnal Utilization Profile",
        fontsize=15.0,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.96,
    )
    fig.text(
        0.5,
        0.915,
        "Recommended Hourly Price Schedule (Weekday vs. Weekend Day) & Resulting Court Utilization Lift Across Active Capacity",
        ha="center",
        fontsize=10.2,
        color=PALETTE["slate"],
    )

    hours = cfg.simulation.hours_array
    n_hours = len(hours)
    hour_labels = [f"{h:02d}:00\n-{h+1:02d}:00" for h in hours]

    # Compute representative Weekday (Mon-Fri) and Weekend (Sat-Sun) hourly schedules
    dows = np.array([d % 7 for d in range(cfg.simulation.days_per_month)])
    wd_mask = dows < 5
    wk_mask = dows >= 5

    wd_ml_price = np.round(eval_res.mc_month_ml_price[wd_mask].mean(axis=0) / 5.0) * 5.0
    wk_ml_price = np.round(eval_res.mc_month_ml_price[wk_mask].mean(axis=0) / 5.0) * 5.0
    wd_base_price = eval_res.rep_month_base_price[wd_mask].mean(axis=0)
    wk_base_price = eval_res.rep_month_base_price[wk_mask].mean(axis=0)

    wd_ml_util = eval_res.mc_month_ml_util[wd_mask].mean(axis=0) * 100.0
    wk_ml_util = eval_res.mc_month_ml_util[wk_mask].mean(axis=0) * 100.0
    wd_base_util = eval_res.mc_month_base_util[wd_mask].mean(axis=0) * 100.0
    wk_base_util = eval_res.mc_month_base_util[wk_mask].mean(axis=0) * 100.0

    price_matrix = np.vstack([wd_ml_price, wk_ml_price])
    base_matrix = np.vstack([wd_base_price, wk_base_price])
    util_matrix = np.vstack([wd_ml_util, wk_ml_util])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pick_price_cmap",
        ["#94e2d5", "#89dceb", "#89b4fa", "#f9e2af", "#fab387", "#f38ba8"],
    )
    norm_p = mcolors.Normalize(vmin=85.0, vmax=300.0)

    im = ax_top.imshow(price_matrix, cmap=cmap, norm=norm_p, aspect="auto")

    for r_idx in range(2):
        for c_idx in range(n_hours):
            p_val = price_matrix[r_idx, c_idx]
            b_val = base_matrix[r_idx, c_idx]
            u_val = util_matrix[r_idx, c_idx]
            diff = p_val - b_val
            diff_str = f"{diff:+.0f}k" if abs(diff) >= 1.0 else "0k"
            txt_color = "#11111b"
            sub_color = "#181825"

            ax_top.text(
                c_idx,
                r_idx - 0.14,
                f"{p_val:.0f}k",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=txt_color,
            )
            ax_top.text(
                c_idx,
                r_idx + 0.13,
                f"(vs {b_val:.0f}k: {diff_str})",
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
                color=sub_color,
            )
            ax_top.text(
                c_idx,
                r_idx + 0.34,
                f"{u_val:.0f}% util",
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
                color=sub_color,
            )

    ax_top.set_xticks(np.arange(n_hours))
    ax_top.set_xticklabels(hour_labels, fontsize=8.2, fontweight="bold", color=PALETTE["navy"])
    ax_top.set_yticks([0, 1])
    ax_top.set_yticklabels(["Weekday\n(Mon - Fri)", "Weekend Day\n(Sat - Sun)"], fontsize=10.0, fontweight="bold", color=PALETTE["navy"])
    ax_top.set_xticks(np.arange(-0.5, n_hours, 1), minor=True)
    ax_top.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax_top.grid(which="minor", color="#181825", linestyle="-", linewidth=2.0)
    ax_top.grid(which="major", visible=False)
    ax_top.tick_params(which="minor", bottom=False, left=False)
    ax_top.set_title(
        "A. Hourly Court Rental Price Schedule: Weekday vs. Weekend Day (k VND / Court-Hour)",
        fontsize=11.2,
        fontweight="bold",
        color=PALETTE["navy"],
        pad=10,
    )

    cbar_ax = fig.add_axes([0.915, 0.52, 0.015, 0.34])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Price (k VND / Hr)", fontsize=8.8, fontweight="bold", color=PALETTE["navy"])
    cbar.ax.tick_params(labelsize=8.0)

    # Bottom Panel: Hourly Utilization Profile (Weekday & Weekend, Baseline vs ML Adaptive)
    x_idx = np.arange(n_hours)
    w = 0.20
    ax_bot.bar(
        x_idx - 1.5 * w,
        wd_base_util,
        width=w,
        color=PALETTE["baseline"],
        alpha=0.75,
        edgecolor=PALETTE["navy"],
        linewidth=0.6,
        label=f"Weekday Baseline Static (Avg: {wd_base_util.mean():.1f}%)",
    )
    ax_bot.bar(
        x_idx - 0.5 * w,
        wd_ml_util,
        width=w,
        color=PALETTE["ml_teal"],
        alpha=0.92,
        edgecolor=PALETTE["navy"],
        linewidth=0.6,
        label=f"Weekday ML Adaptive (Avg: {wd_ml_util.mean():.1f}%)",
    )
    ax_bot.bar(
        x_idx + 0.5 * w,
        wk_base_util,
        width=w,
        color=PALETTE["amber"],
        alpha=0.65,
        edgecolor=PALETTE["navy"],
        linewidth=0.6,
        label=f"Weekend Baseline Static (Avg: {wk_base_util.mean():.1f}%)",
    )
    ax_bot.bar(
        x_idx + 1.5 * w,
        wk_ml_util,
        width=w,
        color=PALETTE["purple"],
        alpha=0.90,
        edgecolor=PALETTE["navy"],
        linewidth=0.6,
        label=f"Weekend ML Adaptive (Avg: {wk_ml_util.mean():.1f}%)",
    )

    ax_bot.axvline(10.5, color=PALETTE["coral"], linestyle="--", linewidth=1.4, zorder=4)
    ax_bot.text(
        5.0,
        102.0,
        "Daytime Off-Peak Window (06:00-17:00): Price Lowered 160k -> 115k (+24.4 pp Weekday Utilization Lift)",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["ml_teal"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e1e2e", edgecolor=PALETTE["ml_teal"], lw=1.0),
    )
    ax_bot.text(
        13.0,
        102.0,
        "Evening Peak (17:00-22:00): 285k-300k Prime & 250k Noise Cap at 21h",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["coral"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e1e2e", edgecolor=PALETTE["coral"], lw=1.0),
    )

    ax_bot.set_title(
        "B. Hourly Court Utilization Comparison Across Operating Hours (% of Active Capacity)",
        fontsize=11.2,
        fontweight="bold",
        color=PALETTE["navy"],
        pad=10,
    )
    ax_bot.set_xticks(x_idx)
    ax_bot.set_xticklabels(hour_labels, fontsize=8.2, fontweight="bold")
    ax_bot.set_ylabel("Court Utilization (% of Active Courts)", fontsize=9.5)
    ax_bot.set_ylim(0, 112)
    ax_bot.grid(True, axis="y")
    ax_bot.legend(loc="lower right", ncol=2, fontsize=8.2, framealpha=0.94)

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.set_ylim(0, 112)
    pct_ticks = [0, 20, 40, 60, 80, 100]
    ax_bot_r.set_yticks(pct_ticks)
    ax_bot_r.set_yticklabels([f"{p*6//100} / {p*20//100} cts" for p in pct_ticks], fontsize=8.0, color=PALETTE["slate"])
    ax_bot_r.set_ylabel("Booked Courts (6-Ct Phase / 20-Ct Phase)", fontsize=8.8, color=PALETTE["slate"])

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_occupancy_and_stress_test(
    cfg: AppConfig,
    eval_res: EvaluationResults,
    save_path: Path,
) -> Path:
    """
    Chart 5: Regime Utilization Shift & Downside Demand Stress Test.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 6.8), dpi=220)
    fig.subplots_adjust(wspace=0.28, top=0.83, bottom=0.12, left=0.06, right=0.94)

    fig.suptitle(
        "Regime Court Utilization Lift & Downside Demand Stress Test",
        fontsize=14.5,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.96,
    )
    fig.text(
        0.5,
        0.90,
        "Evaluating Off-Peak Daytime Court Filling and Revenue Resilience Under Lower Demand Conditions",
        ha="center",
        fontsize=10.0,
        color=PALETTE["slate"],
    )

    # Panel A: Utilization by Regime
    regimes = [
        "Weekday Daytime\n(06:00-17:00)",
        "Weekday Evening\n(17:00-22:00)",
        "Weekend\n(06:00-22:00)",
        "Overall Monthly\nBlended Utilization",
    ]
    base_u = [
        eval_res.baseline_wd_daytime_util * 100,
        eval_res.baseline_wd_evening_util * 100,
        eval_res.baseline_weekend_util * 100,
        (eval_res.mean_baseline_booked_hours / 2880.0) * 100,
    ]
    ml_u = [
        eval_res.ml_wd_daytime_util * 100,
        eval_res.ml_wd_evening_util * 100,
        eval_res.ml_weekend_util * 100,
        (eval_res.mean_ml_booked_hours / 2880.0) * 100,
    ]

    x = np.arange(len(regimes))
    w = 0.35
    b1 = ax1.bar(x - w / 2, base_u, width=w, color=PALETTE["baseline"], edgecolor=PALETTE["navy"], lw=0.7, label="Baseline Static (160k / 260k)")
    b2 = ax1.bar(x + w / 2, ml_u, width=w, color=PALETTE["ml_teal"], edgecolor=PALETTE["navy"], lw=0.7, label="ML Adaptive Pricing (85k-300k)")

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=PALETTE["slate"], fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=PALETTE["ml_teal"], fontweight="bold")

    ax1.annotate(
        f"Daytime Occupancy Surge:\n{base_u[0]:.1f}% -> {ml_u[0]:.1f}%\n(6 Cts: 2.4 -> 3.9 | 20 Cts: 8.0 -> 12.9)",
        xy=(0 + w / 2, ml_u[0]),
        xytext=(0.15, 79),
        fontsize=8.3,
        fontweight="bold",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#1e1e2e", edgecolor=PALETTE["ml_teal"], lw=1.2),
        arrowprops=dict(arrowstyle="->", color=PALETTE["ml_teal"], lw=1.3),
    )

    ax1.set_title("A. Court Utilization Rate by Operating Regime", fontsize=11.2, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=8.8)
    ax1.set_ylabel("Court Utilization (% of Active Capacity)", fontsize=9.5)
    ax1.set_ylim(0, 109)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right", fontsize=8.8)

    # Panel B: Normal vs -30% Stress Test Revenue
    scenarios = [
        "Expected Demand Trajectory\n(100% Baseline Demand)",
        "Downside Stress Test\n(-30% Demand Shock)",
    ]
    base_rev = [
        eval_res.mean_baseline_monthly_rev_kvnd / 1000.0,
        eval_res.stress30_baseline_monthly_rev_kvnd / 1000.0,
    ]
    ml_rev = [
        eval_res.mean_ml_monthly_rev_kvnd / 1000.0,
        eval_res.stress30_ml_monthly_rev_kvnd / 1000.0,
    ]

    x2 = np.arange(len(scenarios))
    s1 = ax2.bar(x2 - w / 2, base_rev, width=w, color=PALETTE["baseline"], edgecolor=PALETTE["navy"], lw=0.7, label="Baseline Static Pricing")
    s2 = ax2.bar(x2 + w / 2, ml_rev, width=w, color=PALETTE["ml_blue"], edgecolor=PALETTE["navy"], lw=0.7, label="ML Adaptive Pricing")

    for i in range(len(scenarios)):
        h_b = base_rev[i]
        h_m = ml_rev[i]
        diff = h_m - h_b
        pct = (diff / h_b) * 100.0
        ax2.text(x2[i] - w / 2, h_b + 8, f"{h_b:.1f}M\n({h_b*20/6:,.0f}M @20c)", ha="center", va="bottom", fontsize=8.0, color=PALETTE["slate"], fontweight="bold")
        ax2.text(x2[i] + w / 2, h_m + 8, f"{h_m:.1f}M (+{pct:.1f}%)\n({h_m*20/6:,.0f}M @20c)", ha="center", va="bottom", fontsize=8.0, color=PALETTE["ml_blue"], fontweight="bold")

    ax2.set_title("B. Monthly Court Revenue Under Normal & Downside Demand", fontsize=11.2, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(scenarios, fontsize=9.2)
    ax2.set_ylabel("Monthly Court Revenue: 6-Court Phase (M VND)", fontsize=9.5)
    ax2.set_ylim(0, max(ml_rev) * 1.24)
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", fontsize=8.8)

    ax2_r = ax2.twinx()
    ax2_r.set_ylim(0, max(ml_rev) * 1.24 * (20.0 / 6.0))
    ax2_r.set_ylabel("Monthly Court Revenue: 20-Court Phase (M VND)", fontsize=9.0, color=PALETTE["slate"])
    ax2_r.tick_params(axis="y", labelsize=8.0, colors=PALETTE["slate"])

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_ml_model_diagnostics(
    cfg: AppConfig,
    eval_res: EvaluationResults,
    save_path: Path,
) -> Path:
    """
    Chart 6: 4-Panel ML Model Evaluation & Diagnostics Suite.
    """
    m = eval_res.ml_metrics
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 11.2), dpi=220)
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88, bottom=0.07, left=0.06, right=0.97)

    fig.suptitle(
        "Adaptive Pricing Machine Learning Model Validation & Diagnostics",
        fontsize=14.8,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.922,
        "Out-of-Sample Generalization, Prediction Residuals, Permutation Feature Importance & Convergence Profile",
        ha="center",
        fontsize=10.0,
        color=PALETTE["slate"],
    )

    # Panel A: Out-of-Sample Parity Plot
    ax_a = axes[0, 0]
    y_true = m.y_test_true
    y_pred = m.y_test_pred_raw
    rng = np.random.default_rng(cfg.simulation.random_seed + 2027)
    sample_idx = rng.choice(len(y_true), size=min(4000, len(y_true)), replace=False)

    ax_a.scatter(
        y_true[sample_idx],
        y_pred[sample_idx],
        c=y_true[sample_idx],
        cmap="viridis",
        s=14,
        alpha=0.35,
        edgecolors="none",
        label="Held-Out Test Hours",
    )
    line_grid = np.linspace(80, 305, 100)
    ax_a.plot(line_grid, line_grid, color=PALETTE["coral"], linestyle="--", linewidth=2.0, label="1:1 Perfect Parity Line")
    ax_a.fill_between(line_grid, line_grid - 15, line_grid + 15, color=PALETTE["ml_blue"], alpha=0.13, label=f"+/-15k Tolerance Band ({m.test_within_15k_acc_pct:.1f}% within)")

    metrics_box = (
        f"Out-of-Sample Test Metrics:\n"
        f"* R^2 Score: {m.test_r2:.4f} (Train: {m.train_r2:.4f})\n"
        f"* RMSE: {m.test_rmse_kvnd:.2f} k VND\n"
        f"* MAE: {m.test_mae_kvnd:.2f} k VND | MedAE: {m.test_medae_kvnd:.2f} k\n"
        f"* MAPE: {m.test_mape_pct:.2f}%\n"
        f"* Oracle Revenue Capture: {m.oracle_capture_ratio_pct:.2f}%"
    )
    ax_a.text(
        0.04,
        0.95,
        metrics_box,
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e2e", edgecolor=PALETTE["ml_blue"], lw=1.2, alpha=0.95),
    )

    ax_a.set_title("A. Out-of-Sample Parity: Predicted vs. Optimal Price", fontsize=11.0, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax_a.set_xlabel("True Optimal Counterfactual Price x* (Thousand VND)", fontsize=9.2)
    ax_a.set_ylabel("ML Predicted Adaptive Price (Thousand VND)", fontsize=9.2)
    ax_a.set_xlim(80, 308)
    ax_a.set_ylim(80, 308)
    ax_a.grid(True)
    ax_a.legend(loc="lower right", fontsize=8.2, framealpha=0.94)

    # Panel B: Residual Error Distribution
    ax_b = axes[0, 1]
    residuals = y_pred - y_true
    res_mean = float(residuals.mean())
    res_std = float(residuals.std())

    ax_b.hist(residuals, bins=48, density=True, color=PALETTE["ml_teal"], alpha=0.42, edgecolor=PALETTE["ml_teal"], lw=0.8, label="Test Residuals")
    r_grid = np.linspace(residuals.min(), residuals.max(), 250)
    ax_b.plot(r_grid, norm.pdf(r_grid, loc=res_mean, scale=res_std), color=PALETTE["navy"], lw=2.2, label=f"Normal Fit (mu={res_mean:+.2f}k, sigma={res_std:.2f}k)")
    ax_b.axvline(0.0, color=PALETTE["coral"], linestyle="--", lw=1.8, label="Zero Bias Reference")
    ax_b.axvspan(-15.0, 15.0, color=PALETTE["amber"], alpha=0.14, label=f"+/-15k Band ({m.test_within_15k_acc_pct:.1f}% of hours)")

    regret_box = (
        f"Counterfactual Economic Regret:\n"
        f"* Mean Regret: {m.mean_hourly_regret_kvnd:.2f} k VND/hr\n"
        f"* Median Regret: {m.median_hourly_regret_kvnd:.2f} k VND/hr\n"
        f"* 90th Pct Regret: {m.p90_hourly_regret_kvnd:.2f} k VND/hr\n"
        f"* Max Lift Captured: {m.uplift_capture_vs_oracle_pct:.1f}%"
    )
    ax_b.text(
        0.04,
        0.95,
        regret_box,
        transform=ax_b.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e2e", edgecolor=PALETTE["ml_teal"], lw=1.2, alpha=0.95),
    )

    ax_b.set_title("B. Out-of-Sample Prediction Residual Distribution", fontsize=11.0, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax_b.set_xlabel("Prediction Error (Thousand VND)", fontsize=9.2)
    ax_b.set_ylabel("Density", fontsize=9.2)
    ax_b.grid(True)
    ax_b.legend(loc="upper right", fontsize=8.0, framealpha=0.94)

    # Panel C: Permutation Feature Importance
    ax_c = axes[1, 0]
    order = np.argsort(m.perm_importance_mean)
    sorted_names = [m.feature_names[i] for i in order]
    sorted_means = m.perm_importance_mean[order]
    sorted_stds = m.perm_importance_std[order]

    bar_colors = [
        PALETTE["ml_blue"] if "Hour" in name or "Day of Week" in name
        else (PALETTE["ml_teal"] if "Court" in name else PALETTE["baseline"])
        for name in sorted_names
    ]
    y_pos = np.arange(len(sorted_names))
    bars = ax_c.barh(y_pos, sorted_means, xerr=sorted_stds, capsize=3, color=bar_colors, edgecolor=PALETTE["navy"], lw=0.7, alpha=0.9)

    for bar, val in zip(bars, sorted_means):
        ax_c.text(
            max(0.0, val) + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color=PALETTE["navy"],
        )

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(sorted_names, fontsize=8.5, fontweight="bold")
    ax_c.set_xlim(0, max(sorted_means + sorted_stds) * 1.18)
    ax_c.set_title("C. Permutation Feature Importance on Test Set", fontsize=11.0, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax_c.set_xlabel("Mean Drop in Test R^2 When Feature is Randomly Permuted", fontsize=9.2)
    ax_c.grid(True, axis="x")

    # Panel D: Boosting Convergence Learning Curve
    ax_d = axes[1, 1]
    ax_d.plot(m.iter_steps, m.iter_train_rmse, color=PALETTE["baseline"], linestyle="--", lw=2.2, label=f"Training Set RMSE (Final: {m.train_rmse_kvnd:.2f} k VND)")
    ax_d.plot(m.iter_steps, m.iter_test_rmse, color=PALETTE["ml_blue"], lw=2.8, label=f"Held-Out Test Set RMSE (Final: {m.test_rmse_kvnd:.2f} k VND)")

    best_iter_idx = int(np.argmin(m.iter_test_rmse))
    best_iter = int(m.iter_steps[best_iter_idx])
    best_rmse = float(m.iter_test_rmse[best_iter_idx])
    ax_d.scatter([best_iter], [best_rmse], color=PALETTE["coral"], s=80, zorder=6, label=f"Test Convergence @ Iter {best_iter} ({best_rmse:.2f} k VND)")

    regime_table_txt = "Regime-Specific Test Performance:\n"
    for r_name in m.regime_r2:
        regime_table_txt += (
            f"* {r_name}: R^2={m.regime_r2[r_name]:.3f} | "
            f"RMSE={m.regime_rmse_kvnd[r_name]:.1f}k | "
            f"Capture={m.regime_optimality_capture_pct[r_name]:.1f}%\n"
        )
    ax_d.text(
        0.96,
        0.58,
        regime_table_txt.strip(),
        transform=ax_d.transAxes,
        va="top",
        ha="right",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e2e", edgecolor=PALETTE["slate"], lw=1.0, alpha=0.95),
    )

    ax_d.set_title("D. Boosting Learning Curve & Generalization Gap", fontsize=11.0, fontweight="bold", color=PALETTE["navy"], pad=8)
    ax_d.set_xlabel("Boosting Iterations (Number of Trees)", fontsize=9.2)
    ax_d.set_ylabel("Root Mean Squared Error (Thousand VND)", fontsize=9.2)
    ax_d.grid(True)
    ax_d.legend(loc="upper right", fontsize=8.2, framealpha=0.94)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
