"""Presentation-ready charts for `Marketing/` (Funnel, Segment Response, ML Budget Allocation & 12 Ancillary Distributions).

Strict visual standards:
  - ZERO icons, emojis, triangles, or arrows anywhere.
  - ZERO inline citations on presentation renders (citations live in REPORT.md and configs/*.toml).
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .config import MarketingAppConfig
from .model import MarketingMLEvaluationResults
from .simulator import MarketingSimulationResults

SEG_COLORS = ["#89b4fa", "#94e2d5", "#fab387", "#cba6f7"]
SEG_SHORT_LABELS = ["Competitive", "Social", "Trend Explorers", "Families"]


def _apply_clean_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#45475a",
            "axes.linewidth": 0.9,
            "axes.labelcolor": "#bac2de",
            "xtick.color": "#bac2de",
            "ytick.color": "#bac2de",
            "text.color": "#cdd6f4",
            "axes.grid": True,
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
        }
    )


def plot_funnel_and_12_distributions(
    cfg: MarketingAppConfig,
    sim: MarketingSimulationResults,
    output_dir: Path,
) -> Path:
    """
    Render `01_marketing_funnel_and_segment_response_distributions.png`:
      - Panel A: Follower-to-Customer Conversion Rate Distribution (~8.57% = 3,600 / 42,000)
      - Panel B: Marketing Cost per Player Visit Distribution (VND/visit) by Customer Segment (x, y, z, w)
      - Panel C: Merged Monthly Player Visits Distributions across the 4 Customer Segments (Total = 7,596 ~ 7,600)
      - Panel D: The 12 Segment-by-Ancillary Per-Visit Spending Distributions (Paddle/Balls, Coaching, F&B x 4 Segments)
    """
    _apply_clean_style()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 12.2), dpi=220)
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.07, wspace=0.26, hspace=0.32)

    fig.suptitle(
        "Marketing Acquisition Funnel, Segment Cost Efficiency & Ancillary Spend Profiles",
        fontsize=15,
        fontweight="bold",
        color="#cdd6f4",
        y=0.965,
    )
    fig.text(
        0.5,
        0.928,
        "Stochastic Follower Conversion Funnel, Segment Acquisition Costs & Per-Visit Ancillary Spending Distributions",
        ha="center",
        fontsize=10.5,
        color="#a6adc8",
    )

    # Panel A: Follower-to-Customer Conversion Rate Distribution
    ax_a = axes[0, 0]
    conv_samples = sim.sampled_conv_rate.ravel() * 100.0
    ax_a.hist(conv_samples, bins=42, color="#89b4fa", alpha=0.78, edgecolor="#11111b", density=True)
    mean_conv = float(np.mean(conv_samples))
    ax_a.axvline(mean_conv, color="#f38ba8", linestyle="-", linewidth=2.2, label=f"Mean Conversion: {mean_conv:.2f}% (3,600 / 42,000)")
    ax_a.set_title("Panel A: Follower-to-Customer Conversion Distribution", fontsize=12, fontweight="bold", color="#cdd6f4", pad=10)
    ax_a.set_xlabel("Monthly Active Unique Customers as % of TikTok & IG Followers (%)", fontsize=10, color="#bac2de")
    ax_a.set_ylabel("Probability Density", fontsize=10, color="#bac2de")
    ax_a.legend(loc="upper right", frameon=True, facecolor="#1e1e2e", edgecolor="#45475a", fontsize=9.5)

    # Panel B: Marketing VND Required per Visit Distribution across the 4 Customer Segments
    ax_b = axes[0, 1]
    vnd_data = [sim.sampled_vnd_per_visit_seg[:, :, i].ravel() for i in range(4)]
    bp = ax_b.boxplot(
        vnd_data,
        vert=False,
        patch_artist=True,
        tick_labels=SEG_SHORT_LABELS,
        widths=0.52,
        showfliers=False,
        whiskerprops=dict(color="#bac2de", linewidth=1.4),
        capprops=dict(color="#bac2de", linewidth=1.4),
    )
    for patch, color in zip(bp["boxes"], SEG_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#11111b")
        patch.set_linewidth(1.1)
    for median in bp["medians"]:
        median.set_color("#11111b")
        median.set_linewidth(2.2)

    base_budgets = [s.baseline_marketing_budget_mvnd_dist.mean for s in cfg.segments]
    mean_vnds = [s.marketing_vnd_per_visit_dist.mean for s in cfg.segments]
    for i, (m_val, b_val, v_arr) in enumerate(zip(mean_vnds, base_budgets, vnd_data)):
        q25, q75 = np.percentile(v_arr, [25, 75])
        whisker_hi = float(np.max(v_arr[v_arr <= q75 + 1.5 * (q75 - q25)]))
        ax_b.text(
            whisker_hi + 220,
            i + 1.0,
            f"{m_val:,.0f} VND/visit (Base: {b_val:.2f}M/mo)",
            fontsize=8.8,
            fontweight="bold",
            color="#cdd6f4",
            va="center",
        )
    ax_b.set_xlim(900, 11100)
    ax_b.tick_params(axis="y", labelsize=9.8)
    ax_b.set_title("Panel B: Marketing Acquisition Cost per Visit by Segment", fontsize=12, fontweight="bold", color="#cdd6f4", pad=10)
    ax_b.set_xlabel("Marketing Spend Required per Player Visit (VND / Visit)", fontsize=10, color="#bac2de")

    # Panel C: Merged Monthly Visits Distributions by Segment
    ax_c = axes[1, 0]
    for i, (label, color) in enumerate(zip(SEG_SHORT_LABELS, SEG_COLORS)):
        vis_s = sim.sampled_base_visits_seg[:, :, i].ravel()
        ax_c.hist(vis_s, bins=35, alpha=0.72, color=color, edgecolor="#11111b", linewidth=0.5, label=f"{label}: Mean {np.mean(vis_s):,.0f} visits/mo")
    ax_c.set_title("Panel C: Monthly Player Visit Distributions by Segment", fontsize=12, fontweight="bold", color="#cdd6f4", pad=10)
    ax_c.set_xlabel("Monthly Player Visits per Segment (Visits / Month)", fontsize=10, color="#bac2de")
    ax_c.set_ylabel("Simulation Frequency", fontsize=10, color="#bac2de")
    ax_c.legend(loc="upper right", frameon=True, facecolor="#1e1e2e", edgecolor="#45475a", fontsize=9.2)

    # Panel D: The 12 Segment-by-Ancillary Per-Visit Spending Distributions (Mean +/- 1 SD Error Bars)
    ax_d = axes[1, 1]
    x = np.arange(4)
    width = 0.25

    pad_means = [np.mean(sim.sampled_paddle_spend_kvnd[:, :, i]) for i in range(4)]
    pad_stds = [np.std(sim.sampled_paddle_spend_kvnd[:, :, i]) for i in range(4)]
    coa_means = [np.mean(sim.sampled_coaching_spend_kvnd[:, :, i]) for i in range(4)]
    coa_stds = [np.std(sim.sampled_coaching_spend_kvnd[:, :, i]) for i in range(4)]
    fnb_means = [np.mean(sim.sampled_fnb_spend_kvnd[:, :, i]) for i in range(4)]
    fnb_stds = [np.std(sim.sampled_fnb_spend_kvnd[:, :, i]) for i in range(4)]

    err_kw = dict(ecolor="#bac2de", lw=1.0, alpha=0.85)
    b1 = ax_d.bar(x - width, pad_means, width, yerr=pad_stds, capsize=3, error_kw=err_kw, color="#89b4fa", edgecolor="#11111b", alpha=0.90, label="Paddle & Balls (15.0M/mo Base)")
    b2 = ax_d.bar(x, coa_means, width, yerr=coa_stds, capsize=3, error_kw=err_kw, color="#a6e3a1", edgecolor="#11111b", alpha=0.90, label="Coaching (115.0M/mo Base)")
    b3 = ax_d.bar(x + width, fnb_means, width, yerr=fnb_stds, capsize=3, error_kw=err_kw, color="#fab387", edgecolor="#11111b", alpha=0.90, label="Food & Drinks (75.0M/mo Base)")

    for bars, stds in ((b1, pad_stds), (b2, coa_stds), (b3, fnb_stds)):
        for rect, s_val in zip(bars, stds):
            h = rect.get_height()
            ax_d.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + s_val + 0.55,
                f"{h:.1f}k",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
                color="#cdd6f4",
            )

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(SEG_SHORT_LABELS, fontsize=10, fontweight="bold")
    ax_d.set_ylim(0, 33.5)
    ax_d.set_title("Panel D: Per-Visit Ancillary Spending by Segment & Service Line", fontsize=12, fontweight="bold", color="#cdd6f4", pad=10)
    ax_d.set_ylabel("Ancillary Spend per Player Visit (kVND / Visit)", fontsize=10, color="#bac2de")
    ax_d.legend(loc="upper left", frameon=True, facecolor="#1e1e2e", edgecolor="#45475a", fontsize=9.0)

    out_path = output_dir / "01_marketing_funnel_and_segment_response_distributions.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def plot_ml_segment_budgets_and_visits(
    sim: MarketingSimulationResults,
    ml_res: MarketingMLEvaluationResults,
    output_dir: Path,
) -> Path:
    """
    Render `02_ml_segment_marketing_budget_and_visits_trajectory.png`:
      - Panel A: 60-Month Stacked Area / Bar of ML-Predicted Monthly Marketing Budget across the 4 Segments vs. 25M Baseline
      - Panel B: Resulting 60-Month Player Visits by Customer Segment & Total Visits vs. Baseline
    """
    _apply_clean_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11.5), dpi=220, sharex=True)
    fig.patch.set_facecolor("#1e1e2e")

    fig.suptitle(
        "Dynamic Segment Marketing Allocation & Player Visit Growth Trajectory",
        fontsize=15,
        fontweight="bold",
        color="#cdd6f4",
        y=0.97,
    )
    fig.text(
        0.5,
        0.935,
        "State-Dependent Marketing Budget Reallocation Across Customer Segments & Resulting Monthly Player Attendance",
        ha="center",
        fontsize=10.5,
        color="#a6adc8",
    )

    months = np.arange(sim.n_months)
    ml_budgets_mean = np.mean(ml_res.ml_mkt_budgets_seg, axis=0)  # (60, 4)
    base_tot_b_sims = np.sum(sim.base_mkt_budgets_seg, axis=2)    # (N, 60)
    base_total_budget = np.mean(base_tot_b_sims, axis=0)
    base_total_b_p05 = np.percentile(base_tot_b_sims, 5, axis=0)
    base_total_b_p95 = np.percentile(base_tot_b_sims, 95, axis=0)

    # Panel A: Stacked Bar of ML Predicted Segment Marketing Budgets
    bottom = np.zeros(sim.n_months)
    for i, (label, color) in enumerate(zip(SEG_SHORT_LABELS, SEG_COLORS)):
        ax1.bar(
            months,
            ml_budgets_mean[:, i],
            bottom=bottom,
            color=color,
            edgecolor="#11111b",
            linewidth=0.4,
            alpha=0.88,
            width=0.72,
            label=f"ML Budget: {label} (5Y Avg: {np.mean(ml_budgets_mean[:, i]):.2f}M/mo)",
        )
        bottom += ml_budgets_mean[:, i]

    ax1.fill_between(
        months,
        base_total_b_p05,
        base_total_b_p95,
        color="#7f849c",
        alpha=0.22,
        label="Baseline Marketing P05-P95 Range",
    )
    ax1.plot(
        months,
        base_total_budget,
        color="#bac2de",
        linestyle="-",
        marker="o",
        markersize=3.8,
        linewidth=1.8,
        alpha=0.85,
        label=f"Baseline Total Marketing (~{np.mean(base_total_budget):.2f}M VND/mo)",
    )
    ax1.axvline(4, color="#a6e3a1", linestyle="--", linewidth=1.4, alpha=0.85)
    ax1.text(
        4.3,
        np.max(bottom) * 0.63,
        "M5 (May-27): App Launch",
        fontsize=8.8,
        fontweight="bold",
        color="#a6e3a1",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e1e2e", edgecolor="#a6e3a1", alpha=0.94),
    )
    ax1.axvline(13, color="#f38ba8", linestyle="--", linewidth=1.5, alpha=0.85)
    ax1.text(
        13.35,
        np.max(bottom) * 0.88,
        "M14 (Feb-28): Venue 2 Opens (20 Courts)",
        fontsize=8.8,
        fontweight="bold",
        color="#f38ba8",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e1e2e", edgecolor="#f38ba8", alpha=0.94),
    )
    ax1.set_ylim(0, float(np.max(bottom)) * 1.18)

    ax1.set_title("Panel A: Monthly Marketing Spend Allocation by Customer Segment", fontsize=12, fontweight="bold", color="#cdd6f4", pad=8)
    ax1.set_ylabel("Marketing Spend (Million VND / Month)", fontsize=10, color="#bac2de")
    ax1.legend(loc="upper left", ncol=3, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a", fontsize=8.8)

    # Panel B: Resulting Monthly Visits by Segment vs. Baseline Total Visits
    ml_visits_mean = np.mean(ml_res.ml_visits_seg, axis=0)  # (60, 4)
    base_visits_total_mean = np.mean(sim.base_visits_total, axis=0)
    base_visits_p05 = np.percentile(sim.base_visits_total, 5, axis=0)
    base_visits_p95 = np.percentile(sim.base_visits_total, 95, axis=0)
    ml_visits_total_mean = np.sum(ml_visits_mean, axis=1)

    ax2.stackplot(
        months,
        ml_visits_mean[:, 0],
        ml_visits_mean[:, 1],
        ml_visits_mean[:, 2],
        ml_visits_mean[:, 3],
        labels=[f"Visits: {lbl}" for lbl in SEG_SHORT_LABELS],
        colors=SEG_COLORS,
        alpha=0.78,
    )
    ax2.fill_between(
        months,
        base_visits_p05,
        base_visits_p95,
        color="#7f849c",
        alpha=0.22,
        label="Baseline Visits P05-P95 Range",
    )
    ax2.plot(
        months,
        ml_visits_total_mean,
        color="#cdd6f4",
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        label=f"ML Total Visits (5Y Avg: {np.mean(ml_visits_total_mean):,.0f}/mo)",
    )
    ax2.plot(
        months,
        base_visits_total_mean,
        color="#9399b2",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        alpha=0.85,
        label=f"Baseline Visits (5Y Avg: {np.mean(base_visits_total_mean):,.0f}/mo)",
    )
    ax2.axvline(4, color="#a6e3a1", linestyle="--", linewidth=1.4, alpha=0.85)
    ax2.axvline(13, color="#f38ba8", linestyle="--", linewidth=1.5, alpha=0.85)

    tick_idx = np.arange(0, sim.n_months, 3)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([sim.month_labels[i] for i in tick_idx], rotation=30, ha="right", fontsize=9)
    ax2.set_title("Panel B: Monthly Player Visits by Customer Segment", fontsize=12, fontweight="bold", color="#cdd6f4", pad=8)
    ax2.set_ylabel("Monthly Player Visits (Visits / Month)", fontsize=10, color="#bac2de")
    ax2.set_xlabel("Projection Timeline (60 Months: Jan 2027 - Dec 2031)", fontsize=10, color="#bac2de")
    ax2.legend(loc="upper left", ncol=3, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a", fontsize=9.0)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out_path = output_dir / "02_ml_segment_marketing_budget_and_visits_trajectory.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_segment_ancillary_revenue_breakdown(
    sim: MarketingSimulationResults,
    ml_res: MarketingMLEvaluationResults,
    output_dir: Path,
) -> Path:
    """
    Render `03_segment_ancillary_revenue_breakdown.png`:
      - Panel A: Paddle Rental & Balls Revenue by Customer Segment (M VND/mo) vs. 15.00M Case Baseline
      - Panel B: Coaching Revenue by Customer Segment (M VND/mo) vs. 115.00M Case Baseline
      - Panel C: Food & Drinks (F&B) Revenue by Customer Segment (M VND/mo) vs. 75.00M Case Baseline
    """
    _apply_clean_style()
    fig, axes = plt.subplots(3, 1, figsize=(18, 13.5), dpi=220, sharex=True)
    fig.patch.set_facecolor("#1e1e2e")

    fig.suptitle(
        "Ancillary Revenue Composition by Customer Segment Across Projection Horizon",
        fontsize=15,
        fontweight="bold",
        color="#cdd6f4",
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Monthly Equipment Rental, Coaching & Hospitality Revenue Decomposed by Customer Segment",
        ha="center",
        fontsize=10.5,
        color="#a6adc8",
    )

    months = np.arange(sim.n_months)
    pad_seg = np.mean(ml_res.ml_paddle_rev_seg, axis=0)      # (60, 4)
    coa_seg = np.mean(ml_res.ml_coaching_rev_seg, axis=0)    # (60, 4)
    fnb_seg = np.mean(ml_res.ml_fnb_rev_seg, axis=0)         # (60, 4)

    base_pad_sims = np.sum(sim.base_paddle_rev_seg, axis=2)
    base_coa_sims = np.sum(sim.base_coaching_rev_seg, axis=2)
    base_fnb_sims = np.sum(sim.base_fnb_rev_seg, axis=2)

    base_pad = np.mean(base_pad_sims, axis=0)
    base_coa = np.mean(base_coa_sims, axis=0)
    base_fnb = np.mean(base_fnb_sims, axis=0)

    panels = [
        (axes[0], pad_seg, base_pad, base_pad_sims, "Panel A: Equipment Rental & Balls Revenue by Customer Segment", 15.0),
        (axes[1], coa_seg, base_coa, base_coa_sims, "Panel B: Coaching Services Revenue by Customer Segment", 115.0),
        (axes[2], fnb_seg, base_fnb, base_fnb_sims, "Panel C: Food & Beverage Revenue by Customer Segment", 75.0),
    ]

    for ax, seg_arr, base_arr, base_sims, title, case_m1 in panels:
        ax.stackplot(
            months,
            seg_arr[:, 0],
            seg_arr[:, 1],
            seg_arr[:, 2],
            seg_arr[:, 3],
            labels=[f"{lbl} (5Y Avg: {np.mean(seg_arr[:, i]):.1f}M)" for i, lbl in enumerate(SEG_SHORT_LABELS)],
            colors=SEG_COLORS,
            alpha=0.80,
        )
        tot_arr = np.sum(seg_arr, axis=1)
        ax.fill_between(
            months,
            np.percentile(base_sims, 5, axis=0),
            np.percentile(base_sims, 95, axis=0),
            color="#7f849c",
            alpha=0.22,
            label="Baseline P05-P95 Range",
        )
        ax.plot(
            months,
            tot_arr,
            color="#cdd6f4",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=f"Solution Total (5Y Avg: {np.mean(tot_arr):.1f}M/mo)",
        )
        ax.plot(
            months,
            base_arr,
            color="#9399b2",
            linewidth=1.6,
            marker="o",
            markersize=3.2,
            alpha=0.82,
            label=f"Baseline Trend-Adjusted (M1: {case_m1:.1f}M/mo)",
        )
        ax.axvline(13, color="#f38ba8", linestyle="--", linewidth=1.4, alpha=0.85)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color="#cdd6f4", pad=6)
        ax.set_ylabel("Million VND / Month", fontsize=9.5, color="#bac2de")
        ax.legend(loc="upper left", ncol=3, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a", fontsize=8.8)

    tick_idx = np.arange(0, sim.n_months, 3)
    axes[2].set_xticks(tick_idx)
    axes[2].set_xticklabels([sim.month_labels[i] for i in tick_idx], rotation=30, ha="right", fontsize=9)
    axes[2].set_xlabel("Projection Timeline (60 Months: Jan 2027 - Dec 2031)", fontsize=10, color="#bac2de")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = output_dir / "03_segment_ancillary_revenue_breakdown.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_marketing_ml_diagnostics(
    sim: MarketingSimulationResults,
    ml_res: MarketingMLEvaluationResults,
    output_dir: Path,
) -> Path:
    """
    Render `04_marketing_ml_diagnostics.png`:
      - Panel A: Out-of-Sample Predicted vs. Oracle Marketing Budgets across the 4 Customer Segments
      - Panel B: Permutation Feature Importance (%) on Held-Out Test Split
    """
    _apply_clean_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.8), dpi=220)
    fig.patch.set_facecolor("#1e1e2e")

    m = ml_res.ml_metrics
    fig.suptitle(
        "Marketing Budget Policy Model Validation & Feature Importance",
        fontsize=14.5,
        fontweight="bold",
        color="#cdd6f4",
        y=0.97,
    )
    fig.text(
        0.5,
        0.92,
        "Out-of-Sample Predicted vs. Optimal Segment Budgets & Permutation Feature Attribution on Held-Out Test Split",
        ha="center",
        fontsize=10.2,
        color="#a6adc8",
    )

    # Panel A: Predicted vs Oracle Scatter across the 4 Segments
    for i, (label, color) in enumerate(zip(SEG_SHORT_LABELS, SEG_COLORS)):
        y_t = ml_res.y_test_true[:, i]
        y_p = ml_res.y_test_pred[:, i]
        seg_key = sim.segment_keys[i]
        r2_s = m.seg_test_r2[seg_key]
        ax1.scatter(
            y_t[::3],
            y_p[::3],
            alpha=0.52,
            s=18,
            color=color,
            label=f"{label} (Test R2 = {r2_s:.3f})",
        )

    lims = [1.5, 24.5]
    ax1.plot(lims, lims, color="#f38ba8", linestyle="--", linewidth=1.8, label="Ideal 1:1 Parity Line")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title("Panel A: Out-of-Sample Predicted vs. Optimal Segment Budgets", fontsize=12, fontweight="bold", color="#cdd6f4", pad=8)
    ax1.set_xlabel("Oracle Optimal Segment Marketing Budget (M VND / Month)", fontsize=10, color="#bac2de")
    ax1.set_ylabel("HistGBR Predicted Segment Marketing Budget (M VND / Month)", fontsize=10, color="#bac2de")
    ax1.legend(loc="upper left", frameon=True, facecolor="#1e1e2e", edgecolor="#45475a", fontsize=9.2)

    # Panel B: Permutation Feature Importance
    order = np.argsort(m.perm_importance_mean)
    sorted_names = [m.feature_names[idx] for idx in order]
    sorted_imp = m.perm_importance_mean[order]
    y_pos = np.arange(len(sorted_names))

    bars = ax2.barh(y_pos, sorted_imp, color="#89b4fa", alpha=0.88, edgecolor="#11111b", height=0.68)
    for bar, val in zip(bars, sorted_imp):
        ax2.text(
            val + 0.4,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#cdd6f4",
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names, fontsize=9.2, fontweight="bold", color="#cdd6f4")
    ax2.set_xlim(0, max(float(np.max(sorted_imp)) * 1.18, 15.0))
    ax2.set_title("Panel B: Permutation Feature Importance on Held-Out Test Split", fontsize=12, fontweight="bold", color="#cdd6f4", pad=8)
    ax2.set_xlabel("Relative Permutation Importance (%)", fontsize=10, color="#bac2de")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    out_path = output_dir / "04_marketing_ml_diagnostics.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path
