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

SEG_COLORS = ["#1D4ED8", "#0D9488", "#D97706", "#7C3AED"]
SEG_SHORT_LABELS = ["Competitive", "Social", "Trend Explorers", "Families"]


def _apply_clean_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#F8FAFC",
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
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "PICK! MARKETING FUNNEL (~8.57% CONVERSION), SEGMENT VND/VISIT & 12 ANCILLARY SPENDING DISTRIBUTIONS",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )
    fig.text(
        0.5,
        0.935,
        "Explicit Assumption: Active Customers (3,600) are a Subset of TikTok/IG Followers (42,000) | 25.00M VND Baseline Split Across 4 Segments | 12 Grounded Ancillary Distributions",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )

    # Panel A: Follower-to-Customer Conversion Rate Distribution (~8.57%)
    ax_a = axes[0, 0]
    conv_samples = sim.sampled_conv_rate.ravel() * 100.0
    ax_a.hist(conv_samples, bins=42, color="#0284C7", alpha=0.78, edgecolor="#0369A1", density=True)
    mean_conv = float(np.mean(conv_samples))
    ax_a.axvline(mean_conv, color="#DC2626", linestyle="-", linewidth=2.2, label=f"Mean Conversion: {mean_conv:.2f}% (3,600 / 42,000)")
    ax_a.set_title("Panel A: Follower-to-Customer Conversion Distribution (~8.57%)", fontsize=12, fontweight="bold", color="#0F172A", pad=10)
    ax_a.set_xlabel("Monthly Active Unique Customers as % of TikTok & IG Followers (%)", fontsize=10, color="#334155")
    ax_a.set_ylabel("Probability Density", fontsize=10, color="#334155")
    ax_a.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

    # Panel B: Marketing VND Required per Visit Distribution across the 4 Customer Segments
    ax_b = axes[0, 1]
    vnd_data = [sim.sampled_vnd_per_visit_seg[:, :, i].ravel() for i in range(4)]
    bp = ax_b.boxplot(
        vnd_data,
        vert=False,
        patch_artist=True,
        tick_labels=SEG_SHORT_LABELS,
        widths=0.55,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], SEG_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#0F172A")
    for median in bp["medians"]:
        median.set_color("#FFFFFF")
        median.set_linewidth(2.0)

    base_budgets = [s.baseline_marketing_budget_mvnd_dist.mean for s in cfg.segments]
    mean_vnds = [s.marketing_vnd_per_visit_dist.mean for s in cfg.segments]
    for i, (m_val, b_val, v_arr) in enumerate(zip(mean_vnds, base_budgets, vnd_data)):
        p95_val = float(np.percentile(v_arr, 95))
        ax_b.text(
            p95_val + 260,
            i + 1.0,
            f"{m_val:,.0f} VND/visit (Base: {b_val:.2f}M/mo)",
            fontsize=9.0,
            fontweight="bold",
            color="#0F172A",
            va="center",
        )
    ax_b.set_xlim(900, 9600)
    ax_b.set_title("Panel B: Marketing Cost per Player Visit by Segment (VND/Visit)", fontsize=12, fontweight="bold", color="#0F172A", pad=10)
    ax_b.set_xlabel("Marketing Spend Required per Player Visit (VND / Visit)", fontsize=10, color="#334155")

    # Panel C: Merged Monthly Visits Distributions by Segment (3,240 | 2,520 | 1,188 | 648 = 7,596 ~ 7,600)
    ax_c = axes[1, 0]
    for i, (label, color) in enumerate(zip(SEG_SHORT_LABELS, SEG_COLORS)):
        vis_s = sim.sampled_base_visits_seg[:, :, i].ravel()
        ax_c.hist(vis_s, bins=35, alpha=0.65, color=color, edgecolor=color, label=f"{label}: Mean {np.mean(vis_s):,.0f} visits/mo")
    ax_c.set_title("Panel C: Merged Monthly Visits Distributions by Segment (Sum = 7,596/mo)", fontsize=12, fontweight="bold", color="#0F172A", pad=10)
    ax_c.set_xlabel("Monthly Player Visits per Segment (Visits / Month)", fontsize=10, color="#334155")
    ax_c.set_ylabel("Frequency across N=300 x 60m Draws", fontsize=10, color="#334155")
    ax_c.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.2)

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

    b1 = ax_d.bar(x - width, pad_means, width, yerr=pad_stds, capsize=3, color="#0284C7", alpha=0.88, label="Paddle & Balls (15.0M/mo Base)")
    b2 = ax_d.bar(x, coa_means, width, yerr=coa_stds, capsize=3, color="#16A34A", alpha=0.88, label="Coaching (115.0M/mo Base)")
    b3 = ax_d.bar(x + width, fnb_means, width, yerr=fnb_stds, capsize=3, color="#EA580C", alpha=0.88, label="Food & Drinks (75.0M/mo Base)")

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
                color="#1E293B",
            )

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(SEG_SHORT_LABELS, fontsize=10, fontweight="bold")
    ax_d.set_ylim(0, 32.5)
    ax_d.set_title("Panel D: 12 Segment-by-Ancillary Spending Distributions (kVND / Visit)", fontsize=12, fontweight="bold", color="#0F172A", pad=10)
    ax_d.set_ylabel("Ancillary Spend per Player Visit (kVND / Visit)", fontsize=10, color="#334155")
    ax_d.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.0)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out_path = output_dir / "01_marketing_funnel_and_segment_response_distributions.png"
    fig.savefig(out_path, dpi=220)
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
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "ML SEGMENT MARKETING BUDGET ALLOCATION & RESULTING PLAYER VISITS BY SEGMENT (JAN 2027 - DEC 2031)",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )
    fig.text(
        0.5,
        0.935,
        "HistGradientBoostingRegressor Predicts Monthly Segment Marketing Budgets from Last Month's Segment Customers, Followers, Trend & 20-Court Capacity",
        ha="center",
        fontsize=10.5,
        color="#475569",
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
            alpha=0.86,
            width=0.72,
            label=f"ML Budget: {label} (5Y Avg: {np.mean(ml_budgets_mean[:, i]):.2f}M/mo)",
        )
        bottom += ml_budgets_mean[:, i]

    ax1.fill_between(
        months,
        base_total_b_p05,
        base_total_b_p95,
        color="#64748B",
        alpha=0.18,
        label="Baseline Marketing P05-P95 Range",
    )
    ax1.plot(
        months,
        base_total_budget,
        color="#64748B",
        linestyle="-",
        marker="o",
        markersize=3.8,
        linewidth=1.8,
        alpha=0.75,
        label=f"Baseline Total Marketing (~{np.mean(base_total_budget):.2f}M VND/mo)",
    )
    ax1.axvline(4, color="#0F172A", linestyle="--", linewidth=1.3, alpha=0.7)
    ax1.text(
        4.3,
        np.max(bottom) * 0.63,
        "M5 (May-27): App Launch",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.92),
    )
    ax1.axvline(13, color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.text(
        13.35,
        np.max(bottom) * 0.88,
        "M14 (Feb-28): Venue 2 Opens (20 Courts)",
        fontsize=8.8,
        fontweight="bold",
        color="#DC2626",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#FCA5A5", alpha=0.92),
    )
    ax1.set_ylim(0, float(np.max(bottom)) * 1.18)

    ax1.set_title("Panel A: ML-Predicted Monthly Marketing Spend by Customer Segment (M VND / Month)", fontsize=12, fontweight="bold", color="#0F172A", pad=8)
    ax1.set_ylabel("Marketing Spend (Million VND / Month)", fontsize=10, color="#334155")
    ax1.legend(loc="upper left", ncol=3, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.8)

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
        color="#64748B",
        alpha=0.20,
        label="Baseline Visits P05-P95 Range",
    )
    ax2.plot(
        months,
        ml_visits_total_mean,
        color="#0F172A",
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        label=f"ML Total Visits (5Y Avg: {np.mean(ml_visits_total_mean):,.0f}/mo)",
    )
    ax2.plot(
        months,
        base_visits_total_mean,
        color="#64748B",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        alpha=0.75,
        label=f"Baseline Visits (5Y Avg: {np.mean(base_visits_total_mean):,.0f}/mo)",
    )
    ax2.axvline(4, color="#0F172A", linestyle="--", linewidth=1.3, alpha=0.7)
    ax2.axvline(13, color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.8)

    tick_idx = np.arange(0, sim.n_months, 3)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([sim.month_labels[i] for i in tick_idx], rotation=30, ha="right", fontsize=9)
    ax2.set_title("Panel B: Resulting Monthly Player Visits Breakdown Across the 4 Customer Segments (Visits / Month)", fontsize=12, fontweight="bold", color="#0F172A", pad=8)
    ax2.set_ylabel("Monthly Player Visits (Visits / Month)", fontsize=10, color="#334155")
    ax2.set_xlabel("Projection Timeline (60 Months: Jan 2027 - Dec 2031)", fontsize=10, color="#334155")
    ax2.legend(loc="upper left", ncol=3, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.0)

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
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "ANCILLARY REVENUE LINES DRIVEN BY THE 12 SEGMENT SPENDING DISTRIBUTIONS (JAN 2027 - DEC 2031)",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Monthly Paddle Rental & Balls, Coaching, and Food & Drinks Revenue Decomposed by Customer Segment vs. Case Exhibit 1 Baselines (15M, 115M, 75M)",
        ha="center",
        fontsize=10.5,
        color="#475569",
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
        (axes[0], pad_seg, base_pad, base_pad_sims, "Panel A: Revenue Line 2 — Paddle Rental & Balls by Segment (Base M1 = 15.00M VND/mo)", 15.0),
        (axes[1], coa_seg, base_coa, base_coa_sims, "Panel B: Revenue Line 3 — Coaching Services by Segment (Base M1 = 115.00M VND/mo)", 115.0),
        (axes[2], fnb_seg, base_fnb, base_fnb_sims, "Panel C: Revenue Line 4 — Food & Drinks (Cafe/F&B) by Segment (Base M1 = 75.00M VND/mo)", 75.0),
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
            color="#64748B",
            alpha=0.18,
            label="Baseline P05-P95 Range",
        )
        ax.plot(
            months,
            tot_arr,
            color="#0F172A",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=f"Solution Total (5Y Avg: {np.mean(tot_arr):.1f}M/mo)",
        )
        ax.plot(
            months,
            base_arr,
            color="#64748B",
            linewidth=1.6,
            marker="o",
            markersize=3.2,
            alpha=0.72,
            label=f"Baseline Trend-Adjusted (M1: {case_m1:.1f}M/mo)",
        )
        ax.axvline(13, color="#DC2626", linestyle="--", linewidth=1.4, alpha=0.8)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color="#0F172A", pad=6)
        ax.set_ylabel("Million VND / Month", fontsize=9.5, color="#334155")
        ax.legend(loc="upper left", ncol=3, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.8)

    tick_idx = np.arange(0, sim.n_months, 3)
    axes[2].set_xticks(tick_idx)
    axes[2].set_xticklabels([sim.month_labels[i] for i in tick_idx], rotation=30, ha="right", fontsize=9)
    axes[2].set_xlabel("Projection Timeline (60 Months: Jan 2027 - Dec 2031)", fontsize=10, color="#334155")

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
    fig.patch.set_facecolor("#FFFFFF")

    m = ml_res.ml_metrics
    fig.suptitle(
        "SEGMENT MARKETING BUDGET ML PREDICTOR — OUT-OF-SAMPLE DIAGNOSTICS & FEATURE IMPORTANCE",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )
    fig.text(
        0.5,
        0.92,
        f"Held-Out Test Split (N={m.n_test_obs:,} Monthly States): R2 = {m.test_r2:.4f} | RMSE = {m.test_rmse_mvnd:.3f} M VND | MAPE = {m.test_mape_pct:.2f}% | Oracle Net Contribution Capture = {m.oracle_net_contrib_capture_pct:.2f}%",
        ha="center",
        fontsize=10.2,
        color="#475569",
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
            alpha=0.45,
            s=18,
            color=color,
            label=f"{label} (Test R2 = {r2_s:.3f})",
        )

    lims = [1.5, 24.5]
    ax1.plot(lims, lims, color="#0F172A", linestyle="--", linewidth=1.8, label="Ideal 1:1 Parity Line")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title("Panel A: Out-of-Sample Predicted vs. Oracle Segment Budgets (M VND/mo)", fontsize=12, fontweight="bold", color="#0F172A", pad=8)
    ax1.set_xlabel("Oracle Optimal Segment Marketing Budget (M VND / Month)", fontsize=10, color="#334155")
    ax1.set_ylabel("HistGBR Predicted Segment Marketing Budget (M VND / Month)", fontsize=10, color="#334155")
    ax1.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.2)

    # Panel B: Permutation Feature Importance
    order = np.argsort(m.perm_importance_mean)
    sorted_names = [m.feature_names[idx] for idx in order]
    sorted_imp = m.perm_importance_mean[order]
    y_pos = np.arange(len(sorted_names))

    bars = ax2.barh(y_pos, sorted_imp, color="#0284C7", alpha=0.86, edgecolor="#0369A1", height=0.68)
    for bar, val in zip(bars, sorted_imp):
        ax2.text(
            val + 0.4,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#0F172A",
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names, fontsize=9.2, fontweight="bold", color="#1E293B")
    ax2.set_xlim(0, max(float(np.max(sorted_imp)) * 1.18, 15.0))
    ax2.set_title("Panel B: Permutation Feature Importance on Held-Out Test Split (%)", fontsize=12, fontweight="bold", color="#0F172A", pad=8)
    ax2.set_xlabel("Relative Permutation Importance (%)", fontsize=10, color="#334155")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    out_path = output_dir / "04_marketing_ml_diagnostics.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path
