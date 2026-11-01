"""Presentation-clean Gantt chart, 20-Court Capacity & 6-to-9 Month Fill-Up Ramp, and Even 7.0B VND CapEx Allocation visualizer (No Icons, No Inline Citations)."""

from __future__ import annotations

from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .config import VenueScalingConfig
from .schedule_model import ScheduledVenueStage, ScheduledVenueTask, VenueScalingResults
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


def generate_all_venue_scaling_charts(
    cfg: VenueScalingConfig,
    res: VenueScalingResults,
    output_dir: Path,
) -> list[Path]:
    """Render all 3 presentation-clean charts for VenueScalingTimeline into outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p1 = plot_venue_scaling_gantt_chart(cfg, res, output_dir)
    p2 = plot_court_capacity_and_fill_up_ramp(cfg, res, output_dir)
    p3 = plot_capex_allocation_and_unit_economics(cfg, res, output_dir)
    return [p1, p2, p3]


def plot_venue_scaling_gantt_chart(
    cfg: VenueScalingConfig,
    res: VenueScalingResults,
    output_dir: Path,
) -> Path:
    """
    Render Chart 1 (01_venue_scaling_gantt_chart.png):
      - Master Construction, Mandatory 28-Day Slab Cure, Fit-Out & Launch Gantt Chart (Q3–Q4 2027: Weeks 1–16, Months 7–10: Jul–Oct 2027)
        plus Entry into the 6-to-9 Month Fill-Up Ramp (Month 11 / Nov 2027 onward).
      - Zero icons, zero diamond markers, zero inline citation keys.
    """
    # Display Stages V1..V5 (the 16-week / 4-month construction & fit-out build) + Stage V6 fill-up summary
    display_rows: list[tuple[str, ScheduledVenueStage | ScheduledVenueTask]] = []
    for st in res.stages:
        display_rows.append(("stage", st))
        for t in res.tasks:
            if t.stage_id == st.stage_id:
                display_rows.append(("task", t))

    n_rows = len(display_rows)
    fig_w = 28.0
    fig_h = 14.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")

    # Timeline x-axis displays Weeks 0..24 from Q4 2027 Kickoff (Oct 2027 – Mar 2028)
    x_left_min = -30.0
    x_right_max = 24.05
    hdr_top = n_rows + 1.65
    hdr_mid = n_rows + 0.65
    hdr_bot = float(n_rows)

    ax.set_xlim(x_left_min, x_right_max)
    ax.set_ylim(-1.65, hdr_top + 0.35)

    month_defs = [
        ("MONTH 10: OCT 2027", "Handover, Permits & Earthworks", 0.0, 4.0, "#181825", "#313244"),
        ("MONTH 11: NOV 2027", "14-Court Slab Pour & 28d Cure", 4.0, 8.0, "#1e1e2e", "#313244"),
        ("MONTH 12: DEC 2027", "Canopy, MEP, LED & Acrylic Prep", 8.0, 12.0, "#181825", "#313244"),
        ("MONTH 13: JAN 2028", "Surfacing, Acoustics & PCCC", 12.0, 16.0, "#291e28", "#452738"),
        ("MONTH 14: FEB 2028", "20 Courts Live / Fill-Up M1", 16.0, 20.0, "#1e2926", "#233b30"),
        ("MONTH 15: MAR 2028+", "Fill-Up Ramp (to Sep-28)", 20.0, 24.0, "#1e292e", "#253b3f"),
    ]

    for m_title, m_sub, m_start, m_end, m_bg, hdr_bg in month_defs:
        ax.add_patch(
            mpatches.Rectangle(
                (m_start, 0.0),
                m_end - m_start,
                n_rows,
                facecolor=m_bg,
                edgecolor="none",
                zorder=0,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (m_start, hdr_mid),
                m_end - m_start,
                hdr_top - hdr_mid,
                facecolor=hdr_bg,
                edgecolor="#585b70",
                linewidth=1.0,
                zorder=4,
            )
        )
        ax.text(
            (m_start + m_end) / 2.0,
            hdr_mid + 0.66,
            m_title,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )
        ax.text(
            (m_start + m_end) / 2.0,
            hdr_mid + 0.28,
            m_sub,
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="normal",
            color="#bac2de",
            zorder=5,
        )

    for w in range(24):
        wk_bg = "#313244" if (w // 4) % 2 == 0 else "#45475a"
        ax.add_patch(
            mpatches.Rectangle(
                (float(w), hdr_bot),
                1.0,
                hdr_mid - hdr_bot,
                facecolor=wk_bg,
                edgecolor="#585b70",
                linewidth=0.7,
                zorder=4,
            )
        )
        ax.axvline(w, ymin=0.06, ymax=0.92, color="#313244", linewidth=0.55, linestyle=":", zorder=1)
        ax.text(
            w + 0.5,
            (hdr_bot + hdr_mid) / 2.0,
            f"W{w + 1}",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )

    for m_boundary in (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0):
        ax.axvline(m_boundary, ymin=0.06, ymax=0.92, color="#585b70", linewidth=1.0, zorder=2)

    col_defs = [
        ("ID", -30.0, -28.7, "center"),
        ("STAGE / CONSTRUCTION & FIT-OUT TASK (WBS)", -28.7, -14.2, "left"),
        ("CORE DELIVERABLE & SPEC", -14.2, -6.0, "left"),
        ("WKS", -6.0, -4.8, "center"),
        ("WINDOW", -4.8, -2.6, "center"),
        ("CAPEX (M VND)", -2.6, 0.0, "center"),
    ]
    ax.add_patch(
        mpatches.Rectangle(
            (x_left_min, hdr_bot),
            -x_left_min,
            hdr_top - hdr_bot,
            facecolor="#11111b",
            edgecolor="#585b70",
            linewidth=1.0,
            zorder=4,
        )
    )
    for c_title, c_x0, c_x1, _ in col_defs:
        ax.text(
            (c_x0 + c_x1) / 2.0,
            (hdr_bot + hdr_top) / 2.0,
            c_title,
            ha="center",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )
        ax.axvline(c_x1, ymin=0.06, ymax=0.95, color="#45475a", linewidth=0.7, zorder=3)

    for row_idx, (rtype, item) in enumerate(display_rows):
        y_top = n_rows - row_idx
        y_bot = y_top - 1.0
        y_mid = y_bot + 0.5

        if rtype == "stage":
            st: ScheduledVenueStage = item  # type: ignore[assignment]
            ax.add_patch(
                mpatches.Rectangle(
                    (x_left_min, y_bot),
                    24.0 - x_left_min,
                    1.0,
                    facecolor=st.light_color,
                    edgecolor="#45475a",
                    linewidth=0.8,
                    zorder=1,
                )
            )
            ax.text(-29.35, y_mid, st.stage_id, ha="center", va="center", fontsize=8.0, fontweight="bold", color=st.color, zorder=5)
            ax.text(-28.5, y_mid, st.name.upper(), ha="left", va="center", fontsize=6.8, fontweight="bold", color="#cdd6f4", zorder=5)
            ax.text(
                -14.0,
                y_mid,
                f"Share of 7,000M Fit-Out CapEx: {st.budget_share_pct:.1f}% ({st.task_count} Tasks)" if st.stage_capex_mvnd > 0 else "Post-Opening 6-to-9 Month Capacity Fill-Up Phase",
                ha="left",
                va="center",
                fontsize=6.9,
                fontweight="bold",
                color=st.color,
                zorder=5,
            )
            ax.text(-5.40, y_mid, f"{st.span_weeks:.1f}w", ha="center", va="center", fontsize=7.5, fontweight="bold", color=st.color, zorder=5)
            ax.text(
                -3.70,
                y_mid,
                f"{st.start_date.strftime('%b %d')}–{st.end_date.strftime('%b %d')}",
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
                color="#bac2de",
                zorder=5,
            )
            ax.text(
                -1.30,
                y_mid,
                f"{st.stage_capex_mvnd:,.1f}M" if st.stage_capex_mvnd > 0 else "Operating Ramp",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#cdd6f4",
                zorder=5,
            )

            disp_start = min(st.start_week, 24.0)
            disp_end = min(st.end_week, 24.0)
            if disp_end > disp_start:
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (disp_start, y_mid - 0.14),
                        disp_end - disp_start,
                        0.28,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=st.color,
                        edgecolor="#11111b",
                        linewidth=1.0,
                        alpha=0.92,
                        zorder=4,
                    )
                )
        else:
            t: ScheduledVenueTask = item  # type: ignore[assignment]
            row_bg = "#1e1e2e" if row_idx % 2 == 0 else "#181825"
            ax.add_patch(
                mpatches.Rectangle(
                    (x_left_min, y_bot),
                    -x_left_min,
                    1.0,
                    facecolor=row_bg,
                    edgecolor="#313244",
                    linewidth=0.6,
                    zorder=1,
                )
            )
            ax.axhline(y_bot, color="#313244", linewidth=0.55, zorder=1)

            id_col = "#f38ba8" if t.is_critical_path else "#a6adc8"
            ax.text(-29.35, y_mid, t.id, ha="center", va="center", fontsize=7.5, fontweight="bold", color=id_col, zorder=5)
            ax.text(
                -28.5,
                y_mid,
                t.name,
                ha="left",
                va="center",
                fontsize=6.9,
                fontweight="bold" if t.is_critical_path else "normal",
                color="#cdd6f4",
                zorder=5,
            )
            ax.text(-14.0, y_mid, t.core_deliverable, ha="left", va="center", fontsize=6.6, color="#bac2de", zorder=5)
            ax.text(-5.40, y_mid, f"{t.duration_weeks:.1f}w", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#cdd6f4", zorder=5)
            ax.text(-3.70, y_mid, f"W{t.start_week + 1:g}–W{t.end_week:g}", ha="center", va="center", fontsize=6.8, color="#a6adc8", zorder=5)
            ax.text(
                -1.30,
                y_mid,
                f"{t.capex_mvnd:,.1f}M" if t.capex_mvnd > 0 else "Fill-Up Ramp",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color="#bac2de",
                zorder=5,
            )

            disp_start = min(t.start_week, 23.8)
            disp_end = min(t.end_week, 24.0)
            disp_w = max(0.2, disp_end - disp_start)

            bar_h = 0.56
            bar_y = y_mid - bar_h / 2.0
            edge_col = "#11111b" if t.is_critical_path else t.stage_color
            edge_w = 1.35 if t.is_critical_path else 0.9
            hatch_pat = "///" if (t.id == "V2.3" or t.is_parallel_track) else None

            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (disp_start, bar_y),
                    disp_w,
                    bar_h,
                    boxstyle="round,pad=0.01,rounding_size=0.08",
                    facecolor=t.stage_color,
                    edgecolor=edge_col,
                    linewidth=edge_w,
                    alpha=0.90 if not t.is_parallel_track else 0.80,
                    hatch=hatch_pat,
                    zorder=4,
                )
            )

            bar_label = f"{t.id} ({t.duration_weeks:.1f}w)" if t.end_week <= 24.0 else f"{t.id} (Continues to W{t.end_week:.0f} / Sep 2028)"
            if disp_w >= 2.8:
                lbl_col = "#cdd6f4" if hatch_pat else "#11111b"
                ax.text(
                    disp_start + disp_w / 2.0,
                    y_mid,
                    bar_label,
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    fontweight="bold",
                    color=lbl_col,
                    bbox=dict(boxstyle="round,pad=0.08", facecolor="#1e1e2e", edgecolor="none", alpha=0.80) if hatch_pat else None,
                    zorder=6,
                )
            elif disp_start >= 22.0:
                ax.text(
                    disp_start - 0.18,
                    y_mid,
                    bar_label,
                    ha="right",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold",
                    color="#cdd6f4",
                    zorder=6,
                )
            else:
                ax.text(
                    disp_end + 0.14,
                    y_mid,
                    bar_label,
                    ha="left",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold",
                    color="#cdd6f4",
                    zorder=6,
                )

    ax.plot([x_left_min, 24.0], [0.0, 0.0], color="#585b70", linewidth=1.1, zorder=5)

    # Bottom summary pill under left WBS table
    ax.text(
        -12.8,
        -0.85,
        "4-MONTH BUILD (OCT 2027 – JAN 2028): 7,000.0M VND Fit-Out Split Evenly = 1,750.0M VND/Mo (M10–M13)  |  "
        "5-YEAR LEASE: 330.0M VND/Mo (from Oct-27)  |  RESERVE: 1,000.0M VND (500M M1 + 500M M10)",
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="bold",
        color="#cdd6f4",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#313244", edgecolor="#585b70", linewidth=1.1),
        zorder=9,
    )

    # Highlight Week 16 (End of Month 13 / Jan 2028 = 14-Court Club Fit-Out Complete, Scaling 6 -> 20 Courts)
    ax.axvline(16.0, ymin=0.04, ymax=0.92, color="#f38ba8", linewidth=2.4, linestyle="--", zorder=7)
    ax.text(
        16.0,
        -0.85,
        "END OF MONTH 13 (W16: JAN 30, 2028) — 14-COURT FIT-OUT COMPLETE / TOTAL CAPACITY SCALES FROM 6 TO 20 COURTS",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#11111b",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f38ba8", edgecolor="#eba0ac", linewidth=1.2),
        zorder=9,
    )

    legend_handles = [
        mpatches.Patch(facecolor="#89b4fa", edgecolor="#11111b", label="Stage 1: Handover, Acoustic Design & PCCC Permits (455.0M / 6.5%)"),
        mpatches.Patch(facecolor="#cba6f7", edgecolor="#11111b", label="Stage 2: Earthworks, 14-Court Slab Pour & 28-Day Cure (2,170.0M / 31.0%)"),
        mpatches.Patch(facecolor="#a6e3a1", edgecolor="#11111b", hatch="///", label="Stage 3: Canopy Roof, 3-Phase MEP, LED, Cafe & Coaching (2,310.0M / 33.0%)"),
        mpatches.Patch(facecolor="#fab387", edgecolor="#11111b", label="Stage 4: Cushioned Acrylic Surfacing & STC-32 Acoustics (1,505.0M / 21.5%)"),
        mpatches.Patch(facecolor="#f38ba8", edgecolor="#11111b", label="Stage 5: PCCC Final Audit, POS Integration & Launch (560.0M / 8.0%)"),
        mpatches.Patch(facecolor="#94e2d5", edgecolor="#11111b", label="Stage 6: 6-to-9 Month Capacity Fill-Up Ramp (Feb 2028 – Sep 2028)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.048),
        ncol=3,
        fontsize=7.8,
        frameon=True,
        facecolor="#181825",
        edgecolor="#45475a",
    )

    ax.set_title(
        "Second Club Facility Construction, Surface Curing & Operational Ramp Schedule\n"
        "Turnkey Engineering Milestones, Parallel Fit-Out Workflows & Capacity Fill-Up Trajectory",
        fontsize=12.0,
        fontweight="bold",
        color="#cdd6f4",
        pad=14,
    )
    ax.axis("off")

    out_path = output_dir / "01_venue_scaling_gantt_chart.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def plot_court_capacity_and_fill_up_ramp(
    cfg: VenueScalingConfig,
    res: VenueScalingResults,
    output_dir: Path,
) -> Path:
    """
    Render Chart 2 (02_court_capacity_and_fill_up_ramp.png):
      - Panel A (Left): Active Court Capacity Step-Up (6 Courts in M1–M10 -> 20 Courts in M11+) &
        Stochastic 6-to-9 Month Fill-Up Ramp (Effective Full-Equivalent Courts & Utilization Band across 60 Months).
      - Panel B (Right): New 14-Court Club Monthly Revenue Ramp Under OUR Model (AdaptivePricing Court Hire +
        Ancillary Services = 1,553.4M VND/mo at steady state) vs. the Case's Untested 1,500M Static Reference.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20.2, 7.6), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(wspace=0.22)

    x_m = np.arange(1, 61)
    tick_idx = [0, 8, 13, 20, 23, 35, 47, 59]
    tick_lbls = [
        "Jan-27\n(M1)",
        "Sep-27\n(M9)",
        "Feb-28\n(M14 Open)",
        "Sep-28\n(M21 Fill)",
        "Dec-28\n(End Y2)",
        "Dec-29\n(End Y3)",
        "Dec-30\n(End Y4)",
        "Dec-31\n(End Y5)",
    ]

    # -------------------------------------------------------------------------
    # PANEL A: Physical Court Capacity (6 -> 20 Courts) & Effective Filled Courts
    # -------------------------------------------------------------------------
    ax_left.set_facecolor("#181825")
    phys_courts = res.active_courts_60m
    eff_courts_mean = 6.0 + 14.0 * res.venue2_fill_factor_mean_60m
    eff_courts_p05 = 6.0 + 14.0 * res.venue2_fill_factor_p05_60m
    eff_courts_p95 = 6.0 + 14.0 * res.venue2_fill_factor_p95_60m

    ax_left.step(
        x_m,
        phys_courts,
        where="post",
        color="#cdd6f4",
        linewidth=2.4,
        linestyle="--",
        label="Total Physical Courts Installed (6 Courts M1–M13; 20 Courts M14+)",
        zorder=4,
    )
    for path in res.sample_fill_paths_60m:
        ax_left.plot(x_m, 6.0 + 14.0 * path, color="#89dceb", alpha=0.14, linewidth=0.8, zorder=2)

    ax_left.fill_between(
        x_m,
        eff_courts_p05,
        eff_courts_p95,
        color="#89dceb",
        alpha=0.25,
        label="6-to-9 Month Fill-Up 90% Monte Carlo Band (P05–P95)",
        zorder=3,
    )
    ax_left.plot(
        x_m,
        eff_courts_mean,
        color="#94e2d5",
        linewidth=2.8,
        label="Effective Steady-State Equivalent Courts Active (Mean Ramp)",
        zorder=5,
    )

    ax_left.axvspan(10, 13.95, color="#f9e2af", alpha=0.14, label="Q4 2027 – Jan 2028 Construction & Curing Window (Months 10–13)", zorder=1)
    ax_left.axvspan(14, 21.5, color="#89b4fa", alpha=0.14, label="6-to-9 Month Fill-Up Window (Months 14–21: Feb-28 to Sep-28)", zorder=1)

    ax_left.set_xticks([t + 1 for t in tick_idx])
    ax_left.set_xticklabels(tick_lbls, fontsize=7.6, fontweight="bold")
    ax_left.set_ylim(0, 23.5)
    ax_left.set_ylabel("Number of Courts (Physical vs. Effective Filled)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_left.set_title(
        "Panel A: Physical Court Capacity Expansion & Effective Utilization Ramp",
        fontsize=10.6,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_left.grid(True, linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_left.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    # -------------------------------------------------------------------------
    # PANEL B: Venue 2 (14 Courts) Monthly Revenue Ramp Using OUR Model
    # -------------------------------------------------------------------------
    ax_right.set_facecolor("#181825")
    for path in res.sample_v2_rev_paths_60m:
        ax_right.plot(x_m, path, color="#a6e3a1", alpha=0.14, linewidth=0.8, zorder=2)

    ax_right.fill_between(
        x_m,
        res.venue2_total_rev_p05_60m,
        res.venue2_total_rev_p95_60m,
        color="#a6e3a1",
        alpha=0.24,
        label="Venue 2 Total Revenue 90% Band (6–9 Month Fill-Up Variance)",
        zorder=3,
    )
    ax_right.plot(
        x_m,
        res.venue2_total_rev_mean_60m,
        color="#a6e3a1",
        linewidth=2.8,
        label=f"Venue 2 Total Revenue (OUR Model: Steady-State = {res.venue2_steady_total_rev_mvnd:,.1f}M/mo)",
        zorder=5,
    )
    ax_right.plot(
        x_m,
        res.venue2_court_hire_mean_60m,
        color="#89b4fa",
        linewidth=2.2,
        linestyle="-.",
        label=f"Venue 2 Court Hire Revenue (14/6 x AdaptivePricing = {res.venue2_steady_court_hire_mvnd:,.1f}M/mo)",
        zorder=4,
    )
    ax_right.plot(
        x_m,
        res.venue2_ancillary_mean_60m,
        color="#cba6f7",
        linewidth=2.0,
        linestyle=":",
        label=f"Venue 2 Ancillary Revenue (Coaching + Cafe + Rentals = {res.venue2_steady_ancillary_mvnd:,.1f}M/mo)",
        zorder=4,
    )
    ax_right.axhline(
        cfg.meta.case_static_ref_new_club_rev_mvnd,
        color="#7f849c",
        linewidth=1.6,
        linestyle="--",
        label="Case Unverified Static Reference (1,500.0M/mo — Not Used)",
        zorder=3,
    )

    ax_right.set_xticks([t + 1 for t in tick_idx])
    ax_right.set_xticklabels(tick_lbls, fontsize=7.6, fontweight="bold")
    ax_right.set_ylim(-50, 1850)
    ax_right.set_ylabel("Venue 2 Monthly Revenue (M VND / Month)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_right.set_title(
        "Panel B: Second Club Monthly Revenue Ramp by Stream",
        fontsize=10.6,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_right.grid(True, linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_right.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    out_path = output_dir / "02_court_capacity_and_fill_up_ramp.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path


def plot_capex_allocation_and_unit_economics(
    cfg: VenueScalingConfig,
    res: VenueScalingResults,
    output_dir: Path,
) -> Path:
    """
    Render Chart 3 (03_capex_allocation_and_unit_economics.png):
      - Panel A (Left): Breakdown of the VND 7,000.0M (7.0 Billion VND) Fit-Out & Launch CapEx by Stage (V1–V5)
        plus the VND 1,000.0M Cash Reserve (= VND 8,000.0M Total Investor Capital).
      - Panel B (Right): Even Monthly Fit-Out CapEx Allocation (7,000.0M / 4 Build Months = 1,750.0M VND/Month
        across Oct 2027 – Jan 2028) & Ongoing Venue 2 Lease (330.0M VND/mo) + Operating Cost Schedule across Year 1 & Year 2.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20.4, 7.6), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(wspace=0.22)

    # -------------------------------------------------------------------------
    # PANEL A: 7,000M VND Fit-Out CapEx Breakdown by Construction Stage
    # -------------------------------------------------------------------------
    ax_left.set_facecolor("#181825")
    build_stages = [s for s in res.stages if s.stage_capex_mvnd > 0]
    s_labels = [s.short_name for s in build_stages] + ["Investor Cash Reserve (Held in Bank)"]
    s_vals = [s.stage_capex_mvnd for s in build_stages] + [cfg.meta.investor_reserve_mvnd]
    s_colors = [s.color for s in build_stages] + ["#7f849c"]

    y_pos = np.arange(len(s_labels))
    bar_h = 0.54

    for idx in range(len(s_labels)):
        ax_left.barh(
            y_pos[idx],
            s_vals[idx],
            height=bar_h,
            color=s_colors[idx],
            edgecolor="#11111b",
            linewidth=1.0,
            alpha=0.92 if idx < len(build_stages) else 0.78,
            hatch="///" if idx == len(build_stages) else None,
            zorder=3,
        )
        pct_7b = (s_vals[idx] / cfg.meta.total_fitout_capex_mvnd) * 100.0
        lbl_txt = (
            f"{s_vals[idx]:,.0f}M VND ({pct_7b:.1f}% of 7,000M Fit-Out)"
            if idx < len(build_stages)
            else f"{s_vals[idx]:,.0f}M VND (Mandatory Cash Reserve — Total = 8,000M)"
        )
        ax_left.text(
            s_vals[idx] + 35.0,
            y_pos[idx],
            lbl_txt,
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(s_labels, fontsize=8.8, fontweight="bold", color="#cdd6f4")
    ax_left.set_ylim(len(s_labels) - 0.25, -0.65)
    ax_left.set_xlim(0, 3450)
    ax_left.set_xlabel("Capital Allocation in Millions of VND (M VND)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_left.set_title(
        "Panel A: Expansion Capital Allocation by Construction Stage & Cash Reserve",
        fontsize=10.6,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_left.grid(axis="x", linestyle="--", alpha=0.8, color="#313244", zorder=1)

    # -------------------------------------------------------------------------
    # PANEL B: Even Monthly CapEx Split (1,750M/mo in M10–M13) & Venue 2 Lease/OpEx
    # -------------------------------------------------------------------------
    ax_right.set_facecolor("#181825")
    # Show Months 8..21 (Aug-27 to Sep-28) covering pre-expansion M8-M9, 4-month build M10-M13, and 8-month fill-up M14-M21
    sub_df = res.monthly_60m_df.iloc[7:21].copy()
    m_lbls = sub_df["month_label"].tolist()
    x_b = np.arange(len(m_lbls))

    capex_b = sub_df["fitout_capex_even_split_mvnd"].to_numpy()
    lease_b = sub_df["venue2_lease_mvnd"].to_numpy()
    ops_b = (
        sub_df["venue2_utilities_mean_mvnd"].to_numpy()
        + sub_df["venue2_maint_ins_mean_mvnd"].to_numpy()
        + sub_df["venue2_staff_wages_mean_mvnd"].to_numpy()
    )
    tot_b = capex_b + lease_b + ops_b

    w = 0.58
    ax_right.bar(
        x_b,
        capex_b,
        width=w,
        color="#f38ba8",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Even Monthly Fit-Out CapEx Split (7,000M / 4 Months = 1,750.0M/mo in Oct 2027 – Jan 2028)",
        zorder=3,
    )
    ax_right.bar(
        x_b,
        lease_b,
        bottom=capex_b,
        width=w,
        color="#89b4fa",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Venue 2 5-Year Fixed Lease (330.0M VND/mo from Oct-27)",
        zorder=3,
    )
    ax_right.bar(
        x_b,
        ops_b,
        bottom=capex_b + lease_b,
        width=w,
        color="#a6e3a1",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Venue 2 Utilities, Upkeep & Staff Payroll (Active from Feb-28 Open)",
        zorder=3,
    )

    for i in range(len(x_b)):
        if tot_b[i] > 0:
            ax_right.text(
                x_b[i],
                tot_b[i] + 35,
                f"{tot_b[i]:,.0f}M",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#cdd6f4",
            )

    ax_right.set_xticks(x_b)
    ax_right.set_xticklabels(m_lbls, fontsize=8.0, fontweight="bold", rotation=25)
    ax_right.set_ylim(0, 2500)
    ax_right.set_ylabel("Monthly Venue 2 Outflow (M VND / Month)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_right.set_title(
        "Panel B: Monthly Construction Capital Deployment & Facility Operating Outflows",
        fontsize=10.6,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_right.grid(axis="y", linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_right.legend(loc="upper right", fontsize=7.6, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    out_path = output_dir / "03_capex_allocation_and_unit_economics.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path
