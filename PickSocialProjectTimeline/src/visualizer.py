"""Production-grade Gantt chart, concurrency profile, and cost estimation & cash-flow allocation visualizer (Presentation-Clean, No Icons or Inline Citations)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .config import TimelineConfig
from .schedule_model import ScheduledTask, StageSummary, TimelineResults
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


def generate_all_timeline_charts(
    cfg: TimelineConfig,
    res: TimelineResults,
    output_dir: Path,
) -> list[Path]:
    """Generate presentation-clean Gantt chart, effort/concurrency profile, and cost/cash-flow allocation chart into outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p1 = plot_master_gantt_chart(cfg, res, output_dir)
    p2 = plot_stage_effort_and_concurrency(cfg, res, output_dir)
    p3 = plot_cost_estimation_and_cashflow_allocation(cfg, res, output_dir)
    return [p1, p2, p3]


def plot_master_gantt_chart(
    cfg: TimelineConfig,
    res: TimelineResults,
    output_dir: Path,
) -> Path:
    """
    Render the Master Production Gantt Chart (01_pick_social_platform_gantt_chart.png):
      - Clean presentation layout: no icons, no milestone diamond clutter, no citation keys on bars
      - Unified precision-aligned header table for WBS columns (including Base/Buffered Cost) + Months 1-6 + Weeks W1-W24
      - Highlighted 4-Month (Week 16 / Apr 25, 2027) Product Complete / Migration boundary
    """
    display_rows: list[tuple[str, StageSummary | ScheduledTask]] = []
    for st_sum in res.stage_summaries:
        display_rows.append(("stage", st_sum))
        for t in res.tasks:
            if t.stage_id == st_sum.stage_id:
                display_rows.append(("task", t))

    n_rows = len(display_rows)
    fig_h = 16.0
    fig_w = 28.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")

    # Coordinate system:
    #   [-26.0, 0.0] : Left WBS structured columns (with Base / Buffered Cost column)
    #   [0.0, 24.0]  : Weeks W1 to W24 (Jan 2027 to Jun 2027)
    x_left_min = -26.0
    x_right_max = 24.05
    hdr_top = n_rows + 1.65
    hdr_mid = n_rows + 0.65
    hdr_bot = float(n_rows)

    ax.set_xlim(x_left_min, x_right_max)
    ax.set_ylim(-1.65, hdr_top + 0.35)

    # Month bands on the Gantt side (Weeks 0..24 -> 6 Months: Jan 2027 .. Jun 2027)
    month_defs = [
        ("MONTH 1: JAN 2027", "Planning & Infra", 0.0, 4.0, "#181825", "#313244"),
        ("MONTH 2: FEB 2027", "Booking & ML Core", 4.0, 8.0, "#1e1e2e", "#313244"),
        ("MONTH 3: MAR 2027", "Payments & Social", 8.0, 12.0, "#181825", "#313244"),
        ("MONTH 4: APR 2027", "Integration & UAT", 12.0, 16.0, "#1e1e2e", "#313244"),
        ("MONTH 5: MAY 2027", "Legacy Migration", 16.0, 20.0, "#291e28", "#452738"),
        ("MONTH 6: JUN 2027+", "Continuous Ops", 20.0, 24.0, "#1e292e", "#253b3f"),
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
            fontsize=8.5,
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
            fontsize=7.5,
            fontweight="normal",
            color="#bac2de",
            zorder=5,
        )

    # Week sub-header cells (W1 .. W24)
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

    # Left Table Header Banner (Unified height from hdr_bot to hdr_top)
    col_defs = [
        ("ID", -26.0, -24.7, "center"),
        ("STAGE / ENGINEERING TASK (WBS)", -24.7, -13.4, "left"),
        ("CORE FEATURE / DELIVERABLE", -13.4, -6.1, "left"),
        ("WKS", -6.1, -4.9, "center"),
        ("WINDOW", -4.9, -3.0, "center"),
        ("COST: BASE / BUF", -3.0, 0.0, "center"),
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
            fontsize=8.2,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )
        ax.axvline(c_x1, ymin=0.06, ymax=0.95, color="#45475a", linewidth=0.7, zorder=3)

    # Render each stage & task row
    for row_idx, (rtype, item) in enumerate(display_rows):
        y_top = n_rows - row_idx
        y_bot = y_top - 1.0
        y_mid = y_bot + 0.5

        if rtype == "stage":
            st: StageSummary = item  # type: ignore[assignment]
            ax.add_patch(
                mpatches.Rectangle(
                    (x_left_min, y_bot),
                    24.0 - x_left_min,
                    1.0,
                    facecolor=st.light_color,
                    edgecolor="#45475a",
                    linewidth=0.9,
                    zorder=1,
                )
            )
            # Bold stage-separator top & bottom rules + left accent bar for clear stage hierarchy
            ax.axhline(y_top, color=st.color, linewidth=1.55, zorder=4)
            ax.axhline(y_bot, color=st.color, linewidth=1.05, zorder=4)
            ax.add_patch(
                mpatches.Rectangle(
                    (x_left_min, y_bot),
                    0.22,
                    1.0,
                    facecolor=st.color,
                    edgecolor="none",
                    zorder=5,
                )
            )
            ax.text(
                -25.35,
                y_mid,
                st.stage_id,
                ha="center",
                va="center",
                fontsize=8.1,
                fontweight="bold",
                color=st.color,
                zorder=5,
            )
            ax.text(
                -24.5,
                y_mid,
                f"{st.name.upper()}",
                ha="left",
                va="center",
                fontsize=7.3,
                fontweight="bold",
                color="#cdd6f4",
                zorder=5,
            )
            ax.text(
                -13.2,
                y_mid,
                f"Effort: {st.sum_person_weeks:.1f} person-wks ({st.task_count} Tasks)",
                ha="left",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color=st.color,
                zorder=5,
            )
            ax.text(
                -5.50,
                y_mid,
                f"{st.span_weeks:.1f}w",
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="bold",
                color=st.color,
                zorder=5,
            )
            ax.text(
                -3.95,
                y_mid,
                f"{st.start_date.strftime('%b %d')}-{st.end_date.strftime('%b %d')}",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color="#bac2de",
                zorder=5,
            )
            ax.text(
                -1.50,
                y_mid,
                f"{st.base_total_cost_mvnd:.1f}M / {st.buffered_total_cost_mvnd:.1f}M",
                ha="center",
                va="center",
                fontsize=7.3,
                fontweight="bold",
                color="#cdd6f4",
                zorder=5,
            )

            # Stage summary span bar on Gantt canvas
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (st.start_week, y_mid - 0.14),
                    st.span_weeks,
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
            t: ScheduledTask = item  # type: ignore[assignment]
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

            id_color = "#f38ba8" if t.is_critical_path else "#a6adc8"
            ax.text(
                -25.35,
                y_mid,
                t.id,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color=id_color,
                zorder=5,
            )
            ax.plot([-24.42, -24.42], [y_bot + 0.18, y_top - 0.18], color=t.stage_color, linewidth=1.1, alpha=0.65, zorder=4)
            ax.text(
                -24.20,
                y_mid,
                t.name,
                ha="left",
                va="center",
                fontsize=7.1,
                fontweight="bold" if t.is_critical_path or t.is_simultaneous_qa else "normal",
                color="#cdd6f4",
                zorder=5,
            )
            ax.text(
                -13.2,
                y_mid,
                t.core_feature,
                ha="left",
                va="center",
                fontsize=7.0,
                color="#bac2de",
                zorder=5,
            )
            ax.text(
                -5.50,
                y_mid,
                f"{t.duration_weeks:.1f}w",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#cdd6f4",
                zorder=5,
            )
            ax.text(
                -3.95,
                y_mid,
                f"W{t.start_week + 1:g}-W{t.end_week:g}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="#a6adc8",
                zorder=5,
            )
            ax.text(
                -1.50,
                y_mid,
                f"{t.base_total_cost_mvnd:.1f}M / {t.buffered_total_cost_mvnd:.1f}M",
                ha="center",
                va="center",
                fontsize=7.1,
                fontweight="bold",
                color="#bac2de",
                zorder=5,
            )

            # Task bar on Gantt canvas
            bar_h = 0.56
            bar_y = y_mid - bar_h / 2.0
            edge_col = "#11111b" if t.is_critical_path else t.stage_color
            edge_w = 1.35 if t.is_critical_path else 0.9
            hatch_pat = "///" if t.is_simultaneous_qa else None
            bar_fill = "#a6e3a1" if (t.id == "T3.6") else t.stage_color
            bar_alpha = 0.85 if t.is_simultaneous_qa else 0.94

            rect = mpatches.FancyBboxPatch(
                (t.start_week, bar_y),
                t.duration_weeks,
                bar_h,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor=bar_fill,
                edgecolor=edge_col,
                linewidth=edge_w,
                alpha=bar_alpha,
                hatch=hatch_pat,
                zorder=4,
            )
            ax.add_patch(rect)

            # Presentation-clean label (Task ID & duration only - no citation keys)
            bar_label = f"{t.id} ({t.duration_weeks:.1f}w)"
            if t.duration_weeks >= 2.5 or t.end_week >= 23.5:
                lbl_bg = "#1e1e2e" if t.is_simultaneous_qa else "none"
                lbl_col = "#cdd6f4" if t.is_simultaneous_qa else "#11111b"
                ax.text(
                    t.start_week + t.duration_weeks / 2.0,
                    y_mid,
                    bar_label,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color=lbl_col,
                    bbox=dict(boxstyle="round,pad=0.10", facecolor=lbl_bg, edgecolor="none", alpha=0.88) if t.is_simultaneous_qa else None,
                    zorder=6,
                )
            else:
                ax.text(
                    t.end_week + 0.14,
                    y_mid,
                    bar_label,
                    ha="left",
                    va="center",
                    fontsize=7.1,
                    fontweight="bold",
                    color="#cdd6f4",
                    zorder=6,
                )

    # Outer bottom border across table + Gantt
    ax.plot([x_left_min, 24.0], [0.0, 0.0], color="#585b70", linewidth=1.1, zorder=5)

    # Total project budget summary pill on bottom left under table
    ax.text(
        -11.1,
        -0.85,
        f"4-MONTH CORE BUILD (S1-S4): {res.core_build_base_cost_mvnd:.2f}M Base / {res.core_build_buffered_cost_mvnd:.2f}M Buffered   |   "
        f"FULL 6-MONTH LIFECYCLE (S1-S6): {res.total_base_cost_mvnd:.2f}M Base + {res.total_buffer_mvnd:.2f}M Buffer (+{res.weighted_buffer_pct:.1f}%) = {res.total_buffered_cost_mvnd:.2f}M VND",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#cdd6f4",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#313244", edgecolor="#585b70", linewidth=1.1),
        zorder=9,
    )

    # Highlight Week 16 (End of Month 4 = Target Product Completion & Entry into Migration) - NO ICONS
    ax.axvline(16.0, ymin=0.04, ymax=0.92, color="#f38ba8", linewidth=2.4, linestyle="--", zorder=7)
    ax.text(
        16.0,
        -0.85,
        "END OF MONTH 4 (W16: APR 25, 2027) - CORE PRODUCT COMPLETE / ENTER MIGRATION PHASE",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#11111b",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f38ba8", edgecolor="#eba0ac", linewidth=1.2),
        zorder=9,
    )

    # Legend at bottom
    legend_handles = [
        mpatches.Patch(facecolor="#89b4fa", edgecolor="#11111b", label="Stage 1: Planning, Arch & Specs (112.0M Base / 130.5M Buf)"),
        mpatches.Patch(facecolor="#cba6f7", edgecolor="#11111b", label="Stage 2: Infra, DevOps & Auth (82.8M Base / 95.2M Buf)"),
        mpatches.Patch(facecolor="#a6e3a1", edgecolor="#11111b", label="Stage 3: Core Dev & Simultaneous QA (362.8M Base / 440.0M Buf)"),
        mpatches.Patch(facecolor="#94e2d5", edgecolor="#11111b", hatch="///", label="Simultaneous QA / Continuous Testing Track"),
        mpatches.Patch(facecolor="#fab387", edgecolor="#11111b", label="Stage 4: Final E2E, Load Test & UAT (118.6M Base / 144.1M Buf)"),
        mpatches.Patch(facecolor="#f38ba8", edgecolor="#11111b", label="Stage 5: Legacy Migration & Cutover (97.0M Base / 116.3M Buf)"),
        mpatches.Patch(facecolor="#74c7ec", edgecolor="#11111b", label="Stage 6: Hypercare, Feedback & Continuous Ops (138.0M Base / 162.9M Buf)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.045),
        ncol=4,
        fontsize=7.7,
        frameon=True,
        facecolor="#181825",
        edgecolor="#45475a",
    )

    ax.set_title(
        "PICK! Social & Adaptive Booking Platform - Production Engineering Schedule & Work Breakdown Structure\n"
        "Phased Engineering Delivery Across Core Platform Build, Legacy Migration & Continuous Operations",
        fontsize=12.2,
        fontweight="bold",
        color="#cdd6f4",
        pad=14,
    )
    ax.axis("off")

    out_path = output_dir / "01_pick_social_platform_gantt_chart.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def plot_stage_effort_and_concurrency(
    cfg: TimelineConfig,
    res: TimelineResults,
    output_dir: Path,
) -> Path:
    """
    Render Chart 2 (02_stage_effort_and_concurrency_profile.png):
      - Presentation-clean labels (no inline academic citations)
      - Panel A: Stage Effort Distribution vs. Industry Standard Benchmark (4-Month Core Build)
      - Panel B: Weekly Active Track Concurrency (Simultaneous Dev + Continuous QA + Migration/Ops)
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18.5, 6.8), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(wspace=0.22)

    core_stages = [s for s in res.stage_summaries if s.stage_id in ("S1", "S2", "S3", "S4")]
    labels = [s.short_name for s in core_stages]
    model_shares = [16.5, 12.5, 54.0, 17.0]
    benchmark_shares = [18.0, 12.0, 52.0, 18.0]

    y_pos = np.arange(len(labels))
    bar_h = 0.34

    ax_left.set_facecolor("#181825")
    ax_left.barh(
        y_pos + bar_h / 2,
        benchmark_shares,
        height=bar_h,
        color="#585b70",
        edgecolor="#7f849c",
        alpha=0.75,
        label="Industry Standard Benchmark",
        zorder=2,
    )
    for idx, st in enumerate(core_stages):
        ax_left.barh(
            y_pos[idx] - bar_h / 2,
            model_shares[idx],
            height=bar_h,
            color=st.color,
            edgecolor="#11111b",
            linewidth=1.0,
            label="PICK! Planned Effort Allocation" if idx == 0 else None,
            zorder=3,
        )
        ax_left.text(
            model_shares[idx] + 0.8,
            y_pos[idx] - bar_h / 2,
            f"PICK!: {model_shares[idx]:.1f}% ({st.span_weeks:.1f}w span)",
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
            color="#cdd6f4",
        )
        ax_left.text(
            benchmark_shares[idx] + 0.8,
            y_pos[idx] + bar_h / 2,
            f"Benchmark: {benchmark_shares[idx]:.1f}%",
            va="center",
            ha="left",
            fontsize=7.8,
            color="#a6adc8",
        )

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(labels, fontsize=9.0, fontweight="bold", color="#cdd6f4")
    ax_left.invert_yaxis()
    ax_left.set_xlim(0, 72)
    ax_left.set_xlabel("Share of Core Engineering Effort (%)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_left.set_title(
        "Panel A: PICK! Social Dev Effort Allocation vs. Industry Benchmark",
        fontsize=10.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_left.grid(axis="x", linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_left.legend(loc="lower right", fontsize=8.2, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    ax_right.set_facecolor("#181825")
    df_c = res.weekly_concurrency_df
    x_w = df_c["week_num"].to_numpy()
    dev_t = df_c["active_arch_infra_dev_tracks"].to_numpy()
    qa_t = df_c["active_qa_testing_tracks"].to_numpy()
    mig_t = df_c["active_migration_ops_tracks"].to_numpy()

    ax_right.bar(x_w, dev_t, color="#89b4fa", edgecolor="#11111b", width=0.72, label="Architecture, Infra & Feature Dev Tracks", zorder=3)
    ax_right.bar(x_w, qa_t, bottom=dev_t, color="#a6e3a1", edgecolor="#11111b", hatch="///", width=0.72, label="Simultaneous QA, Load & UAT Testing Tracks", zorder=3)
    ax_right.bar(x_w, mig_t, bottom=dev_t + qa_t, color="#f38ba8", edgecolor="#11111b", width=0.72, label="Legacy Migration, Shadow Run & Continuous Ops", zorder=3)

    ax_right.axvline(16.5, color="#f38ba8", linewidth=2.0, linestyle="--", zorder=5)
    ax_right.text(
        16.3,
        4.65,
        "End Apr-27 (W16):\nPICK! Social Launch & Migration",
        ha="right",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color="#f38ba8",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e1e2e", edgecolor="#f38ba8"),
    )

    ax_right.set_xticks(x_w[::2])
    ax_right.set_xticklabels([f"W{w}" for w in x_w[::2]], fontsize=8.2, fontweight="bold")
    ax_right.set_ylim(0, 5.2)
    ax_right.set_ylabel("Simultaneous Active Workstreams", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_right.set_xlabel("Project Timeline Weeks", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_right.set_title(
        "Panel B: Simultaneous Development, Continuous QA & Migration Concurrency",
        fontsize=10.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_right.grid(axis="y", linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_right.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    out_path = output_dir / "02_stage_effort_and_concurrency_profile.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path


def plot_cost_estimation_and_cashflow_allocation(
    cfg: TimelineConfig,
    res: TimelineResults,
    output_dir: Path,
) -> Path:
    """
    Render Chart 3 (03_cost_estimation_and_cashflow_allocation.png):
      - Presentation-clean layout: no icons, no inline citation keys
      - Panel A (Left): Stage-by-Stage Cost Estimation Breakdown (Base Labor & Infra vs. Risk Contingency Buffer, in M VND)
      - Panel B (Right): Our Leveled Cost Allocation Across Project Duration (Jan-27 to Jun-27 + Ongoing M7+ Platform OpEx):
          * Bi-Weekly Sprint Active Base Drawdown (M1-M6)
          * Stage 3 Milestone Acceptance Holdback Release (M4 UAT, M5 Cutover, M6 Hypercare)
          * Conditional Contingency Reserve Allocation (29.58M VND/mo across M1-M6)
          * Recurring Monthly Platform & Community Operating Expenses (16.50M VND/mo in M5-M6, 26.50M VND/mo in M7+ Steady-State)
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20.4, 7.8), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(wspace=0.22)

    # --------------------------------------------------------------------------
    # PANEL A: Stage-by-Stage Cost Estimation (Base Cost + Contingency Buffer)
    # --------------------------------------------------------------------------
    ax_left.set_facecolor("#181825")
    stages = list(res.stage_summaries)
    s_labels = [s.short_name for s in stages]
    base_costs = np.array([s.base_total_cost_mvnd for s in stages])
    buf_costs = np.array([s.contingency_buffer_mvnd for s in stages])
    tot_costs = np.array([s.buffered_total_cost_mvnd for s in stages])
    s_colors = [s.color for s in stages]

    y_pos = np.arange(len(stages))
    bar_h = 0.54

    for idx, st in enumerate(stages):
        ax_left.barh(
            y_pos[idx],
            base_costs[idx],
            height=bar_h,
            color=s_colors[idx],
            edgecolor="#11111b",
            linewidth=1.0,
            alpha=0.92,
            label="Base Cost (Loaded Engineering + Direct Cloud/Infra)" if idx == 0 else None,
            zorder=3,
        )
        ax_left.barh(
            y_pos[idx],
            buf_costs[idx],
            left=base_costs[idx],
            height=bar_h,
            color="#f9e2af",
            edgecolor="#11111b",
            linewidth=1.0,
            hatch="///",
            alpha=0.88,
            label="Task Risk Contingency Buffer (+15% to +25%)" if idx == 0 else None,
            zorder=3,
        )
        ax_left.text(
            tot_costs[idx] + 6.5,
            y_pos[idx],
            f"{base_costs[idx]:.1f}M Base + {buf_costs[idx]:.1f}M Buf (+{st.effective_buffer_pct:.1f}%) = {tot_costs[idx]:.1f}M",
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(s_labels, fontsize=8.8, fontweight="bold", color="#cdd6f4")
    ax_left.set_ylim(6.75, -0.65)
    ax_left.set_xlim(0, 745)
    ax_left.set_xlabel("Estimated Cost (M VND)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_left.set_title(
        "Panel A: Stage Cost Estimation Breakdown: Base Cost vs. Risk Contingency",
        fontsize=10.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_left.grid(axis="x", linestyle="--", alpha=0.8, color="#313244", zorder=1)

    # Summary box inside Panel A (bottom center, 100% inside axes frame)
    summary_text = (
        f"4-Month Core Build (Stages 1-4, W1-W16):  Base {res.core_build_base_cost_mvnd:.2f}M  |  Buffer {res.core_build_buffer_mvnd:.2f}M  |  Buffered {res.core_build_buffered_cost_mvnd:.2f}M VND\n"
        f"Full 6-Month Lifecycle (S1-S6, W1-W24):  Labor {res.total_base_labor_cost_mvnd:.2f}M + Cloud/Infra {res.total_direct_infra_cost_mvnd:.2f}M = "
        f"Base {res.total_base_cost_mvnd:.2f}M  |  Buf (+{res.weighted_buffer_pct:.1f}%) {res.total_buffer_mvnd:.2f}M  |  Buffered {res.total_buffered_cost_mvnd:,.2f}M"
    )
    ax_left.text(
        0.50,
        0.025,
        summary_text,
        transform=ax_left.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.1,
        fontweight="bold",
        color="#cdd6f4",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#1e1e2e", edgecolor="#585b70", linewidth=1.0),
        zorder=6,
    )
    ax_left.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    # --------------------------------------------------------------------------
    # PANEL B: Our Leveled Cost Allocation Across Project Duration + Recurring Platform OpEx
    # --------------------------------------------------------------------------
    ax_right.set_facecolor("#181825")
    df_m = res.monthly_cost_allocation_df

    # 6 project months (Jan-27..Jun-27) + 7th bar showing ongoing Jul-27+ steady-state monthly platform OpEx
    m_labels = [f"{r['month_label']}\n({r['weeks_window']})" for _, r in df_m.iterrows()] + ["Jul-27+\n(Steady Ops)"]
    x_m = np.arange(len(m_labels))

    sprint_arr = np.append(df_m["leveled_sprint_active_base_mvnd"].to_numpy(), 0.0)
    hb_arr = np.append(df_m["leveled_milestone_holdback_release_mvnd"].to_numpy(), 0.0)
    committed_dev_arr = sprint_arr + hb_arr
    reserve_arr = np.append(df_m["leveled_contingency_reserve_mvnd"].to_numpy(), 0.0)
    rec_opex_arr = np.append(df_m["recurring_platform_opex_mvnd"].to_numpy(), res.steady_state_monthly_platform_opex_mvnd)

    total_base_with_opex = committed_dev_arr + rec_opex_arr
    total_buffered_with_opex = committed_dev_arr + reserve_arr + rec_opex_arr

    w = 0.56
    # Stack 1: Bi-Weekly Sprint Active Base Drawdown
    ax_right.bar(
        x_m,
        sprint_arr,
        width=w,
        color="#89b4fa",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Sprint Active Base Drawdown (Bi-Weekly Earned Value)",
        zorder=3,
    )
    # Stack 2: Stage 3 Milestone Acceptance Holdback Release (Apr-27 UAT, May-27 Launch, Jun-27 Hypercare)
    ax_right.bar(
        x_m,
        hb_arr,
        bottom=sprint_arr,
        width=w,
        color="#74c7ec",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Stage 3 Milestone Holdback Release (Apr / May / Jun Gates)",
        zorder=3,
    )
    # Stack 3: Conditional Contingency Reserve (29.58M/mo across Jan-Jun 2027)
    ax_right.bar(
        x_m,
        reserve_arr,
        bottom=committed_dev_arr,
        width=w,
        color="#f9e2af",
        edgecolor="#11111b",
        linewidth=1.0,
        hatch="///",
        alpha=0.90,
        label="Conditional Task Contingency Reserve (29.58M/mo)",
        zorder=3,
    )
    # Stack 4: Recurring Monthly PICK! Social & Community OpEx (May-Jun 27: 16.50M/mo; Jul-27+: 26.50M/mo)
    ax_right.bar(
        x_m,
        rec_opex_arr,
        bottom=committed_dev_arr + reserve_arr,
        width=w,
        color="#a6e3a1",
        edgecolor="#11111b",
        linewidth=1.0,
        label="PICK! Social Cloud & Community OpEx (16.5M May-Jun; 26.5M Jul-27+)",
        zorder=4,
    )

    # Annotate each month bar cleanly
    for idx in range(len(x_m)):
        if idx < 6:
            # Inside base bar label
            ax_right.text(
                x_m[idx],
                sprint_arr[idx] / 2.0,
                f"Sprint\n{sprint_arr[idx]:.1f}M",
                ha="center",
                va="center",
                fontsize=7.1,
                fontweight="bold",
                color="#11111b",
                zorder=6,
            )
            if hb_arr[idx] > 0:
                ax_right.text(
                    x_m[idx],
                    sprint_arr[idx] + hb_arr[idx] / 2.0,
                    f"+{hb_arr[idx]:.1f}M HB",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    fontweight="bold",
                    color="#11111b",
                    zorder=6,
                )
            top_val = total_buffered_with_opex[idx]
            base_val = total_base_with_opex[idx]
            opex_note = f"\n(incl. {rec_opex_arr[idx]:.1f}M OpEx)" if rec_opex_arr[idx] > 0 else ""
            ax_right.text(
                x_m[idx],
                top_val + 4.0,
                f"Base: {base_val:.1f}M\nMax: {top_val:.1f}M{opex_note}",
                ha="center",
                va="bottom",
                fontsize=7.1,
                fontweight="bold",
                color="#cdd6f4",
                zorder=6,
            )
        else:
            # Jul-27+ steady-state recurring platform OpEx bar
            ax_right.text(
                x_m[idx],
                rec_opex_arr[idx] + 4.5,
                f"PICK! Social OpEx\n{rec_opex_arr[idx]:.2f}M / mo\n(Cloud 8.5M + Uni 8.0M\n+ Maint 10.0M)",
                ha="center",
                va="bottom",
                fontsize=7.1,
                fontweight="bold",
                color="#a6e3a1",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e1e2e", edgecolor="#a6e3a1", linewidth=0.9),
                zorder=6,
            )

    ax_right.axvline(5.5, color="#585b70", linewidth=1.3, linestyle="--", zorder=2)

    # Summary callout inside Panel B
    alloc_summary = (
        f"6-Month PICK! Social Dev (Jan-Jun 27):  Committed Base 139.9M-166.0M/mo (Avg 151.9M/mo, SD 8.4M)  |  +29.58M/mo Buffer Reserve\n"
        f"Recurring PICK! Social OpEx:  {res.launch_monthly_platform_opex_mvnd:.2f}M/mo at Launch (May-Jun 27)  |  {res.steady_state_monthly_platform_opex_mvnd:.2f}M/mo Steady-State (Jul-27+)"
    )
    ax_right.text(
        0.50,
        0.025,
        alloc_summary,
        transform=ax_right.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
        color="#cdd6f4",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#1e1e2e", edgecolor="#585b70", linewidth=1.0),
        zorder=6,
    )

    ax_right.set_xticks(x_m)
    ax_right.set_xticklabels(m_labels, fontsize=8.1, fontweight="bold", color="#cdd6f4")
    ax_right.set_ylim(-28, 265)
    ax_right.set_ylabel("Monthly Allocated Cost Outflow (M VND)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_right.set_xlabel("Build & Migration Schedule vs. Steady-State Operations", fontsize=9.3, fontweight="bold", color="#bac2de")
    ax_right.set_title(
        "Panel B: Leveled Monthly Cost Allocation & Recurring Platform Operating Expenses",
        fontsize=10.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_right.grid(axis="y", linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_right.legend(loc="upper right", fontsize=7.6, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    fig.suptitle(
        "PICK! Social & Adaptive Booking Platform - Cost Estimation, Risk Contingency & Leveled Cash Allocation",
        fontsize=12.5,
        fontweight="bold",
        color="#cdd6f4",
        y=0.99,
    )

    out_path = output_dir / "03_cost_estimation_and_cashflow_allocation.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path
