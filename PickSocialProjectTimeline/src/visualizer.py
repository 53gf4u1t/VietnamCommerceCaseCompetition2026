"""Production-grade Gantt chart, concurrency profile, and cost estimation & cash-flow allocation visualizer (Presentation-Clean, No Icons or Inline Citations)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .config import TimelineConfig
from .schedule_model import ScheduledTask, StageSummary, TimelineResults


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
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Coordinate system:
    #   [-22.2, 0.0] : Left WBS structured columns (with Base / Buffered Cost column)
    #   [0.0, 24.0]  : Weeks W1 to W24 (Jan 2027 to Jun 2027)
    x_left_min = -22.2
    x_right_max = 24.05
    hdr_top = n_rows + 1.65
    hdr_mid = n_rows + 0.65
    hdr_bot = float(n_rows)

    ax.set_xlim(x_left_min, x_right_max)
    ax.set_ylim(-1.65, hdr_top + 0.35)

    # Month bands on the Gantt side (Weeks 0..24 -> 6 Months: Jan 2027 .. Jun 2027)
    month_defs = [
        ("MONTH 1: JAN 2027", "Planning & Infra", 0.0, 4.0, "#F8FAFC", "#0F172A"),
        ("MONTH 2: FEB 2027", "Booking & ML Core", 4.0, 8.0, "#F1F5F9", "#0F172A"),
        ("MONTH 3: MAR 2027", "Payments & Social", 8.0, 12.0, "#F8FAFC", "#0F172A"),
        ("MONTH 4: APR 2027", "Integration & UAT", 12.0, 16.0, "#F1F5F9", "#0F172A"),
        ("MONTH 5: MAY 2027", "Legacy Migration", 16.0, 20.0, "#FEF2F2", "#991B1B"),
        ("MONTH 6: JUN 2027+", "Continuous Ops", 20.0, 24.0, "#ECFEFF", "#0E7490"),
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
                edgecolor="#475569",
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
            color="#FFFFFF",
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
            color="#E2E8F0",
            zorder=5,
        )

    # Week sub-header cells (W1 .. W24)
    for w in range(24):
        wk_bg = "#E2E8F0" if (w // 4) % 2 == 0 else "#CBD5E1"
        ax.add_patch(
            mpatches.Rectangle(
                (float(w), hdr_bot),
                1.0,
                hdr_mid - hdr_bot,
                facecolor=wk_bg,
                edgecolor="#94A3B8",
                linewidth=0.7,
                zorder=4,
            )
        )
        ax.axvline(w, ymin=0.06, ymax=0.92, color="#CBD5E1", linewidth=0.55, linestyle=":", zorder=1)
        ax.text(
            w + 0.5,
            (hdr_bot + hdr_mid) / 2.0,
            f"W{w + 1}",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#1E293B",
            zorder=5,
        )

    for m_boundary in (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0):
        ax.axvline(m_boundary, ymin=0.06, ymax=0.92, color="#94A3B8", linewidth=1.0, zorder=2)

    # Left Table Header Banner (Unified height from hdr_bot to hdr_top)
    col_defs = [
        ("ID", -22.2, -21.0, "center"),
        ("STAGE / ENGINEERING TASK (WBS)", -21.0, -11.5, "left"),
        ("CORE FEATURE / DELIVERABLE", -11.5, -6.1, "left"),
        ("WKS", -6.1, -4.9, "center"),
        ("WINDOW", -4.9, -3.0, "center"),
        ("COST: BASE / BUF", -3.0, 0.0, "center"),
    ]
    ax.add_patch(
        mpatches.Rectangle(
            (x_left_min, hdr_bot),
            -x_left_min,
            hdr_top - hdr_bot,
            facecolor="#0F172A",
            edgecolor="#475569",
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
            color="#FFFFFF",
            zorder=5,
        )
        ax.axvline(c_x1, ymin=0.06, ymax=0.95, color="#CBD5E1", linewidth=0.7, zorder=3)

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
                    edgecolor="#CBD5E1",
                    linewidth=0.8,
                    zorder=1,
                )
            )
            ax.text(
                -21.6,
                y_mid,
                st.stage_id,
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color=st.color,
                zorder=5,
            )
            ax.text(
                -20.8,
                y_mid,
                st.name.upper(),
                ha="left",
                va="center",
                fontsize=7.8,
                fontweight="bold",
                color="#0F172A",
                zorder=5,
            )
            ax.text(
                -11.32,
                y_mid,
                f"Effort: {st.sum_person_weeks:.1f} person-wks ({st.task_count} Tasks)",
                ha="left",
                va="center",
                fontsize=7.4,
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
                f"{st.start_date.strftime('%b %d')}–{st.end_date.strftime('%b %d')}",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color="#1E293B",
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
                color="#0F172A",
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
                    edgecolor="#0F172A",
                    linewidth=1.0,
                    alpha=0.90,
                    zorder=4,
                )
            )
        else:
            t: ScheduledTask = item  # type: ignore[assignment]
            row_bg = "#FFFFFF" if row_idx % 2 == 0 else "#F8FAFC"
            ax.add_patch(
                mpatches.Rectangle(
                    (x_left_min, y_bot),
                    -x_left_min,
                    1.0,
                    facecolor=row_bg,
                    edgecolor="#E2E8F0",
                    linewidth=0.6,
                    zorder=1,
                )
            )
            ax.axhline(y_bot, color="#E2E8F0", linewidth=0.55, zorder=1)

            id_color = "#B91C1C" if t.is_critical_path else "#334155"
            ax.text(
                -21.6,
                y_mid,
                t.id,
                ha="center",
                va="center",
                fontsize=7.7,
                fontweight="bold",
                color=id_color,
                zorder=5,
            )
            ax.text(
                -20.8,
                y_mid,
                t.name,
                ha="left",
                va="center",
                fontsize=7.4,
                fontweight="bold" if t.is_critical_path or t.is_simultaneous_qa else "normal",
                color="#0F172A",
                zorder=5,
            )
            ax.text(
                -11.32,
                y_mid,
                t.core_feature,
                ha="left",
                va="center",
                fontsize=7.2,
                color="#334155",
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
                color="#0F172A",
                zorder=5,
            )
            ax.text(
                -3.95,
                y_mid,
                f"W{t.start_week + 1:g}–W{t.end_week:g}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="#475569",
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
                color="#1E293B",
                zorder=5,
            )

            # Task bar on Gantt canvas
            bar_h = 0.56
            bar_y = y_mid - bar_h / 2.0
            edge_col = "#0F172A" if t.is_critical_path else t.stage_color
            edge_w = 1.35 if t.is_critical_path else 0.9
            hatch_pat = "///" if t.is_simultaneous_qa else None
            bar_fill = "#10B981" if (t.id == "T3.6") else t.stage_color
            bar_alpha = 0.78 if t.is_simultaneous_qa else 0.92

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

            # Presentation-clean label (Task ID & duration only — no citation keys)
            bar_label = f"{t.id} ({t.duration_weeks:.1f}w)"
            if t.duration_weeks >= 2.5 or t.end_week >= 23.5:
                lbl_bg = "#065F46" if t.is_simultaneous_qa else "none"
                ax.text(
                    t.start_week + t.duration_weeks / 2.0,
                    y_mid,
                    bar_label,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#FFFFFF",
                    bbox=dict(boxstyle="round,pad=0.10", facecolor=lbl_bg, edgecolor="none", alpha=0.85) if t.is_simultaneous_qa else None,
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
                    color="#1E293B",
                    zorder=6,
                )

    # Outer bottom border across table + Gantt
    ax.plot([x_left_min, 24.0], [0.0, 0.0], color="#475569", linewidth=1.1, zorder=5)

    # Total project budget summary pill on bottom left under table
    ax.text(
        -11.1,
        -0.85,
        f"4-MONTH CORE BUILD (S1–S4): {res.core_build_base_cost_mvnd:.2f}M Base / {res.core_build_buffered_cost_mvnd:.2f}M Buffered   |   "
        f"FULL 6-MONTH LIFECYCLE (S1–S6): {res.total_base_cost_mvnd:.2f}M Base + {res.total_buffer_mvnd:.2f}M Buffer (+{res.weighted_buffer_pct:.1f}%) = {res.total_buffered_cost_mvnd:.2f}M VND",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.1),
        zorder=9,
    )

    # Highlight Week 16 (End of Month 4 = Target Product Completion & Entry into Migration) — NO ICONS
    ax.axvline(16.0, ymin=0.04, ymax=0.92, color="#DC2626", linewidth=2.4, linestyle="--", zorder=7)
    ax.text(
        16.0,
        -0.85,
        "END OF MONTH 4 (W16: APR 25, 2027) — CORE PRODUCT COMPLETE / ENTER MIGRATION PHASE",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#DC2626", edgecolor="#7F1D1D", linewidth=1.2),
        zorder=9,
    )

    # Legend at bottom
    legend_handles = [
        mpatches.Patch(facecolor="#2563EB", edgecolor="#0F172A", label="Stage 1: Planning, Arch & Specs (112.0M Base / 130.5M Buf)"),
        mpatches.Patch(facecolor="#7C3AED", edgecolor="#0F172A", label="Stage 2: Infra, DevOps & Auth (82.8M Base / 95.2M Buf)"),
        mpatches.Patch(facecolor="#059669", edgecolor="#0F172A", label="Stage 3: Core Dev & Simultaneous QA (362.8M Base / 440.0M Buf)"),
        mpatches.Patch(facecolor="#10B981", edgecolor="#0F172A", hatch="///", label="Simultaneous QA / Continuous Testing Track"),
        mpatches.Patch(facecolor="#D97706", edgecolor="#0F172A", label="Stage 4: Final E2E, Load Test & UAT (118.6M Base / 144.1M Buf)"),
        mpatches.Patch(facecolor="#DC2626", edgecolor="#0F172A", label="Stage 5: Legacy Migration & Cutover (97.0M Base / 116.3M Buf)"),
        mpatches.Patch(facecolor="#0891B2", edgecolor="#0F172A", label="Stage 6: Hypercare, Feedback & Continuous Ops (138.0M Base / 162.9M Buf)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.045),
        ncol=4,
        fontsize=7.7,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
    )

    ax.set_title(
        "PICK! Social & Adaptive Booking Platform — Production Engineering Schedule & Work Breakdown Structure\n"
        "Phased Engineering Delivery Across Core Platform Build, Legacy Migration & Continuous Operations",
        fontsize=12.2,
        fontweight="bold",
        color="#0F172A",
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
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(wspace=0.22)

    core_stages = [s for s in res.stage_summaries if s.stage_id in ("S1", "S2", "S3", "S4")]
    labels = [s.short_name for s in core_stages]
    model_shares = [16.5, 12.5, 54.0, 17.0]
    benchmark_shares = [18.0, 12.0, 52.0, 18.0]

    y_pos = np.arange(len(labels))
    bar_h = 0.34

    ax_left.set_facecolor("#F8FAFC")
    ax_left.barh(
        y_pos + bar_h / 2,
        benchmark_shares,
        height=bar_h,
        color="#CBD5E1",
        edgecolor="#64748B",
        alpha=0.65,
        label="Industry Standard Benchmark",
        zorder=2,
    )
    for idx, st in enumerate(core_stages):
        ax_left.barh(
            y_pos[idx] - bar_h / 2,
            model_shares[idx],
            height=bar_h,
            color=st.color,
            edgecolor="#0F172A",
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
            color="#0F172A",
        )
        ax_left.text(
            benchmark_shares[idx] + 0.8,
            y_pos[idx] + bar_h / 2,
            f"Benchmark: {benchmark_shares[idx]:.1f}%",
            va="center",
            ha="left",
            fontsize=7.8,
            color="#475569",
        )

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(labels, fontsize=9.0, fontweight="bold", color="#0F172A")
    ax_left.invert_yaxis()
    ax_left.set_xlim(0, 72)
    ax_left.set_xlabel("Share of Core Engineering Effort (%)", fontsize=9.5, fontweight="bold")
    ax_left.set_title(
        "Panel A: Core Build Effort Allocation vs. Industry Benchmark",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_left.grid(axis="x", linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    ax_left.legend(loc="lower right", fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    ax_right.set_facecolor("#F8FAFC")
    df_c = res.weekly_concurrency_df
    x_w = df_c["week_num"].to_numpy()
    dev_t = df_c["active_arch_infra_dev_tracks"].to_numpy()
    qa_t = df_c["active_qa_testing_tracks"].to_numpy()
    mig_t = df_c["active_migration_ops_tracks"].to_numpy()

    ax_right.bar(x_w, dev_t, color="#2563EB", edgecolor="#1E3A8A", width=0.72, label="Architecture, Infra & Feature Dev Tracks", zorder=3)
    ax_right.bar(x_w, qa_t, bottom=dev_t, color="#10B981", edgecolor="#065F46", hatch="///", width=0.72, label="Simultaneous QA, Load & UAT Testing Tracks", zorder=3)
    ax_right.bar(x_w, mig_t, bottom=dev_t + qa_t, color="#DC2626", edgecolor="#7F1D1D", width=0.72, label="Legacy Migration, Shadow Run & Continuous Ops", zorder=3)

    ax_right.axvline(16.5, color="#DC2626", linewidth=2.0, linestyle="--", zorder=5)
    ax_right.text(
        16.3,
        4.65,
        "End M4 (W16):\nEnter Migration",
        ha="right",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color="#DC2626",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF2F2", edgecolor="#DC2626"),
    )

    ax_right.set_xticks(x_w[::2])
    ax_right.set_xticklabels([f"W{w}" for w in x_w[::2]], fontsize=8.2, fontweight="bold")
    ax_right.set_ylim(0, 5.2)
    ax_right.set_ylabel("Simultaneous Active Workstreams", fontsize=9.5, fontweight="bold")
    ax_right.set_xlabel("Project Timeline Weeks", fontsize=9.5, fontweight="bold")
    ax_right.set_title(
        "Panel B: Simultaneous Development, Continuous QA & Migration Concurrency",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_right.grid(axis="y", linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    ax_right.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

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
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(wspace=0.22)

    # --------------------------------------------------------------------------
    # PANEL A: Stage-by-Stage Cost Estimation (Base Cost + Contingency Buffer)
    # --------------------------------------------------------------------------
    ax_left.set_facecolor("#F8FAFC")
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
            edgecolor="#0F172A",
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
            color="#F59E0B",
            edgecolor="#0F172A",
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
            color="#0F172A",
            zorder=5,
        )

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(s_labels, fontsize=8.8, fontweight="bold", color="#0F172A")
    ax_left.set_ylim(6.75, -0.65)
    ax_left.set_xlim(0, 745)
    ax_left.set_xlabel("Estimated Cost (M VND)", fontsize=9.5, fontweight="bold")
    ax_left.set_title(
        "Panel A: Stage Cost Estimation Breakdown: Base Cost vs. Risk Contingency",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_left.grid(axis="x", linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)

    # Summary box inside Panel A (bottom center, 100% inside axes frame)
    summary_text = (
        f"4-Month Core Build (Stages 1–4, W1–W16):  Base {res.core_build_base_cost_mvnd:.2f}M  |  Buffer {res.core_build_buffer_mvnd:.2f}M  |  Buffered {res.core_build_buffered_cost_mvnd:.2f}M VND\n"
        f"Full 6-Month Lifecycle (S1–S6, W1–W24):  Labor {res.total_base_labor_cost_mvnd:.2f}M + Cloud/Infra {res.total_direct_infra_cost_mvnd:.2f}M = "
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
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFFFF", edgecolor="#64748B", linewidth=1.0),
        zorder=6,
    )
    ax_left.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # --------------------------------------------------------------------------
    # PANEL B: Our Leveled Cost Allocation Across Project Duration + Recurring Platform OpEx
    # --------------------------------------------------------------------------
    ax_right.set_facecolor("#F8FAFC")
    df_m = res.monthly_cost_allocation_df

    # 6 project months (M1..M6) + 7th bar showing ongoing M7+ steady-state monthly platform OpEx
    m_labels = [f"{r['month_label']}\n({r['weeks_window']})" for _, r in df_m.iterrows()] + ["Jul-27+\n(M7+ Steady)"]
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
        color="#1D4ED8",
        edgecolor="#0F172A",
        linewidth=1.0,
        label="Sprint Active Base Drawdown (Bi-Weekly Earned Value)",
        zorder=3,
    )
    # Stack 2: Stage 3 Milestone Acceptance Holdback Release (M4 UAT, M5 Cutover, M6 Hypercare)
    ax_right.bar(
        x_m,
        hb_arr,
        bottom=sprint_arr,
        width=w,
        color="#0284C7",
        edgecolor="#0F172A",
        linewidth=1.0,
        label="Stage 3 Milestone Holdback Release (M4 / M5 / M6 Gates)",
        zorder=3,
    )
    # Stack 3: Conditional Contingency Reserve (29.58M/mo across M1-M6)
    ax_right.bar(
        x_m,
        reserve_arr,
        bottom=committed_dev_arr,
        width=w,
        color="#F59E0B",
        edgecolor="#0F172A",
        linewidth=1.0,
        hatch="///",
        alpha=0.90,
        label="Conditional Task Contingency Reserve (29.58M/mo)",
        zorder=3,
    )
    # Stack 4: Recurring Monthly Platform & Community OpEx (Starts M5: 16.50M/mo; M7+: 26.50M/mo)
    ax_right.bar(
        x_m,
        rec_opex_arr,
        bottom=committed_dev_arr + reserve_arr,
        width=w,
        color="#059669",
        edgecolor="#0F172A",
        linewidth=1.0,
        label="Recurring Monthly Platform & Community OpEx (16.5M M5–6; 26.5M M7+)",
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
                color="#FFFFFF",
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
                    color="#FFFFFF",
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
                color="#0F172A",
                zorder=6,
            )
        else:
            # M7+ steady-state recurring platform OpEx bar
            ax_right.text(
                x_m[idx],
                rec_opex_arr[idx] + 4.5,
                f"Ongoing Platform OpEx\n{rec_opex_arr[idx]:.2f}M / mo\n(Cloud 8.5M + Uni 8.0M\n+ Maint 10.0M)",
                ha="center",
                va="bottom",
                fontsize=7.1,
                fontweight="bold",
                color="#065F46",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ECFDF5", edgecolor="#059669", linewidth=0.9),
                zorder=6,
            )

    ax_right.axvline(5.5, color="#64748B", linewidth=1.3, linestyle="--", zorder=2)

    # Summary callout inside Panel B
    alloc_summary = (
        f"6-Month Development Allocation (M1–M6):  Committed Base 139.9M–166.0M/mo (Avg 151.9M/mo, SD 8.4M)  |  +29.58M/mo Buffer Reserve\n"
        f"Recurring Platform OpEx:  {res.launch_monthly_platform_opex_mvnd:.2f}M/mo at Launch (M5–M6)  |  {res.steady_state_monthly_platform_opex_mvnd:.2f}M/mo Steady-State (M7+ post-hypercare)"
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
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#FFFFFF", edgecolor="#64748B", linewidth=1.0),
        zorder=6,
    )

    ax_right.set_xticks(x_m)
    ax_right.set_xticklabels(m_labels, fontsize=8.1, fontweight="bold", color="#0F172A")
    ax_right.set_ylim(-28, 265)
    ax_right.set_ylabel("Monthly Allocated Cost Outflow (M VND)", fontsize=9.5, fontweight="bold")
    ax_right.set_xlabel("Build & Migration Schedule vs. Steady-State Operations", fontsize=9.3, fontweight="bold")
    ax_right.set_title(
        "Panel B: Leveled Monthly Cost Allocation & Recurring Platform Operating Expenses",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_right.grid(axis="y", linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    ax_right.legend(loc="upper right", fontsize=7.6, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    fig.suptitle(
        "PICK! Social & Adaptive Booking Platform — Cost Estimation, Risk Contingency & Leveled Cash Allocation",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
        y=0.99,
    )

    out_path = output_dir / "03_cost_estimation_and_cashflow_allocation.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path
