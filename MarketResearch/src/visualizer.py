"""Publication-grade visualizations for `MarketResearch/` (Section 1: Our Market & Section 2: Our Customers).

Strict Visual & Formatting Standards:
  - 100% ASCII only (`ord(c) <= 127`). Zero Unicode arrows, triangles, or emojis.
  - Zero cryptic abbreviations (always spell out `courts`, `customers`, `visits`, `Marketing Cost`, `Exhibit`, `Average`, `Ancillary`, `year`).
  - Zero fake personal names in Customer Profiles (use segment archetypes and clean bullet points).
  - Generous whitespace and cell padding around all text, callouts, bars, and legends.
  - Designed in Catppuccin Mocha palette so `theme.py` maps every artist cleanly to Catppuccin Latte.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .config import MarketResearchConfig
from .model import MarketPositioningAnalysis
import theme as _theme  # noqa: F401


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


def plot_hcmc_market_positioning_bubble_chart(
    cfg: MarketResearchConfig,
    analysis: MarketPositioningAnalysis,
    output_dir: Path,
) -> Path:
    """
    Render `01_hcmc_market_positioning_bubble_chart.png` (Spacious 2-Panel Layout):
      - Panel A (Left): 2D Current Market Positioning Bubble Plot of HCMC Pickleball Clubs
          * X-Axis: Peak Court Rental Price (kVND / court-hour)
          * Y-Axis: Combined Multi-Platform Social Reach (TikTok + Instagram + Facebook, in thousands)
          * Bubble Area: Estimated Annual Venue Revenue (Billion VND / year)
      - Panel B (Right): Vietnam TAM, HCMC SAM, Prime Catchment & Obtainable SOM Market Sizing (2026 vs 2029)
    """
    _apply_clean_style()
    fig = plt.figure(figsize=(22.0, 11.4), dpi=220)
    fig.patch.set_facecolor("#1e1e2e")

    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.20, 1.0],
        left=0.048,
        right=0.978,
        top=0.865,
        bottom=0.095,
        wspace=0.33,
    )

    fig.suptitle(
        "Section 1 (Our Market): HCMC Pickleball Competitive Landscape & Vietnam TAM - SAM - SOM Sizing",
        fontsize=15.5,
        fontweight="bold",
        color="#cdd6f4",
        y=0.955,
    )
    fig.text(
        0.5,
        0.914,
        "Current HCMC Club Positioning (Peak Court Price vs. Multi-Platform Social Attention & Revenue Size) and 2026-2029 Market Growth Forecast",
        ha="center",
        fontsize=10.4,
        color="#a6adc8",
    )

    # ==========================================================================
    # PANEL A (Left): HCMC Competitive Positioning Bubble Plot (Current Market)
    # ==========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#181825")

    # Shaded zone for <260k Peak Price Undercutting
    ax_a.axvspan(122, 258.5, color="#30212b", alpha=0.24, zorder=1)
    ax_a.axvline(
        260.0,
        color="#f9e2af",
        linestyle="--",
        linewidth=1.6,
        zorder=2,
    )
    ax_a.text(
        180.0,
        101.5,
        "Price-Undercut Zone (< 260k VND/hour Peak)\nWhere Cheaper Local Rivals Compete on Peak Price",
        ha="center",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color="#f38ba8",
        bbox=dict(boxstyle="round,pad=0.36", facecolor="#1e1e2e", edgecolor="#f38ba8", linewidth=1.1, alpha=0.95),
        zorder=6,
    )

    # Horizontal reference line for HCMC Peer Average Reach (18.6k followers)
    ax_a.axhline(
        analysis.avg_peer_followers_k,
        color="#89b4fa",
        linestyle=":",
        linewidth=1.5,
        zorder=2,
    )
    ax_a.text(
        125.0,
        analysis.avg_peer_followers_k + 0.9,
        f"HCMC Peer Average: {analysis.avg_peer_followers_k:.1f}k Followers\n(PICK! = 42.0k, {analysis.pick_follower_multiple_vs_peers:.2f}x Peer Average)",
        ha="left",
        va="bottom",
        fontsize=7.9,
        fontweight="bold",
        color="#89b4fa",
        zorder=5,
    )

    # Bubble area scale proportional to annual_revenue_mvnd
    size_scale = 0.088

    for comp in cfg.competitors:
        x = comp.peak_price_kvnd
        y = comp.social_followers_k
        s = max(comp.annual_revenue_mvnd * size_scale, 165.0)

        if comp.id == "pick_baseline":
            fc = "#f9e2af"
            ec = "#11111b"
            lw = 2.4
            z = 9
        elif comp.peak_price_kvnd < 260.0:
            fc = "#f38ba8"
            ec = "#11111b"
            lw = 1.5
            z = 6
        else:
            fc = "#89b4fa"
            ec = "#11111b"
            lw = 1.5
            z = 6

        ax_a.scatter(
            [x],
            [y],
            s=s,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            alpha=0.88,
            zorder=z,
        )

    # Plot PICK! Year 3 Target ("Where We Should Be") Bubble & Trajectory Line
    target_x = 286.0
    target_y = 92.2
    target_rev_mvnd = 33231.02
    ax_a.plot(
        [260.0, target_x],
        [42.0, target_y],
        color="#a6e3a1",
        linestyle="-.",
        linewidth=2.3,
        zorder=7,
    )
    ax_a.scatter(
        [target_x],
        [target_y],
        s=target_rev_mvnd * size_scale,
        facecolor="#a6e3a1",
        edgecolor="#11111b",
        linewidth=2.5,
        alpha=0.90,
        zorder=9,
    )
    ax_a.plot(
        [target_x, 256.0],
        [target_y, 80.5],
        color="#a6e3a1",
        linewidth=1.2,
        linestyle="-",
        zorder=8,
    )
    ax_a.text(
        254.0,
        80.5,
        "WHERE WE SHOULD BE: PICK! Year 3 Target (20 courts)\n286k peak, 115k daytime, 92.2k followers, 33.23B VND/year",
        ha="right",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#a6e3a1",
        linespacing=1.22,
        bbox=dict(boxstyle="round,pad=0.36", facecolor="#1e1e2e", edgecolor="#a6e3a1", linewidth=1.6, alpha=0.96),
        zorder=11,
    )

    # Widely spaced callout target coordinates (tx, ty, ha, va) across (-15, 105) Y-axis
    label_targets: dict[str, tuple[float, float, str, str]] = {
        "pick_baseline": (244.0, 62.0, "right", "center"),
        "hcmc_djoy": (228.0, 48.0, "right", "center"),
        "hcmc_002": (214.0, 38.0, "right", "center"),
        "hcmc_usc": (298.0, 36.5, "center", "bottom"),
        "hcmc_rudal": (226.0, 28.5, "right", "center"),
        "hcmc_pickoland": (274.0, 22.8, "left", "center"),
        "hcmc_ese": (276.0, 13.5, "left", "center"),
        "hcmc_hitpark": (226.0, 13.5, "right", "center"),
        "hcmc_thanhda": (226.0, 3.8, "left", "center"),
        "hcmc_happyura": (125.0, 9.8, "left", "bottom"),
    }

    for comp in cfg.competitors:
        tx, ty, ha_align, va_align = label_targets[comp.id]
        rev_b = comp.annual_revenue_mvnd / 1000.0

        if comp.id == "pick_baseline":
            lbl = (
                f"WHERE WE ARE TODAY: PICK! Baseline (6 courts)\n"
                f"260k peak, 160k off-peak, 42.0k followers, {rev_b:.2f}B VND/year"
            )
            txt_col = "#f9e2af"
            box_kw = dict(boxstyle="round,pad=0.36", facecolor="#1e1e2e", edgecolor="#f9e2af", linewidth=1.5, alpha=0.96)
            fsize = 8.3
            fweight = "bold"
        else:
            lbl = (
                f"{comp.name} ({comp.courts} courts)\n"
                f"{comp.peak_price_kvnd:.0f}k peak, {comp.social_followers_k:.1f}k followers, {rev_b:.2f}B/year"
            )
            txt_col = "#cdd6f4"
            box_kw = dict(boxstyle="round,pad=0.26", facecolor="#1e1e2e", edgecolor="#45475a", linewidth=0.9, alpha=0.93)
            fsize = 7.5
            fweight = "normal"

        ax_a.plot(
            [comp.peak_price_kvnd, tx],
            [comp.social_followers_k, ty],
            color="#6c7086",
            linewidth=0.95,
            linestyle="-",
            zorder=4,
        )
        ax_a.text(
            tx,
            ty,
            lbl,
            ha=ha_align,
            va=va_align,
            fontsize=fsize,
            fontweight=fweight,
            color=txt_col,
            linespacing=1.22,
            bbox=box_kw,
            zorder=10,
        )

    legend_handles = [
        mpatches.Patch(facecolor="#f9e2af", edgecolor="#11111b", label="Where We Are Today: PICK! Baseline (6 courts, 260k peak, 42.0k followers, 7.68B/year)"),
        mpatches.Patch(facecolor="#a6e3a1", edgecolor="#11111b", label="Where We Should Be: PICK! Year 3 Target (20 courts, 286k peak, 92.2k followers, 33.23B/year)"),
        mpatches.Patch(facecolor="#f38ba8", edgecolor="#11111b", label="HCMC Rivals Pricing Peak < 260k VND/hour (Cheaper Peak Tier)"),
        mpatches.Patch(facecolor="#89b4fa", edgecolor="#11111b", label="HCMC Clubs Pricing Peak >= 260k VND/hour (Parity / Premium Tier)"),
    ]
    ax_a.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.012),
        frameon=True,
        facecolor="#1e1e2e",
        edgecolor="#45475a",
        fontsize=7.6,
        title="Club Category (Bubble Area = Annual Revenue in Billion VND/year)",
        title_fontsize=7.9,
    )

    ax_a.set_xlim(122, 336)
    ax_a.set_ylim(-14.5, 104.5)
    ax_a.set_title(
        "Panel A: HCMC Competitive Positioning (Peak Court Price vs. Multi-Platform Reach & Revenue)",
        fontsize=11.6,
        fontweight="bold",
        color="#cdd6f4",
        pad=12,
    )
    ax_a.set_xlabel("Peak Court Rental Price (kVND / Court-Hour)", fontsize=10.2, color="#bac2de")
    ax_a.set_ylabel("Combined Social Media Attention: TikTok + IG + FB Followers (Thousands)", fontsize=10.2, color="#bac2de")

    # ==========================================================================
    # PANEL B (Right): Vietnam TAM, HCMC SAM & SOM Market Sizing (2026 vs 2029)
    # ==========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#181825")

    tiers = [
        "TAM\n(Vietnam National)",
        "SAM\n(HCMC Venue Market)",
        "Prime Catchment SAM\n(Core HCMC Hubs)",
        "SOM\n(PICK! Market Share)",
    ]
    vals_2026 = [
        analysis.tam_2026_bvnd,
        analysis.sam_2026_bvnd,
        analysis.catchment_2026_bvnd,
        analysis.som_2026_bvnd,
    ]
    vals_2029 = [
        analysis.tam_2029_bvnd,
        analysis.sam_2029_bvnd,
        analysis.catchment_2029_bvnd,
        analysis.som_2029_bvnd,
    ]
    growth_notes = [
        f"+{cfg.sizing.tam_cagr_2026_2029_pct:.1f}% CAGR",
        f"+{cfg.sizing.sam_cagr_2026_2029_pct:.1f}% CAGR",
        f"+{cfg.sizing.catchment_cagr_2026_2029_pct:.1f}% CAGR",
        f"{analysis.som_share_of_sam_2029_pct:.2f}% of SAM",
    ]

    y_pos = np.arange(len(tiers))[::-1]
    bar_h = 0.34

    ax_b.barh(
        y_pos + bar_h / 2,
        vals_2026,
        height=bar_h,
        color="#89b4fa",
        edgecolor="#11111b",
        linewidth=1.1,
        label="2026 Current Market Size (Billion VND/year)",
        zorder=4,
    )
    ax_b.barh(
        y_pos - bar_h / 2,
        vals_2029,
        height=bar_h,
        color="#a6e3a1",
        edgecolor="#11111b",
        linewidth=1.1,
        label="2029 Projected Market Size (Billion VND/year)",
        zorder=4,
    )

    ax_b.set_xscale("log")
    ax_b.set_xlim(3.5, 450000.0)
    ax_b.set_ylim(-0.65, 3.65)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(tiers, fontsize=9.8, fontweight="bold", color="#cdd6f4")

    for y_val, v26, v29, gnote in zip(y_pos, vals_2026, vals_2029, growth_notes):
        lbl_26 = (
            f"2026: {v26:,.2f}B VND ({analysis.som_share_of_sam_2026_pct:.2f}% of SAM)"
            if v26 < 20
            else f"2026: {v26:,.0f}B VND"
        )
        ax_b.text(
            v26 * 1.15,
            y_val + bar_h / 2,
            lbl_26,
            va="center",
            ha="left",
            fontsize=8.6,
            fontweight="bold",
            color="#89b4fa",
        )
        ax_b.text(
            v29 * 1.15,
            y_val - bar_h / 2,
            (f"2029: {v29:,.2f}B VND ({gnote})" if v29 < 50 else f"2029: {v29:,.0f}B VND ({gnote})"),
            va="center",
            ha="left",
            fontsize=8.6,
            fontweight="bold",
            color="#a6e3a1",
        )

    ax_b.set_title(
        "Panel B: Vietnam TAM, HCMC SAM & Obtainable SOM Sizing (Log Scale, Billion VND/Year)",
        fontsize=11.6,
        fontweight="bold",
        color="#cdd6f4",
        pad=12,
    )
    ax_b.set_xlabel("Annual Market Size in Billion VND / Year (Logarithmic Scale)", fontsize=10.2, color="#bac2de")
    ax_b.legend(loc="upper right", frameon=True, facecolor="#1e1e2e", edgecolor="#45475a", fontsize=8.4)

    out_path = output_dir / "01_hcmc_market_positioning_bubble_chart.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_customer_personas_and_mutual_demand(
    cfg: MarketResearchConfig,
    analysis: MarketPositioningAnalysis,
    output_dir: Path,
) -> Path:
    """
    Render `02_customer_personas_and_mutual_demand.png`:
      - Top Half (Panel A): Spacious, bullet-point 4-Segment Customer Profile & Derived Demand Table
        (Zero fake personal names, zero cryptic abbreviations, generous cell padding).
      - Bottom Left (Panel B): Per-Visit Spend Breakdown by Segment (Court Hire, Coaching, Food & Drinks,
        Paddle & Balls in kVND/visit, reconciling Case Exhibit 1 & Exhibit 4).
      - Bottom Right (Panel C): Segment Share Comparison (Customer Share %, Monthly Visit Share %, Revenue Share %)
        & Implied Marketing Cost per Visit (VND/visit).
    """
    _apply_clean_style()
    fig = plt.figure(figsize=(22.4, 14.2), dpi=220)
    fig.patch.set_facecolor("#1e1e2e")

    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.26, 1.0],
        width_ratios=[1.0, 1.08],
        left=0.042,
        right=0.978,
        top=0.898,
        bottom=0.062,
        wspace=0.20,
        hspace=0.26,
    )

    fig.suptitle(
        "Section 2 (Our Customers): HCMC Customer Profiles, Unit Economics & Derived Mutual Demand",
        fontsize=15.5,
        fontweight="bold",
        color="#cdd6f4",
        y=0.966,
    )
    fig.text(
        0.5,
        0.932,
        "4 Customer Segments (3,600 Unique Customers & 7,596 Visits/Month), Per-Visit Spend Breakdown & Data-Driven Priority Focus",
        ha="center",
        fontsize=10.4,
        color="#a6adc8",
    )

    # ==========================================================================
    # PANEL A (Top Half, Spans Both Columns): Spacious Bullet-Point Grid Table
    # ==========================================================================
    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.set_facecolor("#1e1e2e")
    ax_tbl.axis("off")

    col_widths = [0.140, 0.160, 0.155, 0.125, 0.160, 0.155, 0.105]
    total_w = sum(col_widths)
    ax_tbl.set_xlim(0.0, total_w)
    ax_tbl.set_ylim(0.0, 1.0)

    ordered_personas = sorted(cfg.personas, key=lambda p: p.priority_rank)

    table_rows: list[tuple[str, list[str]]] = [
        (
            "banner",
            [
                "PANEL A: CUSTOMER SEGMENT PROFILES, RETENTION FRICTIONS & DERIVED MUTUAL DEMAND (ORDERED BY STRATEGIC PRIORITY)",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        ),
        (
            "header",
            [
                "Priority & Segment\n(Share of Customers)",
                "Customer Profile &\nDemographics",
                "Playing Schedule &\nGroup Behavior",
                "Core Need\n(Case Exhibit 4)",
                "Primary Retention &\nBooking Friction",
                "Derived Customer Demand\n(What Players Need)",
                "Baseline Unit\nEconomics",
            ],
        ),
    ]

    row_styles_by_id = {
        "social": "row_priority1",
        "trend_explorers": "row_priority2",
        "competitive": "row_anchor",
        "families": "row_niche",
    }

    for p in ordered_personas:
        r_style = row_styles_by_id[p.id]
        prio_tag = (
            "PRIORITY #1: Core Volume"
            if p.priority_rank == 1
            else "PRIORITY #2: High Churn"
            if p.priority_rank == 2
            else "Tier 2: Frequency Anchor"
            if p.priority_rank == 3
            else "Tier 3: Coaching Niche"
        )
        c0 = (
            f"{prio_tag}\n"
            f"{p.segment_name}\n"
            f"- {p.customer_share_pct:.0f}% of customers\n"
            f"- {p.baseline_customers:,} customers/month"
        )
        c1 = f"{p.persona_title}\n{p.demographics}"
        c2 = p.schedule_habit
        c3 = (
            f"\"{p.main_case_need}\""
            if len(p.main_case_need) <= 24
            else "\"Improve skills and\nfind good opponents\""
            if p.id == "competitive"
            else "\"Play with friends and\nenjoy the atmosphere\""
            if p.id == "social"
            else "\"Try something seen\nonline; many visit\nonce or twice\""
            if p.id == "trend_explorers"
            else "\"A welcoming activity\nand beginner lessons\""
        )
        c4 = p.pain_points
        c5 = p.derived_demand
        c6 = (
            f"- {p.visits_per_month:.1f} visits/month\n"
            f"- {p.baseline_visits_monthly:,} total visits ({p.visit_share_pct:.1f}%)\n"
            f"- {p.total_spend_per_visit_kvnd:.0f}k VND Spend/visit\n"
            f"- Marketing Cost:\n  {p.marketing_cost_per_visit_vnd:,.0f} VND/visit"
        )
        table_rows.append((r_style, [c0, c1, c2, c3, c4, c5, c6]))

    # Total reconciliation + Mutual Demand synthesis footer row
    table_rows.append(
        (
            "footer",
            [
                "TOTAL / AVERAGE\nAll 4 Segments (100%)\n3,600 customers/month",
                "- Urban HCMC players\n- Gen-Z, Millennials &\n  young households",
                "- 7,596 visits/month\n- 2.11 visits/customer\n- 60% empty daytime",
                "Skill Match + Social\nAtmosphere + Easy\nTrial + Coaching",
                "- 89% first-visit churn\n  in Trend Explorers\n- Zero loyalty lock-in",
                "- 4-player skill matching\n- Affordable daytime rates\n- Beginner progression\n- Social cafe community",
                "- 641.7M VND/month\n  (Exhibit 1: 640M)\n- 84.5k Average/visit\n- Marketing Cost:\n  3,291 VND/visit",
            ],
        )
    )

    row_weights = [0.58, 0.78, 1.48, 1.48, 1.48, 1.48, 1.24]
    tot_weight = sum(row_weights)
    curr_y = 1.0

    for (rtype, cells), rw in zip(table_rows, row_weights):
        rh = rw / tot_weight
        ry = curr_y - rh
        curr_y = ry

        if rtype == "banner":
            ax_tbl.add_patch(
                mpatches.Rectangle(
                    (0.0, ry),
                    total_w,
                    rh,
                    facecolor="#162038",
                    edgecolor="#45475a",
                    linewidth=1.0,
                )
            )
            ax_tbl.text(
                0.012,
                ry + rh * 0.5,
                cells[0],
                ha="left",
                va="center",
                fontsize=9.4,
                fontweight="bold",
                color="#f5f7ff",
            )
            continue

        cx = 0.0
        for c_idx, (cw, cell_txt) in enumerate(zip(col_widths, cells)):
            if rtype == "header":
                bg = "#2b3048"
                fg = "#f5f7ff"
                fw = "bold"
                fs = 8.3
                ha = "center"
            elif rtype == "footer":
                bg = "#1f332c"
                fg = "#a6e3a1" if c_idx in (0, 5, 6) else "#cdd6f4"
                fw = "bold"
                fs = 7.7
                ha = "center" if c_idx in (0, 3) else "left"
            else:
                if rtype == "row_priority1":
                    bg = "#1f2e30" if c_idx in (0, 5) else "#181825"
                    accent_col = "#94e2d5"
                elif rtype == "row_priority2":
                    bg = "#302623" if c_idx in (0, 5) else "#1e1e2e"
                    accent_col = "#fab387"
                elif rtype == "row_anchor":
                    bg = "#24273a" if c_idx in (0, 5) else "#181825"
                    accent_col = "#89b4fa"
                else:
                    bg = "#292338" if c_idx in (0, 5) else "#1e1e2e"
                    accent_col = "#cba6f7"

                if c_idx == 0:
                    fg = accent_col
                    fw = "bold"
                    ha = "left"
                elif c_idx == 3:
                    fg = "#f9e2af"
                    fw = "bold"
                    ha = "center"
                elif c_idx == 4:
                    fg = "#f38ba8"
                    fw = "normal"
                    ha = "left"
                elif c_idx == 5:
                    fg = accent_col
                    fw = "bold"
                    ha = "left"
                elif c_idx == 6:
                    fg = "#cdd6f4"
                    fw = "bold"
                    ha = "left"
                else:
                    fg = "#cdd6f4"
                    fw = "normal"
                    ha = "left"
                fs = 7.75

            ax_tbl.add_patch(
                mpatches.Rectangle(
                    (cx, ry),
                    cw,
                    rh,
                    facecolor=bg,
                    edgecolor="#45475a",
                    linewidth=0.9,
                )
            )

            tx = cx + cw * 0.5 if ha == "center" else cx + 0.009
            ax_tbl.text(
                tx,
                ry + rh * 0.5,
                cell_txt,
                ha=ha,
                va="center",
                fontsize=fs,
                fontweight=fw,
                color=fg,
                linespacing=1.32,
            )
            cx += cw

    # ==========================================================================
    # PANEL B (Bottom Left): Pure Stacked Bar Chart of Per-Visit Spend Breakdown
    # ==========================================================================
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_facecolor("#181825")

    segs = cfg.personas  # Case Exhibit 4 order: Competitive, Social, Trend Explorers, Families
    seg_labels = [
        f"{p.segment_name}\n({p.customer_share_pct:.0f}% Customers, {p.visits_per_month:.1f} visits/month)"
        for p in segs
    ]
    x_pos = np.arange(len(segs))
    bar_w = 0.46

    court_vals = np.array([p.court_hire_per_visit_kvnd for p in segs])
    coach_vals = np.array([p.coaching_per_visit_kvnd for p in segs])
    fnb_vals = np.array([p.fnb_per_visit_kvnd for p in segs])
    pad_vals = np.array([p.paddle_balls_per_visit_kvnd for p in segs])

    ax_b.bar(x_pos, court_vals, width=bar_w, color="#89b4fa", edgecolor="#11111b", linewidth=1.0, label="Court Hire (435M/month Baseline)")
    ax_b.bar(x_pos, coach_vals, width=bar_w, bottom=court_vals, color="#a6e3a1", edgecolor="#11111b", linewidth=1.0, label="Coaching (115M/month Baseline)")
    ax_b.bar(
        x_pos,
        fnb_vals,
        width=bar_w,
        bottom=court_vals + coach_vals,
        color="#f9e2af",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Food & Drinks (75M/month Baseline)",
    )
    ax_b.bar(
        x_pos,
        pad_vals,
        width=bar_w,
        bottom=court_vals + coach_vals + fnb_vals,
        color="#cba6f7",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Paddle & Balls (15M/month Baseline)",
    )

    for i, p in enumerate(segs):
        ax_b.text(
            i,
            p.total_spend_per_visit_kvnd + 2.2,
            f"Total: {p.total_spend_per_visit_kvnd:.0f}k VND/visit\n(Court {p.court_hire_per_visit_kvnd:.1f}k + Ancillary {p.ancillary_per_visit_kvnd:.1f}k)",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#cdd6f4",
            linespacing=1.22,
        )
        ax_b.text(
            i,
            p.court_hire_per_visit_kvnd * 0.5,
            f"Court Hire: {p.court_hire_per_visit_kvnd:.1f}k",
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            color="#11111b",
        )
        if p.coaching_per_visit_kvnd >= 9.0:
            ax_b.text(
                i,
                p.court_hire_per_visit_kvnd + p.coaching_per_visit_kvnd * 0.5,
                f"Coaching: {p.coaching_per_visit_kvnd:.1f}k",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#11111b",
            )
        if p.fnb_per_visit_kvnd >= 9.5:
            ax_b.text(
                i,
                p.court_hire_per_visit_kvnd + p.coaching_per_visit_kvnd + p.fnb_per_visit_kvnd * 0.5,
                f"{p.fnb_per_visit_kvnd:.1f}k",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#11111b",
            )

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(seg_labels, fontsize=8.4, fontweight="bold", color="#cdd6f4")
    ax_b.set_ylim(0, 126)
    ax_b.set_ylabel("Spend per Player Visit (kVND / Visit)", fontsize=10.0, color="#bac2de")
    ax_b.set_title(
        "Panel B: Per-Visit Spend Breakdown by Segment (Case Exhibit 1 & Exhibit 4 Reconciled)",
        fontsize=11.5,
        fontweight="bold",
        color="#cdd6f4",
        pad=12,
    )
    ax_b.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor="#1e1e2e",
        edgecolor="#45475a",
        fontsize=7.9,
    )

    # ==========================================================================
    # PANEL C (Bottom Right): Customer vs. Visit vs. Revenue Share (%) &
    #                         Marketing Cost per Visit Priority Justification
    # ==========================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_facecolor("#181825")

    tot_rev = analysis.total_baseline_revenue_mvnd
    cust_shares = np.array([p.customer_share_pct for p in segs])
    visit_shares = np.array([p.visit_share_pct for p in segs])
    rev_shares = np.array([(p.baseline_monthly_revenue_mvnd / tot_rev) * 100.0 for p in segs])

    gw = 0.24
    ax_c.bar(
        x_pos - gw,
        cust_shares,
        width=gw,
        color="#89b4fa",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Share of Unique Customers (% of 3,600/month)",
    )
    ax_c.bar(
        x_pos,
        visit_shares,
        width=gw,
        color="#94e2d5",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Share of Monthly Visits (% of 7,596/month)",
    )
    ax_c.bar(
        x_pos + gw,
        rev_shares,
        width=gw,
        color="#a6e3a1",
        edgecolor="#11111b",
        linewidth=1.0,
        label="Share of Baseline Revenue (% of 641.7M/month)",
    )

    for idx_bar in range(len(segs)):
        for offset, val, col_hex in [
            (-gw, cust_shares[idx_bar], "#89b4fa"),
            (0.0, visit_shares[idx_bar], "#94e2d5"),
            (gw, rev_shares[idx_bar], "#a6e3a1"),
        ]:
            ax_c.text(
                idx_bar + offset,
                val + 1.1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.7,
                fontweight="bold",
                color=col_hex,
            )

        p = segs[idx_bar]
        top_bar_h = max(cust_shares[idx_bar], visit_shares[idx_bar], rev_shares[idx_bar])
        if p.id == "social":
            badge_txt = f"PRIORITY #1 (Largest Customer Pool)\nMarketing Cost: {p.marketing_cost_per_visit_vnd:,.0f} VND/visit"
            b_edge = "#94e2d5"
            b_col = "#94e2d5"
        elif p.id == "trend_explorers":
            badge_txt = f"PRIORITY #2 (4.09x Marketing Cost!)\nMarketing Cost: {p.marketing_cost_per_visit_vnd:,.0f} VND/visit"
            b_edge = "#f38ba8"
            b_col = "#f38ba8"
        elif p.id == "competitive":
            badge_txt = f"High-Frequency Anchor (4.5 visits/month)\nMarketing Cost: {p.marketing_cost_per_visit_vnd:,.0f} VND/visit"
            b_edge = "#89b4fa"
            b_col = "#89b4fa"
        else:
            badge_txt = f"Coaching Niche (38.6k Ancillary/visit)\nMarketing Cost: {p.marketing_cost_per_visit_vnd:,.0f} VND/visit"
            b_edge = "#cba6f7"
            b_col = "#cba6f7"

        ax_c.text(
            idx_bar,
            top_bar_h + 7.2,
            badge_txt,
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=b_col,
            linespacing=1.22,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#1e1e2e", edgecolor=b_edge, linewidth=1.1, alpha=0.95),
        )

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(seg_labels, fontsize=8.4, fontweight="bold", color="#cdd6f4")
    ax_c.set_ylim(0, 74)
    ax_c.set_ylabel("Segment Share of Club Total (%)", fontsize=10.0, color="#bac2de")
    ax_c.set_title(
        "Panel C: Customer, Visit & Revenue Share (%) + Marketing Cost per Visit Priority Justification",
        fontsize=11.5,
        fontweight="bold",
        color="#cdd6f4",
        pad=12,
    )
    ax_c.legend(
        loc="upper right",
        frameon=True,
        facecolor="#1e1e2e",
        edgecolor="#45475a",
        fontsize=7.9,
    )

    out_path = output_dir / "02_customer_personas_and_mutual_demand.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path
