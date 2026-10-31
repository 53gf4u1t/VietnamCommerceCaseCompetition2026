"""Clean, tight-cropped 3-statement spreadsheet table renderer, double-bar comparison charts, and Monte Carlo cohort fan charts."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from .config import FinancialModelConfig
from .model import FinancialModelResults

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
})

EXCEL_GRID_BORDER = "#CBD5E1"
SECTION_BANNER = "#1E3A8A"
SUBHEADER_BG = "#E2E8F0"
SUBTOTAL_BG = "#F8FAFC"
TOTAL_HIGHLIGHT_BG = "#ECFDF5"
PROFIT_HIGHLIGHT_BG = "#EFF6FF"
LINKED_BG = "#F5F3FF"

INPUT_BLUE = "#0000FF"
LINKED_PURPLE = "#6D28D9"
FORMULA_BLACK = "#0F172A"
FORMULA_DARKGRAY = "#334155"
MUTED_SLATE = "#64748B"
EMERALD_TEXT = "#047857"


def _render_pure_table(
    rows: list[tuple[str, list[str]]],
    col_widths: list[float],
    align_fn,
    style_fn,
    fig_width: float,
    row_height_in: float,
    out_path: Path,
    font_size: float = 8.2,
) -> Path:
    """
    Render ONLY the pure spreadsheet table (from row 1 to last row, col 1 to last col)
    with zero Excel UI chrome, formula bars, row numbers, or column letters.
    """
    n_rows = len(rows)
    fig_height = max(2.0, n_rows * row_height_in)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=240)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    total_w = sum(col_widths)
    row_h = 1.0 / n_rows

    ax.set_xlim(0.0, total_w)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for r_idx, (rstyle, cells) in enumerate(rows):
        ry = 1.0 - (r_idx + 1) * row_h

        if rstyle == "sec_hdr":
            ax.add_patch(
                patches.Rectangle(
                    (0.0, ry),
                    total_w,
                    row_h,
                    facecolor=SECTION_BANNER,
                    edgecolor=EXCEL_GRID_BORDER,
                    linewidth=0.8,
                )
            )
            ax.text(
                0.008 * total_w,
                ry + row_h * 0.5,
                cells[0],
                ha="left",
                va="center",
                fontsize=font_size + 0.5,
                fontweight="bold",
                color="#FFFFFF",
            )
            continue

        cx = 0.0
        for c_idx, (cw, ctext) in enumerate(zip(col_widths, cells)):
            bg, fg, fw = style_fn(r_idx, c_idx, rstyle)

            ax.add_patch(
                patches.Rectangle(
                    (cx, ry),
                    cw,
                    row_h,
                    facecolor=bg,
                    edgecolor=EXCEL_GRID_BORDER,
                    linewidth=0.55,
                )
            )

            if rstyle in ("subtotal", "total", "grand_total"):
                ax.plot([cx, cx + cw], [ry + row_h, ry + row_h], color="#334155", linewidth=1.0)
                if rstyle in ("total", "grand_total"):
                    ax.plot([cx, cx + cw], [ry + row_h * 0.08, ry + row_h * 0.08], color="#0F172A", linewidth=1.1)
                    ax.plot([cx, cx + cw], [ry, ry], color="#0F172A", linewidth=1.1)
                else:
                    ax.plot([cx, cx + cw], [ry, ry], color="#334155", linewidth=1.0)

            ha = align_fn(c_idx, rstyle)
            if ha == "right":
                tx = cx + cw - 0.005 * total_w
            elif ha == "center":
                tx = cx + cw * 0.5
            else:
                tx = cx + 0.005 * total_w

            ax.text(
                tx,
                ry + row_h * 0.5,
                ctext,
                ha=ha,
                va="center",
                fontsize=font_size,
                fontweight=fw,
                color=fg,
            )
            cx += cw

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out_path


def plot_drivers_and_cost_breakdown_sheet(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Render Sheet 1: Pure table of AdaptivePricing Court Hire Link + Full Path C Income Statement
    (including Membership Subscriptions starting Month 5, Dynamic Payment & Event COGS,
    Platform Build Allocation in Months 1–6, and Recurring Platform OpEx in Month 5+).
    """
    col_widths = [0.115, 0.255, 0.105, 0.072, 0.076, 0.078, 0.082, 0.085, 0.062, 0.070]
    ap = res.adaptive

    rows: list[tuple[str, list[str]]] = [
        ("sec_hdr", [
            "PART 1 — COURT HIRE REVENUE INTEGRATION FROM AdaptivePricing/outputs/ (6 Courts | 2,880 Available Hrs/Month)",
            "", "", "", "", "", "", "", "", "",
        ]),
        ("col_hdr", [
            "Pricing Regime",
            "Operating Window & Schedule",
            "Available Hrs",
            "Case Static",
            "Case Rev (M)",
            "ML Adaptive",
            "ML Rev (M VND)",
            "Rev Lift (M)",
            "Lift %",
            "Upstream Source",
        ]),
    ]

    for reg in res.regimes:
        lift_m = reg.ml_revenue_mvnd - reg.case_revenue_mvnd
        lift_p = (lift_m / reg.case_revenue_mvnd) * 100.0 if reg.case_revenue_mvnd > 0 else 0.0
        rows.append((
            "linked",
            [
                "Court Regime",
                f"  {reg.regime_name}",
                f"{reg.available_court_hours:,.0f} hrs",
                f"{reg.case_utilisation_rate*100:.0f}% @ {reg.case_price_kvnd:.0f}k",
                f"{reg.case_revenue_mvnd:,.2f}",
                f"{reg.ml_utilisation_rate*100:.1f}% @ {reg.ml_avg_price_kvnd:.0f}k",
                f"{reg.ml_revenue_mvnd:,.2f}",
                f"{lift_m:+,.2f}",
                f"{lift_p:+.1f}%",
                "AdaptivePricing",
            ],
        ))

    rows.extend([
        ("subtotal", [
            "COURT HIRE TOTAL",
            "Total Court Hire (AdaptivePricing ML vs Case Exhibit 1)",
            f"{res.total_available_court_hours:,.0f} hrs",
            "65.9% (~1,900h)",
            f"{res.case_court_hire_mvnd:,.2f}",
            f"{res.blended_utilisation_rate*100:.1f}% ({res.total_booked_court_hours:,.0f}h)",
            f"{res.court_hire_mvnd:,.2f}",
            f"{ap.lift_vs_case_mvnd:+,.2f}",
            f"{ap.lift_vs_case_pct:+.2f}%",
            "Verified CSV",
        ]),
        ("blank", ["", "", "", "", "", "", "", "", "", ""]),
        ("sec_hdr", [
            "PART 2 — INTEGRATED INCOME STATEMENT STRUCTURE: REVENUE (INCL. PATH C SUBSCRIPTIONS M5+), COGS, FACILITY OpEx & SG&A",
            "", "", "", "", "", "", "", "", "",
        ]),
        ("col_hdr", [
            "Template Section",
            "Line Item (AdaptivePricing + Path C Platform & Subscriptions)",
            "Cost Behavior",
            "Case Base",
            "M1 Build (Jan-27)",
            "M5 Launch (May-27)",
            "M24 End Y2 (Dec-28)",
            "5Y Total (60M)",
            "% 5Y Rev",
            "Config / Source",
        ]),
    ])

    for li in res.line_items:
        if li.section == "SUMMARY":
            if "GROSS PROFIT" in li.template_category:
                rstyle = "total"
            elif "OPERATING PROFIT" in li.template_category:
                rstyle = "grand_total"
            else:
                rstyle = "subtotal"
        elif "New" in li.template_category or "AdaptivePricing" in li.line_item or "Payment gateway" in li.line_item:
            rstyle = "linked"
        else:
            rstyle = "input"

        src_lbl = (
            "AdaptivePricing"
            if "Court hire" in li.line_item
            else (
                "customers.toml"
                if "Membership subscriptions" in li.line_item
                else (
                    "cogs.toml"
                    if li.section == "COGS"
                    else (
                        "opex.toml"
                        if li.section in ("OPEX", "SGA")
                        else "Reconciled Sum"
                    )
                )
            )
        )
        prefix = "  " if li.section != "SUMMARY" else ""
        rows.append((
            rstyle,
            [
                li.template_category,
                f"{prefix}{li.line_item}",
                li.behavior,
                f"{li.case_baseline_monthly_mvnd:,.2f}",
                f"{li.m1_build_monthly_mvnd:,.2f}",
                f"{li.m5_launch_monthly_mvnd:,.2f}",
                f"{li.m24_end_y2_monthly_mvnd:,.2f}",
                f"{li.five_year_total_mvnd:,.2f}",
                f"{li.pct_of_5y_revenue:.2f}%",
                src_lbl,
            ],
        ))

    def _style_s1(r_idx: int, c_idx: int, rstyle: str) -> tuple[str, str, str]:
        if rstyle == "col_hdr":
            return SUBHEADER_BG, FORMULA_BLACK, "bold"
        if rstyle == "subtotal":
            return SUBTOTAL_BG, FORMULA_BLACK, "bold"
        if rstyle == "total":
            return PROFIT_HIGHLIGHT_BG, "#1E40AF", "bold"
        if rstyle == "grand_total":
            return TOTAL_HIGHLIGHT_BG, EMERALD_TEXT, "bold"
        if rstyle == "linked":
            if c_idx in (4, 5, 6, 7):
                return LINKED_BG, LINKED_PURPLE, "bold"
            return LINKED_BG, FORMULA_BLACK, "normal"
        if rstyle == "blank":
            return "#FFFFFF", FORMULA_BLACK, "normal"
        bg = "#FFFFFF" if r_idx % 2 == 0 else "#F8FAFC"
        if c_idx in (3, 4, 5, 6):
            return bg, INPUT_BLUE, ("bold" if c_idx == 6 else "normal")
        if c_idx in (7, 8):
            return bg, FORMULA_DARKGRAY, ("bold" if c_idx == 7 else "normal")
        if c_idx == 9:
            return bg, MUTED_SLATE, "normal"
        return bg, FORMULA_BLACK, "normal"

    def _align_s1(c_idx: int, rstyle: str) -> str:
        if c_idx in (3, 4, 5, 6, 7, 8):
            return "right"
        if c_idx in (2, 9) or rstyle == "col_hdr":
            return "center"
        return "left"

    return _render_pure_table(
        rows=rows,
        col_widths=col_widths,
        align_fn=_align_s1,
        style_fn=_style_s1,
        fig_width=21.0,
        row_height_in=0.33,
        out_path=output_dir / "01_revenue_cogs_opex_sga_sheet.png",
        font_size=7.9,
    )


def plot_fs_month_projection_sheet(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Render Sheet 2: Pure table of FS-Month (60-Month / 5-Year 3-Statement Model:
    Income Statement, Balance Sheet, and Cash Flow Statement showing Year 1 monthly
    transition from M1–M4 Core Build to M5+ Platform Launch plus M24 End Y2 & M60 End Y5).
    """
    col_widths = [0.220, 0.038] + [0.039] * 12 + [0.048, 0.048, 0.054, 0.062, 0.040]
    y1_months = list(res.month_labels[:12])

    rows: list[tuple[str, list[str]]] = [
        ("col_hdr", ["3-STATEMENT MONTHLY MODEL (60M HORIZON)", "Unit"] + y1_months + ["Dec-28 (M24)", "Dec-31 (M60)", "FY1 Total", "5Y (60M)", "% Rev"]),
    ]

    for _, df_row in res.fs_month_df.iterrows():
        rtype = str(df_row["row_type"])
        label = str(df_row["line_item"])
        unit = str(df_row["unit"])
        code = str(df_row["row_code"])

        if rtype == "header":
            rows.append(("sec_hdr", [label] + [""] * 18))
            continue

        m_cells = []
        for m in y1_months:
            val = float(df_row[m])
            if rtype == "pct":
                m_cells.append(f"{val:.1f}%")
            elif code == "MEM_CNT":
                m_cells.append(f"{val:,.0f}")
            else:
                m_cells.append(f"{val:,.1f}")

        m24_val = float(df_row["Dec-28"])
        m24_str = f"{m24_val:.1f}%" if rtype == "pct" else (f"{m24_val:,.0f}" if code == "MEM_CNT" else f"{m24_val:,.1f}")

        m60_val = float(df_row[res.month_labels[-1]])
        m60_str = f"{m60_val:.1f}%" if rtype == "pct" else (f"{m60_val:,.0f}" if code == "MEM_CNT" else f"{m60_val:,.1f}")

        fy_val = float(df_row["FY1_Total"])
        fy_str = f"{fy_val:.1f}%" if rtype == "pct" else (f"{fy_val:,.0f}" if code == "MEM_CNT" else f"{fy_val:,.1f}")

        tot5y_val = float(df_row["5Y_Total"])
        tot5y_str = f"{tot5y_val:.1f}%" if rtype == "pct" else (f"{tot5y_val:,.0f}" if code == "MEM_CNT" else f"{tot5y_val:,.1f}")

        pct_val = df_row["Pct_of_Rev"]
        pct_str = "-" if pct_val is None or pd.isna(pct_val) else f"{float(pct_val):.1f}%"

        rows.append((rtype, [label, unit] + m_cells + [m24_str, m60_str, fy_str, tot5y_str, pct_str]))

    def _style_s2(r_idx: int, c_idx: int, rstyle: str) -> tuple[str, str, str]:
        if rstyle == "col_hdr":
            return SUBHEADER_BG, FORMULA_BLACK, "bold"
        if rstyle == "subtotal":
            return SUBTOTAL_BG, FORMULA_BLACK, "bold"
        if rstyle == "total":
            return PROFIT_HIGHLIGHT_BG, "#1E40AF", "bold"
        if rstyle == "grand_total":
            return TOTAL_HIGHLIGHT_BG, EMERALD_TEXT, "bold"
        if rstyle == "pct":
            return "#F8FAFC", "#475569", "bold"
        if rstyle == "input_linked":
            return LINKED_BG, (LINKED_PURPLE if 2 <= c_idx <= 15 else FORMULA_DARKGRAY), "bold"
        bg = "#FFFFFF" if r_idx % 2 == 0 else "#F8FAFC"
        if rstyle == "input" and 2 <= c_idx <= 15:
            return bg, INPUT_BLUE, "normal"
        if c_idx >= 16 or rstyle in ("formula", "unit"):
            return bg, FORMULA_DARKGRAY, ("bold" if c_idx == 17 else "normal")
        return bg, FORMULA_BLACK, "normal"

    def _align_s2(c_idx: int, rstyle: str) -> str:
        if c_idx == 0:
            return "left"
        if c_idx == 1 or rstyle == "col_hdr":
            return "center"
        return "right"

    return _render_pure_table(
        rows=rows,
        col_widths=col_widths,
        align_fn=_align_s2,
        style_fn=_style_s2,
        fig_width=23.2,
        row_height_in=0.28,
        out_path=output_dir / "02_fs_month_60m_projection_sheet.png",
        font_size=7.1,
    )


def plot_fs_annual_and_structure_dashboard(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Render Sheet 3: Pure table of FS-Annual 5-Year 3-Statement Model
    (Income Statement, Balance Sheet, and Cash Flow Statement for FY 2027 to FY 2031).
    """
    col_widths = [0.245, 0.042, 0.064, 0.068, 0.068, 0.082, 0.082, 0.082, 0.082, 0.082, 0.088, 0.045]

    rows: list[tuple[str, list[str]]] = [
        ("col_hdr", [
            "3-STATEMENT ANNUAL MODEL (FY 2027 TO FY 2031)",
            "Unit",
            "Case M1",
            "M1 Build",
            "M5 Launch",
            "FY 2027 (Y1)",
            "FY 2028 (Y2)",
            "FY 2029 (Y3)",
            "FY 2030 (Y4)",
            "FY 2031 (Y5)",
            "5Y Total / End",
            "% Rev",
        ]),
    ]

    for _, df_row in res.fs_annual_df.iterrows():
        rtype = str(df_row["row_type"])
        label = str(df_row["line_item"])
        unit = str(df_row["unit"])
        code = str(df_row["row_code"])

        if rtype == "header":
            rows.append(("sec_hdr", [label] + [""] * 11))
            continue

        def _fmt(v: float) -> str:
            if rtype == "pct":
                return f"{v:.2f}%"
            if code == "MEM_CNT":
                return f"{v:,.0f}"
            return f"{v:,.2f}"

        c_m1 = _fmt(float(df_row["Case_M1_Baseline"]))
        m_m1 = _fmt(float(df_row["Model_M1_Build"]))
        m_m5 = _fmt(float(df_row["Model_M5_Launch"]))

        y_cells = [_fmt(float(df_row[y_lbl])) for y_lbl in res.year_labels]
        tot5_str = _fmt(float(df_row["5Y_Total"]))

        pct_val = df_row["Pct_of_Rev"]
        pct_str = "-" if pct_val is None or pd.isna(pct_val) else f"{float(pct_val):.1f}%"

        rows.append((rtype, [label, unit, c_m1, m_m1, m_m5] + y_cells + [tot5_str, pct_str]))

    def _style_s3(r_idx: int, c_idx: int, rstyle: str) -> tuple[str, str, str]:
        if rstyle == "col_hdr":
            return SUBHEADER_BG, FORMULA_BLACK, "bold"
        if rstyle == "subtotal":
            return SUBTOTAL_BG, FORMULA_BLACK, "bold"
        if rstyle == "total":
            return PROFIT_HIGHLIGHT_BG, "#1E40AF", "bold"
        if rstyle == "grand_total":
            return TOTAL_HIGHLIGHT_BG, EMERALD_TEXT, "bold"
        if rstyle == "pct":
            return "#F8FAFC", "#475569", "bold"
        if rstyle == "input_linked":
            return LINKED_BG, (LINKED_PURPLE if c_idx >= 3 else INPUT_BLUE), "bold"
        bg = "#FFFFFF" if r_idx % 2 == 0 else "#F8FAFC"
        if rstyle == "input" and c_idx in (2, 3, 4):
            return bg, INPUT_BLUE, ("bold" if c_idx == 4 else "normal")
        if c_idx >= 5:
            return bg, FORMULA_DARKGRAY, ("bold" if c_idx == 10 else "normal")
        return bg, FORMULA_BLACK, "normal"

    def _align_s3(c_idx: int, rstyle: str) -> str:
        if c_idx == 0:
            return "left"
        if c_idx == 1 or rstyle == "col_hdr":
            return "center"
        return "right"

    return _render_pure_table(
        rows=rows,
        col_widths=col_widths,
        align_fn=_align_s3,
        style_fn=_style_s3,
        fig_width=21.5,
        row_height_in=0.29,
        out_path=output_dir / "03_fs_annual_5y_projection_sheet.png",
        font_size=7.4,
    )


def plot_revenue_costs_profit_comparison_charts(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 4: Double-Bar & Profit-Line Comparison Charts (Baseline vs. Model)
      - Baseline bars dimmed down to grey and low opacity to emphasize our solution
      - Model Revenue plotted in vibrant GREEN above 0; Model Costs plotted in vibrant RED below 0
      - Profit (EBITDA / Net Income) plotted as solid lines with 'o' markers (no profit number tooltips;
        dimmed grey line + 'o' markers for Baseline vs. bold line + 'o' markers for Model)
      - Panel A: 5-Year Annual Comparison (FY 2027 to FY 2031)
      - Panel B: Monthly Trajectory Across Year 1 & Year 2 (Months 1–24: Jan-27 to Dec-28, showing M1–M4 Build, M5 Launch, and smooth ramp to 1,500 members by Dec-28)
    """
    fig, (ax_ann, ax_mon) = plt.subplots(2, 1, figsize=(19.0, 13.4), dpi=240)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(hspace=0.32)

    BASE_BAR_GREY = "#CBD5E1"
    BASE_BAR_EDGE = "#94A3B8"
    BASE_BAR_ALPHA = 0.42
    BASE_TEXT_GREY = "#94A3B8"

    MOD_REV_GREEN = "#10B981"
    MOD_REV_EDGE = "#064E3B"

    MOD_COST_RED = "#EF4444"
    MOD_COST_EDGE = "#7F1D1D"

    BASE_PROFIT_LINE = "#94A3B8"
    BASE_PROFIT_ALPHA = 0.50
    MOD_PROFIT_LINE = "#0F172A"
    MOD_PROFIT_MARKER = "#10B981"

    # =========================================================================
    # PANEL A: 5-Year Annual Double-Bar & Profit-Line Chart (FY 2027 – FY 2031)
    # =========================================================================
    ax_ann.set_facecolor("#F8FAFC")
    mc = res.mc
    n_years = len(res.year_labels)
    x_yr = np.arange(n_years)
    bar_w = 0.34

    base_ann_rev = mc.base_annual_rev_mean
    mod_ann_rev = res.annual_revenue_arr

    base_ann_cost = -mc.base_annual_costs_mean
    mod_ann_cost = -res.annual_total_costs_arr

    base_ann_profit = mc.base_annual_ebitda_mean
    mod_ann_profit = res.annual_ebitda_arr

    base_rev_yerr = np.vstack([
        np.maximum(0.0, base_ann_rev - mc.base_annual_rev_p05),
        np.maximum(0.0, mc.base_annual_rev_p95 - base_ann_rev),
    ])
    mod_rev_yerr = np.vstack([
        np.maximum(0.0, mod_ann_rev - mc.annual_rev_p05),
        np.maximum(0.0, mc.annual_rev_p95 - mod_ann_rev),
    ])
    base_cost_yerr = np.vstack([
        np.maximum(0.0, mc.base_annual_costs_p95 - mc.base_annual_costs_mean),
        np.maximum(0.0, mc.base_annual_costs_mean - mc.base_annual_costs_p05),
    ])
    mod_cost_yerr = np.vstack([
        np.maximum(0.0, mc.annual_costs_p95 - res.annual_total_costs_arr),
        np.maximum(0.0, res.annual_total_costs_arr - mc.annual_costs_p05),
    ])

    ax_ann.bar(
        x_yr - bar_w / 2,
        base_ann_rev,
        width=bar_w,
        yerr=base_rev_yerr,
        capsize=3.5,
        error_kw=dict(ecolor="#64748B", lw=1.0, alpha=0.65),
        color=BASE_BAR_GREY,
        edgecolor=BASE_BAR_EDGE,
        alpha=BASE_BAR_ALPHA,
        linewidth=0.9,
        label=f"Baseline Stochastic Revenue (~{base_ann_rev.mean():,.0f}M / yr, P05-P95)",
        zorder=2,
    )
    ax_ann.bar(
        x_yr + bar_w / 2,
        mod_ann_rev,
        width=bar_w,
        yerr=mod_rev_yerr,
        capsize=3.5,
        error_kw=dict(ecolor="#064E3B", lw=1.1, alpha=0.80),
        color=MOD_REV_GREEN,
        edgecolor=MOD_REV_EDGE,
        linewidth=1.1,
        label="Model Revenue: AdaptivePricing + PICK! Social + 20 Courts (Green > 0, P05-P95)",
        zorder=3,
    )

    ax_ann.bar(
        x_yr - bar_w / 2,
        base_ann_cost,
        width=bar_w,
        yerr=base_cost_yerr,
        capsize=3.5,
        error_kw=dict(ecolor="#64748B", lw=1.0, alpha=0.65),
        color=BASE_BAR_GREY,
        edgecolor=BASE_BAR_EDGE,
        alpha=BASE_BAR_ALPHA,
        linewidth=0.9,
        label=f"Baseline Stochastic Costs (~{base_ann_cost.mean():,.0f}M / yr, P05-P95)",
        zorder=2,
    )
    ax_ann.bar(
        x_yr + bar_w / 2,
        mod_ann_cost,
        width=bar_w,
        yerr=mod_cost_yerr,
        capsize=3.5,
        error_kw=dict(ecolor="#7F1D1D", lw=1.1, alpha=0.80),
        color=MOD_COST_RED,
        edgecolor=MOD_COST_EDGE,
        linewidth=1.1,
        label="Model Costs: COGS + OpEx/SG&A + Platform Build/OpEx (Red < 0, P05-P95)",
        zorder=3,
    )

    # Profit Statistical Range Bands (P05-P95) + Solid '-' lines with 'o' markers
    ax_ann.fill_between(
        x_yr - bar_w / 2,
        mc.base_annual_ebitda_p05,
        mc.base_annual_ebitda_p95,
        color=BASE_PROFIT_LINE,
        alpha=0.18,
        zorder=4,
    )
    ax_ann.fill_between(
        x_yr + bar_w / 2,
        mc.annual_ebitda_p05,
        mc.annual_ebitda_p95,
        color="#F59E0B",
        alpha=0.22,
        zorder=4,
    )
    ax_ann.plot(
        x_yr - bar_w / 2,
        base_ann_profit,
        linestyle="-",
        linewidth=2.0,
        color=BASE_PROFIT_LINE,
        alpha=BASE_PROFIT_ALPHA,
        marker="o",
        markersize=7.0,
        markerfacecolor="#E2E8F0",
        markeredgecolor=BASE_PROFIT_LINE,
        markeredgewidth=1.6,
        label=f"Baseline Operating Profit Line (~+{base_ann_profit.mean():,.0f}M / yr, P05-P95 Band)",
        zorder=5,
    )
    ax_ann.plot(
        x_yr + bar_w / 2,
        mod_ann_profit,
        linestyle="-",
        linewidth=2.8,
        color=MOD_PROFIT_LINE,
        marker="o",
        markersize=8.2,
        markerfacecolor=MOD_PROFIT_MARKER,
        markeredgecolor=MOD_PROFIT_LINE,
        markeredgewidth=1.8,
        label=f"Model Operating Profit Line (Y3 = +{mod_ann_profit[2]:,.0f}M vs 3,500M Investor Target)",
        zorder=6,
    )

    ax_ann.axhline(0, color="#0F172A", linewidth=1.4, zorder=4)

    for i in range(n_years):
        ax_ann.text(
            i - bar_w / 2,
            mc.base_annual_rev_p95[i] + 220,
            f"+{base_ann_rev[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=7.9,
            fontweight="normal",
            color=BASE_TEXT_GREY,
            alpha=0.80,
        )
        ax_ann.text(
            i + bar_w / 2,
            mc.annual_rev_p95[i] + 220,
            f"+{mod_ann_rev[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#065F46",
        )
        ax_ann.text(
            i - bar_w / 2,
            -mc.base_annual_costs_p95[i] - 280,
            f"{base_ann_cost[i]:,.0f}M",
            ha="center",
            va="top",
            fontsize=7.9,
            fontweight="normal",
            color=BASE_TEXT_GREY,
            alpha=0.80,
        )
        ax_ann.text(
            i + bar_w / 2,
            -mc.annual_costs_p95[i] - 280,
            f"{mod_ann_cost[i]:,.0f}M",
            ha="center",
            va="top",
            fontsize=8.3,
            fontweight="bold",
            color="#7F1D1D",
        )

    ax_ann.set_xticks(x_yr)
    ax_ann.set_xticklabels(res.year_labels, fontsize=9.8, fontweight="bold", color="#0F172A")
    ax_ann.set_ylabel("Million VND / Year", fontsize=10.2, fontweight="bold", color="#0F172A")
    ax_ann.set_ylim(-float(np.max(mc.annual_costs_p95)) * 1.25, float(np.max(mc.annual_rev_p95)) * 1.25)
    ax_ann.grid(axis="y", linestyle="--", alpha=0.65, color="#CBD5E1", zorder=1)
    ax_ann.legend(loc="upper left", ncol=2, fontsize=8.1, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_ann.set_title(
        "Panel A: 5-Year Annual Projections (FY 2027 – FY 2031) — Model Revenue (Green > 0) & Costs (Red < 0) vs. Stochastic Baseline & P05-P95 Ranges",
        fontsize=11.8,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )

    # =========================================================================
    # PANEL B: Monthly Trajectory Across Years 1 & 2 (Months 1–24: Jan-27 to Dec-28)
    # =========================================================================
    ax_mon.set_facecolor("#F8FAFC")
    m24_labels = list(res.month_labels[:24])
    n_m = len(m24_labels)
    x_m = np.arange(n_m)
    bar_wm = 0.38

    base_mon_rev = mc.base_total_rev_mean[:24]
    mod_mon_rev = res.monthly_total_rev_arr[:24]

    base_mon_cost = -mc.base_total_costs_mean[:24]
    mod_mon_cost = -res.monthly_total_costs_arr[:24]

    base_mon_profit = mc.base_ebitda_mean[:24]
    mod_mon_profit = res.monthly_ebitda_arr[:24]

    base_m_rev_yerr = np.vstack([
        np.maximum(0.0, base_mon_rev - mc.base_total_rev_p05[:24]),
        np.maximum(0.0, mc.base_total_rev_p95[:24] - base_mon_rev),
    ])
    mod_m_rev_yerr = np.vstack([
        np.maximum(0.0, mod_mon_rev - mc.total_rev_p05[:24]),
        np.maximum(0.0, mc.total_rev_p95[:24] - mod_mon_rev),
    ])

    ax_mon.bar(
        x_m - bar_wm / 2,
        base_mon_rev,
        width=bar_wm,
        yerr=base_m_rev_yerr,
        capsize=2.0,
        error_kw=dict(ecolor="#64748B", lw=0.75, alpha=0.55),
        color=BASE_BAR_GREY,
        edgecolor=BASE_BAR_EDGE,
        alpha=BASE_BAR_ALPHA,
        linewidth=0.7,
        label=f"Baseline Stochastic Monthly Revenue (~+{base_mon_rev.mean():.1f}M / mo, P05-P95)",
        zorder=2,
    )
    ax_mon.bar(
        x_m + bar_wm / 2,
        mod_mon_rev,
        width=bar_wm,
        yerr=mod_m_rev_yerr,
        capsize=2.0,
        error_kw=dict(ecolor="#064E3B", lw=0.80, alpha=0.70),
        color=MOD_REV_GREEN,
        edgecolor=MOD_REV_EDGE,
        linewidth=0.85,
        label="Model Monthly Revenue (M1–M4 Build; M5 App Launch; M14 20-Court Open, P05-P95)",
        zorder=3,
    )

    ax_mon.bar(
        x_m - bar_wm / 2,
        base_mon_cost,
        width=bar_wm,
        color=BASE_BAR_GREY,
        edgecolor=BASE_BAR_EDGE,
        alpha=BASE_BAR_ALPHA,
        linewidth=0.7,
        label=f"Baseline Stochastic Monthly Costs (~{base_mon_cost.mean():.1f}M / mo)",
        zorder=2,
    )
    ax_mon.bar(
        x_m + bar_wm / 2,
        mod_mon_cost,
        width=bar_wm,
        color=MOD_COST_RED,
        edgecolor=MOD_COST_EDGE,
        linewidth=0.85,
        label="Model Monthly Costs (M1–M6 Leveled Build + M5+ Platform + M10+ Venue 2)",
        zorder=3,
    )

    # Profit Statistical Bands (P05-P95) + Solid '-' lines with 'o' markers
    ax_mon.fill_between(
        x_m - bar_wm / 2,
        mc.base_ebitda_p05[:24],
        mc.base_ebitda_p95[:24],
        color=BASE_PROFIT_LINE,
        alpha=0.18,
        zorder=4,
    )
    ax_mon.fill_between(
        x_m + bar_wm / 2,
        mc.ebitda_p05[:24],
        mc.ebitda_p95[:24],
        color="#F59E0B",
        alpha=0.22,
        zorder=4,
    )
    ax_mon.plot(
        x_m - bar_wm / 2,
        base_mon_profit,
        linestyle="-",
        linewidth=1.8,
        color=BASE_PROFIT_LINE,
        alpha=BASE_PROFIT_ALPHA,
        marker="o",
        markersize=5.2,
        markerfacecolor="#E2E8F0",
        markeredgecolor=BASE_PROFIT_LINE,
        markeredgewidth=1.3,
        label=f"Baseline Stochastic Monthly Profit (~+{base_mon_profit.mean():.1f}M / mo, P05-P95 Band)",
        zorder=5,
    )
    ax_mon.plot(
        x_m + bar_wm / 2,
        mod_mon_profit,
        linestyle="-",
        linewidth=2.4,
        color=MOD_PROFIT_LINE,
        marker="o",
        markersize=6.2,
        markerfacecolor=MOD_PROFIT_MARKER,
        markeredgecolor=MOD_PROFIT_LINE,
        markeredgewidth=1.5,
        label=f"Model Monthly Operating Profit (Ramps to +{mod_mon_profit[-1]:,.0f}M/mo by M24, P05-P95 Band)",
        zorder=6,
    )

    ax_mon.axhline(0, color="#0F172A", linewidth=1.4, zorder=4)
    ax_mon.axvline(3.5, color="#DC2626", linewidth=1.6, linestyle="--", zorder=4)
    ax_mon.axvline(5.5, color="#0284C7", linewidth=1.4, linestyle=":", zorder=4)

    for i in range(0, n_m, 2):
        ax_mon.text(
            i + bar_wm / 2,
            mod_mon_rev[i] + 22,
            f"{mod_mon_rev[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="#065F46",
        )
        ax_mon.text(
            i + bar_wm / 2,
            mod_mon_cost[i] - 25,
            f"{mod_mon_cost[i]:,.0f}M",
            ha="center",
            va="top",
            fontsize=6.8,
            fontweight="bold",
            color="#7F1D1D",
        )

    ax_mon.set_xticks(x_m)
    ax_mon.set_xticklabels(m24_labels, fontsize=7.8, fontweight="bold", color="#0F172A", rotation=25)
    ax_mon.set_ylabel("Million VND / Month", fontsize=10.2, fontweight="bold", color="#0F172A")
    ax_mon.set_ylim(float(np.min(mod_mon_cost)) * 1.26, float(np.max(mod_mon_rev)) * 1.26)
    ax_mon.grid(axis="y", linestyle="--", alpha=0.65, color="#CBD5E1", zorder=1)
    ax_mon.legend(loc="upper left", ncol=2, fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_mon.set_title(
        "Panel B: Monthly Build, Launch & Growth Trajectory (Months 1–24: Jan-27 to Dec-28) — Smooth Subscription Ramp Starting Month 5 (May-27)",
        fontsize=11.8,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )

    out_path = output_dir / "04_revenue_costs_profit_comparison_charts.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path


def plot_membership_and_monte_carlo_distributions(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 5 (05_membership_and_monte_carlo_distributions.png):
      - Panel A (Top Left): Smooth Active Paying Subscribers Trajectory S_t (Months 1–60: Jan-27 to Dec-31)
        with 40 Monte Carlo simulation lines, P05–P95 confidence band, Train vs. Held-Out Test mean,
        and Case Page 3 End-of-Year-2 benchmark (1,500 paying members at M24 / Dec-28).
      - Panel B (Top Right): Monthly Membership Revenue & Total Revenue Trajectory (M VND/month)
        with P05–P95 band and -30% lower-demand stress test comparison.
      - Panel C (Bottom Left): Monthly Subscriber Flow Dynamics (Monthly New Subscribers Acquired vs.
        Churned Subscribers & Net Monthly Member Additions from Month 5 onward).
      - Panel D (Bottom Right): Annual Operating Profit (EBITDA) Across FY 2027–FY 2031 (Base Plan P05–P95
        vs. -30% Lower Demand Stress Test vs. Investor Minh Tran's 3,500M VND Year 3 Target).
    """
    mc = res.mc
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(20.0, 12.6), dpi=240)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(wspace=0.20, hspace=0.30)

    x_m = np.arange(1, 61)
    tick_idx = [0, 5, 11, 23, 35, 47, 59]
    tick_lbls = [
        "Jan-27\n(M1)",
        "Jun-27\n(M6)",
        "Dec-27\n(End Y1)",
        "Dec-28\n(End Y2)",
        "Dec-29\n(End Y3)",
        "Dec-30\n(End Y4)",
        "Dec-31\n(End Y5)",
    ]

    # -------------------------------------------------------------------------
    # PANEL A: Smooth Active Paying Subscribers S_t (N=300 Monte Carlo Trajectories)
    # -------------------------------------------------------------------------
    ax_a.set_facecolor("#F8FAFC")
    for path in mc.sample_subscribers_paths:
        ax_a.plot(x_m, path, color="#3B82F6", alpha=0.14, linewidth=0.8, zorder=2)

    ax_a.fill_between(
        x_m,
        mc.active_subscribers_p05,
        mc.active_subscribers_p95,
        color="#2563EB",
        alpha=0.22,
        label="90% Monte Carlo Band (P05–P95)",
        zorder=3,
    )
    ax_a.plot(
        x_m,
        mc.Active_subscribers_train_mean,
        color="#1D4ED8",
        linewidth=2.5,
        label=f"Train Split Mean (n={mc.n_train} sims)",
        zorder=5,
    )
    ax_a.plot(
        x_m,
        mc.active_subscribers_test_mean,
        color="#0F172A",
        linestyle="--",
        linewidth=2.0,
        label=f"Held-Out Test Split Mean (n={mc.n_test} sims)",
        zorder=6,
    )
    ax_a.plot(
        x_m,
        mc.stress_active_subscribers_mean,
        color="#DC2626",
        linestyle="-.",
        linewidth=2.0,
        label="Stress Scenario (-30% Demand) Subscribers",
        zorder=4,
    )

    ax_a.axvline(5, color="#059669", linewidth=1.6, linestyle="--", zorder=4)
    ax_a.scatter([24], [mc.m24_end_y2_subscribers], color="#059669", edgecolor="#0F172A", s=75, zorder=7)
    ax_a.annotate(
        f"End of Year 2 (Dec-28):\n{mc.m24_end_y2_subscribers:,.0f} Paying Members\n({mc.m24_end_y2_membership_rev_mvnd:,.1f}M VND/mo)",
        xy=(24, mc.m24_end_y2_subscribers),
        xytext=(27.5, mc.m24_end_y2_subscribers - 360),
        fontsize=8.0,
        fontweight="bold",
        color="#065F46",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.0),
        arrowprops=dict(arrowstyle="-", color="#059669", linewidth=1.2),
        zorder=8,
    )

    ax_a.set_xticks([t + 1 for t in tick_idx])
    ax_a.set_xticklabels(tick_lbls, fontsize=7.8, fontweight="bold")
    ax_a.set_ylim(-50, float(np.max(mc.active_subscribers_p95)) * 1.16)
    ax_a.set_ylabel("Active Paying Members (Subscribers)", fontsize=9.5, fontweight="bold")
    ax_a.set_title(
        "Panel A: Smooth Monthly Subscriber Cohort Evolution S_t (Starts Month 5 / May-27)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_a.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    ax_a.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # PANEL B: Total Monthly Revenue & Membership Revenue Trajectory (M VND/mo)
    # -------------------------------------------------------------------------
    ax_b.set_facecolor("#F8FAFC")
    for path in mc.sample_total_rev_paths:
        ax_b.plot(x_m, path, color="#10B981", alpha=0.12, linewidth=0.8, zorder=2)
    for b_path in mc.sample_base_rev_paths:
        ax_b.plot(x_m, b_path, color="#94A3B8", alpha=0.10, linewidth=0.7, zorder=1)

    ax_b.fill_between(
        x_m,
        mc.total_rev_p05,
        mc.total_rev_p95,
        color="#10B981",
        alpha=0.22,
        label="Total Monthly Revenue P05–P95 Band",
        zorder=3,
    )
    ax_b.plot(
        x_m,
        mc.total_rev_mean,
        color="#047857",
        linewidth=2.6,
        label="Total Monthly Revenue Mean (Court + Ancillary + Subscriptions)",
        zorder=5,
    )
    ax_b.fill_between(
        x_m,
        mc.membership_rev_p05,
        mc.membership_rev_p95,
        color="#6D28D9",
        alpha=0.18,
        zorder=3,
    )
    ax_b.plot(
        x_m,
        mc.membership_rev_mean,
        color="#6D28D9",
        linewidth=2.3,
        label="Membership Subscription Revenue (@ 399k VND/member/mo)",
        zorder=5,
    )
    ax_b.plot(
        x_m,
        mc.stress_total_rev_mean,
        color="#DC2626",
        linestyle="-.",
        linewidth=1.9,
        label="Stress Scenario (-30% Demand) Total Revenue",
        zorder=4,
    )
    ax_b.fill_between(
        x_m,
        mc.base_total_rev_p05,
        mc.base_total_rev_p95,
        color="#64748B",
        alpha=0.20,
        label="Baseline Stochastic Revenue P05–P95 Band",
        zorder=2,
    )
    ax_b.plot(
        x_m,
        mc.base_total_rev_mean,
        color="#64748B",
        linestyle="-",
        linewidth=1.7,
        alpha=0.80,
        label=f"Baseline Stochastic Revenue (~{mc.base_total_rev_mean.mean():.1f}M/mo)",
        zorder=3,
    )

    ax_b.set_xticks([t + 1 for t in tick_idx])
    ax_b.set_xticklabels(tick_lbls, fontsize=7.8, fontweight="bold")
    ax_b.set_ylim(0, float(np.max(mc.total_rev_p95)) * 1.15)
    ax_b.set_ylabel("Monthly Revenue (M VND)", fontsize=9.5, fontweight="bold")
    ax_b.set_title(
        "Panel B: Monthly Total Revenue & Membership Subscription Revenue Trajectory (60 Months)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_b.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    ax_b.legend(loc="upper left", fontsize=7.4, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # PANEL C: Monthly Subscriber Acquisition vs. Monthly Churn Flow Dynamics
    # -------------------------------------------------------------------------
    ax_c.set_facecolor("#F8FAFC")
    net_adds = mc.new_subscribers_mean - mc.churned_subscribers_mean
    ax_c.bar(
        x_m,
        mc.new_subscribers_mean,
        color="#10B981",
        edgecolor="#065F46",
        width=0.68,
        alpha=0.85,
        label="Gross New Subscribers Acquired (+ / mo)",
        zorder=3,
    )
    ax_c.bar(
        x_m,
        -mc.churned_subscribers_mean,
        color="#EF4444",
        edgecolor="#7F1D1D",
        width=0.68,
        alpha=0.85,
        label="Churned Subscribers (- / mo, ~4.2% Monthly Churn)",
        zorder=3,
    )
    ax_c.plot(
        x_m,
        net_adds,
        color="#0F172A",
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        label="Net Monthly Subscriber Additions (New - Churned)",
        zorder=5,
    )
    ax_c.axhline(0, color="#0F172A", linewidth=1.2, zorder=4)

    ax_c.set_xticks([t + 1 for t in tick_idx])
    ax_c.set_xticklabels(tick_lbls, fontsize=7.8, fontweight="bold")
    ax_c.set_ylim(-115, 345)
    ax_c.set_ylabel("Subscribers / Month", fontsize=9.5, fontweight="bold")
    ax_c.set_title(
        "Panel C: Monthly Subscriber Stock-Flow Dynamics (Gross New Sign-Ups vs. Monthly Churn)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_c.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    ax_c.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # PANEL D: Annual Operating Profit (FY27–FY31) vs. Investor Minh Tran 3,500M Target
    # -------------------------------------------------------------------------
    ax_d.set_facecolor("#F8FAFC")
    x_y = np.arange(5)
    w_y = 0.26

    base_y_ebitda = mc.base_annual_ebitda_mean
    mod_y_ebitda = res.annual_ebitda_arr
    stress_y_ebitda = np.array([mc.stress_ebitda_mean[(y * 12):((y + 1) * 12)].sum() for y in range(5)])

    base_y_yerr = np.vstack([
        np.maximum(0.0, base_y_ebitda - mc.base_annual_ebitda_p05),
        np.maximum(0.0, mc.base_annual_ebitda_p95 - base_y_ebitda),
    ])
    mod_y_yerr = np.vstack([
        np.maximum(0.0, mod_y_ebitda - mc.annual_ebitda_p05),
        np.maximum(0.0, mc.annual_ebitda_p95 - mod_y_ebitda),
    ])
    stress_y_yerr = np.vstack([
        np.maximum(0.0, stress_y_ebitda - mc.stress_annual_ebitda_p05),
        np.maximum(0.0, mc.stress_annual_ebitda_p95 - stress_y_ebitda),
    ])

    ax_d.bar(
        x_y - w_y,
        base_y_ebitda,
        width=w_y,
        yerr=base_y_yerr,
        capsize=3.0,
        error_kw=dict(ecolor="#475569", lw=1.0, alpha=0.75),
        color="#CBD5E1",
        edgecolor="#64748B",
        alpha=0.62,
        label=f"Baseline Stochastic EBITDA (~+{base_y_ebitda.mean():,.0f}M / yr, P05-P95)",
        zorder=2,
    )
    ax_d.bar(
        x_y,
        mod_y_ebitda,
        width=w_y,
        yerr=mod_y_yerr,
        capsize=3.0,
        error_kw=dict(ecolor="#064E3B", lw=1.1, alpha=0.85),
        color="#10B981",
        edgecolor="#064E3B",
        linewidth=1.1,
        label="Path C Integrated Model (Mean Annual EBITDA, P05-P95)",
        zorder=4,
    )
    ax_d.bar(
        x_y + w_y,
        stress_y_ebitda,
        width=w_y,
        yerr=stress_y_yerr,
        capsize=3.0,
        error_kw=dict(ecolor="#78350F", lw=1.0, alpha=0.80),
        color="#F59E0B",
        edgecolor="#78350F",
        linewidth=1.0,
        hatch="///",
        label="-30% Lower Demand Stress Test (Case Page 4, P05-P95)",
        zorder=3,
    )

    ax_d.axhline(
        3500.0,
        color="#DC2626",
        linewidth=2.2,
        linestyle="--",
        label="Investor Minh Tran Year 3 Target (3,500M VND / yr)",
        zorder=6,
    )

    for i in range(5):
        ax_d.text(
            x_y[i] - w_y,
            mc.base_annual_ebitda_p95[i] + 150,
            f"{base_y_ebitda[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="normal",
            color="#475569",
        )
        ax_d.text(
            x_y[i],
            mc.annual_ebitda_p95[i] + 180,
            f"{mod_y_ebitda[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#065F46",
        )
        ax_d.text(
            x_y[i] + w_y,
            mc.stress_annual_ebitda_p95[i] + 180 if stress_y_ebitda[i] >= 0 else mc.stress_annual_ebitda_p05[i] - 950,
            f"{stress_y_ebitda[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#92400E",
        )

    ax_d.set_xticks(x_y)
    ax_d.set_xticklabels(res.year_labels, fontsize=8.8, fontweight="bold")
    ax_d.set_ylim(min(-2600.0, float(np.min(mc.stress_annual_ebitda_p05)) * 1.45), float(np.max(mc.annual_ebitda_p95)) * 1.18)
    ax_d.set_ylabel("Annual Operating Profit / EBITDA (M VND / Year)", fontsize=9.5, fontweight="bold")
    ax_d.set_title(
        f"Panel D: Annual Operating Profit vs. Minh Tran 3,500M Y3 Target ({mc.prob_exceed_investor_3500m_y3_pct:.1f}% MC Pass Rate)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_d.grid(axis="y", linestyle="--", alpha=0.6, color="#CBD5E1", zorder=1)
    ax_d.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    fig.suptitle(
        "PICK! Integrated Solution (Path C + Path A + ML Marketing) — Monte Carlo Subscription Cohort Dynamics & Profit Robustness (2027-2031)",
        fontsize=12.6,
        fontweight="bold",
        color="#0F172A",
        y=0.99,
    )

    out_path = output_dir / "05_membership_and_monte_carlo_distributions.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def plot_monthly_cash_balance_and_externals(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Render `06_monthly_cash_balance_and_externals.png`:
      - Panel A: 60-Month Ending Cash Balance Trajectory (Mean, P05-P95 Band, -30% Stress Cash & Stochastic Case Baseline with P05-P95 Band)
      - Panel B: Monthly Cash Flow Components (Operating Cash Flow OCF, Venue 2 Fit-Out CapEx -1,750M/mo in M10-M13, +500M M1 & +7,500M M10 Equity)
      - Panel C: External Macro Pickleball Interest Trend Multiplier I_t (externals.toml) & Legal/Safety Risk Provision
    """
    mc = res.mc
    x_m = np.arange(1, 61)
    tick_idx = list(range(0, 60, 4)) + [59]
    tick_lbls = [res.month_labels[i] for i in tick_idx]

    fig, axes = plt.subplots(3, 1, figsize=(18.5, 13.5), dpi=220, sharex=True)
    fig.subplots_adjust(hspace=0.26, top=0.92, bottom=0.06)

    # Panel A: 60-Month Ending Cash Balance Trajectory
    ax1 = axes[0]
    ax1.set_facecolor("#F8FAFC")
    for path in mc.sample_cash_paths:
        ax1.plot(x_m, path, color="#0284C7", alpha=0.12, linewidth=0.8, zorder=2)
    for b_c_path in mc.sample_base_cash_paths:
        ax1.plot(x_m, b_c_path, color="#94A3B8", alpha=0.10, linewidth=0.7, zorder=1)

    ax1.fill_between(
        x_m,
        mc.ending_cash_p05,
        mc.ending_cash_p95,
        color="#38BDF8",
        alpha=0.24,
        label="Solution Ending Cash P05-P95 Confidence Band (300 Sims)",
        zorder=3,
    )
    ax1.fill_between(
        x_m,
        mc.base_ending_cash_p05,
        mc.base_ending_cash_p95,
        color="#94A3B8",
        alpha=0.22,
        label="Baseline Ending Cash P05-P95 Statistical Band (300 Sims)",
        zorder=2,
    )
    ax1.plot(
        x_m,
        res.monthly_ending_cash_model,
        color="#0369A1",
        linewidth=2.6,
        marker="o",
        markersize=3.8,
        label=f"Solution Ending Cash Balance (M60: {res.ending_cash_60m_model_mvnd:,.1f}M | Min Cash: {res.min_cash_balance_model_mvnd:,.1f}M in {res.min_cash_month_label})",
        zorder=6,
    )
    ax1.plot(
        x_m,
        mc.stress_ending_cash_mean,
        color="#DC2626",
        linestyle="-.",
        linewidth=2.0,
        label=f"-30% Lower Demand Stress Cash Balance (M60: {mc.stress_ending_cash_mean[-1]:,.1f}M VND)",
        zorder=5,
    )
    ax1.plot(
        x_m,
        res.monthly_ending_cash_case,
        color="#64748B",
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        alpha=0.78,
        label=f"Baseline Stochastic Ending Cash (M60: {res.ending_cash_60m_case_mvnd:,.1f}M VND)",
        zorder=4,
    )
    ax1.axvline(10, color="#0F172A", linestyle="--", linewidth=1.3, alpha=0.75)
    ax1.axvline(14, color="#059669", linestyle="--", linewidth=1.4, alpha=0.8)
    ax1.set_title(
        "Panel A: 60-Month Ending Cash Balance Trajectory (Includes +500M M1 & +7,500M M10 Investor Equity + -7,000M Venue 2 CapEx in M10-M13)",
        fontsize=11.2,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax1.set_ylabel("Ending Cash Balance (M VND)", fontsize=9.8, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1")
    ax1.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # Panel B: Monthly Cash Flow Components (OCF, CapEx ICF, Equity FCF, Net Cash Flow)
    ax2 = axes[1]
    ax2.set_facecolor("#F8FAFC")
    ax2.bar(x_m - 0.2, res.monthly_ocf_model, width=0.38, color="#10B981", alpha=0.85, label="Operating Cash Flow (OCF = Net Income + D&A)")
    ax2.bar(x_m + 0.2, res.monthly_icf_capex_model, width=0.38, color="#EF4444", alpha=0.85, label="Venue 2 Fit-Out CapEx Outflow (-1,750M/mo in M10-M13)")
    ax2.bar(x_m, res.monthly_fcf_equity_model, width=0.35, color="#6366F1", alpha=0.75, label="Investor Minh Tran Equity (+500M in M1 & +7,500M in M10)")
    ax2.plot(x_m, res.monthly_net_cash_flow_model, color="#0F172A", linewidth=2.0, marker="o", markersize=3.5, label="Net Monthly Cash Flow (OCF + ICF + FCF)")
    ax2.axhline(0, color="#0F172A", linewidth=1.1)
    ax2.set_title(
        "Panel B: Monthly Cash Flow Waterfall Components (OCF, Evenly Split Venue 2 Fit-Out CapEx & Investor Equity)",
        fontsize=11.2,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax2.set_ylabel("Monthly Cash Flow (M VND / Mo)", fontsize=9.8, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1")
    ax2.legend(loc="upper right", ncol=2, fontsize=8.5, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # Panel C: External Pickleball Interest Trend Multiplier I_t & Legal/Ward Noise Compliance Provision (externals.toml)
    ax3 = axes[2]
    ax3.set_facecolor("#F8FAFC")
    ax3_twin = ax3.twinx()

    ax3.fill_between(x_m, mc.trend_index_p05, mc.trend_index_p95, color="#F59E0B", alpha=0.22, label="Pickleball Interest Trend Index I_t (P05-P95 Band)")
    ax3.plot(x_m, mc.trend_index_mean, color="#D97706", linewidth=2.4, marker="o", markersize=3.5, label="Cumulative Pickleball Interest Trend Multiplier I_t (externals.toml)")
    ax3.axhline(1.0, color="#64748B", linestyle=":", linewidth=1.3)

    ax3_twin.plot(
        x_m,
        res.monthly_external_legal_safety_sga_arr,
        color="#7C3AED",
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=3.2,
        label="Monthly Legal/Ward Noise & U-18 Safety Cost (M VND/mo)",
    )

    ax3.set_xticks([t + 1 for t in tick_idx])
    ax3.set_xticklabels(tick_lbls, fontsize=8.5, fontweight="bold")
    ax3.set_ylim(0.65, 1.35)
    ax3_twin.set_ylim(0, 22.0)
    ax3.set_ylabel("Macro Pickleball Trend Index (I_t)", fontsize=9.8, fontweight="bold", color="#B45309")
    ax3_twin.set_ylabel("Legal/Noise & U-18 Safety (M VND/mo)", fontsize=9.5, fontweight="bold", color="#6D28D9")
    ax3.set_xlabel("60-Month Projection Horizon (Jan 2027 - Dec 2031)", fontsize=10.0, fontweight="bold")
    ax3.set_title(
        "Panel C: External Macro Pickleball Interest Trend Multiplier I_t & Ward Noise/Safety Compliance Cost (externals.toml)",
        fontsize=11.2,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax3.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2, fontsize=8.5, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    fig.suptitle(
        "PICK! 60-Month Cash Balance Evolution, Expansion Financing & External Risk Factors (Jan 2027 - Dec 2031)",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )

    out_path = output_dir / "06_monthly_cash_balance_and_externals.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path
