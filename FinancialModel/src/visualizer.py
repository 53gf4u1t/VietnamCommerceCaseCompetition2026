"""Clean, tight-cropped 3-statement spreadsheet table renderer, double-bar comparison charts, and Monte Carlo cohort fan charts."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from .config import FinancialModelConfig
from .model import FinancialModelResults
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

EXCEL_GRID_BORDER = "#313244"
SECTION_BANNER = "#162038"
TOP_COL_HEADER_BG = "#2b3048"
SUBSECTION_BANNER_BG = "#262b40"
SUBHEADER_BG = "#313244"
SUBTOTAL_BG = "#252638"
TOTAL_HIGHLIGHT_BG = "#1f332c"
PROFIT_HIGHLIGHT_BG = "#222c44"
LINKED_BG = "#292338"

INPUT_BLUE = "#89b4fa"
LINKED_PURPLE = "#cba6f7"
FORMULA_BLACK = "#cdd6f4"
FORMULA_DARKGRAY = "#bac2de"
MUTED_SLATE = "#a6adc8"
EMERALD_TEXT = "#a6e3a1"
BANNER_WHITE = "#f5f7ff"


def _row_weight(rstyle: str) -> float:
    """Return proportional row height multiplier by hierarchical row style."""
    if rstyle == "stmt_gap":
        return 0.48
    if rstyle == "sec_hdr":
        return 1.38
    if rstyle == "col_hdr":
        return 1.24
    if rstyle == "sub_hdr":
        return 1.08
    if rstyle in ("total", "grand_total"):
        return 1.18
    if rstyle == "subtotal":
        return 1.08
    return 1.00


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
    Render a hierarchical financial table with variable row heights, visual gaps
    between major statements (`stmt_gap`), high-contrast executive banners (`sec_hdr`),
    subsection group headers (`sub_hdr`), and indented child line items.
    """
    weights = [_row_weight(rstyle) for rstyle, _ in rows]
    total_weight = sum(weights)
    fig_height = max(2.0, total_weight * row_height_in)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("#1e1e2e")

    total_w = sum(col_widths)
    ax.set_xlim(0.0, total_w)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    curr_top_y = 1.0
    for r_idx, ((rstyle, cells), w_r) in enumerate(zip(rows, weights)):
        row_h = w_r / total_weight
        ry = curr_top_y - row_h
        curr_top_y = ry

        if rstyle == "stmt_gap":
            # Clean breathing-room gap between statements/sections
            ax.add_patch(
                patches.Rectangle(
                    (0.0, ry),
                    total_w,
                    row_h,
                    facecolor="#1e1e2e",
                    edgecolor="none",
                    linewidth=0.0,
                )
            )
            continue

        if rstyle == "sec_hdr":
            ax.add_patch(
                patches.Rectangle(
                    (0.0, ry),
                    total_w,
                    row_h,
                    facecolor=SECTION_BANNER,
                    edgecolor="#45475a",
                    linewidth=1.1,
                )
            )
            # Bold left accent strip on every major statement banner
            ax.add_patch(
                patches.Rectangle(
                    (0.0, ry),
                    0.0045 * total_w,
                    row_h,
                    facecolor="#89b4fa",
                    edgecolor="none",
                )
            )
            ax.text(
                0.007 * total_w,
                ry + row_h * 0.5,
                cells[0],
                ha="left",
                va="center",
                fontsize=font_size + 0.8,
                fontweight="bold",
                color=BANNER_WHITE,
            )
            # If period column labels are provided in cells[1:], render them on the banner
            if any(c.strip() for c in cells[1:]):
                cx_hdr = col_widths[0]
                for c_idx in range(1, len(col_widths)):
                    cw = col_widths[c_idx]
                    ctext = cells[c_idx] if c_idx < len(cells) else ""
                    if ctext:
                        ha_hdr = "center" if c_idx == 1 else "right"
                        tx_hdr = (cx_hdr + cw * 0.5) if ha_hdr == "center" else (cx_hdr + cw - 0.005 * total_w)
                        ax.text(
                            tx_hdr,
                            ry + row_h * 0.5,
                            ctext,
                            ha=ha_hdr,
                            va="center",
                            fontsize=font_size - 0.3,
                            fontweight="bold",
                            color=BANNER_WHITE,
                        )
                    cx_hdr += cw
            continue

        if rstyle == "sub_hdr":
            ax.add_patch(
                patches.Rectangle(
                    (0.0, ry),
                    total_w,
                    row_h,
                    facecolor=SUBSECTION_BANNER_BG,
                    edgecolor="#45475a",
                    linewidth=0.75,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (0.0045 * total_w, ry),
                    0.0032 * total_w,
                    row_h,
                    facecolor="#cba6f7",
                    edgecolor="none",
                )
            )
            ax.text(
                0.011 * total_w,
                ry + row_h * 0.5,
                cells[0],
                ha="left",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color=FORMULA_BLACK,
            )
            continue

        cx = 0.0
        for c_idx, (cw, ctext) in enumerate(zip(col_widths, cells)):
            bg, fg, fw = style_fn(r_idx, c_idx, rstyle)
            is_indented_cell = c_idx in (0, 1) and ctext.startswith("  ")
            is_deep_submetric = c_idx == 0 and (
                rstyle == "pct" or "Check (" in ctext or "Memo:" in ctext
            )

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

            # Left accent strip on total and grand_total rows in column 0
            if c_idx == 0 and rstyle in ("total", "grand_total"):
                strip_col = "#a6e3a1" if rstyle == "grand_total" else "#89b4fa"
                ax.add_patch(
                    patches.Rectangle(
                        (0.0, ry),
                        0.0042 * total_w,
                        row_h,
                        facecolor=strip_col,
                        edgecolor="none",
                    )
                )
            elif is_deep_submetric:
                ax.plot(
                    [cx + 0.022 * total_w, cx + 0.022 * total_w],
                    [ry + row_h * 0.15, ry + row_h * 0.85],
                    color="#585b70",
                    linewidth=0.9,
                )
                ax.plot(
                    [cx + 0.022 * total_w, cx + 0.027 * total_w],
                    [ry + row_h * 0.50, ry + row_h * 0.50],
                    color="#585b70",
                    linewidth=0.9,
                )
            # Clear stepped vertical indent guide for child line items
            elif is_indented_cell and rstyle not in ("col_hdr", "subtotal", "total", "grand_total"):
                guide_offset = 0.015 * total_w if c_idx == 0 else 0.009 * total_w
                ax.plot(
                    [cx + guide_offset, cx + guide_offset],
                    [ry + row_h * 0.12, ry + row_h * 0.88],
                    color="#585b70",
                    linewidth=0.95,
                )

            if rstyle in ("subtotal", "total", "grand_total"):
                ax.plot([cx, cx + cw], [ry + row_h, ry + row_h], color="#45475a", linewidth=1.15)
                if rstyle in ("total", "grand_total"):
                    rule_col = "#a6e3a1" if rstyle == "grand_total" else "#89b4fa"
                    ax.plot([cx, cx + cw], [ry + row_h * 0.10, ry + row_h * 0.10], color=rule_col, linewidth=1.2)
                    ax.plot([cx, cx + cw], [ry, ry], color=rule_col, linewidth=1.4)
                else:
                    ax.plot([cx, cx + cw], [ry, ry], color="#45475a", linewidth=1.15)

            ha = align_fn(c_idx, rstyle)
            disp_text = ctext.lstrip(" ") if (c_idx in (0, 1) and ha == "left") else ctext
            if ha == "right":
                tx = cx + cw - 0.005 * total_w
            elif ha == "center":
                tx = cx + cw * 0.5
            else:
                if c_idx == 0:
                    if is_deep_submetric:
                        tx = cx + 0.030 * total_w
                    elif is_indented_cell:
                        tx = cx + 0.021 * total_w
                    elif rstyle == "subtotal":
                        tx = cx + 0.013 * total_w
                    elif rstyle in ("total", "grand_total"):
                        tx = cx + 0.008 * total_w
                    else:
                        tx = cx + 0.007 * total_w
                elif c_idx == 1 and is_indented_cell:
                    tx = cx + 0.014 * total_w
                else:
                    tx = cx + 0.006 * total_w

            f_sz = font_size + 0.25 if rstyle in ("total", "grand_total", "col_hdr") else font_size
            ax.text(
                tx,
                ry + row_h * 0.5,
                disp_text,
                ha=ha,
                va="center",
                fontsize=f_sz,
                fontweight=fw,
                color=fg,
            )
            cx += cw

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def _line_item_source_label(li) -> str:
    """Return a clean, client-facing Source label for each Income Statement line item."""
    if "D&A" in li.template_category:
        return "MOF Circular 45/2013"
    if "CIT" in li.template_category:
        return "Vietnam CIT Law (20%)"
    if li.section == "SUMMARY":
        return "Calculated Subtotal"
    if "Venue 1 (" in li.line_item and "Court hire" in li.line_item:
        return "Adaptive Pricing Model"
    if "Venue 2 (" in li.line_item and "Court hire" in li.line_item:
        return "Adaptive Pricing & Venue 2"
    if "PICK! Social subscriptions" in li.line_item:
        return "Case Exhibit 3 & Assumptions"
    if li.section == "REVENUE":
        return "Case Exhibit 1 & Segment Model"
    if "Coach fees" in li.line_item or "Food and drink supplies" in li.line_item:
        return "Case Exhibit 1 Ratios"
    if "Payment gateway" in li.line_item:
        return "Case Exhibit 1 & Assumptions"
    if "Partner courts" in li.line_item:
        return "Case Exhibit 3 & Event Plan"
    if "Venue 1 lease" in li.line_item or "Booking system" in li.line_item:
        return "Case Exhibit 1"
    if "Venue 2 lease" in li.line_item:
        return "Venue 2 Lease Terms"
    if "Marketing" in li.line_item:
        return "Case Exhibit 1 & Marketing Model"
    if "PICK! Social" in li.line_item:
        return "Platform Build Schedule"
    if "Ward noise" in li.line_item:
        return "Risk & Compliance Assumptions"
    return "Case Exhibit 1 & Venue 2 Plan"


_THREE_STMT_SUBHEADERS: dict[str, str] = {
    "REV_1A": "REVENUE STREAMS (COURT HIRE, PICK! SOCIAL & ANCILLARY)",
    "COGS_1": "DIRECT COST OF GOODS SOLD (VARIABLE COGS)",
    "OPEX_1A": "FACILITY OPERATING EXPENSES (FIXED & SEMI-FIXED OPEX)",
    "SGA_1": "SELLING, GENERAL, ADMIN, PLATFORM & COMPLIANCE (SG&A)",
    "DA_EXP": "DEPRECIATION (D&A), TAXABLE INCOME (EBT) & CORPORATE INCOME TAX",
    "BS_CASH": "ASSETS - CURRENT ASSETS (CASH, INVENTORY & LEASE DEPOSIT)",
    "BS_PPE": "ASSETS - LONG-TERM (FIXED) ASSETS (PP&E NET OF D&A)",
    "BS_AP": "LIABILITIES - CURRENT OPERATING & TAX PAYABLES",
    "BS_CAP": "OWNER'S EQUITY - CONTRIBUTED CAPITAL & RETAINED EARNINGS",
    "CF_START": "OPERATING ACTIVITIES (NET INCOME, D&A & WORKING CAPITAL)",
    "CF_CAPEX": "INVESTING ACTIVITIES (VENUE 2 TURNKEY FIT-OUT CAPEX)",
    "CF_EQ": "FINANCING ACTIVITIES & ENDING CASH RESERVES",
    "TX_EBT": "CORPORATE INCOME TAX SCHEDULE (VIETNAM CIT LAW 67/2025/QH15 @ 20.0%)",
    "TX_VAT_OUT": "VALUE-ADDED TAX SCHEDULE (8.0% OUTPUT VAT VS. 10.0% INPUT VAT - LAW 48/2024/QH15)",
}


def plot_drivers_and_cost_breakdown_sheet(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Render Sheet 1: Hierarchical single-table Income Statement Structure
    (Revenue, COGS, Facility OpEx, SG&A, D&A & Taxes across Case Baseline,
    PICK! Social Dev, PICK! Social Launch, 20-Court Steady State, 5-Year Total, % of Revenue & Source).
    """
    col_widths = [0.110, 0.215, 0.095, 0.066, 0.090, 0.098, 0.088, 0.068, 0.058, 0.112]
    empty_10 = [""] * 10

    rows: list[tuple[str, list[str]]] = [
        ("sec_hdr", [
            "INCOME STATEMENT STRUCTURE: REVENUE, COGS, FACILITY OPEX, SG&A, D&A & TAXES (M VND)",
            "", "", "", "", "", "", "", "", "",
        ]),
        ("col_hdr", [
            "Category",
            "Line Item",
            "Cost Behavior",
            "Case Baseline",
            "PICK! Social Dev (Jan-27)",
            "PICK! Social Launch (May-27)",
            "20-Court Steady (Dec-28)",
            "5-Year Total",
            "% of Revenue",
            "Source",
        ]),
    ]

    seen_sections: set[str] = set()
    for li in res.line_items:
        if li.section == "REVENUE" and "REVENUE" not in seen_sections:
            seen_sections.add("REVENUE")
            rows.append(("sub_hdr", ["SECTION 1 - REVENUE STREAMS (COURT HIRE, PICK! SOCIAL & ANCILLARY)"] + [""] * 9))
        elif li.section == "COGS" and "COGS" not in seen_sections:
            seen_sections.add("COGS")
            rows.append(("stmt_gap", empty_10))
            rows.append(("sub_hdr", ["SECTION 2 - DIRECT COST OF GOODS SOLD (VARIABLE COGS) & GROSS PROFIT"] + [""] * 9))
        elif li.section == "OPEX" and "OPEX" not in seen_sections:
            seen_sections.add("OPEX")
            rows.append(("stmt_gap", empty_10))
            rows.append(("sub_hdr", ["SECTION 3 - FACILITY OPERATING EXPENSES (FIXED & SEMI-FIXED OPEX)"] + [""] * 9))
        elif li.section == "SGA" and "SGA" not in seen_sections:
            seen_sections.add("SGA")
            rows.append(("stmt_gap", empty_10))
            rows.append(("sub_hdr", ["SECTION 4 - SELLING, GENERAL, ADMIN, PLATFORM & COMPLIANCE (SG&A)"] + [""] * 9))
        elif "D&A" in li.template_category and "DA_TAX" not in seen_sections:
            seen_sections.add("DA_TAX")
            rows.append(("stmt_gap", empty_10))
            rows.append(("sub_hdr", ["SECTION 5 - DEPRECIATION (D&A), TAXABLE INCOME (EBT), VIETNAM CIT (20%) & NET INCOME"] + [""] * 9))

        if li.section == "SUMMARY":
            if "GROSS PROFIT" in li.template_category:
                rstyle = "total"
            elif "OPERATING PROFIT" in li.template_category or "NET INCOME" in li.template_category:
                rstyle = "grand_total"
            else:
                rstyle = "subtotal"
        elif "New" in li.template_category or "PICK! Social" in li.template_category or "Adaptive Pricing" in li.line_item or "Payment gateway" in li.line_item:
            rstyle = "linked"
        else:
            rstyle = "input"

        src_lbl = _line_item_source_label(li)
        prefix = "  " if li.section != "SUMMARY" else ""
        rows.append((
            rstyle,
            [
                f"{prefix}{li.template_category}" if li.section != "SUMMARY" else li.template_category,
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
            return TOP_COL_HEADER_BG, BANNER_WHITE, "bold"
        if rstyle == "subtotal":
            return SUBTOTAL_BG, FORMULA_BLACK, "bold"
        if rstyle == "total":
            return PROFIT_HIGHLIGHT_BG, "#89b4fa", "bold"
        if rstyle == "grand_total":
            return TOTAL_HIGHLIGHT_BG, EMERALD_TEXT, "bold"
        if rstyle == "linked":
            if c_idx in (4, 5, 6, 7):
                return LINKED_BG, LINKED_PURPLE, "bold"
            return LINKED_BG, FORMULA_BLACK, "normal"
        if rstyle == "blank":
            return "#181825", FORMULA_BLACK, "normal"
        bg = "#1e1e2e" if r_idx % 2 == 0 else "#181825"
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
    Render Sheet 2: Hierarchical table of FS-Month (Monthly 3-Statement Model:
    Income Statement, Balance Sheet, Cash Flow Statement & Tax Schedule across Year 1 months
    plus End of Year 2 & End of Year 5 monthly snapshots - strictly monthly figures, no totals).
    """
    col_widths = [0.268, 0.044] + [0.046] * 12 + [0.068, 0.068]
    y1_months = list(res.month_labels[:12])
    period_hdrs = ["Unit"] + y1_months + ["Dec-28 (End Y2)", "Dec-31 (End Y5)"]

    rows: list[tuple[str, list[str]]] = [
        ("col_hdr", ["FINANCIAL STATEMENTS - MONTHLY (YEAR 1 + END Y2 & END Y5 SNAPSHOTS)"] + period_hdrs),
    ]

    for _, df_row in res.fs_month_df.iterrows():
        rtype = str(df_row["row_type"])
        label = str(df_row["line_item"])
        unit = str(df_row["unit"])
        code = str(df_row["row_code"])

        if rtype == "header":
            if code != "KI_HDR":
                rows.append(("stmt_gap", [""] * 16))
                rows.append(("sec_hdr", [label] + period_hdrs))
            else:
                rows.append(("sec_hdr", [label] + [""] * 15))
            continue

        if code in _THREE_STMT_SUBHEADERS:
            rows.append(("sub_hdr", [_THREE_STMT_SUBHEADERS[code]] + [""] * 15))

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

        rows.append((rtype, [label, unit] + m_cells + [m24_str, m60_str]))

    def _style_s2(r_idx: int, c_idx: int, rstyle: str) -> tuple[str, str, str]:
        if rstyle == "col_hdr":
            return TOP_COL_HEADER_BG, BANNER_WHITE, "bold"
        if rstyle == "subtotal":
            return SUBTOTAL_BG, FORMULA_BLACK, "bold"
        if rstyle == "total":
            return PROFIT_HIGHLIGHT_BG, "#89b4fa", "bold"
        if rstyle == "grand_total":
            return TOTAL_HIGHLIGHT_BG, EMERALD_TEXT, "bold"
        if rstyle == "pct":
            return "#232436", "#a6adc8", "bold"
        if rstyle == "input_linked":
            return LINKED_BG, (LINKED_PURPLE if 2 <= c_idx <= 15 else FORMULA_DARKGRAY), "bold"
        bg = "#1e1e2e" if r_idx % 2 == 0 else "#181825"
        if rstyle == "input" and 2 <= c_idx <= 15:
            return bg, INPUT_BLUE, "normal"
        if rstyle in ("formula", "unit"):
            return bg, FORMULA_DARKGRAY, "normal"
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
        fig_width=22.0,
        row_height_in=0.28,
        out_path=output_dir / "02_fs_month_60m_projection_sheet.png",
        font_size=7.3,
    )


def plot_fs_annual_and_structure_dashboard(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Render Sheet 3: Hierarchical table of FS-Annual 5-Year 3-Statement Model
    (Standard Annual Income Statement, Balance Sheet, Cash Flow Statement & Tax Schedule for FY 2027 to FY 2031).
    """
    col_widths = [0.365, 0.060, 0.115, 0.115, 0.115, 0.115, 0.115]
    period_hdrs = [
        "Unit",
        "FY 2027 (Y1)",
        "FY 2028 (Y2)",
        "FY 2029 (Y3)",
        "FY 2030 (Y4)",
        "FY 2031 (Y5)",
    ]

    rows: list[tuple[str, list[str]]] = [
        ("col_hdr", ["FINANCIAL STATEMENTS - ANNUAL (FY 2027 TO FY 2031)"] + period_hdrs),
    ]

    for _, df_row in res.fs_annual_df.iterrows():
        rtype = str(df_row["row_type"])
        label = str(df_row["line_item"])
        unit = str(df_row["unit"])
        code = str(df_row["row_code"])

        if rtype == "header":
            if code != "KI_HDR":
                rows.append(("stmt_gap", [""] * 7))
                rows.append(("sec_hdr", [label] + period_hdrs))
            else:
                rows.append(("sec_hdr", [label] + [""] * 6))
            continue

        if code in _THREE_STMT_SUBHEADERS:
            rows.append(("sub_hdr", [_THREE_STMT_SUBHEADERS[code]] + [""] * 6))

        def _fmt(v: float) -> str:
            if rtype == "pct":
                return f"{v:.2f}%"
            if code == "MEM_CNT":
                return f"{v:,.0f}"
            return f"{v:,.2f}"

        y_cells = [_fmt(float(df_row[y_lbl])) for y_lbl in res.year_labels]
        rows.append((rtype, [label, unit] + y_cells))

    def _style_s3(r_idx: int, c_idx: int, rstyle: str) -> tuple[str, str, str]:
        if rstyle == "col_hdr":
            return TOP_COL_HEADER_BG, BANNER_WHITE, "bold"
        if rstyle == "subtotal":
            return SUBTOTAL_BG, FORMULA_BLACK, "bold"
        if rstyle == "total":
            return PROFIT_HIGHLIGHT_BG, "#89b4fa", "bold"
        if rstyle == "grand_total":
            return TOTAL_HIGHLIGHT_BG, EMERALD_TEXT, "bold"
        if rstyle == "pct":
            return "#232436", "#a6adc8", "bold"
        if rstyle == "input_linked":
            return LINKED_BG, (LINKED_PURPLE if c_idx >= 2 else FORMULA_BLACK), "bold"
        bg = "#1e1e2e" if r_idx % 2 == 0 else "#181825"
        if rstyle == "input" and c_idx >= 2:
            return bg, INPUT_BLUE, "normal"
        if c_idx >= 2:
            return bg, FORMULA_DARKGRAY, "normal"
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
        fig_width=18.5,
        row_height_in=0.29,
        out_path=output_dir / "03_fs_annual_5y_projection_sheet.png",
        font_size=7.8,
    )


def plot_revenue_costs_profit_comparison_charts(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Generate Chart 4: Stacked Revenue-by-Line, Costs-by-Category & Profit-Line Comparison Charts (Baseline vs. Model)
      - Baseline bars dimmed down to grey and low opacity to emphasize our solution
      - Model Revenue above 0 separated into stacked revenue lines (Venue 1 Court Hire, Venue 2 Court Hire,
        Memberships, Coaching, Food & Drinks, Paddle & Balls)
      - Model Costs below 0 separated into stacked cost & tax categories (Costs (COGS), Facility OpEx,
        Expenses (SG&A), D&A & 20% CIT)
      - Operating Profit (EBITDA) and Net Income (After D&A & 20% Vietnam CIT) plotted as lines with markers
      - Panel A: 5-Year Annual Comparison (FY 2027 to FY 2031)
      - Panel B: Monthly Trajectory Across Year 1 & Year 2 (Months 1-24: Jan-27 to Dec-28)
    """
    fig, (ax_ann, ax_mon) = plt.subplots(2, 1, figsize=(19.4, 14.0), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.30, top=0.92, bottom=0.06)
    fig.suptitle(
        "Annual & Monthly Revenue, Cost Structure & Profit Comparison",
        fontsize=14.5,
        fontweight="bold",
        color="#cdd6f4",
        y=0.965,
    )

    BASE_BAR_GREY = "#585b70"
    BASE_BAR_EDGE = "#7f849c"
    BASE_BAR_ALPHA = 0.50
    BASE_TEXT_GREY = "#a6adc8"

    # Catppuccin Mocha palette for the 6 Revenue Lines (positive stacked bars > 0)
    REV_LINES_SPEC = [
        ("Venue 1 Court Hire", res.monthly_venue1_court_hire_arr, "#89b4fa"),
        ("Venue 2 Court Hire", res.monthly_venue2_court_hire_arr, "#74c7ec"),
        ("Memberships (PICK! Social)", res.monthly_membership_rev_arr, "#a6e3a1"),
        ("Coaching", res.monthly_coaching_arr, "#94e2d5"),
        ("Food & Drinks", res.monthly_food_drinks_arr, "#f9e2af"),
        ("Paddle & Balls", res.monthly_paddle_balls_arr, "#cba6f7"),
    ]

    # Catppuccin Mocha palette for the 4 Cost & Tax Categories (negative stacked bars < 0)
    COST_CATS_SPEC = [
        ("Costs (COGS)", res.monthly_total_cogs_arr, "#fab387"),
        ("Facility OpEx (Lease & Utilities)", res.monthly_total_opex_arr, "#f38ba8"),
        ("Expenses (SG&A & Platform)", res.monthly_total_sga_arr, "#eba0ac"),
        ("D&A & 20% Vietnam CIT", res.monthly_da_arr + res.monthly_taxes_arr, "#b4befe"),
    ]

    BASE_PROFIT_LINE = "#9399b2"
    BASE_PROFIT_ALPHA = 0.72
    MOD_EBITDA_LINE = "#f9e2af"
    MOD_EBITDA_MARKER = "#a6e3a1"
    MOD_NETINC_LINE = "#cdd6f4"
    MOD_NETINC_MARKER = "#89b4fa"

    # =========================================================================
    # PANEL A: 5-Year Annual Stacked Revenue-by-Line & Cost-by-Category Chart
    # =========================================================================
    ax_ann.set_facecolor("#181825")
    mc = res.mc
    n_years = len(res.year_labels)
    x_yr = np.arange(n_years)
    bar_w = 0.35

    base_ann_rev = mc.base_annual_rev_mean
    mod_ann_rev = res.annual_revenue_arr

    base_ann_cost = -mc.base_annual_costs_mean
    mod_ann_total_outflow = res.annual_total_costs_arr + res.annual_da_arr + res.annual_taxes_arr

    base_ann_profit = mc.base_annual_ebitda_mean
    mod_ann_ebitda = res.annual_ebitda_arr
    mod_ann_netinc = res.annual_net_income_arr

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

    # Baseline status-quo bars (left position x_yr - bar_w/2)
    ax_ann.bar(
        x_yr - bar_w / 2,
        base_ann_rev,
        width=bar_w,
        yerr=base_rev_yerr,
        capsize=3.2,
        error_kw=dict(ecolor="#a6adc8", lw=0.95, alpha=0.70),
        color=BASE_BAR_GREY,
        edgecolor=BASE_BAR_EDGE,
        alpha=BASE_BAR_ALPHA,
        linewidth=0.85,
        label="Case Baseline Revenue & Costs (Dimmed Grey)",
        zorder=2,
    )
    ax_ann.bar(
        x_yr - bar_w / 2,
        base_ann_cost,
        width=bar_w,
        yerr=base_cost_yerr,
        capsize=3.2,
        error_kw=dict(ecolor="#a6adc8", lw=0.95, alpha=0.70),
        color=BASE_BAR_GREY,
        edgecolor=BASE_BAR_EDGE,
        alpha=BASE_BAR_ALPHA,
        linewidth=0.85,
        zorder=2,
    )

    # Model Stacked Revenue Lines above 0 (right position x_yr + bar_w/2)
    pos_bottom_ann = np.zeros(n_years, dtype=float)
    for idx_r, (r_label, r_mon_arr, r_color) in enumerate(REV_LINES_SPEC):
        r_ann_arr = np.array([float(np.sum(r_mon_arr[y * 12 : (y + 1) * 12])) for y in range(n_years)])
        is_top_rev = idx_r == len(REV_LINES_SPEC) - 1
        ax_ann.bar(
            x_yr + bar_w / 2,
            r_ann_arr,
            bottom=pos_bottom_ann,
            width=bar_w,
            yerr=mod_rev_yerr if is_top_rev else None,
            capsize=3.5 if is_top_rev else 0.0,
            error_kw=dict(ecolor="#cdd6f4", lw=1.05, alpha=0.85) if is_top_rev else {},
            color=r_color,
            edgecolor="#11111b",
            linewidth=0.75,
            label=f"Rev: {r_label}",
            zorder=3,
        )
        pos_bottom_ann += r_ann_arr

    # Model Stacked Cost & Tax Categories below 0 (right position x_yr + bar_w/2)
    neg_bottom_ann = np.zeros(n_years, dtype=float)
    for c_label, c_mon_arr, c_color in COST_CATS_SPEC:
        c_ann_arr = -np.array([float(np.sum(c_mon_arr[y * 12 : (y + 1) * 12])) for y in range(n_years)])
        ax_ann.bar(
            x_yr + bar_w / 2,
            c_ann_arr,
            bottom=neg_bottom_ann,
            width=bar_w,
            color=c_color,
            edgecolor="#11111b",
            linewidth=0.75,
            label=f"Cost: {c_label}",
            zorder=3,
        )
        neg_bottom_ann += c_ann_arr

    # Profit Statistical Range Bands (P05-P95) + Operating Profit & Net Income lines
    ax_ann.fill_between(
        x_yr + bar_w / 2,
        mc.annual_ebitda_p05,
        mc.annual_ebitda_p95,
        color="#f9e2af",
        alpha=0.18,
        zorder=4,
    )
    ax_ann.plot(
        x_yr - bar_w / 2,
        base_ann_profit,
        linestyle="-",
        linewidth=1.9,
        color=BASE_PROFIT_LINE,
        alpha=BASE_PROFIT_ALPHA,
        marker="o",
        markersize=6.5,
        markerfacecolor="#313244",
        markeredgecolor=BASE_PROFIT_LINE,
        markeredgewidth=1.5,
        label="Case Baseline Operating Profit (EBITDA)",
        zorder=5,
    )
    ax_ann.plot(
        x_yr + bar_w / 2,
        mod_ann_ebitda,
        linestyle="-",
        linewidth=2.7,
        color=MOD_EBITDA_LINE,
        marker="o",
        markersize=8.0,
        markerfacecolor=MOD_EBITDA_MARKER,
        markeredgecolor="#11111b",
        markeredgewidth=1.7,
        label="Model Operating Profit (EBITDA)",
        zorder=6,
    )
    ax_ann.plot(
        x_yr + bar_w / 2,
        mod_ann_netinc,
        linestyle="--",
        linewidth=2.3,
        color=MOD_NETINC_LINE,
        marker="s",
        markersize=6.8,
        markerfacecolor=MOD_NETINC_MARKER,
        markeredgecolor="#11111b",
        markeredgewidth=1.5,
        label="Model Net Income (After D&A & 20% CIT)",
        zorder=7,
    )

    ax_ann.axhline(0, color="#cdd6f4", linewidth=1.4, zorder=4)

    for i in range(n_years):
        ax_ann.text(
            i - bar_w / 2,
            base_ann_rev[i] * 0.52,
            f"+{base_ann_rev[i]:,.0f}M",
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#cdd6f4",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="#1e1e2e", edgecolor="none", alpha=0.75),
            zorder=8,
        )
        ax_ann.text(
            i + bar_w / 2,
            mc.annual_rev_p95[i] + 350,
            f"+{mod_ann_rev[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#a6e3a1",
        )
        ax_ann.text(
            i - bar_w / 2,
            -mc.base_annual_costs_p95[i] - 320,
            f"{base_ann_cost[i]:,.0f}M",
            ha="center",
            va="top",
            fontsize=7.8,
            fontweight="normal",
            color=BASE_TEXT_GREY,
            alpha=0.90,
        )
        ax_ann.text(
            i + bar_w / 2,
            -mod_ann_total_outflow[i] - 320,
            f"-{mod_ann_total_outflow[i]:,.0f}M",
            ha="center",
            va="top",
            fontsize=8.3,
            fontweight="bold",
            color="#f38ba8",
        )

    ax_ann.set_xticks(x_yr)
    ax_ann.set_xticklabels(res.year_labels, fontsize=9.8, fontweight="bold", color="#cdd6f4")
    ax_ann.set_ylabel("Million VND / Year", fontsize=10.2, fontweight="bold", color="#bac2de")
    ax_ann.set_ylim(-float(np.max(mod_ann_total_outflow)) * 1.28, float(np.max(mc.annual_rev_p95)) * 1.44)
    ax_ann.grid(axis="y", linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_ann.legend(loc="upper left", ncol=4, fontsize=7.5, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")
    ax_ann.set_title(
        "Annual Revenue by Line, Costs by Category & Profit Comparison",
        fontsize=11.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=12,
    )

    # =========================================================================
    # PANEL B: Monthly Trajectory Across Years 1 & 2 (Months 1-24: Jan-27 to Dec-28)
    # =========================================================================
    ax_mon.set_facecolor("#181825")
    m24_labels = list(res.month_labels[:24])
    n_m = len(m24_labels)
    x_m = np.arange(n_m)
    bar_wm = 0.38

    base_mon_rev = mc.base_total_rev_mean[:24]
    mod_mon_rev = res.monthly_total_rev_arr[:24]

    base_mon_cost = -mc.base_total_costs_mean[:24]
    mod_mon_total_outflow = (res.monthly_total_costs_arr + res.monthly_da_arr + res.monthly_taxes_arr)[:24]

    base_mon_profit = mc.base_ebitda_mean[:24]
    mod_mon_ebitda = res.monthly_ebitda_arr[:24]
    mod_mon_netinc = res.monthly_net_income_arr[:24]

    ax_mon.bar(
        x_m - bar_wm / 2,
        base_mon_rev,
        width=bar_wm,
        color=BASE_BAR_GREY,
        edgecolor=BASE_BAR_EDGE,
        alpha=BASE_BAR_ALPHA,
        linewidth=0.65,
        label="Case Baseline Monthly Revenue & Costs",
        zorder=2,
    )
    ax_mon.bar(
        x_m - bar_wm / 2,
        base_mon_cost,
        width=bar_wm,
        color=BASE_BAR_GREY,
        edgecolor=BASE_BAR_EDGE,
        alpha=BASE_BAR_ALPHA,
        linewidth=0.65,
        zorder=2,
    )

    pos_bottom_mon = np.zeros(n_m, dtype=float)
    for r_label, r_mon_arr, r_color in REV_LINES_SPEC:
        r_slice = r_mon_arr[:24]
        ax_mon.bar(
            x_m + bar_wm / 2,
            r_slice,
            bottom=pos_bottom_mon,
            width=bar_wm,
            color=r_color,
            edgecolor="#11111b",
            linewidth=0.60,
            label=f"Rev: {r_label}",
            zorder=3,
        )
        pos_bottom_mon += r_slice

    neg_bottom_mon = np.zeros(n_m, dtype=float)
    for c_label, c_mon_arr, c_color in COST_CATS_SPEC:
        c_slice = -c_mon_arr[:24]
        ax_mon.bar(
            x_m + bar_wm / 2,
            c_slice,
            bottom=neg_bottom_mon,
            width=bar_wm,
            color=c_color,
            edgecolor="#11111b",
            linewidth=0.60,
            label=f"Cost: {c_label}",
            zorder=3,
        )
        neg_bottom_mon += c_slice

    ax_mon.fill_between(
        x_m + bar_wm / 2,
        mc.ebitda_p05[:24],
        mc.ebitda_p95[:24],
        color="#f9e2af",
        alpha=0.18,
        zorder=4,
    )
    ax_mon.plot(
        x_m - bar_wm / 2,
        base_mon_profit,
        linestyle="-",
        linewidth=1.7,
        color=BASE_PROFIT_LINE,
        alpha=BASE_PROFIT_ALPHA,
        marker="o",
        markersize=4.8,
        markerfacecolor="#313244",
        markeredgecolor=BASE_PROFIT_LINE,
        markeredgewidth=1.2,
        label="Case Baseline Monthly Operating Profit",
        zorder=5,
    )
    ax_mon.plot(
        x_m + bar_wm / 2,
        mod_mon_ebitda,
        linestyle="-",
        linewidth=2.3,
        color=MOD_EBITDA_LINE,
        marker="o",
        markersize=5.8,
        markerfacecolor=MOD_EBITDA_MARKER,
        markeredgecolor="#11111b",
        markeredgewidth=1.4,
        label="Model Monthly Operating Profit (EBITDA)",
        zorder=6,
    )
    ax_mon.plot(
        x_m + bar_wm / 2,
        mod_mon_netinc,
        linestyle="--",
        linewidth=2.0,
        color=MOD_NETINC_LINE,
        marker="s",
        markersize=5.0,
        markerfacecolor=MOD_NETINC_MARKER,
        markeredgecolor="#11111b",
        markeredgewidth=1.3,
        label="Model Monthly Net Income (After D&A & 20% CIT)",
        zorder=7,
    )

    ax_mon.axhline(0, color="#cdd6f4", linewidth=1.4, zorder=4)
    ax_mon.axvline(3.5, color="#f38ba8", linewidth=1.5, linestyle="--", zorder=4)
    ax_mon.axvline(12.5, color="#89b4fa", linewidth=1.5, linestyle=":", zorder=4)

    for i in range(0, n_m, 2):
        ax_mon.text(
            i + bar_wm / 2,
            mod_mon_rev[i] + 24,
            f"{mod_mon_rev[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="#a6e3a1",
        )
        ax_mon.text(
            i + bar_wm / 2,
            -mod_mon_total_outflow[i] - 26,
            f"-{mod_mon_total_outflow[i]:,.0f}M",
            ha="center",
            va="top",
            fontsize=6.8,
            fontweight="bold",
            color="#f38ba8",
        )

    ax_mon.set_xticks(x_m)
    ax_mon.set_xticklabels(m24_labels, fontsize=7.8, fontweight="bold", color="#cdd6f4", rotation=25)
    ax_mon.set_ylabel("Million VND / Month", fontsize=10.2, fontweight="bold", color="#bac2de")
    ax_mon.set_ylim(-float(np.max(mod_mon_total_outflow)) * 1.32, float(np.max(mod_mon_rev)) * 1.44)
    ax_mon.grid(axis="y", linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_mon.legend(loc="upper left", ncol=4, fontsize=7.3, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")
    ax_mon.set_title(
        "Monthly Revenue by Line, Costs by Category & Profit Trajectory",
        fontsize=11.8,
        fontweight="bold",
        color="#cdd6f4",
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
      - Panel A (Top Left): Smooth Active Paying Subscribers Trajectory S_t (Months 1-60: Jan-27 to Dec-31)
        with 40 Monte Carlo simulation lines, P05-P95 confidence band, Train vs. Held-Out Test mean,
        and Case Page 3 End-of-Year-2 benchmark (1,500 paying members at M24 / Dec-28).
      - Panel B (Top Right): Monthly Membership Revenue & Total Revenue Trajectory (M VND/month)
        with P05-P95 band and -30% lower-demand stress test comparison.
      - Panel C (Bottom Left): Monthly Subscriber Flow Dynamics (Monthly New Subscribers Acquired vs.
        Churned Subscribers & Net Monthly Member Additions from Month 5 onward).
      - Panel D (Bottom Right): Annual Operating Profit (EBITDA) Across FY 2027-FY 2031 (Base Plan P05-P95
        vs. -30% Lower Demand Stress Test vs. Investor Minh Tran's 3,500M VND Year 3 Target).
    """
    mc = res.mc
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(20.0, 12.6), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(wspace=0.20, hspace=0.28, top=0.915, bottom=0.07)

    x_m = np.arange(1, 61)
    tick_idx = [0, 5, 11, 23, 35, 47, 59]
    tick_lbls = [
        "Jan-27 (Start)",
        "Jun-27 (End Dev)",
        "Dec-27 (End Y1)",
        "Dec-28 (End Y2)",
        "Dec-29 (End Y3)",
        "Dec-30 (End Y4)",
        "Dec-31 (End Y5)",
    ]

    # -------------------------------------------------------------------------
    # PANEL A: Smooth Active Paying Subscribers S_t (N=300 Monte Carlo Trajectories)
    # -------------------------------------------------------------------------
    ax_a.set_facecolor("#181825")
    for path in mc.sample_subscribers_paths:
        ax_a.plot(x_m, path, color="#89b4fa", alpha=0.14, linewidth=0.8, zorder=2)

    ax_a.fill_between(
        x_m,
        mc.active_subscribers_p05,
        mc.active_subscribers_p95,
        color="#89b4fa",
        alpha=0.22,
        label="90% Monte Carlo Band (P05-P95)",
        zorder=3,
    )
    ax_a.plot(
        x_m,
        mc.Active_subscribers_train_mean,
        color="#89b4fa",
        linewidth=2.5,
        label=f"Train Split Mean (n={mc.n_train} sims)",
        zorder=5,
    )
    ax_a.plot(
        x_m,
        mc.active_subscribers_test_mean,
        color="#cdd6f4",
        linestyle="--",
        linewidth=2.0,
        label=f"Held-Out Test Split Mean (n={mc.n_test} sims)",
        zorder=6,
    )
    ax_a.plot(
        x_m,
        mc.stress_active_subscribers_mean,
        color="#f38ba8",
        linestyle="-.",
        linewidth=2.0,
        label="Stress Scenario (-30% Demand) Subscribers",
        zorder=4,
    )

    ax_a.axvline(5, color="#a6e3a1", linewidth=1.6, linestyle="--", zorder=4)
    ax_a.scatter([24], [mc.m24_end_y2_subscribers], color="#a6e3a1", edgecolor="#11111b", s=75, zorder=7)
    ax_a.annotate(
        f"End of Year 2 (Dec-28):\n{mc.m24_end_y2_subscribers:,.0f} Paying Members\n({mc.m24_end_y2_membership_rev_mvnd:,.1f}M VND/mo)",
        xy=(24, mc.m24_end_y2_subscribers),
        xytext=(13.5, mc.m24_end_y2_subscribers + 620),
        fontsize=8.0,
        fontweight="bold",
        color="#a6e3a1",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#1e1e2e", edgecolor="#a6e3a1", linewidth=1.0),
        arrowprops=dict(arrowstyle="->", color="#a6e3a1", linewidth=1.3),
        zorder=8,
    )

    ax_a.set_xticks([t + 1 for t in tick_idx])
    ax_a.set_xticklabels(tick_lbls, fontsize=7.8, fontweight="bold", rotation=18, ha="right")
    ax_a.set_ylim(-50, float(np.max(mc.active_subscribers_p95)) * 1.20)
    ax_a.set_ylabel("Active Paying Members (Subscribers)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_a.set_title(
        "Active Paying Subscriber Cohort Evolution",
        fontsize=10.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_a.grid(True, linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_a.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    # -------------------------------------------------------------------------
    # PANEL B: Total Monthly Revenue & Membership Revenue Trajectory (M VND/mo)
    # -------------------------------------------------------------------------
    ax_b.set_facecolor("#181825")
    for path in mc.sample_total_rev_paths:
        ax_b.plot(x_m, path, color="#a6e3a1", alpha=0.12, linewidth=0.8, zorder=2)
    for b_path in mc.sample_base_rev_paths:
        ax_b.plot(x_m, b_path, color="#7f849c", alpha=0.10, linewidth=0.7, zorder=1)

    ax_b.fill_between(
        x_m,
        mc.total_rev_p05,
        mc.total_rev_p95,
        color="#a6e3a1",
        alpha=0.22,
        label="Total Monthly Revenue P05-P95 Band",
        zorder=3,
    )
    ax_b.plot(
        x_m,
        mc.total_rev_mean,
        color="#a6e3a1",
        linewidth=2.6,
        label="Total Monthly Revenue Mean (Court + Ancillary + PICK! Social)",
        zorder=5,
    )
    ax_b.fill_between(
        x_m,
        mc.membership_rev_p05,
        mc.membership_rev_p95,
        color="#cba6f7",
        alpha=0.22,
        zorder=3,
    )
    ax_b.plot(
        x_m,
        mc.membership_rev_mean,
        color="#cba6f7",
        linewidth=2.3,
        label="PICK! Social Subscription Revenue (@ 399k VND/member/mo)",
        zorder=5,
    )
    ax_b.plot(
        x_m,
        mc.stress_total_rev_mean,
        color="#f38ba8",
        linestyle="-.",
        linewidth=1.9,
        label="Stress Scenario (-30% Demand) Total Revenue",
        zorder=4,
    )
    ax_b.fill_between(
        x_m,
        mc.base_total_rev_p05,
        mc.base_total_rev_p95,
        color="#7f849c",
        alpha=0.20,
        label="Case Baseline Revenue P05-P95 Band",
        zorder=2,
    )
    ax_b.plot(
        x_m,
        mc.base_total_rev_mean,
        color="#9399b2",
        linestyle="-",
        linewidth=1.7,
        alpha=0.85,
        label=f"Case Baseline Revenue (~{mc.base_total_rev_mean.mean():.1f}M/mo)",
        zorder=3,
    )

    ax_b.set_xticks([t + 1 for t in tick_idx])
    ax_b.set_xticklabels(tick_lbls, fontsize=7.8, fontweight="bold", rotation=18, ha="right")
    ax_b.set_ylim(0, float(np.max(mc.total_rev_p95)) * 1.22)
    ax_b.set_ylabel("Monthly Revenue (M VND)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_b.set_title(
        "Monthly Total Revenue & PICK! Social Subscription Trajectory",
        fontsize=10.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_b.grid(True, linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_b.legend(loc="upper left", fontsize=7.4, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    # -------------------------------------------------------------------------
    # PANEL C: Monthly Subscriber Acquisition vs. Monthly Churn Flow Dynamics
    # -------------------------------------------------------------------------
    ax_c.set_facecolor("#181825")
    net_adds = mc.new_subscribers_mean - mc.churned_subscribers_mean
    ax_c.bar(
        x_m,
        mc.new_subscribers_mean,
        color="#a6e3a1",
        edgecolor="#11111b",
        width=0.68,
        alpha=0.88,
        label="Gross New Subscribers Acquired (+ / mo)",
        zorder=3,
    )
    ax_c.bar(
        x_m,
        -mc.churned_subscribers_mean,
        color="#f38ba8",
        edgecolor="#11111b",
        width=0.68,
        alpha=0.88,
        label="Churned Subscribers (- / mo, ~4.2% Monthly Churn)",
        zorder=3,
    )
    ax_c.plot(
        x_m,
        net_adds,
        color="#f9e2af",
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        label="Net Monthly Subscriber Additions (New - Churned)",
        zorder=5,
    )
    ax_c.axhline(0, color="#cdd6f4", linewidth=1.2, zorder=4)

    ax_c.set_xticks([t + 1 for t in tick_idx])
    ax_c.set_xticklabels(tick_lbls, fontsize=7.8, fontweight="bold", rotation=18, ha="right")
    ax_c.set_ylim(-125, 415)
    ax_c.set_ylabel("Subscribers / Month", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_c.set_title(
        "Monthly Subscriber Acquisition vs. Churn Flow Dynamics",
        fontsize=10.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_c.grid(True, linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_c.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    # -------------------------------------------------------------------------
    # PANEL D: Annual Operating Profit (FY27-FY31) vs. Investor Minh Tran 3,500M Target
    # -------------------------------------------------------------------------
    ax_d.set_facecolor("#181825")
    x_y = np.arange(5)
    w_y = 0.24
    gap_y = 0.27

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
        x_y - gap_y,
        base_y_ebitda,
        width=w_y,
        yerr=base_y_yerr,
        capsize=3.0,
        error_kw=dict(ecolor="#a6adc8", lw=1.0, alpha=0.75),
        color="#585b70",
        edgecolor="#7f849c",
        alpha=0.68,
        label=f"Case Baseline EBITDA (~+{base_y_ebitda.mean():,.0f}M / yr, P05-P95)",
        zorder=2,
    )
    ax_d.bar(
        x_y,
        mod_y_ebitda,
        width=w_y,
        yerr=mod_y_yerr,
        capsize=3.0,
        error_kw=dict(ecolor="#cdd6f4", lw=1.1, alpha=0.85),
        color="#a6e3a1",
        edgecolor="#11111b",
        linewidth=1.1,
        label="Integrated Solution Model (Mean Annual EBITDA, P05-P95)",
        zorder=4,
    )
    ax_d.bar(
        x_y + gap_y,
        stress_y_ebitda,
        width=w_y,
        yerr=stress_y_yerr,
        capsize=3.0,
        error_kw=dict(ecolor="#cdd6f4", lw=1.0, alpha=0.80),
        color="#fab387",
        edgecolor="#11111b",
        linewidth=1.0,
        hatch="///",
        label="-30% Lower Demand Stress Test (Case Page 4, P05-P95)",
        zorder=3,
    )

    ax_d.axhline(
        3500.0,
        color="#f38ba8",
        linewidth=2.2,
        linestyle="--",
        label="Investor Minh Tran Year 3 Target (3,500M VND / yr)",
        zorder=6,
    )

    for i in range(5):
        ax_d.text(
            x_y[i] - gap_y - 0.02,
            mc.base_annual_ebitda_p95[i] + 240,
            f"{base_y_ebitda[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="normal",
            color="#a6adc8",
        )
        ax_d.text(
            x_y[i],
            mc.annual_ebitda_p95[i] + 280,
            f"{mod_y_ebitda[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color="#a6e3a1",
        )
        stress_lbl_y = (
            3780.0
            if 2900.0 <= mc.stress_annual_ebitda_p95[i] <= 3650.0
            else (mc.stress_annual_ebitda_p95[i] + 260 if stress_y_ebitda[i] >= 0 else mc.stress_annual_ebitda_p05[i] - 1050)
        )
        ax_d.text(
            x_y[i] + gap_y + 0.03,
            stress_lbl_y,
            f"{stress_y_ebitda[i]:,.0f}M",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="#fab387",
        )

    ax_d.set_xticks(x_y)
    ax_d.set_xticklabels(res.year_labels, fontsize=8.8, fontweight="bold")
    ax_d.set_ylim(min(-2900.0, float(np.min(mc.stress_annual_ebitda_p05)) * 1.55), float(np.max(mc.annual_ebitda_p95)) * 1.36)
    ax_d.set_ylabel("Annual Operating Profit / EBITDA (M VND / Year)", fontsize=9.5, fontweight="bold", color="#bac2de")
    ax_d.set_title(
        "Annual Operating Profit Across Scenarios vs. Investor Target",
        fontsize=10.8,
        fontweight="bold",
        color="#cdd6f4",
        pad=10,
    )
    ax_d.grid(axis="y", linestyle="--", alpha=0.8, color="#313244", zorder=1)
    ax_d.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    fig.suptitle(
        "Membership Cohort Dynamics, Revenue Trajectory & Operating Profit Robustness",
        fontsize=14.2,
        fontweight="bold",
        color="#cdd6f4",
        y=0.962,
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
      - Panel A: 60-Month Ending Cash Balance Trajectory (Mean, P05-P95 Band, -30% Stress Cash & Case Baseline with P05-P95 Band)
      - Panel B: Monthly Cash Flow Components (Operating Cash Flow OCF, Venue 2 Fit-Out CapEx -1,750M/mo in Oct-27-Jan-28, +500M Jan-27 & +7,500M Oct-27 Equity)
      - Panel C: External Macro Pickleball Interest Trend Multiplier I_t (externals.toml) & Legal/Safety Risk Provision
    """
    mc = res.mc
    x_m = np.arange(1, 61)
    tick_idx = list(range(0, 60, 4)) + [59]
    tick_lbls = [res.month_labels[i] for i in tick_idx]

    fig, axes = plt.subplots(3, 1, figsize=(18.5, 13.5), dpi=220, sharex=True)
    fig.patch.set_facecolor("#1e1e2e")
    fig.subplots_adjust(hspace=0.26, top=0.92, bottom=0.06)

    # Panel A: 60-Month Ending Cash Balance Trajectory
    ax1 = axes[0]
    ax1.set_facecolor("#181825")
    for path in mc.sample_cash_paths:
        ax1.plot(x_m, path, color="#89b4fa", alpha=0.12, linewidth=0.8, zorder=2)
    for b_c_path in mc.sample_base_cash_paths:
        ax1.plot(x_m, b_c_path, color="#7f849c", alpha=0.10, linewidth=0.7, zorder=1)

    ax1.fill_between(
        x_m,
        mc.ending_cash_p05,
        mc.ending_cash_p95,
        color="#89b4fa",
        alpha=0.24,
        label="Solution Cash Reserves P05-P95 Confidence Band (300 Sims)",
        zorder=3,
    )
    ax1.fill_between(
        x_m,
        mc.base_ending_cash_p05,
        mc.base_ending_cash_p95,
        color="#7f849c",
        alpha=0.22,
        label="Case Baseline Cash Reserves P05-P95 Band (300 Sims)",
        zorder=2,
    )
    ax1.plot(
        x_m,
        res.monthly_ending_cash_model,
        color="#89b4fa",
        linewidth=2.6,
        marker="o",
        markersize=3.8,
        label=f"Solution Cash Reserves (Dec-31: {res.ending_cash_60m_model_mvnd:,.1f}M | Min Cash: {res.min_cash_balance_model_mvnd:,.1f}M in {res.min_cash_month_label})",
        zorder=6,
    )
    ax1.plot(
        x_m,
        mc.stress_ending_cash_mean,
        color="#f38ba8",
        linestyle="-.",
        linewidth=2.0,
        label=f"-30% Lower Demand Stress Cash Reserves (Dec-31: {mc.stress_ending_cash_mean[-1]:,.1f}M VND)",
        zorder=5,
    )
    ax1.plot(
        x_m,
        res.monthly_ending_cash_case,
        color="#9399b2",
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        alpha=0.82,
        label=f"Case Baseline Cash Reserves (Dec-31: {res.ending_cash_60m_case_mvnd:,.1f}M VND)",
        zorder=4,
    )
    ax1.axvline(10, color="#f9e2af", linestyle="--", linewidth=1.3, alpha=0.85)
    ax1.text(
        9.6,
        29500,
        "Oct-27: +7,500M Equity\n& Venue 2 Build Start",
        ha="right",
        fontsize=8.0,
        fontweight="bold",
        color="#f9e2af",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e1e2e", edgecolor="#f9e2af", alpha=0.92),
    )
    ax1.axvline(14, color="#a6e3a1", linestyle="--", linewidth=1.4, alpha=0.85)
    ax1.text(
        14.4,
        21500,
        "Feb-28: Venue 2 Opens\n(20 Courts Active)",
        ha="left",
        fontsize=8.0,
        fontweight="bold",
        color="#a6e3a1",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e1e2e", edgecolor="#a6e3a1", alpha=0.92),
    )
    ax1.set_ylim(-2500, float(np.max(mc.ending_cash_p95)) * 1.18)
    ax1.set_title(
        "Cumulative Cash Reserves Trajectory Across Scenarios",
        fontsize=11.2,
        fontweight="bold",
        color="#cdd6f4",
        pad=8,
    )
    ax1.set_ylabel("Cash Reserves (M VND)", fontsize=9.8, fontweight="bold", color="#bac2de")
    ax1.grid(True, linestyle="--", alpha=0.8, color="#313244")
    ax1.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    # Panel B: Monthly Cash Flow Components (OCF, CapEx ICF, Equity FCF, April CIT Settlement, Net Cash Flow)
    ax2 = axes[1]
    ax2.set_facecolor("#181825")
    ax2.bar(x_m - 0.2, res.monthly_ocf_model, width=0.38, color="#a6e3a1", edgecolor="#11111b", alpha=0.88, label="Operating Cash Flow (OCF, Net of April CIT Payment)")
    ax2.bar(x_m + 0.2, res.monthly_icf_capex_model, width=0.38, color="#f38ba8", edgecolor="#11111b", alpha=0.88, label="Venue 2 Fit-Out CapEx Outflow (-1,750M/mo in Oct-27-Jan-28)")
    ax2.bar(x_m, -res.monthly_cit_cash_paid_arr, width=0.38, color="#fab387", edgecolor="#11111b", alpha=0.92, label="April Annual CIT Settlement (-20% Prior FY Tax)")
    ax2.bar(x_m, res.monthly_fcf_equity_model, width=0.35, color="#cba6f7", edgecolor="#11111b", alpha=0.80, label="Investor Minh Tran Equity (+500M in Jan-27 & +7,500M in Oct-27)")
    ax2.plot(x_m, res.monthly_net_cash_flow_model, color="#f9e2af", linewidth=2.0, marker="o", markersize=3.5, label="Net Monthly Cash Flow (OCF + ICF + FCF)")
    ax2.axhline(0, color="#cdd6f4", linewidth=1.1)
    ax2.set_title(
        "Monthly Operating, Investing & Financing Cash Flow Components",
        fontsize=11.2,
        fontweight="bold",
        color="#cdd6f4",
        pad=8,
    )
    ax2.set_ylabel("Monthly Cash Flow (M VND / Mo)", fontsize=9.8, fontweight="bold", color="#bac2de")
    ax2.grid(True, linestyle="--", alpha=0.8, color="#313244")
    ax2.legend(loc="upper right", ncol=2, fontsize=8.5, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    # Panel C: External Pickleball Interest Trend Multiplier I_t & Legal/Ward Noise Compliance Provision (externals.toml)
    ax3 = axes[2]
    ax3.set_facecolor("#181825")
    ax3_twin = ax3.twinx()

    ax3.fill_between(x_m, mc.trend_index_p05, mc.trend_index_p95, color="#f9e2af", alpha=0.22, label="Macro Pickleball Trend Index (P05-P95 Band)")
    ax3.plot(x_m, mc.trend_index_mean, color="#fab387", linewidth=2.4, marker="o", markersize=3.5, label="Macro Pickleball Demand Trend Multiplier")
    ax3.axhline(1.0, color="#7f849c", linestyle=":", linewidth=1.3)

    ax3_twin.plot(
        x_m,
        res.monthly_external_legal_safety_sga_arr,
        color="#cba6f7",
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=3.2,
        label="Monthly Legal, Ward Noise & Youth Safety Reserve (M VND/mo)",
    )

    ax3.set_xticks([t + 1 for t in tick_idx])
    ax3.set_xticklabels(tick_lbls, fontsize=8.5, fontweight="bold")
    ax3.set_ylim(0.65, 1.35)
    ax3_twin.set_ylim(0, 22.0)
    ax3.set_ylabel("Macro Pickleball Trend Index", fontsize=9.8, fontweight="bold", color="#fab387")
    ax3_twin.set_ylabel("Legal, Noise & Safety Reserve (M VND/mo)", fontsize=9.5, fontweight="bold", color="#cba6f7")
    ax3.set_xlabel("60-Month Projection Horizon (Jan 2027 - Dec 2031)", fontsize=10.0, fontweight="bold", color="#bac2de")
    ax3.set_title(
        "Macro Market Demand Trend & Regulatory Compliance Provisions",
        fontsize=11.2,
        fontweight="bold",
        color="#cdd6f4",
        pad=8,
    )
    ax3.grid(True, linestyle="--", alpha=0.8, color="#313244")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2, fontsize=8.5, frameon=True, facecolor="#1e1e2e", edgecolor="#45475a")

    fig.suptitle(
        "Cash Reserves Evolution, Capital Deployment & Macro Risk Provisions",
        fontsize=13.5,
        fontweight="bold",
        color="#cdd6f4",
        y=0.97,
    )

    out_path = output_dir / "06_monthly_cash_balance_and_externals.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def plot_executive_solution_master_roadmap(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    output_dir: Path,
) -> Path:
    """
    Render `00_executive_solution_master_roadmap.png`:
    A true, staggered multi-row Master Gantt Chart across Months M1-M24 (Jan 2027 - Dec 2028)
    plus FY 2029 - FY 2031 (Years 3-5 Steady-State), showing the 4 Solution Pillars as Stage Headers
    and their staggered sub-activities stepping across the timeline with short, punchy labels.
    """
    # Structure: 4 Pillar Header Rows + 13 Staggered Sub-Task Rows = 17 rows total
    # Timeline X-axis [0.0 .. 28.0]:
    #   0.0 .. 12.0  : Year 1 Months M1..M12 (Jan 2027 .. Dec 2027)
    #   12.0 .. 24.0 : Year 2 Months M13..M24 (Jan 2028 .. Dec 2028)
    #   24.0 .. 28.0 : Years 3-5 Steady-State (FY 2029 Target Year + FY 2030-2031)
    rows_spec = [
        # Pillar 1: 5.1 Adaptive Court Pricing
        {
            "type": "pillar",
            "id": "5.1",
            "name": "PILLAR 1: ADAPTIVE COURT PRICING (ML DYNAMIC TARIFF ENGINE)",
            "window": "Jan 27 - Dec 31",
            "metric": "+23.9M -> +79.6M/mo Lift",
            "start": 0.0,
            "end": 28.0,
            "color": "#89b4fa",
            "row_bg": "#24273a",
        },
        {
            "type": "task",
            "id": "5.1a",
            "name": "Venue 1 Adaptive Pricing (6 Courts): 160k->115k Daytime & 285k-300k Peak",
            "window": "Jan 27 - Dec 31",
            "metric": "66.6% Daytime Fill (460.8M/mo)",
            "start": 0.0,
            "end": 28.0,
            "color": "#89b4fa",
            "hatch": False,
            "bar_text": "5.1a Venue 1 Adaptive Pricing (6 Courts: 115k Daytime / 285k-300k Peak -> 460.8M/mo Continuous)",
        },
        {
            "type": "task",
            "id": "5.1b",
            "name": "Venue 2 Adaptive Pricing (+14 Courts): Dynamic Tariff Schedule",
            "window": "Feb 28 - Dec 31",
            "metric": "+1,075.1M/mo (1,535.8M Total)",
            "start": 13.0,
            "end": 28.0,
            "color": "#74c7ec",
            "hatch": False,
            "bar_text": "5.1b Venue 2 Adaptive Pricing (+14 Courts: +1,075.1M/mo -> 1,535.8M/mo Combined 20-Court Yield)",
        },
        # Pillar 2: 5.2 Segment-Optimized Marketing
        {
            "type": "pillar",
            "id": "5.2",
            "name": "PILLAR 2: SEGMENT-OPTIMIZED MARKETING & QUARTERLY TOURNAMENTS",
            "window": "Jan 27 - Dec 31",
            "metric": "7,596 -> 21,618 Visits/mo",
            "start": 0.0,
            "end": 28.0,
            "color": "#fab387",
            "row_bg": "#302623",
        },
        {
            "type": "task",
            "id": "5.2a",
            "name": "Continuous ML Segment Budgeting & 4-Persona Content",
            "window": "Jan 27 - Dec 31",
            "metric": "42M-82M/mo -> 21,618 Visits/mo",
            "start": 0.0,
            "end": 28.0,
            "color": "#f9e2af",
            "hatch": False,
            "bar_text": "5.2a Continuous ML Segment Budgeting & 4-Persona Content (42M-82M/mo -> 7,596 to 21,618 Visits/mo)",
        },
        {
            "type": "task",
            "id": "5.2b",
            "name": "Venue 1 Quarterly Tournaments (Mar/Jun/Sep/Dec) & Uni Partnerships",
            "window": "May 27 - Dec 31",
            "metric": "192 Players/Qtr & +15% Reach",
            "start": 4.0,
            "end": 28.0,
            "color": "#fab387",
            "hatch": False,
            "bar_text": "5.2b Venue 1 Seasonal Tournaments (Mar/Jun/Sep/Dec) & University Beginner Leagues",
        },
        {
            "type": "task",
            "id": "5.2c",
            "name": "Venue 2 Flagship Tournaments, Coaching Studio & Specialty Cafe Campaigns",
            "window": "Feb 28 - Dec 31",
            "metric": "420 Players/Qtr | 581M/mo Ancil",
            "start": 13.0,
            "end": 28.0,
            "color": "#eba0ac",
            "hatch": False,
            "bar_text": "5.2c Venue 2 Flagship Tournaments & 2-Club Ancillary Engine (581.1M/mo Coaching, F&B & Rentals)",
        },
        # Pillar 3: 5.3 PICK! Social Community & Membership Network
        {
            "type": "pillar",
            "id": "5.3",
            "name": "PILLAR 3: PICK! SOCIAL COMMUNITY & MEMBERSHIP NETWORK (399K/MO)",
            "window": "Jan 27 - Dec 31",
            "metric": "1,489 Y2 -> 2,218 Y5 Members",
            "start": 0.0,
            "end": 28.0,
            "color": "#a6e3a1",
            "row_bg": "#212e28",
        },
        {
            "type": "task",
            "id": "5.3a",
            "name": "Core Platform Build & Simultaneous QA",
            "window": "Jan 27 - Apr 27",
            "metric": "676.2M Base (4.0 Months)",
            "start": 0.0,
            "end": 4.0,
            "color": "#cba6f7",
            "hatch": True,
            "bar_text": "5.3a 4-Mo Core Build",
        },
        {
            "type": "task",
            "id": "5.3b",
            "name": "Legacy Booking Migration, Public Cutover & Hypercare",
            "window": "May 27 - Jun 27",
            "metric": "351 Launch Members (May-27)",
            "start": 4.0,
            "end": 6.0,
            "color": "#f38ba8",
            "hatch": False,
            "bar_text": "5.3b Cutover (2m)",
        },
        {
            "type": "task",
            "id": "5.3c",
            "name": "Live PICK! Social Operations & Continuous Dev",
            "window": "May 27 - Dec 31",
            "metric": "645M/mo (Y2) -> 948M/mo (Y5)",
            "start": 4.0,
            "end": 28.0,
            "color": "#a6e3a1",
            "hatch": False,
            "bar_text": "5.3c Live PICK! Social Membership & Matchmaking (351 May-27 -> 807 Dec-27 -> 1,489 Dec-28 -> 2,218 Dec-31)",
        },
        # Pillar 4: 5.4 Phased 14-Court Second Venue Expansion
        {
            "type": "pillar",
            "id": "5.4",
            "name": "PILLAR 4: PHASED 14-COURT SECOND VENUE EXPANSION (8.0B EQUITY)",
            "window": "Jan 27 - Dec 31",
            "metric": "20 Courts | 12.42B Y3 EBITDA",
            "start": 0.0,
            "end": 28.0,
            "color": "#cba6f7",
            "row_bg": "#292338",
        },
        {
            "type": "task",
            "id": "5.4a",
            "name": "Venue 1 6-Court Operations & Phase 1 Proof Gate",
            "window": "Jan 27 - Sep 27",
            "metric": "+500M Tranche 1 (9 Months)",
            "start": 0.0,
            "end": 9.0,
            "color": "#94e2d5",
            "hatch": True,
            "bar_text": "5.4a Phase 1 Proof Gate (M1-M9: 684 Members)",
        },
        {
            "type": "task",
            "id": "5.4b",
            "name": "Venue 2 14-Court Construction",
            "window": "Oct 27 - Jan 28",
            "metric": "+7,500M Tranche 2 (7.0B Fit-Out)",
            "start": 9.0,
            "end": 13.0,
            "color": "#f38ba8",
            "hatch": False,
            "bar_text": "5.4b 14-Court Build (4m)",
        },
        {
            "type": "task",
            "id": "5.4c",
            "name": "Venue 2 Grand Opening (20 Courts Live) & 6-to-9 Month Fill-Up Ramp",
            "window": "Feb 28 - Sep 28",
            "metric": "Ramp 28% -> 100% Steady-State",
            "start": 13.0,
            "end": 21.0,
            "color": "#fab387",
            "hatch": True,
            "bar_text": "5.4c 6-9 Mo Fill-Up Ramp (Feb-Sep 28)",
        },
        {
            "type": "task",
            "id": "5.4d",
            "name": "Full 20-Court Steady-State Operations Across Both Flagship HCMC Clubs",
            "window": "Oct 28 - Dec 31",
            "metric": "Y3 EBITDA: 12.42B (3.55x Target)",
            "start": 21.0,
            "end": 28.0,
            "color": "#cba6f7",
            "hatch": False,
            "bar_text": "5.4d 20-Court Steady-State (Y3: 12.42B EBITDA)",
        },
    ]

    n_rows = len(rows_spec)
    fig, ax = plt.subplots(figsize=(21.5, 11.2), dpi=240)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")

    x_left_min = -12.5
    x_right_max = 28.05
    hdr_top = n_rows + 1.65
    hdr_mid = n_rows + 0.65
    hdr_bot = float(n_rows)

    ax.set_xlim(x_left_min, x_right_max)
    ax.set_ylim(-1.45, hdr_top + 0.35)

    # Top timeline banners across [0.0 .. 28.0]
    period_banners = [
        ("Q1 2027 (JAN-MAR)", "Pricing & App Build", 0.0, 3.0, "#181825", "#313244"),
        ("Q2 2027 (APR-JUN)", "PICK! Social Launch", 3.0, 6.0, "#291e28", "#452738"),
        ("Q3 2027 (JUL-SEP)", "Waitlist Proof Gate", 6.0, 9.0, "#181825", "#313244"),
        ("Q4 2027 (OCT-DEC)", "+7.5B & V2 Build", 9.0, 12.0, "#1e1e2e", "#313244"),
        ("Q1 2028 (JAN-MAR)", "20 Courts Open", 12.0, 15.0, "#1e2926", "#233b30"),
        ("Q2 2028 (APR-JUN)", "6-9 Mo Fill-Up Ramp", 15.0, 18.0, "#181825", "#313244"),
        ("Q3 2028 (JUL-SEP)", "100% Steady Fill", 18.0, 21.0, "#1e1e2e", "#313244"),
        ("Q4 2028 (OCT-DEC)", "1,489 Y2 Members", 21.0, 24.0, "#181825", "#313244"),
        ("FY 2029 - FY 2031 (Y3-Y5)", "Y3: 12.42B EBITDA (3.55x)", 24.0, 28.0, "#1e292e", "#253b3f"),
    ]
    for p_title, p_sub, x0, x1, bg_c, hdr_c in period_banners:
        ax.add_patch(
            patches.Rectangle((x0, 0.0), x1 - x0, n_rows, facecolor=bg_c, edgecolor="none", zorder=0)
        )
        ax.add_patch(
            patches.Rectangle(
                (x0, hdr_mid),
                x1 - x0,
                hdr_top - hdr_mid,
                facecolor=hdr_c,
                edgecolor="#585b70",
                linewidth=1.0,
                zorder=4,
            )
        )
        ax.text(
            (x0 + x1) / 2.0,
            hdr_mid + 0.66,
            p_title,
            ha="center",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )
        ax.text(
            (x0 + x1) / 2.0,
            hdr_mid + 0.28,
            p_sub,
            ha="center",
            va="center",
            fontsize=6.6,
            fontweight="normal",
            color="#bac2de",
            zorder=5,
        )

    # Sub-header cells: Months M1..M24 + Y3, Y4, Y5
    sub_cells = [(float(m), 1.0, f"M{m + 1}") for m in range(24)] + [
        (24.0, 1.333, "Y3 (FY29)"),
        (25.333, 1.333, "Y4 (FY30)"),
        (26.666, 1.334, "Y5 (FY31)"),
    ]
    for idx_c, (cx, cw, clbl) in enumerate(sub_cells):
        c_bg = "#313244" if (idx_c // 3) % 2 == 0 else "#45475a"
        ax.add_patch(
            patches.Rectangle(
                (cx, hdr_bot),
                cw,
                hdr_mid - hdr_bot,
                facecolor=c_bg,
                edgecolor="#585b70",
                linewidth=0.7,
                zorder=4,
            )
        )
        ax.axvline(cx, ymin=0.06, ymax=0.92, color="#313244", linewidth=0.55, linestyle=":", zorder=1)
        ax.text(
            cx + cw / 2.0,
            (hdr_bot + hdr_mid) / 2.0,
            clbl,
            ha="center",
            va="center",
            fontsize=6.9,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )

    for qb in (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 28.0):
        ax.axvline(qb, ymin=0.06, ymax=0.92, color="#585b70", linewidth=1.0, zorder=2)

    # Left WBS Table Header
    col_defs = [
        ("SOLUTION PILLAR / PHASED INITIATIVE (WBS)", -12.5, 0.0, "left"),
    ]
    ax.add_patch(
        patches.Rectangle(
            (x_left_min, hdr_bot),
            -x_left_min,
            hdr_top - hdr_bot,
            facecolor="#11111b",
            edgecolor="#585b70",
            linewidth=1.0,
            zorder=4,
        )
    )
    for c_title, cx0, cx1, _ in col_defs:
        ax.text(
            (cx0 + cx1) / 2.0,
            (hdr_bot + hdr_top) / 2.0,
            c_title,
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color="#cdd6f4",
            zorder=5,
        )
        ax.axvline(cx1, ymin=0.06, ymax=0.95, color="#45475a", linewidth=0.7, zorder=3)

    # Render each Pillar Header Row & Task Bar Row
    for r_idx, r in enumerate(rows_spec):
        y_top = n_rows - r_idx
        y_bot = y_top - 1.0
        y_mid = y_bot + 0.5

        if r["type"] == "pillar":
            ax.add_patch(
                patches.Rectangle(
                    (x_left_min, y_bot),
                    28.0 - x_left_min,
                    1.0,
                    facecolor=r["row_bg"],
                    edgecolor="#45475a",
                    linewidth=0.9,
                    zorder=1,
                )
            )
            ax.axhline(y_top, color=r["color"], linewidth=1.55, zorder=4)
            ax.axhline(y_bot, color=r["color"], linewidth=1.05, zorder=4)
            ax.add_patch(
                patches.Rectangle((x_left_min, y_bot), 0.22, 1.0, facecolor=r["color"], edgecolor="none", zorder=5)
            )
            ax.text(-12.1, y_mid, r["name"], ha="left", va="center", fontsize=7.8, fontweight="bold", color="#cdd6f4", zorder=5)

            # Thin summary span bar on Gantt side (just like 01_pick_social & 01_venue_scaling)
            ax.add_patch(
                patches.FancyBboxPatch(
                    (r["start"], y_mid - 0.13),
                    r["end"] - r["start"],
                    0.26,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    facecolor=r["color"],
                    edgecolor="#11111b",
                    linewidth=0.95,
                    alpha=0.90,
                    zorder=4,
                )
            )
        else:
            row_bg = "#1e1e2e" if r_idx % 2 == 0 else "#181825"
            ax.add_patch(
                patches.Rectangle(
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

            ax.plot([-11.95, -11.95], [y_bot + 0.16, y_top - 0.16], color=r["color"], linewidth=1.15, alpha=0.8, zorder=4)
            ax.plot([-11.95, -11.70], [y_mid, y_mid], color=r["color"], linewidth=1.0, alpha=0.8, zorder=4)
            ax.text(-11.55, y_mid, r["name"], ha="left", va="center", fontsize=7.4, fontweight="bold", color="#cdd6f4", zorder=5)

            bar_h = 0.56
            bar_y = y_mid - bar_h / 2.0
            bw = r["end"] - r["start"]
            is_h = bool(r["hatch"])
            ax.add_patch(
                patches.FancyBboxPatch(
                    (r["start"], bar_y),
                    bw,
                    bar_h,
                    boxstyle="round,pad=0.01,rounding_size=0.08",
                    facecolor=r["color"],
                    edgecolor="#11111b",
                    linewidth=1.2,
                    alpha=0.92,
                    hatch="///" if is_h else None,
                    zorder=4,
                )
            )

    ax.plot([x_left_min, 28.0], [0.0, 0.0], color="#585b70", linewidth=1.1, zorder=5)

    # Key vertical milestone markers (May-27 Platform Launch at x=4.0, Oct-27 V2 Build at x=9.0, Feb-28 V2 Open at x=13.0)
    ax.axvline(4.0, ymin=0.05, ymax=0.92, color="#a6e3a1", linewidth=2.0, linestyle="--", zorder=6)
    ax.axvline(9.0, ymin=0.05, ymax=0.92, color="#f38ba8", linewidth=2.0, linestyle="--", zorder=6)
    ax.axvline(13.0, ymin=0.05, ymax=0.92, color="#fab387", linewidth=2.0, linestyle="--", zorder=6)

    # Bottom summary pills
    ax.text(
        4.0,
        -0.78,
        "M5 (MAY 2027):\nPICK! SOCIAL LAUNCH",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#11111b",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#a6e3a1", edgecolor="#11111b", linewidth=1.1),
        zorder=9,
    )
    ax.text(
        9.0,
        -0.78,
        "M10 (OCT 2027):\n+7.5B & V2 BUILD START",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#11111b",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f38ba8", edgecolor="#11111b", linewidth=1.1),
        zorder=9,
    )
    ax.text(
        14.2,
        -0.78,
        "M14 (FEB 2028):\nVENUE 2 OPENS (20 COURTS)",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#11111b",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fab387", edgecolor="#11111b", linewidth=1.1),
        zorder=9,
    )
    ax.text(
        23.5,
        -0.78,
        "FY 2029 (YEAR 3): 33.23B REV & 12.42B EBITDA\n(3.55x MINH TRAN 3.50B TARGET | 1.72x AT -30% STRESS)",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#11111b",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#74c7ec", edgecolor="#11111b", linewidth=1.1),
        zorder=9,
    )

    ax.set_title(
        "Solution Implementation Schedule",
        fontsize=14.0,
        fontweight="bold",
        color="#cdd6f4",
        pad=14,
    )
    ax.axis("off")

    out_path = output_dir / "00_executive_solution_master_roadmap.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path
