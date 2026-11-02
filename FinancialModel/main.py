"""FinancialModel component runner (invoked by root CaseComp/main.py)."""

from __future__ import annotations

from pathlib import Path

try:
    from FinancialModel.src.config import load_config
    from FinancialModel.src.model import build_financial_model, FinancialModelResults
    from FinancialModel.src.visualizer import (
        plot_executive_solution_master_roadmap,
        plot_drivers_and_cost_breakdown_sheet,
        plot_fs_month_projection_sheet,
        plot_fs_annual_and_structure_dashboard,
        plot_revenue_costs_profit_comparison_charts,
        plot_membership_and_monte_carlo_distributions,
        plot_monthly_cash_balance_and_externals,
    )
    from FinancialModel.src.reporter import export_all_outputs
except ImportError:
    from src.config import load_config
    from src.model import build_financial_model, FinancialModelResults
    from src.visualizer import (
        plot_executive_solution_master_roadmap,
        plot_drivers_and_cost_breakdown_sheet,
        plot_fs_month_projection_sheet,
        plot_fs_annual_and_structure_dashboard,
        plot_revenue_costs_profit_comparison_charts,
        plot_membership_and_monte_carlo_distributions,
        plot_monthly_cash_balance_and_externals,
    )
    from src.reporter import export_all_outputs


def run_financial_model(
    config_path: Path | None = None,
    output_dir: Path | None = None,
    adaptive_output_dir: Path | None = None,
    timeline_output_dir: Path | None = None,
    venue_scaling_output_dir: Path | None = None,
    marketing_output_dir: Path | None = None,
) -> FinancialModelResults:
    """Execute the PICK! 5-Year (60-Month) 3-Statement Financial Model using central `configs/`."""
    base_dir = Path(__file__).resolve().parent
    cfg_path = config_path or (base_dir.parent / "configs")
    out_dir = output_dir or (base_dir / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FinancialModel 1/4] Loading central distribution configs from {cfg_path.relative_to(base_dir.parent)}/...")
    cfg = load_config(cfg_path)

    print("[FinancialModel 2/4] Verifying upstream outputs & running 300 Monte Carlo 60-month (5Y) Financial Statements simulations...")
    res = build_financial_model(
        cfg,
        adaptive_output_dir=adaptive_output_dir,
        timeline_output_dir=timeline_output_dir,
        venue_scaling_output_dir=venue_scaling_output_dir,
        marketing_output_dir=marketing_output_dir,
    )

    print(f"[FinancialModel 3/4] Rendering Financial Statements (FS-Month, FS-Annual & Taxes) tables, stacked revenue/cost charts, Monte Carlo cohort & cash reserves plots in {out_dir.relative_to(base_dir.parent)}...")
    c0 = plot_executive_solution_master_roadmap(cfg, res, out_dir)
    s1 = plot_drivers_and_cost_breakdown_sheet(cfg, res, out_dir)
    s2 = plot_fs_month_projection_sheet(cfg, res, out_dir)
    s3 = plot_fs_annual_and_structure_dashboard(cfg, res, out_dir)
    c4 = plot_revenue_costs_profit_comparison_charts(cfg, res, out_dir)
    c5 = plot_membership_and_monte_carlo_distributions(cfg, res, out_dir)
    c6 = plot_monthly_cash_balance_and_externals(cfg, res, out_dir)

    print(f"[FinancialModel 4/4] Exporting FS-Month, FS-Annual, Vietnam Tax Estimations CSV, cash reserves CSV, Monte Carlo cohort CSV & REPORT.md to {out_dir.relative_to(base_dir.parent)}...")
    exported = export_all_outputs(cfg, res, [c0, s1, s2, s3, c4, c5, c6], out_dir)

    ap = res.adaptive
    mc = res.mc
    print("\n" + "=" * 86)
    print("PICK! 5-YEAR (60-MONTH) FINANCIAL STATEMENTS MODEL (FS-MONTH & FS-ANNUAL) - SUMMARY")
    print("=" * 86)
    print(f"  Verified AdaptivePricing Files : {', '.join(ap.verified_files)}")
    print(f"  Verified Timeline Cost Files   : {', '.join(res.timeline_cost.verified_files)}")
    print(f"  Verified VenueScaling Files    : {', '.join(res.venue_scaling.verified_files)}")
    print(f"  Verified Marketing Files       : {', '.join(res.marketing.verified_files)}")
    print(f"  Revenue: Court Hire (5Y Avg)   : {res.court_hire_mvnd:8.2f} M VND/mo (M1 Venue 1 = {res.venue1_court_hire_m1_mvnd:.2f}M | +14 Courts Opens M14)")
    print(
        f"  Revenue: Subscriptions (M5+)   : M5 Launch = {res.monthly_membership_rev_arr[4]:6.2f}M ({mc.m5_launch_subscribers:,.0f} members) -> "
        f"M24 End Y2 = {res.m24_membership_rev_mvnd:6.2f}M ({mc.m24_end_y2_subscribers:,.0f} members)"
    )
    print(
        f"  Revenue: 12-Dist Ancillaries   : Paddle/Balls = {res.paddle_balls_mvnd:6.2f}M/mo | "
        f"Coaching = {res.coaching_mvnd:6.2f}M/mo | F&B = {res.food_drinks_mvnd:6.2f}M/mo"
    )
    print(f"  5-Year Cumulative Revenue      : {res.five_year_revenue_mvnd:9.2f} M VND  |  5-Year Operating Costs: {res.five_year_total_costs_mvnd:9.2f} M VND")
    print(f"  5-Year Cumulative EBITDA       : {res.five_year_ebitda_mvnd:9.2f} M VND  |  Year 3 Annual EBITDA: {mc.y3_annual_ebitda_mean_mvnd:8.2f} M VND (Target >= 3,500M)")
    print(f"  5-Year D&A / 20% Vietnam CIT   : D&A = {res.five_year_da_mvnd:7.2f} M VND | Accrued 20% CIT = {res.five_year_taxes_mvnd:7.2f} M VND | Net Income = {res.five_year_net_income_mvnd:8.2f} M VND")
    print(f"  Year 3 Stress (-30% Demand)    : {mc.y3_annual_ebitda_stress_mvnd:9.2f} M VND  |  Minh Tran Y3 Pass Rate: {mc.prob_exceed_investor_3500m_y3_pct:5.1f}%")
    print(f"  Ending Cash Reserves (Month 60): {res.ending_cash_60m_model_mvnd:9.2f} M VND (vs Case Baseline: {res.ending_cash_60m_case_mvnd:.2f} M VND | Min Cash: {res.min_cash_balance_model_mvnd:.2f} M in {res.min_cash_month_label})")
    print("-" * 86)
    print(f"  Rendered Sheet & Chart PNGs    : {s1.name}, {s2.name}, {s3.name}, {c4.name}, {c5.name}, {c6.name}")
    print(
        f"  60-Month & 5-Year CSVs         : {exported['fs_month_csv'].name}, {exported['fs_annual_csv'].name}, "
        f"{exported['tax_csv'].name}, {exported['cash_csv'].name}, {exported['cohort_csv'].name}, {exported['breakdown_csv'].name}"
    )
    print(f"  Markdown Report                : {exported['report_path'].name}")
    print("=" * 86)
    return res
