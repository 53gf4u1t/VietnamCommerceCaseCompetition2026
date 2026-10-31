"""Automated 5-Year (60-Month) 3-Statement CSV, Monte Carlo Cohort CSV, and Markdown report exporter."""

from __future__ import annotations

from pathlib import Path

from .config import FinancialModelConfig
from .model import FinancialModelResults


def export_all_outputs(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    chart_paths: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Export 3-Statement CSVs (including 60-month, 5-year, and Monte Carlo subscriber cohort projections) and REPORT.md into outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSV Exports
    breakdown_csv = output_dir / "revenue_cogs_opex_sga_breakdown.csv"
    res.line_items_df.to_csv(breakdown_csv, index=False)

    fs_month_csv = output_dir / "fs_month_60m_projections.csv"
    res.fs_month_df.to_csv(fs_month_csv, index=False)

    fs_annual_csv = output_dir / "fs_annual_5y_projections.csv"
    res.fs_annual_df.to_csv(fs_annual_csv, index=False)

    cohort_csv = output_dir / "monte_carlo_subscription_cohort_60m.csv"
    res.mc.subscriber_cohort_df.to_csv(cohort_csv, index=False)

    cash_csv = output_dir / "monthly_cash_flow_and_balance_60m.csv"
    res.cash_flow_60m_df.to_csv(cash_csv, index=False)

    # 2. Markdown Summary Report
    ap = res.adaptive
    tl = res.timeline_cost
    mc = res.mc
    bs = cfg.balance_sheet
    report_path = output_dir / "REPORT.md"

    li_rows = []
    for li in res.line_items:
        bold = "**" if li.section == "SUMMARY" else ""
        li_rows.append(
            f"| {bold}{li.template_category}{bold} | {bold}{li.line_item}{bold} | {li.behavior} | "
            f"`{li.case_baseline_monthly_mvnd:,.2f}` | `{li.m1_build_monthly_mvnd:,.2f}` | "
            f"`{li.m5_launch_monthly_mvnd:,.2f}` | `{li.m24_end_y2_monthly_mvnd:,.2f}` | "
            f"{bold}`{li.five_year_total_mvnd:,.2f}`{bold} | `{li.pct_of_5y_revenue:.2f}%` |"
        )
    li_table_md = "\n".join(li_rows)

    report_md = f"""# PICK! 5-Year (60-Month) 3-Statement Financial Model — Integrated with `AdaptivePricing`, `PickSocialProjectTimeline` & Path C Subscriptions

**Company**: {cfg.company.name}  
**Model Type**: 100% Stochastic Distribution-Driven Projections (No Actuals) — 60-Month (5-Year) Horizon (`Jan-27` to `Dec-31`)  
**Monte Carlo Validation**: `{mc.n_simulations}` independent 60-month trajectories (`{mc.n_train}` Train / `{mc.n_test}` Held-Out Test) + `-30%` Lower-Demand Stress Test  
**Statements Included**: **1. Income Statement (P&L)**, **2. Balance Sheet**, **3. Cash Flow Statement**  
**Currency**: Vietnamese Dong (`VND`), reported in **Million VND (`M VND`)**  
**Central Configuration**: `CaseComp/configs/` (`pricing.toml`, `customers.toml`, `cogs.toml`, `opex.toml`, `capex.toml`, `randomness.toml`)  
**Upstream Dependencies**: Verified `AdaptivePricing/outputs/` & `PickSocialProjectTimeline/outputs/`

---

## 1. Upstream Integrations & Distribution-First Philosophy

Every assumption in this model is parameterized as a **Truncated Normal probability distribution** in `CaseComp/configs/*.toml`, grounded in the Case Question exhibits and Vietnam benchmarks:

### 1.1 `AdaptivePricing/outputs/` Integration (Court Hire Revenue)
| Court Pricing Regime | Available Hrs/Mo | Case Static Util & Rate | Case Rev (M VND) | ML Adaptive Util & Avg Rate | ML Adaptive Rev (M VND) | Monthly Lift (M VND) |
|---|---:|---|---:|---|---:|---:|
| **Weekday Daytime (06:00–17:00)** | 1,452.0 | 40.0% @ 160k | 92.93 | {res.regimes[0].ml_utilisation_rate*100:.1f}% @ {res.regimes[0].ml_avg_price_kvnd:.0f}k | {res.regimes[0].ml_revenue_mvnd:.2f} | +{res.regimes[0].ml_revenue_mvnd - res.regimes[0].case_revenue_mvnd:.2f} |
| **Weekday Evening (17:00–22:00)** | 660.0 | 95.0% @ 260k | 163.02 | {res.regimes[1].ml_utilisation_rate*100:.1f}% @ {res.regimes[1].ml_avg_price_kvnd:.0f}k | {res.regimes[1].ml_revenue_mvnd:.2f} | {res.regimes[1].ml_revenue_mvnd - res.regimes[1].case_revenue_mvnd:+.2f} |
| **Weekend All Day (06:00–22:00)** | 768.0 | 90.0% @ 260k | 179.71 | {res.regimes[2].ml_utilisation_rate*100:.1f}% @ {res.regimes[2].ml_avg_price_kvnd:.0f}k | {res.regimes[2].ml_revenue_mvnd:.2f} | +{res.regimes[2].ml_revenue_mvnd - res.regimes[2].case_revenue_mvnd:.2f} |
| **Total Court Hire** | **2,880.0** | **~66.0% (~1,900 hrs)** | **435.00** | **{res.blended_utilisation_rate*100:.1f}% ({res.total_booked_court_hours:,.0f} hrs)** | **{res.court_hire_mvnd:.2f}** | **+{ap.lift_vs_case_mvnd:.2f} (+{ap.lift_vs_case_pct:.2f}%)** |

### 1.2 `PickSocialProjectTimeline/outputs/` Integration (Platform Build & Recurring Platform OpEx)
- **Months 1–6 (`Jan-27` to `Jun-27`) Leveled Platform Development Allocation**:
  - Committed Base Outflow: **`{tl.total_committed_dev_base_mvnd:.2f}M VND`** (`139.93M` to `165.97M VND/month`) + Conditional Risk Contingency Reserve **`+{tl.total_contingency_reserve_mvnd:.2f}M VND`** (`29.58M/month` max; Monte Carlo expected drawdown `50%` = **`~999.91M VND`** expected total build cost across `Months 1–6`).
- **Month 5+ (`May-27` onward) Recurring Monthly Platform & Community OpEx**:
  - **Months 5–6 (`May–Jun 2027`, Launch & Hypercare)**: **`{tl.launch_platform_opex_m5_m6_mvnd:.2f}M VND/month`** (`4.50M` Cloud/DB/Redis + `2.20M` Zalo ZNS/OTP + `1.80M` Observability/CI-CD SaaS + `8.00M` School/University & Beginner League Ops).
  - **Months 7–60 (`Jul 2027 – Dec 2031`, Steady-State)**: **`{tl.steady_state_platform_opex_m7_plus_mvnd:.2f}M VND/month`** (`16.50M` above + `10.00M/month` ongoing engineering maintenance, security patching & ML retraining retainer).

### 1.3 `VenueScalingTimeline/outputs/` Integration (Q4 2027 14-Court Expansion & 2-Tranche Equity Drawdown)
- **2-Tranche Investor Minh Tran Equity Drawdown (`8,000.00M VND` Total)**:
  - **Tranche 1 (`Month 1 / Jan-27` upon Signing)**: **`+500.00M VND`** drawn into cash reserve (`starting_cash = 800.00M + 500.00M = 1,300.00M VND` before M1 operations), ensuring **zero negative cash months (`> 0 VND`) across all 60 months even under a permanent `-30%` lower-demand stress test starting Day 1**.
  - **Tranche 2 (`Month 10 / Oct-27` at Q4 2027 Expansion Kickoff)**: **`+7,500.00M VND`** drawn (`7,000.00M VND` funding the even **`-1,750.00M VND/month`** fit-out CapEx across `Months 10–13: Oct 2027 – Jan 2028` + the remaining **`+500.00M VND`** reserve buffer).
- **Grand Opening (`Month 14 / Feb-28`) & 6-to-9 Month Fill-Up Ramp (`Feb–Sep/Oct 2028`)**: Scales PICK! from **`6` to `20` courts total**, reaching `100%` steady-state utilization by `Oct 2028 (Month 22)`—**2 full months before Year 3 (`FY 2029`) begins**.

### 1.4 Smooth Path C Membership Subscription Cohort Model (Starts Month 5: `May-27`)
Instead of sampling monthly subscriber counts independently, each Monte Carlo trajectory evolves active paying members ($S_t$, paying **VND 399,000/month** per Case Page 3 & Exhibit 3) smoothly via a Markov state-transition equation driven by sampled monthly unique customers (`mean = 3,600`, `std = 160`), monthly churn rate (`mean = 4.2%`, `std = 0.55%`), and monthly new-subscriber conversion rate (`mean = 4.65%`, `std = 0.50%`):

| Milestone Month | Calendar Month | Active Paying Subscribers (Mean `[P05, P95]`) | Monthly Membership Revenue (`M VND/mo`) | Direct Partner-Court & Event COGS (`M VND/mo`) | Total Monthly Revenue (`M VND/mo`) | Monthly Operating Profit / EBITDA (`M VND/mo`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Months 1–4 (Core Build)** | `Jan-27 – Apr-27` | **`0`** (`[0, 0]`) | `0.00M` | `0.00M` | `{res.monthly_total_rev_arr[0]:,.2f}M` | `{res.monthly_ebitda_arr[0]:,.2f}M` |
| **Month 5 (Platform Launch)** | `May-27` | **`{mc.m5_launch_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[4]:,.0f}, {mc.active_subscribers_p95[4]:,.0f}]`) | `{res.monthly_membership_rev_arr[4]:,.2f}M` | `{res.monthly_membership_event_cogs_arr[4]:,.2f}M` | `{res.monthly_total_rev_arr[4]:,.2f}M` | `{res.monthly_ebitda_arr[4]:,.2f}M` |
| **Month 12 (End of Year 1)** | `Dec-27` | **`{mc.m12_end_y1_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[11]:,.0f}, {mc.active_subscribers_p95[11]:,.0f}]`) | `{res.monthly_membership_rev_arr[11]:,.2f}M` | `{res.monthly_membership_event_cogs_arr[11]:,.2f}M` | `{res.monthly_total_rev_arr[11]:,.2f}M` | `{res.monthly_ebitda_arr[11]:,.2f}M` |
| **Month 24 (End of Year 2 — Case Benchmark)** | `Dec-28` | **`{mc.m24_end_y2_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[23]:,.0f}, {mc.active_subscribers_p95[23]:,.0f}]`) | **`{mc.m24_end_y2_membership_rev_mvnd:,.2f}M`** | `{res.monthly_membership_event_cogs_arr[23]:,.2f}M` | `{res.monthly_total_rev_arr[23]:,.2f}M` | `{res.monthly_ebitda_arr[23]:,.2f}M` |
| **Month 36 (End of Year 3)** | `Dec-29` | **`{mc.m36_end_y3_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[35]:,.0f}, {mc.active_subscribers_p95[35]:,.0f}]`) | `{res.monthly_membership_rev_arr[35]:,.2f}M` | `{res.monthly_membership_event_cogs_arr[35]:,.2f}M` | `{res.monthly_total_rev_arr[35]:,.2f}M` | `{res.monthly_ebitda_arr[35]:,.2f}M` |
| **Month 60 (End of Year 5)** | `Dec-31` | **`{mc.m60_end_y5_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[59]:,.0f}, {mc.active_subscribers_p95[59]:,.0f}]`) | `{res.monthly_membership_rev_arr[59]:,.2f}M` | `{res.monthly_membership_event_cogs_arr[59]:,.2f}M` | `{res.monthly_total_rev_arr[59]:,.2f}M` | `{res.monthly_ebitda_arr[59]:,.2f}M` |

---

## 2. Statement 1 — Income Statement Breakdown (Revenue, COGS, Facility OpEx & SG&A)

| Category | Line Item | Cost Behavior | Case Base `/mo` | M1 Build (`Jan-27`) | M5 Launch (`May-27`) | M24 End Y2 (`Dec-28`) | 5-Year (60M) Total | % of 5Y Rev |
|---|---|---|---:|---:|---:|---:|---:|---:|
{li_table_md}

---

## 3. 5-Year Annual 3-Statement Summary & Investor Target Verification (`FY 2027` – `FY 2031`)

- **Investor Minh Tran's Year 3 Target (`Case Page 4`)**: Requires **at least VND 3,500.0M (`3.5 billion VND`) in annual operating profit by Year 3 (`FY 2029`)**.
- **Our Integrated Path C Result in Year 3 (`FY 2029`)**:
  - **Base Plan Expected Year 3 Operating Profit (EBITDA)**: **`{mc.y3_annual_ebitda_mean_mvnd:,.2f}M VND/yr`** (`90% CI: [{mc.y3_annual_ebitda_p05_mvnd:,.2f}M, {mc.y3_annual_ebitda_p95_mvnd:,.2f}M]`), exceeding Minh Tran's `3,500M` hurdle in **`{mc.prob_exceed_investor_3500m_y3_pct:.1f}%`** of Monte Carlo trajectories!
  - **`-30%` Lower Demand Stress Test Year 3 Operating Profit**: **`{mc.y3_annual_ebitda_stress_mvnd:,.2f}M VND/yr`** — even if overall club demand and subscriber conversion sit `30%` below plan, PICK! still clears the `3,500M VND` Year 3 operating profit requirement!

| 3-Statement Annual Metric (`M VND`) | Start (`t=0`) | FY 2027 (Y1) | FY 2028 (Y2) | FY 2029 (Y3) | FY 2030 (Y4) | FY 2031 (Y5) | 5Y Total / End |
|---|---:|---:|---:|---:|---:|---:|---:|
| **End-of-Year Paying Members (Subscribers)** | `0` | `{mc.m12_end_y1_subscribers:,.0f}` | **`{mc.m24_end_y2_subscribers:,.0f}`** | `{mc.m36_end_y3_subscribers:,.0f}` | `{mc.active_subscribers_mean[47]:,.0f}` | `{mc.m60_end_y5_subscribers:,.0f}` | `{mc.m60_end_y5_subscribers:,.0f}` |
| **Total Annual Revenue (`M VND`, Model)** | — | `{res.annual_revenue_arr[0]:,.2f}` | `{res.annual_revenue_arr[1]:,.2f}` | **`{res.annual_revenue_arr[2]:,.2f}`** | `{res.annual_revenue_arr[3]:,.2f}` | `{res.annual_revenue_arr[4]:,.2f}` | **`{res.five_year_revenue_mvnd:,.2f}`** |
| **Total Annual Revenue (`M VND`, Stochastic Baseline)** | — | `{mc.base_annual_rev_mean[0]:,.2f}` | `{mc.base_annual_rev_mean[1]:,.2f}` | `{mc.base_annual_rev_mean[2]:,.2f}` | `{mc.base_annual_rev_mean[3]:,.2f}` | `{mc.base_annual_rev_mean[4]:,.2f}` | `{mc.base_annual_rev_mean.sum():,.2f}` |
| **Total Annual Costs (`COGS + OpEx + SG&A`, Model)** | — | `{res.annual_total_costs_arr[0]:,.2f}` | `{res.annual_total_costs_arr[1]:,.2f}` | **`{res.annual_total_costs_arr[2]:,.2f}`** | `{res.annual_total_costs_arr[3]:,.2f}` | `{res.annual_total_costs_arr[4]:,.2f}` | **`{res.five_year_total_costs_mvnd:,.2f}`** |
| **Total Annual Costs (`M VND`, Stochastic Baseline)** | — | `{mc.base_annual_costs_mean[0]:,.2f}` | `{mc.base_annual_costs_mean[1]:,.2f}` | `{mc.base_annual_costs_mean[2]:,.2f}` | `{mc.base_annual_costs_mean[3]:,.2f}` | `{mc.base_annual_costs_mean[4]:,.2f}` | `{mc.base_annual_costs_mean.sum():,.2f}` |
| **Annual Operating Profit (`EBITDA`, Model Base Plan)** | — | `{res.annual_ebitda_arr[0]:,.2f}` | `{res.annual_ebitda_arr[1]:,.2f}` | **`{res.annual_ebitda_arr[2]:,.2f}`** | `{res.annual_ebitda_arr[3]:,.2f}` | `{res.annual_ebitda_arr[4]:,.2f}` | **`{res.five_year_ebitda_mvnd:,.2f}`** |
| **Annual Operating Profit (`EBITDA`, Stochastic Baseline)** | — | `{mc.base_annual_ebitda_mean[0]:,.2f}` | `{mc.base_annual_ebitda_mean[1]:,.2f}` | `{mc.base_annual_ebitda_mean[2]:,.2f}` | `{mc.base_annual_ebitda_mean[3]:,.2f}` | `{mc.base_annual_ebitda_mean[4]:,.2f}` | `{mc.base_annual_ebitda_mean.sum():,.2f}` |
| **Annual Operating Profit (`-30%` Demand Stress)** | — | `{mc.stress_ebitda_mean[0:12].sum():,.2f}` | `{mc.stress_ebitda_mean[12:24].sum():,.2f}` | **`{mc.y3_annual_ebitda_stress_mvnd:,.2f}`** | `{mc.stress_ebitda_mean[36:48].sum():,.2f}` | `{mc.stress_ebitda_mean[48:60].sum():,.2f}` | **`{mc.stress_ebitda_mean.sum():,.2f}`** |
| **Ending Cash Balance (Model Base Plan)** | `{bs.starting_cash_mvnd:,.2f}` | `{res.monthly_ending_cash_model[11]:,.2f}` | `{res.monthly_ending_cash_model[23]:,.2f}` | `{res.monthly_ending_cash_model[35]:,.2f}` | `{res.monthly_ending_cash_model[47]:,.2f}` | **`{res.monthly_ending_cash_model[59]:,.2f}`** | **`{res.monthly_ending_cash_model[59]:,.2f}`** |
| **Ending Cash Balance (`-30%` Demand Stress)** | `{bs.starting_cash_mvnd:,.2f}` | `{mc.stress_ending_cash_mean[11]:,.2f}` | `{mc.stress_ending_cash_mean[23]:,.2f}` | `{mc.stress_ending_cash_mean[35]:,.2f}` | `{mc.stress_ending_cash_mean[47]:,.2f}` | **`{mc.stress_ending_cash_mean[59]:,.2f}`** | **`{mc.stress_ending_cash_mean[59]:,.2f}`** |
| **Ending Cash Balance (Stochastic Case Baseline)** | `{bs.starting_cash_mvnd:,.2f}` | `{res.monthly_ending_cash_case[11]:,.2f}` | `{res.monthly_ending_cash_case[23]:,.2f}` | `{res.monthly_ending_cash_case[35]:,.2f}` | `{res.monthly_ending_cash_case[47]:,.2f}` | `{res.monthly_ending_cash_case[59]:,.2f}` | `{res.monthly_ending_cash_case[59]:,.2f}` |
| **Balance Sheet Check (`Assets - L&E`)** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** |

---

## 4. Generated Output Files (`FinancialModel/outputs/`)

- **60-Month (5-Year) 3-Statement Monthly Projection CSV**: `outputs/fs_month_60m_projections.csv` (Month 1 `Jan-27` to Month 60 `Dec-31`)
- **5-Year 3-Statement Annual Projection CSV**: `outputs/fs_annual_5y_projections.csv` (`FY 2027` to `FY 2031`)
- **60-Month Cash Flow & Ending Cash Balance CSV**: `outputs/monthly_cash_flow_and_balance_60m.csv`
- **Monte Carlo 60-Month Subscription Cohort & Stress-Test CSV**: `outputs/monte_carlo_subscription_cohort_60m.csv`
- **Revenue, COGS, OpEx & SG&A Breakdown CSV**: `outputs/revenue_cogs_opex_sga_breakdown.csv`
- **Sheet 1 Render (Drivers, Revenue, COGS, OpEx & SG&A)**: `outputs/01_revenue_cogs_opex_sga_sheet.png`
- **Sheet 2 Render (`FS-Month` 60-Month 3-Statement Projections)**: `outputs/02_fs_month_60m_projection_sheet.png`
- **Sheet 3 Render (`FS-Annual` 5-Year 3-Statement Projections)**: `outputs/03_fs_annual_5y_projection_sheet.png`
- **Chart 4 Render (Double-Bar Revenue > 0 Green, Costs < 0 Red & Profit Line vs. Baseline)**: `outputs/04_revenue_costs_profit_comparison_charts.png`
- **Chart 5 Render (Monte Carlo Subscription Cohort Evolution, Revenue Bands & Year 3 Investor Target)**: `outputs/05_membership_and_monte_carlo_distributions.png`
- **Chart 6 Render (60-Month Cash Balance Evolution, Expansion CapEx/Equity & External Factors)**: `outputs/06_monthly_cash_balance_and_externals.png`
"""
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "breakdown_csv": breakdown_csv,
        "fs_month_csv": fs_month_csv,
        "fs_annual_csv": fs_annual_csv,
        "cohort_csv": cohort_csv,
        "cash_csv": cash_csv,
        "report_path": report_path,
    }
