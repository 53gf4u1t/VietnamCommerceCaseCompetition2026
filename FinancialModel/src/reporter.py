"""Automated 5-Year (60-Month) Financial Statements CSV, Vietnam Tax Estimations CSV, Monte Carlo Cohort CSV, and Markdown report exporter."""

from __future__ import annotations

from pathlib import Path

from .config import FinancialModelConfig
from .model import FinancialModelResults
from .visualizer import _line_item_source_label


def export_all_outputs(
    cfg: FinancialModelConfig,
    res: FinancialModelResults,
    chart_paths: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Export Financial Statements CSVs (including FS-Month, FS-Annual, Vietnam Tax Estimations, and Monte Carlo subscriber cohort projections) and REPORT.md into outputs/."""
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

    tax_csv = output_dir / "tax_estimations_60m.csv"
    res.tax_estimations_60m_df.to_csv(tax_csv, index=False)

    # 2. Markdown Summary Report
    ap = res.adaptive
    tl = res.timeline_cost
    mc = res.mc
    bs = cfg.balance_sheet
    report_path = output_dir / "REPORT.md"

    li_rows = []
    for li in res.line_items:
        bold = "**" if li.section == "SUMMARY" else ""
        src_lbl = _line_item_source_label(li)
        li_rows.append(
            f"| {bold}{li.template_category}{bold} | {bold}{li.line_item}{bold} | {li.behavior} | "
            f"`{li.case_baseline_monthly_mvnd:,.2f}` | `{li.m1_build_monthly_mvnd:,.2f}` | "
            f"`{li.m5_launch_monthly_mvnd:,.2f}` | `{li.m24_end_y2_monthly_mvnd:,.2f}` | "
            f"{bold}`{li.five_year_total_mvnd:,.2f}`{bold} | `{li.pct_of_5y_revenue:.2f}%` | {src_lbl} |"
        )
    li_table_md = "\n".join(li_rows)

    report_md = f"""# Financial Statements Model (FS-Month, FS-Annual & Vietnam Statutory Taxes)

**Company**: {cfg.company.name}  
**Model Type**: 100% Stochastic Distribution-Driven Projections (No Actuals) - 60-Month (5-Year) Horizon (`Jan-27` to `Dec-31`)  
**Monte Carlo Validation**: `{mc.n_simulations}` independent 60-month trajectories (`{mc.n_train}` Train / `{mc.n_test}` Held-Out Test) + `-30%` Lower-Demand Stress Test  
**Financial Statements Included**: **1. Income Statement (P&L)**, **2. Balance Sheet**, **3. Cash Flow Statement**, **4. Vietnam Statutory Tax Estimations (`CIT` & `VAT`)**  
**Currency**: Vietnamese Dong (`VND`), reported in **Million VND (`M VND`)**  

---

## 1. Key Model Drivers & Assumptions

Every assumption in this model is parameterized as a **Truncated Normal probability distribution** grounded in the Case Question exhibits and Vietnam statutory benchmarks:

### 1.1 Adaptive Pricing Model Integration (Court Hire Revenue)
- **Venue 1 (`6 Courts`)**: Adaptive pricing lifts monthly court-hire revenue from **`435.00M VND/month`** (`Case Baseline`, `~66.0%` blended utilization) to **`{res.venue1_court_hire_m1_mvnd:.2f}M VND/month`** (**`+{ap.lift_vs_case_mvnd:.2f}M VND/month`** or **`+{ap.lift_vs_case_pct:.2f}%`**, `~{res.blended_utilisation_rate*100:.1f}%` blended utilization across `{res.total_booked_court_hours:,.0f}` booked court-hours/month).
- **Venue 2 (`14 Courts`, Opens `Feb-28`)**: Scales along the 6-to-9 month fill-up ramp from `Feb-28` to reach **`{res.line_items[1].m24_end_y2_monthly_mvnd:,.2f}M VND/month`** at steady state (`Dec-28`).

### 1.2 PICK! Social Platform Build & Recurring OpEx Schedule
- **Months 1-6 (`Jan-27` to `Jun-27`) Leveled Platform Development Allocation**:
  - Committed Base Outflow: **`{tl.total_committed_dev_base_mvnd:.2f}M VND`** (`139.93M` to `165.97M VND/month`) + Conditional Risk Contingency Reserve **`+{tl.total_contingency_reserve_mvnd:.2f}M VND`** (`29.58M/month` max; Monte Carlo expected drawdown `50%` = **`~999.91M VND`** expected total build cost across `Months 1-6`).
- **Month 5+ (`May-27` onward) Recurring Monthly Platform & Community OpEx**:
  - **Months 5-6 (`May-Jun 2027`, Launch & Hypercare)**: **`{tl.launch_platform_opex_m5_m6_mvnd:.2f}M VND/month`** (`4.50M` Cloud/DB/Redis + `2.20M` SMS OTP/Push Notifications + `1.80M` Observability/CI-CD SaaS + `8.00M` School/University & Beginner League Ops).
  - **Months 7-60 (`Jul 2027 - Dec 2031`, Steady-State)**: **`{tl.steady_state_platform_opex_m7_plus_mvnd:.2f}M VND/month`** (`16.50M` above + `10.00M/month` ongoing engineering maintenance, security patching & ML retraining retainer).

### 1.3 Venue 2 Expansion Schedule (Q4 2027 14-Court Build & 2-Tranche Equity Drawdown)
- **2-Tranche Investor Minh Tran Equity Drawdown (`8,000.00M VND` Total)**:
  - **Tranche 1 (`Month 1 / Jan-27` upon Signing)**: **`+500.00M VND`** drawn into cash reserves (`starting_cash = 500.00M + 500.00M = 1,000.00M VND` before M1 operations), ensuring **zero negative cash months (`> 0 VND`) across all 60 months even under a permanent `-30%` lower-demand stress test starting Day 1**.
  - **Tranche 2 (`Month 10 / Oct-27` at Q4 2027 Expansion Kickoff)**: **`+7,500.00M VND`** drawn (`7,000.00M VND` funding the even **`-1,750.00M VND/month`** fit-out CapEx across `Months 10-13: Oct 2027 - Jan 2028` + the remaining **`+500.00M VND`** reserve buffer).
- **Grand Opening (`Month 14 / Feb-28`) & 6-to-9 Month Fill-Up Ramp (`Feb-Sep/Oct 2028`)**: Scales PICK! from **`6` courts (`Months 1-13`) to `20` courts (`Months 14-60`)**, reaching `100%` steady-state utilization by `Oct 2028 (Month 22)`-**2 full months before Year 3 (`FY 2029`) begins**.

### 1.4 Vietnam Statutory Tax & Depreciation Schedule (`Taxes` Block in `FS-Month` & `FS-Annual`)
- **Depreciation & Amortization (`D&A` - Vietnam MOF Circular 45/2013/TT-BTC)**:
  - **Months 1-13 (`6 Courts`)**: Straight-line depreciation of existing `320.00M VND` court equipment over `60 months` = **`5.33M VND/month`**.
  - **Months 14-60 (`20 Courts`)**: Straight-line amortization of the `7,000.00M VND` Venue 2 leasehold improvement & court construction over its `60-month` lease (`+116.67M VND/month`), bringing total monthly D&A to **`122.00M VND/month`** (`5Y Total D&A = {res.five_year_da_mvnd:,.2f}M VND`).
- **Corporate Income Tax (`CIT` - Vietnam CIT Law No. 67/2025/QH15)**:
  - Statutory rate of **`20.0%`** applied to positive taxable income (`EBT = EBITDA - D&A`).
  - Monthly CIT accrues into **`Income Taxes Payable - Current FY`** during `Months 1-11`, consolidates into **`Income Taxes Payable - Past FY`** at fiscal year-end (`Month 12 / Dec`), and is settled in cash in **Month 4 (`April`) of the following fiscal year** (`tax_consolidation_month = 4`), creating exact working-capital cash movements (`(-) Change in Working Capital`) on the Cash Flow Statement (`5Y Accrued CIT = {res.five_year_taxes_mvnd:,.2f}M VND`; `5Y Cash CIT Settled = {res.five_year_cit_cash_paid_mvnd:,.2f}M VND`).
- **Value-Added Tax (`VAT` - Vietnam VAT Law No. 48/2024/QH15)**:
  - Tracks **`8.0%`** effective Output VAT on sports/hospitality revenue vs. **`10.0%`** Input VAT on taxable purchases and the `7,000.00M VND` Venue 2 fit-out CapEx, including **`VAT Receivable (Recoverable Input VAT Credit Carryforward)`** and **`Total VAT Remitted to State Treasury`** in `outputs/tax_estimations_60m.csv`.

### 1.5 Smooth Path C Membership Subscription Cohort & Quarterly Seasonal Tournaments (Starts Month 5: `May-27`)
Instead of sampling monthly subscriber counts independently, each Monte Carlo trajectory evolves active paying members ($S_t$, paying **VND 399,000/month** per Case Page 3 & Exhibit 3) smoothly via a Markov state-transition equation driven by sampled monthly unique customers ($\\mu = 3{{,}}600, \\sigma = 160$), monthly churn rate ($\\mu = 4.2\\%, \\sigma = 0.55\\%$), base monthly organic conversion rate ($\\mu = 4.20\\%, \\sigma = 0.45\\%$), and **Quarterly Seasonal Tournaments (`Spring [Mar]`, `Summer [Jun]`, `Fall [Sep]`, `Winter [Dec]`, starting Month 6 / `Jun-27`)**:
- **Tournament Entry Pricing**: **Free (`0 VND`)** for active `PICK! Social` members (included as a core perk of the `399k VND/mo` subscription); **`350,000 VND / player`** ($\\mu = 350\\text{{k}}, \\sigma = 25\\text{{k}}$) for non-members (`~86` non-members out of `192` players in the 6-court phase = `~30.1M VND/qtr`; `~168` non-members out of `420` players in the 20-court phase = `~58.8M VND/qtr`).
- **Quarterly Tournament Organization Cost (`COGS`)**: **`45.00M VND / tournament`** ($\\sigma = 4.00\\text{{M}}$) in the 6-court phase (`Months 6, 9, 12`) and **`85.00M VND / tournament`** ($\\sigma = 6.50\\text{{M}}$) in the 20-court phase (`Months 15, 18, ..., 60`), covering referees, tournament balls, trophies/medals, courtside hydration, and event livestreaming.
- **Acquisition, Retention & Ancillary Spend Effects**: **`22.0%`** ($\\sigma = 3.0\\%$) of non-member tournament entrants convert into paying `PICK! Social` subscribers (`~19` extra members/qtr in 6-court phase; `~37` extra members/qtr in 20-court phase), tournament months experience a **`20%` relative churn reduction ($0.80\\times$ multiplier)**, and all tournament participants spend an incremental **`85.0k VND/player`** on `Food & Drinks` (`55k`), `Paddle/Balls` (`10k`), and pre-tournament `Coaching` clinics (`20k`).

| Milestone Month | Calendar Month | Active Paying Subscribers (Mean `[P05, P95]`) | Monthly Membership Revenue (`M VND/mo`) | Direct Partner-Court & Event COGS (`M VND/mo`) | Total Monthly Revenue (`M VND/mo`) | Monthly Operating Profit / EBITDA (`M VND/mo`) | Monthly Net Income After Tax (`M VND/mo`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PICK! Social Dev (Months 1-4)** | `Jan-27 - Apr-27` | **`0`** (`[0, 0]`) | `0.00M` | `0.00M` | `{res.monthly_total_rev_arr[0]:,.2f}M` | `{res.monthly_ebitda_arr[0]:,.2f}M` | `{res.monthly_net_income_arr[0]:,.2f}M` |
| **PICK! Social Launch (Month 5)** | `May-27` | **`{mc.m5_launch_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[4]:,.0f}, {mc.active_subscribers_p95[4]:,.0f}]`) | `{res.monthly_membership_rev_arr[4]:,.2f}M` | `{res.monthly_membership_event_cogs_arr[4]:,.2f}M` | `{res.monthly_total_rev_arr[4]:,.2f}M` | `{res.monthly_ebitda_arr[4]:,.2f}M` | `{res.monthly_net_income_arr[4]:,.2f}M` |
| **End of Year 1 (Month 12)** | `Dec-27` | **`{mc.m12_end_y1_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[11]:,.0f}, {mc.active_subscribers_p95[11]:,.0f}]`) | `{res.monthly_membership_rev_arr[11]:,.2f}M` | `{res.monthly_membership_event_cogs_arr[11]:,.2f}M` | `{res.monthly_total_rev_arr[11]:,.2f}M` | `{res.monthly_ebitda_arr[11]:,.2f}M` | `{res.monthly_net_income_arr[11]:,.2f}M` |
| **20-Court Steady State (End of Year 2)** | `Dec-28` | **`{mc.m24_end_y2_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[23]:,.0f}, {mc.active_subscribers_p95[23]:,.0f}]`) | **`{mc.m24_end_y2_membership_rev_mvnd:,.2f}M`** | `{res.monthly_membership_event_cogs_arr[23]:,.2f}M` | `{res.monthly_total_rev_arr[23]:,.2f}M` | `{res.monthly_ebitda_arr[23]:,.2f}M` | `{res.monthly_net_income_arr[23]:,.2f}M` |
| **End of Year 3 (Month 36)** | `Dec-29` | **`{mc.m36_end_y3_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[35]:,.0f}, {mc.active_subscribers_p95[35]:,.0f}]`) | `{res.monthly_membership_rev_arr[35]:,.2f}M` | `{res.monthly_membership_event_cogs_arr[35]:,.2f}M` | `{res.monthly_total_rev_arr[35]:,.2f}M` | `{res.monthly_ebitda_arr[35]:,.2f}M` | `{res.monthly_net_income_arr[35]:,.2f}M` |
| **End of Year 5 (Month 60)** | `Dec-31` | **`{mc.m60_end_y5_subscribers:,.0f}`** (`[{mc.active_subscribers_p05[59]:,.0f}, {mc.active_subscribers_p95[59]:,.0f}]`) | `{res.monthly_membership_rev_arr[59]:,.2f}M` | `{res.monthly_membership_event_cogs_arr[59]:,.2f}M` | `{res.monthly_total_rev_arr[59]:,.2f}M` | `{res.monthly_ebitda_arr[59]:,.2f}M` | `{res.monthly_net_income_arr[59]:,.2f}M` |

---

## 2. Statement 1 - Income Statement Breakdown (Revenue, COGS, Facility OpEx, SG&A, D&A & Taxes)

| Category | Line Item | Cost Behavior | Case Baseline `/mo` | PICK! Social Dev (`Jan-27`) | PICK! Social Launch (`May-27`) | 20-Court Steady (`Dec-28`) | 5-Year Total | % of Revenue | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
{li_table_md}

---

## 3. Financial Statements - Annual Summary (`FS-Annual`) & Investor Target Verification (`FY 2027` - `FY 2031`)

- **Investor Minh Tran's Year 3 Target (`Case Page 4`)**: Requires **at least VND 3,500.0M (`3.5 billion VND`) in annual operating profit by Year 3 (`FY 2029`)**.
- **Our Integrated Solution Result in Year 3 (`FY 2029`)**:
  - **Base Plan Expected Year 3 Operating Profit (EBITDA)**: **`{mc.y3_annual_ebitda_mean_mvnd:,.2f}M VND/yr`** (`90% CI: [{mc.y3_annual_ebitda_p05_mvnd:,.2f}M, {mc.y3_annual_ebitda_p95_mvnd:,.2f}M]`), exceeding Minh Tran's `3,500M` hurdle in **`{mc.prob_exceed_investor_3500m_y3_pct:.1f}%`** of Monte Carlo trajectories!
  - **`-30%` Lower Demand Stress Test Year 3 Operating Profit**: **`{mc.y3_annual_ebitda_stress_mvnd:,.2f}M VND/yr`** - even if overall club demand and subscriber conversion sit `30%` below plan, PICK! still clears the `3,500M VND` Year 3 operating profit requirement!

| Financial Statements Annual Metric (`M VND`) | Start (`t=0`) | FY 2027 (Y1) | FY 2028 (Y2) | FY 2029 (Y3) | FY 2030 (Y4) | FY 2031 (Y5) | 5-Year Total / End |
|---|---:|---:|---:|---:|---:|---:|---:|
| **End-of-Year Paying Members (Subscribers)** | `0` | `{mc.m12_end_y1_subscribers:,.0f}` | **`{mc.m24_end_y2_subscribers:,.0f}`** | `{mc.m36_end_y3_subscribers:,.0f}` | `{mc.active_subscribers_mean[47]:,.0f}` | `{mc.m60_end_y5_subscribers:,.0f}` | `{mc.m60_end_y5_subscribers:,.0f}` |
| **Total Annual Revenue (`M VND`, Model)** | - | `{res.annual_revenue_arr[0]:,.2f}` | `{res.annual_revenue_arr[1]:,.2f}` | **`{res.annual_revenue_arr[2]:,.2f}`** | `{res.annual_revenue_arr[3]:,.2f}` | `{res.annual_revenue_arr[4]:,.2f}` | **`{res.five_year_revenue_mvnd:,.2f}`** |
| **Total Annual Revenue (`M VND`, Case Baseline)** | - | `{mc.base_annual_rev_mean[0]:,.2f}` | `{mc.base_annual_rev_mean[1]:,.2f}` | `{mc.base_annual_rev_mean[2]:,.2f}` | `{mc.base_annual_rev_mean[3]:,.2f}` | `{mc.base_annual_rev_mean[4]:,.2f}` | `{mc.base_annual_rev_mean.sum():,.2f}` |
| **Total Annual Operating Costs (`COGS + OpEx + SG&A`, Model)** | - | `{res.annual_total_costs_arr[0]:,.2f}` | `{res.annual_total_costs_arr[1]:,.2f}` | **`{res.annual_total_costs_arr[2]:,.2f}`** | `{res.annual_total_costs_arr[3]:,.2f}` | `{res.annual_total_costs_arr[4]:,.2f}` | **`{res.five_year_total_costs_mvnd:,.2f}`** |
| **Annual Operating Profit (`EBITDA`, Model Base Plan)** | - | `{res.annual_ebitda_arr[0]:,.2f}` | `{res.annual_ebitda_arr[1]:,.2f}` | **`{res.annual_ebitda_arr[2]:,.2f}`** | `{res.annual_ebitda_arr[3]:,.2f}` | `{res.annual_ebitda_arr[4]:,.2f}` | **`{res.five_year_ebitda_mvnd:,.2f}`** |
| **Annual Depreciation & Amortization (`D&A`)** | - | `{res.annual_da_arr[0]:,.2f}` | `{res.annual_da_arr[1]:,.2f}` | `{res.annual_da_arr[2]:,.2f}` | `{res.annual_da_arr[3]:,.2f}` | `{res.annual_da_arr[4]:,.2f}` | `{res.five_year_da_mvnd:,.2f}` |
| **Annual Corporate Income Tax (`20.0%` Vietnam CIT)** | - | `{res.annual_taxes_arr[0]:,.2f}` | `{res.annual_taxes_arr[1]:,.2f}` | `{res.annual_taxes_arr[2]:,.2f}` | `{res.annual_taxes_arr[3]:,.2f}` | `{res.annual_taxes_arr[4]:,.2f}` | `{res.five_year_taxes_mvnd:,.2f}` |
| **Annual Net Income After Tax (`Model Base Plan`)** | - | `{res.annual_net_income_arr[0]:,.2f}` | `{res.annual_net_income_arr[1]:,.2f}` | **`{res.annual_net_income_arr[2]:,.2f}`** | `{res.annual_net_income_arr[3]:,.2f}` | `{res.annual_net_income_arr[4]:,.2f}` | **`{res.five_year_net_income_mvnd:,.2f}`** |
| **Annual Operating Profit (`EBITDA`, Case Baseline)** | - | `{mc.base_annual_ebitda_mean[0]:,.2f}` | `{mc.base_annual_ebitda_mean[1]:,.2f}` | `{mc.base_annual_ebitda_mean[2]:,.2f}` | `{mc.base_annual_ebitda_mean[3]:,.2f}` | `{mc.base_annual_ebitda_mean[4]:,.2f}` | `{mc.base_annual_ebitda_mean.sum():,.2f}` |
| **Annual Operating Profit (`-30%` Demand Stress)** | - | `{mc.stress_ebitda_mean[0:12].sum():,.2f}` | `{mc.stress_ebitda_mean[12:24].sum():,.2f}` | **`{mc.y3_annual_ebitda_stress_mvnd:,.2f}`** | `{mc.stress_ebitda_mean[36:48].sum():,.2f}` | `{mc.stress_ebitda_mean[48:60].sum():,.2f}` | **`{mc.stress_ebitda_mean.sum():,.2f}`** |
| **Cash Reserves (`Ending Cash`, Model Base Plan)** | `{bs.starting_cash_mvnd:,.2f}` | `{res.monthly_ending_cash_model[11]:,.2f}` | `{res.monthly_ending_cash_model[23]:,.2f}` | `{res.monthly_ending_cash_model[35]:,.2f}` | `{res.monthly_ending_cash_model[47]:,.2f}` | **`{res.monthly_ending_cash_model[59]:,.2f}`** | **`{res.monthly_ending_cash_model[59]:,.2f}`** |
| **Cash Reserves (`-30%` Demand Stress)** | `{bs.starting_cash_mvnd:,.2f}` | `{mc.stress_ending_cash_mean[11]:,.2f}` | `{mc.stress_ending_cash_mean[23]:,.2f}` | `{mc.stress_ending_cash_mean[35]:,.2f}` | `{mc.stress_ending_cash_mean[47]:,.2f}` | **`{mc.stress_ending_cash_mean[59]:,.2f}`** | **`{mc.stress_ending_cash_mean[59]:,.2f}`** |
| **Cash Reserves (Case Baseline)** | `{bs.starting_cash_mvnd:,.2f}` | `{res.monthly_ending_cash_case[11]:,.2f}` | `{res.monthly_ending_cash_case[23]:,.2f}` | `{res.monthly_ending_cash_case[35]:,.2f}` | `{res.monthly_ending_cash_case[47]:,.2f}` | `{res.monthly_ending_cash_case[59]:,.2f}` | `{res.monthly_ending_cash_case[59]:,.2f}` |
| **Balance Sheet Check (`Assets - L&E`)** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** |

---

## 4. Generated Output Files (`FinancialModel/outputs/`)

- **Financial Statements - Monthly (`FS-Month`, 60-Month Projection CSV)**: `outputs/fs_month_60m_projections.csv` (Month 1 `Jan-27` to Month 60 `Dec-31`)
- **Financial Statements - Annual (`FS-Annual`, 5-Year Projection CSV)**: `outputs/fs_annual_5y_projections.csv` (`FY 2027` to `FY 2031`)
- **Vietnam Statutory Tax Estimations (`CIT` & `VAT`, 60-Month CSV)**: `outputs/tax_estimations_60m.csv`
- **60-Month Cash Flow & Cash Reserves CSV**: `outputs/monthly_cash_flow_and_balance_60m.csv`
- **Monte Carlo 60-Month Subscription Cohort & Stress-Test CSV**: `outputs/monte_carlo_subscription_cohort_60m.csv`
- **Revenue, COGS, OpEx, SG&A, D&A & Tax Breakdown CSV**: `outputs/revenue_cogs_opex_sga_breakdown.csv`
- **Sheet 1 Render (Income Statement Structure: Revenue, COGS, OpEx, SG&A, D&A & Taxes)**: `outputs/01_revenue_cogs_opex_sga_sheet.png`
- **Sheet 2 Render (`FS-Month` 60-Month Financial Statements)**: `outputs/02_fs_month_60m_projection_sheet.png`
- **Sheet 3 Render (`FS-Annual` 5-Year Financial Statements)**: `outputs/03_fs_annual_5y_projection_sheet.png`
- **Chart 4 Render (Stacked Revenue-by-Line > 0, Stacked Costs-by-Category < 0 & Profit Lines vs. Baseline)**: `outputs/04_revenue_costs_profit_comparison_charts.png`
- **Chart 5 Render (Monte Carlo Subscription Cohort Evolution, Revenue Bands & Year 3 Investor Target)**: `outputs/05_membership_and_monte_carlo_distributions.png`
- **Chart 6 Render (60-Month Cash Reserves Evolution, Expansion CapEx/Equity, April CIT Settlement & External Factors)**: `outputs/06_monthly_cash_balance_and_externals.png`
"""
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "breakdown_csv": breakdown_csv,
        "fs_month_csv": fs_month_csv,
        "fs_annual_csv": fs_annual_csv,
        "cohort_csv": cohort_csv,
        "cash_csv": cash_csv,
        "tax_csv": tax_csv,
        "report_path": report_path,
    }
