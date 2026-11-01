# Financial Statements Model (FS-Month, FS-Annual & Vietnam Statutory Taxes)

**Company**: PICK! Pickleball Club (Ho Chi Minh City)  
**Model Type**: 100% Stochastic Distribution-Driven Projections (No Actuals) — 60-Month (5-Year) Horizon (`Jan-27` to `Dec-31`)  
**Monte Carlo Validation**: `300` independent 60-month trajectories (`240` Train / `60` Held-Out Test) + `-30%` Lower-Demand Stress Test  
**Financial Statements Included**: **1. Income Statement (P&L)**, **2. Balance Sheet**, **3. Cash Flow Statement**, **4. Vietnam Statutory Tax Estimations (`CIT` & `VAT`)**  
**Currency**: Vietnamese Dong (`VND`), reported in **Million VND (`M VND`)**  
**Central Configuration**: `CaseComp/configs/` (`pricing.toml`, `customers.toml`, `cogs.toml`, `opex.toml`, `capex.toml`, `randomness.toml`, `externals.toml`)  
**Upstream Dependencies**: Verified `AdaptivePricing/outputs/`, `PickSocialProjectTimeline/outputs/`, `VenueScalingTimeline/outputs/` & `Marketing/outputs/`

---

## 1. Upstream Integrations & Distribution-First Philosophy

Every assumption in this model is parameterized as a **Truncated Normal probability distribution** in `CaseComp/configs/*.toml`, grounded in the Case Question exhibits and Vietnam statutory benchmarks:

### 1.1 `AdaptivePricing/outputs/` Integration (Court Hire Revenue)
| Court Pricing Regime | Available Hrs/Mo | Case Static Util & Rate | Case Rev (M VND) | ML Adaptive Util & Avg Rate | ML Adaptive Rev (M VND) | Monthly Lift (M VND) |
|---|---:|---|---:|---|---:|---:|
| **Weekday Daytime (06:00–17:00)** | 1,452.0 | 40.0% @ 160k | 92.93 | 66.6% @ 115k | 111.20 | +18.28 |
| **Weekday Evening (17:00–22:00)** | 660.0 | 95.0% @ 260k | 163.02 | 87.9% @ 290k | 167.58 | +4.56 |
| **Weekend All Day (06:00–22:00)** | 768.0 | 90.0% @ 260k | 179.71 | 83.8% @ 283k | 181.97 | +2.26 |
| **Total Court Hire (Venue 1 — 6 Courts)** | **2,880.0** | **~66.0% (~1,900 hrs)** | **435.00** | **76.1% (2,191 hrs)** | **457.35** | **+25.75 (+5.92%)** |

### 1.2 `PickSocialProjectTimeline/outputs/` Integration (Platform Build & Recurring Platform OpEx)
- **Months 1–6 (`Jan-27` to `Jun-27`) Leveled Platform Development Allocation**:
  - Committed Base Outflow: **`911.18M VND`** (`139.93M` to `165.97M VND/month`) + Conditional Risk Contingency Reserve **`+177.36M VND`** (`29.58M/month` max; Monte Carlo expected drawdown `50%` = **`~999.91M VND`** expected total build cost across `Months 1–6`).
- **Month 5+ (`May-27` onward) Recurring Monthly Platform & Community OpEx**:
  - **Months 5–6 (`May–Jun 2027`, Launch & Hypercare)**: **`16.50M VND/month`** (`4.50M` Cloud/DB/Redis + `2.20M` Zalo ZNS/OTP + `1.80M` Observability/CI-CD SaaS + `8.00M` School/University & Beginner League Ops).
  - **Months 7–60 (`Jul 2027 – Dec 2031`, Steady-State)**: **`26.50M VND/month`** (`16.50M` above + `10.00M/month` ongoing engineering maintenance, security patching & ML retraining retainer).

### 1.3 `VenueScalingTimeline/outputs/` Integration (Q4 2027 14-Court Expansion & 2-Tranche Equity Drawdown)
- **2-Tranche Investor Minh Tran Equity Drawdown (`8,000.00M VND` Total)**:
  - **Tranche 1 (`Month 1 / Jan-27` upon Signing)**: **`+500.00M VND`** drawn into cash reserves (`starting_cash = 800.00M + 500.00M = 1,300.00M VND` before M1 operations), ensuring **zero negative cash months (`> 0 VND`) across all 60 months even under a permanent `-30%` lower-demand stress test starting Day 1**.
  - **Tranche 2 (`Month 10 / Oct-27` at Q4 2027 Expansion Kickoff)**: **`+7,500.00M VND`** drawn (`7,000.00M VND` funding the even **`-1,750.00M VND/month`** fit-out CapEx across `Months 10–13: Oct 2027 – Jan 2028` + the remaining **`+500.00M VND`** reserve buffer).
- **Grand Opening (`Month 14 / Feb-28`) & 6-to-9 Month Fill-Up Ramp (`Feb–Sep/Oct 2028`)**: Scales PICK! from **`6` courts (`Months 1–13`) to `20` courts (`Months 14–60`)**, reaching `100%` steady-state utilization by `Oct 2028 (Month 22)`—**2 full months before Year 3 (`FY 2029`) begins**.

### 1.4 Vietnam Statutory Tax & Depreciation Schedule (`Taxes` Block in `FS-Month` & `FS-Annual`)
- **Depreciation & Amortization (`D&A` — Vietnam MOF Circular 45/2013/TT-BTC)**:
  - **Months 1–13 (`6 Courts`)**: Straight-line depreciation of existing `320.00M VND` court equipment over `60 months` = **`5.33M VND/month`**.
  - **Months 14–60 (`20 Courts`)**: Straight-line amortization of the `7,000.00M VND` Venue 2 leasehold improvement & court construction over its `60-month` lease (`+116.67M VND/month`), bringing total monthly D&A to **`122.00M VND/month`** (`5Y Total D&A = 5,803.33M VND`).
- **Corporate Income Tax (`CIT` — Vietnam CIT Law No. 67/2025/QH15)**:
  - Statutory rate of **`20.0%`** applied to positive taxable income (`EBT = EBITDA - D&A`).
  - Monthly CIT accrues into **`Income Taxes Payable — Current FY`** during `Months 1–11`, consolidates into **`Income Taxes Payable — Past FY`** at fiscal year-end (`Month 12 / Dec`), and is settled in cash in **Month 4 (`April`) of the following fiscal year** (`tax_consolidation_month = 4`), creating exact working-capital cash movements (`(-) Change in Working Capital`) on the Cash Flow Statement (`5Y Accrued CIT = 8,540.52M VND`; `5Y Cash CIT Settled = 6,094.16M VND`).
- **Value-Added Tax (`VAT` — Vietnam VAT Law No. 48/2024/QH15)**:
  - Tracks **`8.0%`** effective Output VAT on sports/hospitality revenue vs. **`10.0%`** Input VAT on taxable purchases and the `7,000.00M VND` Venue 2 fit-out CapEx, including **`VAT Receivable (Recoverable Input VAT Credit Carryforward)`** and **`Total VAT Remitted to State Treasury`** in `outputs/tax_estimations_60m.csv`.

### 1.5 Smooth Path C Membership Subscription Cohort Model (Starts Month 5: `May-27`)
Instead of sampling monthly subscriber counts independently, each Monte Carlo trajectory evolves active paying members ($S_t$, paying **VND 399,000/month** per Case Page 3 & Exhibit 3) smoothly via a Markov state-transition equation driven by sampled monthly unique customers (`mean = 3,600`, `std = 160`), monthly churn rate (`mean = 4.2%`, `std = 0.55%`), and monthly new-subscriber conversion rate (`mean = 4.65%`, `std = 0.50%`):

| Milestone Month | Calendar Month | Active Paying Subscribers (Mean `[P05, P95]`) | Monthly Membership Revenue (`M VND/mo`) | Direct Partner-Court & Event COGS (`M VND/mo`) | Total Monthly Revenue (`M VND/mo`) | Monthly Operating Profit / EBITDA (`M VND/mo`) | Monthly Net Income After Tax (`M VND/mo`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Months 1–4 (Core Build)** | `Jan-27 – Apr-27` | **`0`** (`[0, 0]`) | `0.00M` | `0.00M` | `711.00M` | `21.49M` | `12.92M` |
| **Month 5 (Platform Launch)** | `May-27` | **`351`** (`[280, 423]`) | `140.17M` | `47.52M` | `875.33M` | `113.76M` | `86.74M` |
| **Month 12 (End of Year 1)** | `Dec-27` | **`829`** (`[785, 880]`) | `330.66M` | `108.87M` | `1,037.73M` | `39.21M` | `27.10M` |
| **Month 24 (End of Year 2 — Case Benchmark)** | `Dec-28` | **`1,483`** (`[1,377, 1,571]`) | **`591.53M`** | `158.08M` | `2,693.38M` | `1,029.77M` | `726.22M` |
| **Month 36 (End of Year 3)** | `Dec-29` | **`1,851`** (`[1,708, 1,981]`) | `738.72M` | `209.02M` | `2,789.32M` | `1,078.38M` | `765.10M` |
| **Month 60 (End of Year 5)** | `Dec-31` | **`2,208`** (`[1,968, 2,427]`) | `881.09M` | `220.75M` | `2,896.53M` | `1,125.47M` | `802.78M` |

---

## 2. Statement 1 — Income Statement Breakdown (Revenue, COGS, Facility OpEx, SG&A, D&A & Taxes)

| Category | Line Item | Cost Behavior | Case Base `/mo` | M1 Build (`Jan-27`) | M5 Launch (`May-27`) | M24 End Y2 (`Dec-28`) | 5-Year (60M) Total | % of 5Y Rev |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Revenue Line 1A (Venue 1) | Court hire: Venue 1 (6 Courts, AdaptivePricing ML) | Revenue (Dynamic) | `435.00` | `457.35` | `473.92` | `463.84` | `27,194.49` | `19.51%` |
| Revenue Line 1B (Venue 2 New) | Court hire: Venue 2 (14 Courts, Opens M14 / Feb-28) | Revenue (Path A Ramp) | `0.00` | `0.00` | `0.00` | `1,082.30` | `46,687.30` | `33.50%` |
| Revenue Line 2 (New Path C) | Membership subscriptions (PICK! Social @ 399k/mo) | Revenue (Recurring M5+) | `0.00` | `0.00` | `140.17` | `591.53` | `35,474.69` | `25.46%` |
| Revenue Line 3 | Paddle rental and balls (Scales 6 -> 20 Courts) | Revenue (Ancillary) | `15.00` | `18.98` | `17.09` | `40.66` | `2,177.69` | `1.56%` |
| Revenue Line 4 | Coaching (Scales 6 -> 20 Courts + Coaching Studio) | Revenue (Services) | `115.00` | `140.85` | `147.72` | `317.72` | `16,855.94` | `12.10%` |
| Revenue Line 5 | Food and drinks (Scales 6 -> 20 Courts + Specialty Cafe) | Revenue (F&B) | `75.00` | `93.83` | `96.44` | `197.33` | `10,969.60` | `7.87%` |
| **TOTAL REVENUE** | **Total Projected Revenue (20-Court Network + Memberships)** | Subtotal | `640.00` | `711.00` | `875.33` | `2,693.38` | **`139,359.70`** | `100.00%` |
| Direct Service Costs | Coach fees (paid per lesson, ~52.17% of Coaching) | Variable (COGS) | `60.00` | `74.15` | `79.08` | `166.99` | `8,803.26` | `6.32%` |
| Sold Inventory | Food and drink supplies (~40.0% of F&B Revenue) | Variable (COGS) | `30.00` | `37.10` | `37.32` | `76.80` | `4,398.90` | `3.16%` |
| Payment & Consumables | Payment gateway fees (~1.50% of Rev) & court consumables | Variable (COGS) | `15.00` | `15.56` | `18.38` | `60.71` | `3,052.05` | `2.19%` |
| Partner Courts & Events (New) | Membership partner courts (@ 200k/hr) & league events | Variable (COGS M5+) | `0.00` | `0.00` | `47.52` | `158.08` | `10,086.37` | `7.24%` |
| **TOTAL COGS** | **Total Cost of Goods Sold (Direct Variable)** | Subtotal | `105.00` | `126.81` | `182.31` | `462.58` | **`26,340.57`** | `18.90%` |
| **GROSS PROFIT** | **Gross Profit (5Y Gross Margin = 81.10%)** | Profit Line | `535.00` | `584.19` | `693.02` | `2,230.80` | **`113,019.13`** | `81.10%` |
| Rent — Venue 1 (6 Courts) | Venue 1 lease (Existing 6-Court Club) | Fixed (OpEx) | `150.00` | `150.00` | `150.00` | `150.00` | `9,000.00` | `6.46%` |
| Rent — Venue 2 (14 Courts New) | Venue 2 lease (New 14-Court Club, Starts M10 / Oct-27) | Fixed (OpEx M10+) | `0.00` | `0.00` | `0.00` | `330.00` | `16,830.00` | `12.08%` |
| Utilities (6 -> 20 Courts) | Electricity, lighting and water (Venue 1 + Venue 2 M14+) | Mostly Fixed (OpEx) | `40.00` | `39.24` | `38.29` | `140.83` | `7,277.63` | `5.22%` |
| Insurance & Upkeep (6 -> 20 Courts) | Maintenance and insurance (Venue 1 + Venue 2 M14+) | Fixed (OpEx) | `20.00` | `20.01` | `19.58` | `73.79` | `3,700.01` | `2.66%` |
| **SUBTOTAL OpEx** | **Subtotal Facility Operating Expenses (OpEx)** | Subtotal | `210.00` | `209.25` | `207.86` | `694.62` | **`36,807.64`** | `26.41%` |
| Staff & Payroll (6 -> 20 Courts) | Staff wages (Venue 1 130M + Venue 2 215M from M14) | Fixed (SG&A) | `130.00` | `129.60` | `129.73` | `409.75` | `20,709.27` | `14.86%` |
| Tech Support & Admin | Booking system and administration (Base) | Fixed (SG&A) | `10.00` | `10.02` | `10.15` | `9.91` | `602.39` | `0.43%` |
| Marketing & Growth | Marketing (Social & Community) | Variable (SG&A) | `25.00` | `48.82` | `43.97` | `50.74` | `3,215.90` | `2.31%` |
| Platform Dev Allocation (New) | Platform build & migration allocation (Months 1–6) | Leveled Dev (M1–M6) | `0.00` | `153.99` | `161.17` | `0.00` | `985.86` | `0.71%` |
| Platform & Community OpEx (New) | Platform cloud, OTP, maintenance & uni ops (Month 5+) | Recurring OpEx (M5+) | `0.00` | `0.00` | `16.70` | `25.74` | `1,461.87` | `1.05%` |
| External Legal/Noise & U-18 Safety (New) | Ward noise-complaint legal handling & U-18 safety (externals.toml) | External Risk (SG&A) | `0.00` | `11.02` | `9.68` | `10.25` | `763.21` | `0.55%` |
| **SUBTOTAL SG&A** | **Subtotal Selling, General, Admin, Platform & External Compliance** | Subtotal | `165.00` | `353.46` | `371.40` | `506.40` | **`27,738.50`** | `19.90%` |
| **OPEX + SG&A** | **Total Operating Expenses (Facility OpEx + SG&A)** | Subtotal | `375.00` | `562.71` | `579.27` | `1,201.03` | **`64,546.15`** | `46.32%` |
| **TOTAL ALL COSTS** | **Total Monthly Operating Costs (COGS + OpEx + SG&A)** | Subtotal | `480.00` | `689.51` | `761.57` | `1,663.60` | **`90,886.72`** | `65.22%` |
| **OPERATING PROFIT** | **Net Operating Income (EBITDA) [5Y Margin = 34.78%]** | Operating Profit | `160.00` | `21.49` | `113.76` | `1,029.77` | **`48,472.98`** | `34.78%` |
| **D&A EXPENSE** | **Depreciation & Amortization (D&A — Circular 45/2013/TT-BTC)** | Non-Cash Expense | `5.33` | `5.33` | `5.33` | `122.00` | **`5,803.33`** | `4.16%` |
| **TAXABLE INCOME (EBT)** | **Net Income Before Taxes (EBT = EBITDA - D&A)** | Pre-Tax Profit | `154.67` | `16.15` | `108.42` | `907.77` | **`42,669.65`** | `30.62%` |
| **VIETNAM CIT (20%)** | **Corporate Income Tax Expense (Vietnam Law 67/2025/QH15 @ 20.0%)** | Statutory Tax | `30.93` | `3.23` | `21.68` | `181.55` | **`8,540.52`** | `6.13%` |
| **NET INCOME** | **Net Income After Tax [5Y Net Margin = 24.49%]** | Bottom-Line Profit | `123.73` | `12.92` | `86.74` | `726.22` | **`34,129.13`** | `24.49%` |

---

## 3. Financial Statements — Annual Summary (`FS-Annual`) & Investor Target Verification (`FY 2027` – `FY 2031`)

- **Investor Minh Tran's Year 3 Target (`Case Page 4`)**: Requires **at least VND 3,500.0M (`3.5 billion VND`) in annual operating profit by Year 3 (`FY 2029`)**.
- **Our Integrated Path C Result in Year 3 (`FY 2029`)**:
  - **Base Plan Expected Year 3 Operating Profit (EBITDA)**: **`12,553.13M VND/yr`** (`90% CI: [11,843.45M, 13,451.89M]`), exceeding Minh Tran's `3,500M` hurdle in **`100.0%`** of Monte Carlo trajectories!
  - **`-30%` Lower Demand Stress Test Year 3 Operating Profit**: **`6,156.59M VND/yr`** — even if overall club demand and subscriber conversion sit `30%` below plan, PICK! still clears the `3,500M VND` Year 3 operating profit requirement!

| Financial Statements Annual Metric (`M VND`) | Start (`t=0`) | FY 2027 (Y1) | FY 2028 (Y2) | FY 2029 (Y3) | FY 2030 (Y4) | FY 2031 (Y5) | 5Y Total / End |
|---|---:|---:|---:|---:|---:|---:|---:|
| **End-of-Year Paying Members (Subscribers)** | `0` | `829` | **`1,483`** | `1,851` | `2,047` | `2,208` | `2,208` |
| **Total Annual Revenue (`M VND`, Model)** | — | `10,493.88` | `26,470.16` | **`33,030.88`** | `34,297.11` | `35,067.67` | **`139,359.70`** |
| **Total Annual Revenue (`M VND`, Stochastic Baseline)** | — | `7,637.08` | `7,571.21` | `7,517.94` | `7,550.61` | `7,457.62` | `37,734.46` |
| **Total Annual Operating Costs (`COGS + OpEx + SG&A`, Model)** | — | `9,284.41` | `18,883.46` | **`20,477.75`** | `20,869.23` | `21,371.87` | **`90,886.72`** |
| **Annual Operating Profit (`EBITDA`, Model Base Plan)** | — | `1,209.46` | `7,586.70` | **`12,553.13`** | `13,427.88` | `13,695.80` | **`48,472.98`** |
| **Annual Depreciation & Amortization (`D&A`)** | — | `64.00` | `1,347.33` | `1,464.00` | `1,464.00` | `1,464.00` | `5,803.33` |
| **Annual Corporate Income Tax (`20.0%` Vietnam CIT)** | — | `235.68` | `1,247.87` | `2,217.83` | `2,392.78` | `2,446.36` | `8,540.52` |
| **Annual Net Income After Tax (`Model Base Plan`)** | — | `909.78` | `4,991.49` | **`8,871.31`** | `9,571.11` | `9,785.44` | **`34,129.13`** |
| **Annual Operating Profit (`EBITDA`, Stochastic Baseline)** | — | `1,855.70` | `1,826.10` | `1,753.21` | `1,793.95` | `1,665.25` | `8,894.21` |
| **Annual Operating Profit (`-30%` Demand Stress)** | — | `-622.65` | `2,755.86` | **`6,156.59`** | `6,755.22` | `6,926.58` | **`21,971.61`** |
| **Cash Reserves (`Ending Cash`, Model Base Plan)** | `500.00` | `4,459.46` | `10,060.48` | `21,365.74` | `32,575.80` | **`43,878.83`** | **`43,878.83`** |
| **Cash Reserves (`-30%` Demand Stress)** | `500.00` | `2,627.35` | `3,551.60` | `9,294.93` | `15,111.63` | **`20,979.97`** | **`20,979.97`** |
| **Cash Reserves (Stochastic Case Baseline)** | `500.00` | `2,355.70` | `3,823.46` | `5,224.25` | `6,680.36` | `7,999.62` | `7,999.62` |
| **Balance Sheet Check (`Assets - L&E`)** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** |

---

## 4. Generated Output Files (`FinancialModel/outputs/`)

- **Financial Statements — Monthly (`FS-Month`, 60-Month Projection CSV)**: `outputs/fs_month_60m_projections.csv` (Month 1 `Jan-27` to Month 60 `Dec-31`)
- **Financial Statements — Annual (`FS-Annual`, 5-Year Projection CSV)**: `outputs/fs_annual_5y_projections.csv` (`FY 2027` to `FY 2031`)
- **Vietnam Statutory Tax Estimations (`CIT` & `VAT`, 60-Month CSV)**: `outputs/tax_estimations_60m.csv`
- **60-Month Cash Flow & Cash Reserves CSV**: `outputs/monthly_cash_flow_and_balance_60m.csv`
- **Monte Carlo 60-Month Subscription Cohort & Stress-Test CSV**: `outputs/monte_carlo_subscription_cohort_60m.csv`
- **Revenue, COGS, OpEx, SG&A, D&A & Tax Breakdown CSV**: `outputs/revenue_cogs_opex_sga_breakdown.csv`
- **Sheet 1 Render (Drivers, Revenue, COGS, OpEx, SG&A, D&A & Taxes)**: `outputs/01_revenue_cogs_opex_sga_sheet.png`
- **Sheet 2 Render (`FS-Month` 60-Month Financial Statements)**: `outputs/02_fs_month_60m_projection_sheet.png`
- **Sheet 3 Render (`FS-Annual` 5-Year Financial Statements)**: `outputs/03_fs_annual_5y_projection_sheet.png`
- **Chart 4 Render (Stacked Revenue-by-Line > 0, Stacked Costs-by-Category < 0 & Profit Lines vs. Baseline)**: `outputs/04_revenue_costs_profit_comparison_charts.png`
- **Chart 5 Render (Monte Carlo Subscription Cohort Evolution, Revenue Bands & Year 3 Investor Target)**: `outputs/05_membership_and_monte_carlo_distributions.png`
- **Chart 6 Render (60-Month Cash Reserves Evolution, Expansion CapEx/Equity, April CIT Settlement & External Factors)**: `outputs/06_monthly_cash_balance_and_externals.png`
