# Financial Statements Model (FS-Month, FS-Annual & Vietnam Statutory Taxes)

**Company**: PICK! Pickleball Club (Ho Chi Minh City)  
**Model Type**: 100% Stochastic Distribution-Driven Projections (No Actuals) — 60-Month (5-Year) Horizon (`Jan-27` to `Dec-31`)  
**Monte Carlo Validation**: `300` independent 60-month trajectories (`240` Train / `60` Held-Out Test) + `-30%` Lower-Demand Stress Test  
**Financial Statements Included**: **1. Income Statement (P&L)**, **2. Balance Sheet**, **3. Cash Flow Statement**, **4. Vietnam Statutory Tax Estimations (`CIT` & `VAT`)**  
**Currency**: Vietnamese Dong (`VND`), reported in **Million VND (`M VND`)**  

---

## 1. Key Model Drivers & Assumptions

Every assumption in this model is parameterized as a **Truncated Normal probability distribution** grounded in the Case Question exhibits and Vietnam statutory benchmarks:

### 1.1 Adaptive Pricing Model Integration (Court Hire Revenue)
- **Venue 1 (`6 Courts`)**: Adaptive pricing lifts monthly court-hire revenue from **`435.00M VND/month`** (`Case Baseline`, `~66.0%` blended utilization) to **`457.35M VND/month`** (**`+25.75M VND/month`** or **`+5.92%`**, `~76.1%` blended utilization across `2,191` booked court-hours/month).
- **Venue 2 (`14 Courts`, Opens `Feb-28`)**: Scales along the 6-to-9 month fill-up ramp from `Feb-28` to reach **`1,060.23M VND/month`** at steady state (`Dec-28`).

### 1.2 PICK! Social Platform Build & Recurring OpEx Schedule
- **Months 1–6 (`Jan-27` to `Jun-27`) Leveled Platform Development Allocation**:
  - Committed Base Outflow: **`911.18M VND`** (`139.93M` to `165.97M VND/month`) + Conditional Risk Contingency Reserve **`+177.36M VND`** (`29.58M/month` max; Monte Carlo expected drawdown `50%` = **`~999.91M VND`** expected total build cost across `Months 1–6`).
- **Month 5+ (`May-27` onward) Recurring Monthly Platform & Community OpEx**:
  - **Months 5–6 (`May–Jun 2027`, Launch & Hypercare)**: **`16.50M VND/month`** (`4.50M` Cloud/DB/Redis + `2.20M` Zalo ZNS/OTP + `1.80M` Observability/CI-CD SaaS + `8.00M` School/University & Beginner League Ops).
  - **Months 7–60 (`Jul 2027 – Dec 2031`, Steady-State)**: **`26.50M VND/month`** (`16.50M` above + `10.00M/month` ongoing engineering maintenance, security patching & ML retraining retainer).

### 1.3 Venue 2 Expansion Schedule (Q4 2027 14-Court Build & 2-Tranche Equity Drawdown)
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
  - Monthly CIT accrues into **`Income Taxes Payable — Current FY`** during `Months 1–11`, consolidates into **`Income Taxes Payable — Past FY`** at fiscal year-end (`Month 12 / Dec`), and is settled in cash in **Month 4 (`April`) of the following fiscal year** (`tax_consolidation_month = 4`), creating exact working-capital cash movements (`(-) Change in Working Capital`) on the Cash Flow Statement (`5Y Accrued CIT = 8,339.97M VND`; `5Y Cash CIT Settled = 5,941.66M VND`).
- **Value-Added Tax (`VAT` — Vietnam VAT Law No. 48/2024/QH15)**:
  - Tracks **`8.0%`** effective Output VAT on sports/hospitality revenue vs. **`10.0%`** Input VAT on taxable purchases and the `7,000.00M VND` Venue 2 fit-out CapEx, including **`VAT Receivable (Recoverable Input VAT Credit Carryforward)`** and **`Total VAT Remitted to State Treasury`** in `outputs/tax_estimations_60m.csv`.

### 1.5 Smooth Path C Membership Subscription Cohort & Quarterly Seasonal Tournaments (Starts Month 5: `May-27`)
Instead of sampling monthly subscriber counts independently, each Monte Carlo trajectory evolves active paying members ($S_t$, paying **VND 399,000/month** per Case Page 3 & Exhibit 3) smoothly via a Markov state-transition equation driven by sampled monthly unique customers (`mean = 3,600`, `std = 160`), monthly churn rate (`mean = 4.2%`, `std = 0.55%`), base monthly organic conversion rate (`mean = 4.20%`, `std = 0.45%`), and **Quarterly Seasonal Tournaments (`Spring [Mar]`, `Summer [Jun]`, `Fall [Sep]`, `Winter [Dec]`, starting Month 6 / `Jun-27`)**:
- **Tournament Entry Pricing**: **Free (`0 VND`)** for active `PICK! Social` members (included as a core perk of the `399k VND/mo` subscription); **`350,000 VND / player`** (`mean = 350k`, `std = 25k`) for non-members (`~86` non-members out of `192` players in the 6-court phase = `~30.1M VND/qtr`; `~168` non-members out of `420` players in the 20-court phase = `~58.8M VND/qtr`).
- **Quarterly Tournament Organization Cost (`COGS`)**: **`45.00M VND / tournament`** (`std = 4.00M`) in the 6-court phase (`Months 6, 9, 12`) and **`85.00M VND / tournament`** (`std = 6.50M`) in the 20-court phase (`Months 15, 18, ..., 60`), covering referees, tournament balls, trophies/medals, courtside hydration, and event livestreaming.
- **Acquisition, Retention & Ancillary Spend Effects**: **`22.0%`** (`std = 3.0%`) of non-member tournament entrants convert into paying `PICK! Social` subscribers (`~19` extra members/qtr in 6-court phase; `~37` extra members/qtr in 20-court phase), tournament months experience a **`20%` relative churn reduction (`0.80x` multiplier)**, and all tournament participants spend an incremental **`85.0k VND/player`** on `Food & Drinks` (`55k`), `Paddle/Balls` (`10k`), and pre-tournament `Coaching` clinics (`20k`).

| Milestone Month | Calendar Month | Active Paying Subscribers (Mean `[P05, P95]`) | Monthly Membership Revenue (`M VND/mo`) | Direct Partner-Court & Event COGS (`M VND/mo`) | Total Monthly Revenue (`M VND/mo`) | Monthly Operating Profit / EBITDA (`M VND/mo`) | Monthly Net Income After Tax (`M VND/mo`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PICK! Social Dev (Months 1–4)** | `Jan-27 – Apr-27` | **`0`** (`[0, 0]`) | `0.00M` | `0.00M` | `711.01M` | `21.48M` | `12.91M` |
| **PICK! Social Launch (Month 5)** | `May-27` | **`351`** (`[280, 423]`) | `140.17M` | `47.53M` | `875.35M` | `113.73M` | `86.71M` |
| **End of Year 1 (Month 12)** | `Dec-27` | **`807`** (`[765, 851]`) | `351.16M` | `159.57M` | `1,071.20M` | `10.18M` | `3.87M` |
| **20-Court Steady State (End of Year 2)** | `Dec-28` | **`1,489`** (`[1,399, 1,571]`) | **`644.93M`** | `263.83M` | `2,749.02M` | `966.59M` | `675.68M` |
| **End of Year 3 (Month 36)** | `Dec-29` | **`1,827`** (`[1,692, 1,961]`) | `786.77M` | `284.32M` | `2,874.10M` | `1,075.85M` | `763.08M` |
| **End of Year 5 (Month 60)** | `Dec-31` | **`2,218`** (`[2,006, 2,437]`) | `948.47M` | `343.58M` | `3,078.56M` | `1,166.37M` | `835.49M` |

---

## 2. Statement 1 — Income Statement Breakdown (Revenue, COGS, Facility OpEx, SG&A, D&A & Taxes)

| Category | Line Item | Cost Behavior | Case Baseline `/mo` | PICK! Social Dev (`Jan-27`) | PICK! Social Launch (`May-27`) | 20-Court Steady (`Dec-28`) | 5-Year Total | % of Revenue | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Revenue Line 1A (Venue 1) | Court hire: Venue 1 (6 Courts, Adaptive Pricing) | Revenue (Dynamic) | `435.00` | `457.35` | `473.92` | `454.38` | `26,978.76` | `19.26%` | Adaptive Pricing Model |
| Revenue Line 1B (Venue 2 New) | Court hire: Venue 2 (14 Courts, Opens Feb-28) | Revenue (Venue 2 Ramp) | `0.00` | `0.00` | `0.00` | `1,060.23` | `46,181.69` | `32.98%` | Adaptive Pricing & Venue 2 |
| Revenue Line 2 (PICK! Social) | PICK! Social subscriptions (399k/mo) & seasonal tournaments | Revenue (PICK! Social Launch+) | `0.00` | `0.00` | `140.17` | `644.93` | `36,220.08` | `25.86%` | Case Exhibit 3 & Assumptions |
| Revenue Line 3 | Paddle rental and balls (Scales 6 -> 20 Courts) | Revenue (Ancillary) | `15.00` | `18.98` | `17.10` | `44.72` | `2,253.39` | `1.61%` | Case Exhibit 1 & Segment Model |
| Revenue Line 4 | Coaching (Scales 6 -> 20 Courts + Coaching Studio) | Revenue (Services) | `115.00` | `140.85` | `147.72` | `325.18` | `17,028.86` | `12.16%` | Case Exhibit 1 & Segment Model |
| Revenue Line 5 | Food and drinks (Scales 6 -> 20 Courts + Specialty Cafe) | Revenue (F&B) | `75.00` | `93.83` | `96.44` | `219.58` | `11,381.87` | `8.13%` | Case Exhibit 1 & Segment Model |
| **TOTAL REVENUE** | **Total Projected Revenue (20-Court Network + PICK! Social)** | Subtotal | `640.00` | `711.01` | `875.35` | `2,749.02` | **`140,044.65`** | `100.00%` | Calculated Subtotal |
| Direct Service Costs | Coach fees (paid per lesson, ~52.17% of Coaching) | Variable (COGS) | `60.00` | `74.15` | `79.08` | `172.25` | `8,904.62` | `6.36%` | Case Exhibit 1 Ratios |
| Sold Inventory | Food and drink supplies (~40.0% of F&B Revenue) | Variable (COGS) | `30.00` | `37.10` | `37.32` | `88.70` | `4,554.86` | `3.25%` | Case Exhibit 1 Ratios |
| Payment & Consumables | Payment gateway fees (~1.50% of Rev) & court consumables | Variable (COGS) | `15.00` | `15.56` | `18.38` | `61.75` | `3,040.08` | `2.17%` | Case Exhibit 1 & Assumptions |
| Partner Courts & Events (New) | Partner courts (@ 200k/hr), member leagues & seasonal tournaments | Variable (COGS, May-27+) | `0.00` | `0.00` | `47.53` | `263.83` | `11,512.56` | `8.22%` | Case Exhibit 3 & Event Plan |
| **TOTAL COGS** | **Total Cost of Goods Sold (Direct Variable)** | Subtotal | `105.00` | `126.81` | `182.31` | `586.53` | **`28,012.12`** | `20.00%` | Calculated Subtotal |
| **GROSS PROFIT** | **Gross Profit (5-Year Gross Margin = 80.00%)** | Profit Line | `535.00` | `584.20` | `693.04` | `2,162.50` | **`112,032.54`** | `80.00%` | Calculated Subtotal |
| Rent — Venue 1 (6 Courts) | Venue 1 lease (Existing 6-Court Club) | Fixed (OpEx) | `150.00` | `150.00` | `150.00` | `150.00` | `9,000.00` | `6.43%` | Case Exhibit 1 |
| Rent — Venue 2 (14 Courts New) | Venue 2 lease (New 14-Court Club, Starts Oct-27) | Fixed (OpEx, Oct-27+) | `0.00` | `0.00` | `0.00` | `330.00` | `16,830.00` | `12.02%` | Venue 2 Lease Terms |
| Utilities (6 -> 20 Courts) | Electricity, lighting and water (Venue 1 + Venue 2 from Feb-28) | Mostly Fixed (OpEx) | `40.00` | `39.24` | `38.29` | `141.88` | `7,296.38` | `5.21%` | Case Exhibit 1 & Venue 2 Plan |
| Insurance & Upkeep (6 -> 20 Courts) | Maintenance and insurance (Venue 1 + Venue 2 from Feb-28) | Fixed (OpEx) | `20.00` | `20.01` | `19.58` | `69.65` | `3,697.98` | `2.64%` | Case Exhibit 1 & Venue 2 Plan |
| **SUBTOTAL OpEx** | **Subtotal Facility Operating Expenses (OpEx)** | Subtotal | `210.00` | `209.25` | `207.86` | `691.54` | **`36,824.36`** | `26.29%` | Calculated Subtotal |
| Staff & Payroll (6 -> 20 Courts) | Staff wages (Venue 1 130M + Venue 2 215M from Feb-28) | Fixed (SG&A) | `130.00` | `129.60` | `129.73` | `404.90` | `20,706.03` | `14.79%` | Case Exhibit 1 & Venue 2 Plan |
| Tech Support & Admin | Booking system and administration (Base) | Fixed (SG&A) | `10.00` | `10.02` | `10.15` | `10.12` | `598.88` | `0.43%` | Case Exhibit 1 |
| Marketing & Growth | Marketing (Social & Community) | Variable (SG&A) | `25.00` | `48.84` | `44.01` | `50.43` | `3,219.84` | `2.30%` | Case Exhibit 1 & Marketing Model |
| PICK! Social Dev (New) | PICK! Social development & migration (Jan–Jun 2027) | PICK! Social Dev (Jan–Jun 27) | `0.00` | `153.99` | `161.17` | `0.00` | `987.70` | `0.71%` | Platform Build Schedule |
| PICK! Social OpEx (New) | PICK! Social cloud, OTP, maintenance & uni ops (May-27+) | PICK! Social OpEx (May-27+) | `0.00` | `0.00` | `16.70` | `26.79` | `1,472.41` | `1.05%` | Platform Build Schedule |
| Legal & Safety Compliance (New) | Ward noise-complaint legal handling & U-18 safety | External Risk (SG&A) | `0.00` | `11.02` | `9.68` | `12.12` | `753.87` | `0.54%` | Risk & Compliance Assumptions |
| **SUBTOTAL SG&A** | **Subtotal Selling, General, Admin, PICK! Social & Compliance** | Subtotal | `165.00` | `353.47` | `371.45` | `504.37` | **`27,738.73`** | `19.81%` | Calculated Subtotal |
| **OPEX + SG&A** | **Total Operating Expenses (Facility OpEx + SG&A)** | Subtotal | `375.00` | `562.72` | `579.31` | `1,195.90` | **`64,563.09`** | `46.10%` | Calculated Subtotal |
| **TOTAL ALL COSTS** | **Total Monthly Operating Costs (COGS + OpEx + SG&A)** | Subtotal | `480.00` | `689.54` | `761.62` | `1,782.43` | **`92,575.21`** | `66.10%` | Calculated Subtotal |
| **OPERATING PROFIT** | **Net Operating Income (EBITDA) [5-Year Margin = 33.90%]** | Operating Profit | `160.00` | `21.48` | `113.73` | `966.59` | **`47,469.45`** | `33.90%` | Calculated Subtotal |
| **D&A EXPENSE** | **Depreciation & Amortization (D&A — Circular 45/2013/TT-BTC)** | Non-Cash Expense | `5.33` | `5.33` | `5.33` | `122.00` | **`5,803.33`** | `4.14%` | MOF Circular 45/2013 |
| **TAXABLE INCOME (EBT)** | **Net Income Before Taxes (EBT = EBITDA - D&A)** | Pre-Tax Profit | `154.67` | `16.14` | `108.39` | `844.59` | **`41,666.11`** | `29.75%` | Calculated Subtotal |
| **VIETNAM CIT (20%)** | **Corporate Income Tax Expense (Vietnam Law 67/2025/QH15 @ 20.0%)** | Statutory Tax | `30.93` | `3.23` | `21.68` | `168.92` | **`8,339.97`** | `5.96%` | Vietnam CIT Law (20%) |
| **NET INCOME** | **Net Income After Tax [5-Year Net Margin = 23.80%]** | Bottom-Line Profit | `123.73` | `12.91` | `86.71` | `675.68` | **`33,326.14`** | `23.80%` | Calculated Subtotal |

---

## 3. Financial Statements — Annual Summary (`FS-Annual`) & Investor Target Verification (`FY 2027` – `FY 2031`)

- **Investor Minh Tran's Year 3 Target (`Case Page 4`)**: Requires **at least VND 3,500.0M (`3.5 billion VND`) in annual operating profit by Year 3 (`FY 2029`)**.
- **Our Integrated Solution Result in Year 3 (`FY 2029`)**:
  - **Base Plan Expected Year 3 Operating Profit (EBITDA)**: **`12,420.45M VND/yr`** (`90% CI: [11,846.38M, 13,017.12M]`), exceeding Minh Tran's `3,500M` hurdle in **`100.0%`** of Monte Carlo trajectories!
  - **`-30%` Lower Demand Stress Test Year 3 Operating Profit**: **`6,008.76M VND/yr`** — even if overall club demand and subscriber conversion sit `30%` below plan, PICK! still clears the `3,500M VND` Year 3 operating profit requirement!

| Financial Statements Annual Metric (`M VND`) | Start (`t=0`) | FY 2027 (Y1) | FY 2028 (Y2) | FY 2029 (Y3) | FY 2030 (Y4) | FY 2031 (Y5) | 5-Year Total / End |
|---|---:|---:|---:|---:|---:|---:|---:|
| **End-of-Year Paying Members (Subscribers)** | `0` | `807` | **`1,489`** | `1,827` | `2,088` | `2,218` | `2,218` |
| **Total Annual Revenue (`M VND`, Model)** | — | `10,612.43` | `26,551.92` | **`33,231.02`** | `34,429.56` | `35,219.73` | **`140,044.65`** |
| **Total Annual Revenue (`M VND`, Case Baseline)** | — | `7,639.92` | `7,551.95` | `7,486.45` | `7,401.84` | `7,282.10` | `37,362.27` |
| **Total Annual Operating Costs (`COGS + OpEx + SG&A`, Model)** | — | `9,429.37` | `19,211.55` | **`20,810.57`** | `21,359.55` | `21,764.18` | **`92,575.21`** |
| **Annual Operating Profit (`EBITDA`, Model Base Plan)** | — | `1,183.06` | `7,340.37` | **`12,420.45`** | `13,070.01` | `13,455.56` | **`47,469.45`** |
| **Annual Depreciation & Amortization (`D&A`)** | — | `64.00` | `1,347.33` | `1,464.00` | `1,464.00` | `1,464.00` | `5,803.33` |
| **Annual Corporate Income Tax (`20.0%` Vietnam CIT)** | — | `230.56` | `1,198.61` | `2,191.29` | `2,321.20` | `2,398.31` | `8,339.97` |
| **Annual Net Income After Tax (`Model Base Plan`)** | — | `888.50` | `4,794.43` | **`8,765.16`** | `9,284.81` | `9,593.24` | **`33,326.14`** |
| **Annual Operating Profit (`EBITDA`, Case Baseline)** | — | `1,879.65` | `1,794.64` | `1,735.78` | `1,654.10` | `1,548.63` | `8,612.80` |
| **Annual Operating Profit (`-30%` Demand Stress)** | — | `-662.33` | `2,533.00` | **`6,008.76`** | `6,449.45` | `6,698.46` | **`21,027.34`** |
| **Cash Reserves (`Ending Cash`, Model Base Plan)** | `500.00` | `4,433.06` | `9,792.87` | `21,014.71` | `31,893.43` | **`43,027.79`** | **`43,027.79`** |
| **Cash Reserves (`-30%` Demand Stress)** | `500.00` | `2,587.67` | `3,290.35` | `8,926.73` | `14,467.23` | **`20,168.60`** | **`20,168.60`** |
| **Cash Reserves (Case Baseline)** | `500.00` | `2,379.65` | `3,811.16` | `5,200.82` | `6,520.56` | `7,751.17` | `7,751.17` |
| **Balance Sheet Check (`Assets - L&E`)** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** | **`0.00`** |

---

## 4. Generated Output Files (`FinancialModel/outputs/`)

- **Financial Statements — Monthly (`FS-Month`, 60-Month Projection CSV)**: `outputs/fs_month_60m_projections.csv` (Month 1 `Jan-27` to Month 60 `Dec-31`)
- **Financial Statements — Annual (`FS-Annual`, 5-Year Projection CSV)**: `outputs/fs_annual_5y_projections.csv` (`FY 2027` to `FY 2031`)
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
