# PICK! 5-Year (60-Month) 3-Statement Financial Model — Integrated with `AdaptivePricing`, `PickSocialProjectTimeline` & Path C Subscriptions

**Company**: PICK! Pickleball Club (Ho Chi Minh City)  
**Model Type**: 100% Stochastic Distribution-Driven Projections (No Actuals) — 60-Month (5-Year) Horizon (`Jan-27` to `Dec-31`)  
**Monte Carlo Validation**: `300` independent 60-month trajectories (`240` Train / `60` Held-Out Test) + `-30%` Lower-Demand Stress Test  
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
| **Weekday Daytime (06:00–17:00)** | 1,452.0 | 40.0% @ 160k | 92.93 | 66.6% @ 115k | 111.20 | +18.28 |
| **Weekday Evening (17:00–22:00)** | 660.0 | 95.0% @ 260k | 163.02 | 87.9% @ 290k | 167.58 | +4.56 |
| **Weekend All Day (06:00–22:00)** | 768.0 | 90.0% @ 260k | 179.71 | 83.8% @ 283k | 181.97 | +2.26 |
| **Total Court Hire** | **2,880.0** | **~66.0% (~1,900 hrs)** | **435.00** | **76.1% (2,191 hrs)** | **1231.36** | **+25.75 (+5.92%)** |

### 1.2 `PickSocialProjectTimeline/outputs/` Integration (Platform Build & Recurring Platform OpEx)
- **Months 1–6 (`Jan-27` to `Jun-27`) Leveled Platform Development Allocation**:
  - Committed Base Outflow: **`911.18M VND`** (`139.93M` to `165.97M VND/month`) + Conditional Risk Contingency Reserve **`+177.36M VND`** (`29.58M/month` max; Monte Carlo expected drawdown `50%` = **`~999.91M VND`** expected total build cost across `Months 1–6`).
- **Month 5+ (`May-27` onward) Recurring Monthly Platform & Community OpEx**:
  - **Months 5–6 (`May–Jun 2027`, Launch & Hypercare)**: **`16.50M VND/month`** (`4.50M` Cloud/DB/Redis + `2.20M` Zalo ZNS/OTP + `1.80M` Observability/CI-CD SaaS + `8.00M` School/University & Beginner League Ops).
  - **Months 7–60 (`Jul 2027 – Dec 2031`, Steady-State)**: **`26.50M VND/month`** (`16.50M` above + `10.00M/month` ongoing engineering maintenance, security patching & ML retraining retainer).

### 1.3 `VenueScalingTimeline/outputs/` Integration (Q4 2027 14-Court Expansion & 2-Tranche Equity Drawdown)
- **2-Tranche Investor Minh Tran Equity Drawdown (`8,000.00M VND` Total)**:
  - **Tranche 1 (`Month 1 / Jan-27` upon Signing)**: **`+500.00M VND`** drawn into cash reserve (`starting_cash = 800.00M + 500.00M = 1,300.00M VND` before M1 operations), ensuring **zero negative cash months (`> 0 VND`) across all 60 months even under a permanent `-30%` lower-demand stress test starting Day 1**.
  - **Tranche 2 (`Month 10 / Oct-27` at Q4 2027 Expansion Kickoff)**: **`+7,500.00M VND`** drawn (`7,000.00M VND` funding the even **`-1,750.00M VND/month`** fit-out CapEx across `Months 10–13: Oct 2027 – Jan 2028` + the remaining **`+500.00M VND`** reserve buffer).
- **Grand Opening (`Month 14 / Feb-28`) & 6-to-9 Month Fill-Up Ramp (`Feb–Sep/Oct 2028`)**: Scales PICK! from **`6` to `20` courts total**, reaching `100%` steady-state utilization by `Oct 2028 (Month 22)`—**2 full months before Year 3 (`FY 2029`) begins**.

### 1.4 Smooth Path C Membership Subscription Cohort Model (Starts Month 5: `May-27`)
Instead of sampling monthly subscriber counts independently, each Monte Carlo trajectory evolves active paying members ($S_t$, paying **VND 399,000/month** per Case Page 3 & Exhibit 3) smoothly via a Markov state-transition equation driven by sampled monthly unique customers (`mean = 3,600`, `std = 160`), monthly churn rate (`mean = 4.2%`, `std = 0.55%`), and monthly new-subscriber conversion rate (`mean = 4.65%`, `std = 0.50%`):

| Milestone Month | Calendar Month | Active Paying Subscribers (Mean `[P05, P95]`) | Monthly Membership Revenue (`M VND/mo`) | Direct Partner-Court & Event COGS (`M VND/mo`) | Total Monthly Revenue (`M VND/mo`) | Monthly Operating Profit / EBITDA (`M VND/mo`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Months 1–4 (Core Build)** | `Jan-27 – Apr-27` | **`0`** (`[0, 0]`) | `0.00M` | `0.00M` | `711.00M` | `21.49M` |
| **Month 5 (Platform Launch)** | `May-27` | **`351`** (`[280, 423]`) | `140.17M` | `14.79M` | `875.33M` | `146.49M` |
| **Month 12 (End of Year 1)** | `Dec-27` | **`1,382`** (`[1,321, 1,449]`) | `551.50M` | `56.27M` | `1,258.57M` | `309.11M` |
| **Month 24 (End of Year 2 — Case Benchmark)** | `Dec-28` | **`2,460`** (`[2,318, 2,589]`) | **`981.68M`** | `54.44M` | `3,083.53M` | `1,607.18M` |
| **Month 36 (End of Year 3)** | `Dec-29` | **`3,124`** (`[2,925, 3,292]`) | `1,246.67M` | `73.81M` | `3,297.27M` | `1,803.51M` |
| **Month 60 (End of Year 5)** | `Dec-31` | **`3,908`** (`[3,535, 4,242]`) | `1,559.35M` | `80.26M` | `3,574.79M` | `2,024.73M` |

---

## 2. Statement 1 — Income Statement Breakdown (Revenue, COGS, Facility OpEx & SG&A)

| Category | Line Item | Cost Behavior | Case Base `/mo` | M1 Build (`Jan-27`) | M5 Launch (`May-27`) | M24 End Y2 (`Dec-28`) | 5-Year (60M) Total | % of 5Y Rev |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Revenue Line 1A (Venue 1) | Court hire: Venue 1 (6 Courts, AdaptivePricing ML) | Revenue (Dynamic) | `435.00` | `457.35` | `473.92` | `463.84` | `27,194.49` | `16.58%` |
| Revenue Line 1B (Venue 2 New) | Court hire: Venue 2 (14 Courts, Opens M14 / Feb-28) | Revenue (Path A Ramp) | `0.00` | `0.00` | `0.00` | `1,082.30` | `46,687.30` | `28.46%` |
| Revenue Line 2 (New Path C) | Membership subscriptions (PICK! Social @ 399k/mo) | Revenue (Recurring M5+) | `0.00` | `0.00` | `140.17` | `981.68` | `60,181.73` | `36.68%` |
| Revenue Line 3 | Paddle rental and balls (Scales 6 -> 20 Courts) | Revenue (Ancillary) | `15.00` | `18.98` | `17.09` | `40.66` | `2,177.69` | `1.33%` |
| Revenue Line 4 | Coaching (Scales 6 -> 20 Courts + Coaching Studio) | Revenue (Services) | `115.00` | `140.85` | `147.72` | `317.72` | `16,855.94` | `10.27%` |
| Revenue Line 5 | Food and drinks (Scales 6 -> 20 Courts + Specialty Cafe) | Revenue (F&B) | `75.00` | `93.83` | `96.44` | `197.33` | `10,969.60` | `6.69%` |
| **TOTAL REVENUE** | **Total Projected Revenue (20-Court Network + Memberships)** | Subtotal | `640.00` | `711.00` | `875.33` | `3,083.53` | **`164,066.74`** | `100.00%` |
| Direct Service Costs | Coach fees (paid per lesson, ~52.17% of Coaching) | Variable (COGS) | `60.00` | `74.15` | `79.08` | `166.99` | `8,803.26` | `5.37%` |
| Sold Inventory | Food and drink supplies (~40.0% of F&B Revenue) | Variable (COGS) | `30.00` | `37.10` | `37.32` | `76.80` | `4,398.90` | `2.68%` |
| Payment & Consumables | Payment gateway fees (~1.50% of Rev) & court consumables | Variable (COGS) | `15.00` | `15.56` | `18.38` | `66.68` | `3,425.56` | `2.09%` |
| Partner Courts & Events (New) | Membership partner courts (@ 200k/hr) & league events | Variable (COGS M5+) | `0.00` | `0.00` | `14.79` | `54.44` | `3,766.64` | `2.30%` |
| **TOTAL COGS** | **Total Cost of Goods Sold (Direct Variable)** | Subtotal | `105.00` | `126.81` | `149.57` | `364.91` | **`20,394.36`** | `12.43%` |
| **GROSS PROFIT** | **Gross Profit (5Y Gross Margin = 87.57%)** | Profit Line | `535.00` | `584.19` | `725.76` | `2,718.61` | **`143,672.38`** | `87.57%` |
| Rent — Venue 1 (6 Courts) | Venue 1 lease (Existing 6-Court Club) | Fixed (OpEx) | `150.00` | `150.00` | `150.00` | `150.00` | `9,000.00` | `5.49%` |
| Rent — Venue 2 (14 Courts New) | Venue 2 lease (New 14-Court Club, Starts M10 / Oct-27) | Fixed (OpEx M10+) | `0.00` | `0.00` | `0.00` | `330.00` | `16,830.00` | `10.26%` |
| Utilities (6 -> 20 Courts) | Electricity, lighting and water (Venue 1 + Venue 2 M14+) | Mostly Fixed (OpEx) | `40.00` | `39.24` | `38.29` | `127.10` | `6,629.53` | `4.04%` |
| Insurance & Upkeep (6 -> 20 Courts) | Maintenance and insurance (Venue 1 + Venue 2 M14+) | Fixed (OpEx) | `20.00` | `20.01` | `19.58` | `62.72` | `3,193.19` | `1.95%` |
| **SUBTOTAL OpEx** | **Subtotal Facility Operating Expenses (OpEx)** | Subtotal | `210.00` | `209.25` | `207.86` | `669.82` | **`35,652.72`** | `21.73%` |
| Staff & Payroll (6 -> 20 Courts) | Staff wages (Venue 1 130M + Venue 2 215M from M14) | Fixed (SG&A) | `130.00` | `129.60` | `129.73` | `344.96` | `17,890.02` | `10.90%` |
| Tech Support & Admin | Booking system and administration (Base) | Fixed (SG&A) | `10.00` | `10.02` | `10.15` | `9.91` | `602.39` | `0.37%` |
| Marketing & Growth | Marketing (Social & Community) | Variable (SG&A) | `25.00` | `48.82` | `43.97` | `50.74` | `3,215.90` | `1.96%` |
| Platform Dev Allocation (New) | Platform build & migration allocation (Months 1–6) | Leveled Dev (M1–M6) | `0.00` | `153.99` | `161.17` | `0.00` | `985.86` | `0.60%` |
| Platform & Community OpEx (New) | Platform cloud, OTP, maintenance & uni ops (Month 5+) | Recurring OpEx (M5+) | `0.00` | `0.00` | `16.70` | `25.74` | `1,461.87` | `0.89%` |
| External Legal/Noise & U-18 Safety (New) | Ward noise-complaint legal handling & U-18 safety (externals.toml) | External Risk (SG&A) | `0.00` | `11.02` | `9.68` | `10.25` | `763.21` | `0.47%` |
| **SUBTOTAL SG&A** | **Subtotal Selling, General, Admin, Platform & External Compliance** | Subtotal | `165.00` | `353.46` | `371.40` | `441.61` | **`24,919.25`** | `15.19%` |
| **OPEX + SG&A** | **Total Operating Expenses (Facility OpEx + SG&A)** | Subtotal | `375.00` | `562.71` | `579.27` | `1,111.43` | **`60,571.97`** | `36.92%` |
| **TOTAL ALL COSTS** | **Total Monthly Operating Costs (COGS + OpEx + SG&A)** | Subtotal | `480.00` | `689.51` | `728.84` | `1,476.34` | **`80,966.33`** | `49.35%` |
| **OPERATING PROFIT** | **Net Operating Income (EBITDA) [5Y Margin = 50.65%]** | Operating Profit | `160.00` | `21.49` | `146.49` | `1,607.18` | **`83,100.41`** | `50.65%` |

---

## 3. 5-Year Annual 3-Statement Summary & Investor Target Verification (`FY 2027` – `FY 2031`)

- **Investor Minh Tran's Year 3 Target (`Case Page 4`)**: Requires **at least VND 3,500.0M (`3.5 billion VND`) in annual operating profit by Year 3 (`FY 2029`)**.
- **Our Integrated Path C Result in Year 3 (`FY 2029`)**:
  - **Base Plan Expected Year 3 Operating Profit (EBITDA)**: **`20,473.28M VND/yr`** (`90% CI: [19,522.92M, 21,555.03M]`), exceeding Minh Tran's `3,500M` hurdle in **`100.0%`** of Monte Carlo trajectories!
  - **`-30%` Lower Demand Stress Test Year 3 Operating Profit**: **`10,285.59M VND/yr`** — even if overall club demand and subscriber conversion sit `30%` below plan, PICK! still clears the `3,500M VND` Year 3 operating profit requirement!

| 3-Statement Annual Metric (`M VND`) | Start (`t=0`) | FY 2027 (Y1) | FY 2028 (Y2) | FY 2029 (Y3) | FY 2030 (Y4) | FY 2031 (Y5) | 5Y Total / End |
|---|---:|---:|---:|---:|---:|---:|---:|
| **End-of-Year Paying Members (Subscribers)** | `0` | `1,382` | **`2,460`** | `3,124` | `3,530` | `3,908` | `3,908` |
| **Total Annual Revenue (`M VND`, Model)** | — | `11,586.69` | `30,330.02` | **`38,481.51`** | `40,867.17` | `42,801.36` | **`164,066.74`** |
| **Total Annual Revenue (`M VND`, Stochastic Baseline)** | — | `7,637.08` | `7,571.21` | `7,517.94` | `7,550.61` | `7,457.62` | `37,734.46` |
| **Total Annual Costs (`COGS + OpEx + SG&A`, Model)** | — | `8,932.93` | `17,196.32` | **`18,008.22`** | `18,233.38` | `18,595.49` | **`80,966.33`** |
| **Total Annual Costs (`M VND`, Stochastic Baseline)** | — | `5,781.38` | `5,745.12` | `5,764.72` | `5,756.66` | `5,792.37` | `28,840.25` |
| **Annual Operating Profit (`EBITDA`, Model Base Plan)** | — | `2,653.76` | `13,133.70` | **`20,473.28`** | `22,633.79` | `24,205.88` | **`83,100.41`** |
| **Annual Operating Profit (`EBITDA`, Stochastic Baseline)** | — | `1,855.70` | `1,826.10` | `1,753.21` | `1,793.95` | `1,665.25` | `8,894.21` |
| **Annual Operating Profit (`-30%` Demand Stress)** | — | `-264.79` | `5,258.96` | **`10,285.59`** | `11,760.62` | `12,816.62` | **`39,857.00`** |
| **Ending Cash Balance (Model Base Plan)** | `500.00` | `5,903.76` | `17,287.46` | `37,760.75` | `60,394.53` | **`84,600.41`** | **`84,600.41`** |
| **Ending Cash Balance (`-30%` Demand Stress)** | `500.00` | `2,985.21` | `6,494.17` | `16,779.76` | `28,540.38` | **`41,357.00`** | **`41,357.00`** |
| **Ending Cash Balance (Stochastic Case Baseline)** | `500.00` | `2,355.70` | `4,181.80` | `5,935.01` | `7,728.96` | `9,394.21` | `9,394.21` |
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
