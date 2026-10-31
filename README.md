# Vietnam Commerce Case Competition — PICK! Strategic & Financial Plan

> **AI Usage Acknowledgement**  
> - **Human Direction & Original Ideas**: Our team defined the overall business strategy, the sequenced growth roadmap, the distribution-based stochastic modeling methodology, the customer-segment funnel logic, and all financial and operational assumptions.  
> - **AI Agent Execution**: We used **OpenAI Codex** and **Google Antigravity** to implement the simulation and machine learning code, benchmark software and construction task estimates, generate charts, and format reports under our direction and verification.

---

## 1. Strategic Overview & Financial Outlook

PICK! has `42,000` social followers and `90%–95%` evening court occupancy, but **`60%` of weekday daytime courts sit empty** and casual visitors churn once viral excitement fades. Expanding venues immediately would multiply daytime vacancy, while launching an e-commerce paddle brand requires manufacturing and distribution capabilities PICK! does not have (`18%` survey intent).

Instead, we execute a **4-stage operational and capital sequence** within the **`8,000.0M VND` investment ceiling**:

| Stage | Timeline | Action | Why It Works |
| :--- | :--- | :--- | :--- |
| **1. Dynamic Pricing & Cash Buffer** | `Months 1–4` (`Jan–Apr 2027`) | Deploy ML time-of-day court pricing on Venue 1 (`6` courts); build `PICK! Social` in-house; draw **`+500.0M VND` (Tranche 1)** into reserve upon signing. | Cuts weekday daytime prices (`160k -> 115k VND/hr`) to fill empty daytime slots (`40.0% -> 64.4%` occupancy), lifting revenue **`+24.92M VND/mo`** immediately while keeping cash `>= 1,000.73M VND`. |
| **2. `PICK! Social` & Segment Marketing** | `Months 5–9` (`May–Sep 2027`) | Launch `PICK! Social` (`399,000 VND/mo` membership, skill matchmaking, leagues) and ML segment marketing across `5` full operating months. | Converts casual visitors into **`1,422` paying subscribers** and builds cash reserves to **`+2,438.49M VND`** before committing to a second venue lease. |
| **3. Second Venue Expansion Build** | `Months 10–13` (`Oct 2027–Jan 2028`) | Draw **`+7,500.0M VND` (Tranche 2)** (`7,000M` fit-out + `500M` reserve); execute `16-week` turnkey build of Venue 2 (`+14` courts, `-1,750M VND/mo` leveled CapEx). | Funded when Venue 1 + `PICK! Social` is already generating `+297M–435M VND/mo` in operating profit. |
| **4. Expanded Network Operations** | `Months 14–60` (`Feb 2028–Dec 2031`) | Open Venue 2 in `Feb 2028` (`20` courts total); pre-built `PICK! Social` community drives 6-to-9 month utilization ramp. | Reaches **`20,473.28M VND` Year 3 operating profit** (**`5.85x`** the `3,500M VND` investor target; **`100%` simulation pass rate**). |

![Revenue, Costs, and Operating Profit Comparison](FinancialModel/outputs/04_revenue_costs_profit_comparison_charts.png)

![Annual Financial Statement Projection](FinancialModel/outputs/03_fs_annual_5y_projection_sheet.png)

### Multi-Year Performance Summary

| Metric | Stochastic Baseline (`6 Courts`) | Our Plan (`Expected Mean`) | Our Plan (`90% CI: P05–P95`) | Permanent `-30%` Demand Stress |
| :--- | :---: | :---: | :---: | :---: |
| **Year 1 (`FY 2027`) EBITDA** | `1,856.4M VND` | **`2,654.21M VND`** | `[2,226M, 3,098M]` | `-265.10M VND` |
| **Year 2 (`FY 2028`) EBITDA** | `1,826.2M VND` | **`13,134.12M VND`** | `[12,240M, 14,009M]` | `+5,259.40M VND` |
| **Year 3 (`FY 2029`) EBITDA (`Target: 3,500M`)** | `1,753.0M VND` | **`20,473.28M VND` (`5.85x`)** | `[19,388M, 21,542M]` (`100% Pass`) | **`+10,285.59M VND` (`2.94x`)** |
| **Year 4 (`FY 2030`) EBITDA** | `1,794.1M VND` | **`22,633.72M VND`** | `[21,495M, 23,777M]` | `+11,761.20M VND` |
| **Year 5 (`FY 2031`) EBITDA** | `1,664.5M VND` | **`24,205.08M VND`** | `[22,953M, 25,388M]` | `+12,817.00M VND` |
| **5Y Cumulative Revenue** | `37,883.4M VND` | **`164,066.74M VND`** | `[161,012M, 167,118M]` | `120,035.40M VND` |
| **5Y Cumulative EBITDA** | `8,894.2M VND` | **`83,100.41M VND` (`50.65%`)** | `[80,046M, 86,152M]` | `39,857.00M VND` |
| **Minimum Monthly Cash (`M1–M60`)** | `+646.97M VND` | **`+1,000.73M VND` (`M3`)** | `P05 Min: +954.57M VND` | **`+217.59M VND` (`M6`)** |
| **Month 60 Ending Cash** | `9,394.21M VND` | **`84,600.41M VND`** | `[81,546M, 87,652M]` | `41,357.00M VND` |

---

## 2. Input Grounding & Empirical Assumptions

Every input is modeled as a **bounded probability distribution (`Truncated Normal`)** in [`configs/`](configs) and reconciles directly to the Case Data Pack:

### 2.1 Court Utilization & Price Elasticity ([`AdaptivePricing/`](AdaptivePricing))
- Grounded in the `06:00–22:00` operating schedule (`40%` weekday daytime vs. `95%` evening / `90%` weekend baseline occupancy) and scaled across active court capacity (`6` courts in `Months 1–13` -> `20` courts in `Months 14–60`).

![Court Utilization & Dynamic Pricing Assumptions](AdaptivePricing/outputs/01_assumptions_distributions.png)

![Price-Demand Elasticity Curves Across Active Capacity](AdaptivePricing/outputs/02_demand_curves_variations.png)

![Hourly Dynamic Pricing Schedule: Weekday vs. Weekend](AdaptivePricing/outputs/04_monthly_calendar_heatmap.png)

### 2.2 Customer Acquisition Funnel & Segment Ancillary Spending ([`Marketing/`](Marketing))
- **Explicit Funnel Assumption**: Our `3,600` monthly unique customers come from our `42,000` TikTok/Instagram followers (`3,600 / 42,000 = 8.571%` conversion, modeled as `N(8.571%, 0.85%)`), generating `7,596` baseline player visits across 4 customer segments.
- **Exact VND Marketing Cost per Visit**: Mapping the `25.0M VND/mo` baseline marketing spend across segments yields **`x = 1,543 VND/visit`** (Competitive), **`y = 3,373 VND/visit`** (Social), **`z = 6,313 VND/visit`** (Trend Explorers), and **`w = 6,173 VND/visit`** (Families).
- **Segment-by-Ancillary Spending Distributions**: Each segment has its own distribution for **Paddle/Ball Rental**, **Coaching**, and **Food & Drinks**. Multiplied by baseline visits, they sum to **`204.80M VND/month`**—matching the `32%` non-court revenue in the Data Pack exactly.

![Customer Acquisition Funnel & Segment Ancillary Spending](Marketing/outputs/01_marketing_funnel_and_segment_response_distributions.png)

### 2.3 Platform Engineering, Cloud & Venue Construction Costs
- **`PICK! Social` Platform Engineering & Cloud Operations**: Scoped across `17` engineering work packages (`911.20M` base + `177.80M` / `19.5%` risk contingency buffer across the 6-month build & migration lifecycle, plus `26.50M VND/mo` steady-state cloud/maintenance OpEx and `1.50%` payment gateway COGS) using *COCOMO II*, *PMI PMBOK 7th Ed.*, *ITviec/TopDev Vietnam Tech Salary Reports*, and *NAPAS/VietQR + Momo/VNPay fee schedules* ([`PickSocialProjectTimeline/`](PickSocialProjectTimeline)).
- **Second Venue Construction & Operating Scale**: Scoped across `15` construction packages (`7,000.0M VND` turnkey fit-out CapEx + `330.0M VND/mo` lease) using *ASBA Courts Manual* and *Vietnam Ministry of Construction benchmarks* ([`VenueScalingTimeline/`](VenueScalingTimeline)); when the 14 new courts open in `Month 14`, electricity/water (`+82.0M/mo`), maintenance/insurance (`+38.0M/mo`), and staff wages (`+215.0M/mo`) scale up automatically alongside variable COGS.

---

## 3. Model Validation, Out-of-Sample Machine Learning & Stress Testing

Static spreadsheets fail because they ignore variance, capacity ceilings, and external shocks. We verify our results through **four mechanisms**:

### 3.1 Baseline Comparison Under Identical Stochastic Shocks
- The **Case Baseline (`6` courts, static pricing, `25M VND/mo` marketing)** runs through the **exact same `N = 300` simulation lines and monthly random draws** as our plan—fluctuating around `628.9M–633.7M VND/mo` revenue and `1,664.5M–1,856.4M VND/yr` profit (`0.0%` chance of hitting `3,500M VND`).

### 3.2 Out-of-Sample Machine Learning Validation
- **`AdaptivePricing` ML (`HistGradientBoostingRegressor`)**: Held-out **Test `R^2 = 0.9922`** (`MAE = 2.15%`), capturing **`98.8%`** of Oracle revenue lift (`+24.92M VND/mo` per `6` courts; `+83.07M VND/mo` across `20` courts).
- **`Marketing` Segment Allocator (`MultiOutputRegressor`)**: Predicts Month `t`'s 4-segment marketing budget from **Month `t-1`'s realized segment metrics** (zero look-ahead bias). Held-out **Test `R^2 = 0.9555`**, capturing **`99.29%`** of Oracle profit and lifting ancillary revenue from **`204.8M -> 500.05M VND/mo`** (`5Y avg`).

![Adaptive Pricing Daily Revenue Lift Comparison](AdaptivePricing/outputs/03_daily_revenue_comparison.png)

![Adaptive Pricing Out-of-Sample ML Validation Diagnostics](AdaptivePricing/outputs/06_ml_model_diagnostics.png)

![Segment Marketing Budget & Visits Trajectory](Marketing/outputs/02_ml_segment_marketing_budget_and_visits_trajectory.png)

![Segment Ancillary Revenue Breakdown](Marketing/outputs/03_segment_ancillary_revenue_breakdown.png)

![Marketing ML Out-of-Sample Diagnostics](Marketing/outputs/04_marketing_ml_diagnostics.png)

### 3.3 Subscriber Cohort Dynamics & External Risk Modeling ([`configs/externals.toml`](configs/externals.toml))
- **Paying Subscribers Only**: Free registered accounts (`1,150`) are excluded from revenue. Paying members (`M_t`) evolve smoothly via monthly churn `N(4.8%, 1.0%)` and non-member conversion `N(8.2%, 1.4%)` (conservative vs. `31%` survey intent): **`351` (`M5`) -> `1,546` (`M12`) -> `2,460` (`M24`) -> `3,905` (`M60`)**.
- **External Factors (`externals.toml`)**: Every trajectory compounds monthly **overall pickleball trend shifts** (`N(+0.3%, 1.8%, [-6%, +5%])`), **competitor court drag**, **weather shocks**, **cost inflation**, and **noise complaint legal handling costs**.

![Membership Growth, Year 3 Profit Distribution & External Risk Shocks](FinancialModel/outputs/05_membership_and_monte_carlo_distributions.png)

### 3.4 Downside Demand Stress Testing
- Even if total demand across all revenue lines is permanently **`30%` lower from Month 1 to Month 60** while fixed leases, payroll, and `7,000M VND` CapEx stay at `100%`:
  1. **Zero Negative Cash Months**: Minimum cash across 60 months is **`+217.59M VND` (`Jun 2027`)**, and stays `>= +1,244.54M VND` through the Q4 2027 Venue 2 construction window.
  2. **Still Clears the Investor Hurdle by `2.94x`**: Stress Year 3 EBITDA reaches **`+10,285.59M VND`** vs. the `3,500M VND` target.

![Monthly Cash Balance Trajectory & Stress Test](FinancialModel/outputs/06_monthly_cash_balance_and_externals.png)

---

## 4. Execution Roadmap, Capital Allocation & Milestone Gates

### 4.1 Phased Equity Drawdowns & Leveled Capital Disbursement
- **Tranche 1 (`+500.0M VND` in `Month 1 / Jan 2027`)**: Drawn upon signing (`500M` existing + `500M` draw = `1,000M VND` starting cash) while `PICK! Social` engineering costs are paid via leveled bi-weekly sprint drawdowns and milestone holdbacks across `Months 1–6` (`Jan–Jun 2027`).
- **Tranche 2 (`+7,500.0M VND` in `Month 10 / Oct 2027`)**: Drawn at Q4 expansion kickoff (`7,000M` fit-out paid in **4 equal `-1,750.0M VND/mo` milestones** across `Oct 2027–Jan 2028` + `500M` reserve), after `PICK! Social` has already built cash reserves to **`+2,438.49M VND`**.

![PICK! Social Production Gantt Schedule](PickSocialProjectTimeline/outputs/01_pick_social_platform_gantt_chart.png)

![PICK! Social Cost Estimation & Leveled Monthly Allocation](PickSocialProjectTimeline/outputs/03_cost_estimation_and_cashflow_allocation.png)

### 4.2 Second Venue Construction Schedule & Network Capacity Scaling
- **Turnkey Critical Path (`Oct 2027–Jan 2028`)**: Sequences site handover, `28-day` concrete slab moisture curing (`Weeks 5–8`), 5-layer ITF acrylic surfacing (`Weeks 9–11`), acoustic barriers, and PCCC fire acceptance (`Weeks 13–14`) prior to the `Month 14 (Feb 2028)` opening.

![Venue 2 Construction Gantt Schedule](VenueScalingTimeline/outputs/01_venue_scaling_gantt_chart.png)

![Network Capacity Scaling & Utilization Ramp](VenueScalingTimeline/outputs/02_court_capacity_and_fill_up_ramp.png)

![Venue 2 Leveled CapEx Allocation & Operating Cost Schedule](VenueScalingTimeline/outputs/03_capex_allocation_and_unit_economics.png)

### 4.3 Operational & Financial Milestone Gates

| Gate | Target Date | Operational & Customer KPI | Financial & Cash KPI |
| :--- | :--- | :--- | :--- |
| **Gate 1: Platform Ready** | `Month 4` (`Apr 2027`) | `PICK! Social` passes `500`-user rush load test; Venue 1 utilization **`>= 68.0%`**. | Core platform build on budget; ending cash **`>= 1,000.0M VND`** (`1,000.73M` realized). |
| **Gate 2: Pre-Expansion Gate** | `Month 9` (`Sep 2027`) | **`>= 1,350` paying subscribers** (`1,422` expected); monthly churn `<= 5.0%`. | Monthly EBITDA **`>= 400.0M VND/mo`** (`434.7M` expected); cash **`>= 2,200.0M VND`** (`2,438.49M` expected). |
| **Gate 3: Network Ramp** | `Month 24` (`Dec 2028`) | `20`-court utilization reaches steady state; paying members exceed **`2,400`** (`2,460` expected). | **Year 2 (`FY 2028`) EBITDA reaches `13,134.12M VND`** (`+5,259.4M` under `-30%` stress). |
| **Gate 4: Investor Target** | `Month 36` (`Dec 2029`) | **`3,167` paying members** (`1,263.6M VND/mo` subscription rev) + **`19,800+` visits/mo**. | **Year 3 (`FY 2029`) EBITDA reaches `20,473.28M VND`** (**`5.85x` hurdle**; `+10,285.6M` under `-30%` stress). |

![Granular Monthly Revenue, COGS, OpEx & SG&A Breakdown](FinancialModel/outputs/01_revenue_cogs_opex_sga_sheet.png)

![Monthly Financial Statement Projection Schedule](FinancialModel/outputs/02_fs_month_60m_projection_sheet.png)

---

## 5. Repository Structure & Reproducibility

Run the entire 5-stage simulation, ML training, and chart-generation pipeline from the root directory with a single command:

```bash
uv run main
```

| Directory / File | Description |
| :--- | :--- |
| [`configs/`](configs) | Central stochastic distribution configs (`pricing.toml`, `customers.toml`, `revenue.toml`, `cogs.toml`, `opex.toml`, `capex.toml`, `externals.toml`, `randomness.toml`). |
| [`AdaptivePricing/`](AdaptivePricing) | Stage 1: Hourly court demand elasticity simulation & `HistGradientBoostingRegressor` dynamic pricing engine. |
| [`PickSocialProjectTimeline/`](PickSocialProjectTimeline) | Stage 2: Software engineering WBS, Gantt schedule, buffered cost estimation, and leveled monthly allocation. |
| [`VenueScalingTimeline/`](VenueScalingTimeline) | Stage 3: Second venue expansion WBS, Gantt schedule, leveled CapEx schedule, and 20-court ramp model. |
| [`Marketing/`](Marketing) | Stage 4: Follower-to-customer funnel, 4-segment `MultiOutputRegressor` budget predictor, and segment ancillary spending distributions. |
| [`FinancialModel/`](FinancialModel) | Stage 5: 60-month Monte Carlo 3-statement financial model, stochastic baseline comparison, and downside demand stress test. |
