# PICK! Marketing Funnel, Segment VND/Visit Response & 12 Ancillary Revenue Distributions Report

## 1. Explicit Team Assumption: Social Follower-to-Customer Funnel (~8.57% Conversion)

Per **Case Page 2 ("PICK! At a glance")** and **Case Exhibit 2 ("Monthly Costs")**, when PICK! spends its baseline **VND 25,000,000 / month** on marketing, it sustains:
- **42,000** TikTok and Instagram followers
- **3,600** monthly unique customers
- **7,596** exact monthly player visits (reported as *"about 7,600 player visits"* in Case Page 2)

> **EXPLICIT TEAM ASSUMPTION (NOTED IN `configs/customers.toml`)**:
> We explicitly assume that PICK!'s active monthly unique customers are strictly a **subset of its TikTok and Instagram followers**. Therefore, the baseline **Follower-to-Customer Conversion Rate** is parameterized as a Truncated Normal distribution centered at:
> $$\eta_0 = \frac{3,600\text{ unique customers}}{42,000\text{ followers}} = 8.5714\% \quad (\text{Mean} = 8.5714\%,\; \text{Std} = 0.38\%,\; \text{Range} = [7.40\%,\; 9.80\%])$$

---

## 2. Segment Marketing Cost per Visit (`x, y, z, w` VND/Visit) & Baseline 25.00M VND Split

Splitting the baseline **25.00M VND/month** marketing budget across the 4 customer segments from **Case Exhibit 4** and merging customer count and visits/month into direct **Monthly Segment Visits Distributions** yields the exact marketing spend required to drive **1 player visit** in each segment:

| Customer Segment | Share of Customers | Baseline Customers | Visits / Customer / Mo | Merged Monthly Visits Dist ($\mu \pm \sigma$) | Baseline Marketing Budget Split | Marketing Cost per Visit (`VND / Visit`) | Marketing Cost per Active Customer (`VND / Cust`) | Strategic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Competitive Players** | 20.0% | 720 | 4.5 | **$3,240 \pm 165$** | **5.00M VND** (20.0%) | **1,543 VND / visit** | 6,944 VND / cust | Lowest cost/visit due to high $4.5\times$/mo repeat frequency |
| **2. Social Players** | 35.0% | 1,260 | 2.0 | **$2,520 \pm 135$** | **8.50M VND** (34.0%) | **3,373 VND / visit** | 6,746 VND / cust | Core group-play & PICK! Social membership audience |
| **3. Trend Explorers** | 30.0% | 1,080 | 1.1 | **$1,188 \pm 95$** | **7.50M VND** (30.0%) | **6,313 VND / visit** | 6,944 VND / cust | Highest cost/visit ($4.09\times$ Competitive) due to $1.1\times$/mo low retention |
| **4. Families** | 15.0% | 540 | 1.2 | **$648 \pm 48$** | **4.00M VND** (16.0%) | **6,173 VND / visit** | 7,407 VND / cust | High acquisition cost per visit, but highest ancillary coaching & F&B spend |
| **Total / Weighted Avg** | **100.0%** | **3,600** | **2.11** | **7,596 (~7,600)** | **25.00M VND (100%)** | **3,291 VND / visit** | **6,944 VND / cust** | Exact reconciliation to Case Page 2, Exhibit 2 & Exhibit 4 |

---

## 3. Grounding the 12 Segment-by-Ancillary Spending Distributions ($3\text{ Lines} \times 4\text{ Segments}$)

To model **Revenue Line 2 (Paddle Rental & Balls = 15.00M VND/mo)**, **Revenue Line 3 (Coaching = 115.00M VND/mo)**, and **Revenue Line 4 (Food & Drinks = 75.00M VND/mo)** dynamically from customer segment visits, we define **12 Truncated Normal distributions** in `configs/customers.toml` ($3\text{ ancillary lines} \times 4\text{ customer segments}$), grounded simultaneously in **Case Exhibit 1** and **Case Exhibit 4**:

| Customer Segment | Baseline Visits / Mo | Dist 1: Paddle & Balls ($\text{kVND/visit} \to \text{M VND/mo}$) | Dist 2: Coaching ($\text{kVND/visit} \to \text{M VND/mo}$) | Dist 3: Food & Drinks ($\text{kVND/visit} \to \text{M VND/mo}$) | Total Ancillary (`kVND/visit`) | Court Hire Share (`kVND/visit`) | Total Spend / Visit (**Case Exhibit 4**) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Competitive Players** | 3,240 | **$0.80 \pm 0.12\text{ k}$** (2.59M/mo) | **$21.30 \pm 1.85\text{ k}$** (69.01M/mo) | **$10.00 \pm 0.95\text{ k}$** (32.40M/mo) | 32.10 k | 62.90 k | **95.00 kVND** (Exh. 4 = 95k) |
| **2. Social Players** | 2,520 | **$1.60 \pm 0.20\text{ k}$** (4.03M/mo) | **$9.52 \pm 0.90\text{ k}$** (23.99M/mo) | **$11.50 \pm 1.05\text{ k}$** (28.98M/mo) | 22.62 k | 57.38 k | **80.00 kVND** (Exh. 4 = 80k) |
| **3. Trend Explorers** | 1,188 | **$5.20 \pm 0.50\text{ k}$** (6.18M/mo) | **$4.80 \pm 0.55\text{ k}$** (5.70M/mo) | **$5.99 \pm 0.65\text{ k}$** (7.12M/mo) | 15.99 k | 49.01 k | **65.00 kVND** (Exh. 4 = 65k) |
| **4. Families** | 648 | **$3.40 \pm 0.35\text{ k}$** (2.20M/mo) | **$25.15 \pm 2.10\text{ k}$** (16.30M/mo) | **$10.03 \pm 0.95\text{ k}$** (6.50M/mo) | 38.58 k | 46.42 k | **85.00 kVND** (Exh. 4 = 85k) |
| **Baseline Monthly Sum** | **7,596** | **15.005M VND/mo** (Exh. 1 = **15.00M**) | **115.001M VND/mo** (Exh. 1 = **115.00M**) | **74.995M VND/mo** (Exh. 1 = **75.00M**) | **205.00M/mo** | **436.70M/mo** | **641.70M VND/mo** (Exh. 1 = **640M**) |

### 3.1 Quarterly Seasonal Tournament Ancillary & Follower Boost (`Mar, Jun, Sep, Dec` from Month 6+)
Starting in **Month 6 (`Jun-27`, Summer Tournament)** after `PICK! Social` launches in `Month 5 (May-27)`, PICK! hosts a flagship seasonal tournament every quarter (`Spring [Mar]`, `Summer [Jun]`, `Fall [Sep]`, `Winter [Dec]`):
- **Tournament Participant Field**: $192 \pm 16$ players in the `6`-court phase (`Months 6, 9, 12`), scaling to $420 \pm 32$ players across the `20`-court network (`Months 15, 18, ..., 60`).
- **Social Follower Boost**: Tournament highlight reels and brackets multiply monthly new follower acquisition by **$1.15\times$ (`+15%`)** in tournament months.
- **Ancillary Spend Lift (`+85.0k VND / participant`)**: Both active members (who enter for **free**) and non-members (who pay **`350,000 VND`** entry fee) generate incremental full-day tournament ancillary spending of **`55.0k VND`** in `Food & Drinks`, **`10.0k VND`** in `Paddle & Balls`, and **`20.0k VND`** in pre-tournament `Coaching` clinics (`~16.3M VND/qtr` in 6-court phase; `~35.7M VND/qtr` in 20-court phase).

---

## 4. 60-Month ML Marketing Allocation & Ancillary Revenue Trajectory Summary

| Horizon Milestone | Active Courts | ML Marketing Spend (Comp / Soc / Trend / Fam = **Total**) | Active Followers (`~8.57%` Conv) | Total Monthly Visits | Paddle & Balls Rev (`M VND`) | Coaching Rev (`M VND`) | Food & Drinks Rev (`M VND`) | Total Ancillary Rev (`M VND`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Month 1 (Jan-27, Build)** | 6 | 10.26M / 16.55M / 13.30M / 8.37M = **48.49M** | 42,871 | 9,196 | 18.54M | 137.59M | 91.67M | **247.80M** |
| **Month 5 (May-27, App Live)** | 6 | 8.54M / 13.88M / 13.12M / 6.76M = **42.30M** | 46,308 | 9,718 | 17.21M | 148.69M | 97.07M | **262.97M** |
| **Month 14 (Feb-28, 20 Courts)** | 20 | 18.45M / 23.93M / 23.88M / 15.97M = **82.24M** | 55,776 | 20,712 | 40.30M | 307.18M | 209.28M | **556.77M** |
| **Month 24 (Dec-28, End Y2)** | 20 | 10.14M / 15.54M / 17.12M / 7.66M = **50.46M** | 71,233 | 21,242 | 45.82M | 333.17M | 224.98M | **603.97M** |
| **Month 36 (Dec-29, End Y3)** | 20 | 10.38M / 16.39M / 17.33M / 7.99M = **52.09M** | 92,170 | 21,618 | 44.94M | 311.18M | 224.98M | **581.10M** |
| **5-Year Monthly Average** | 17.7 | 10.86M / 16.88M / 17.46M / 8.24M = **53.44M** | 87,787 | 18,916 | 37.96M | 286.94M | 191.76M | **516.66M** |
