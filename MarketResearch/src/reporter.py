"""CSV and Markdown report exporter for `MarketResearch/` (`outputs/`)."""

from __future__ import annotations

import csv
from pathlib import Path

from .config import MarketResearchConfig
from .model import MarketPositioningAnalysis


def export_market_research_outputs(
    cfg: MarketResearchConfig,
    analysis: MarketPositioningAnalysis,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """
    Export all 3 CSV datasets and `REPORT.md` into `MarketResearch/outputs/`:
      1. `hcmc_competitor_positioning.csv`
      2. `tam_sam_som_market_sizing.csv`
      3. `customer_personas_and_demand.csv`
      4. `REPORT.md`
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Competitor positioning CSV
    comp_csv = output_dir / "hcmc_competitor_positioning.csv"
    with comp_csv.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "name",
                "category",
                "district",
                "courts",
                "offpeak_price_kvnd",
                "peak_price_kvnd",
                "blended_price_kvnd",
                "daytime_util_pct",
                "peak_util_pct",
                "blended_util_pct",
                "facebook_followers_k",
                "tiktok_followers_k",
                "instagram_followers_k",
                "social_followers_k",
                "annual_revenue_mvnd",
                "annual_revenue_bvnd",
                "revenue_per_court_mvnd",
                "positioning_note",
            ]
        )
        for c in cfg.competitors:
            writer.writerow(
                [
                    c.id,
                    c.name,
                    c.category,
                    c.district,
                    c.courts,
                    f"{c.offpeak_price_kvnd:.1f}",
                    f"{c.peak_price_kvnd:.1f}",
                    f"{c.blended_price_kvnd:.1f}",
                    f"{c.daytime_util_pct:.1f}",
                    f"{c.peak_util_pct:.1f}",
                    f"{c.blended_util_pct:.1f}",
                    f"{c.facebook_followers_k:.1f}",
                    f"{c.tiktok_followers_k:.1f}",
                    f"{c.instagram_followers_k:.1f}",
                    f"{c.social_followers_k:.1f}",
                    f"{c.annual_revenue_mvnd:.2f}",
                    f"{c.annual_revenue_mvnd / 1000.0:.3f}",
                    f"{c.annual_revenue_mvnd / c.courts:.2f}",
                    c.positioning_note,
                ]
            )

    # 2. TAM / SAM / SOM Market Sizing CSV
    sizing_csv = output_dir / "tam_sam_som_market_sizing.csv"
    with sizing_csv.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "year",
                "tam_vietnam_bvnd",
                "sam_hcmc_bvnd",
                "sam_prime_catchment_bvnd",
                "som_pick_bvnd",
                "som_share_of_hcmc_sam_pct",
                "som_share_of_prime_catchment_pct",
            ]
        )
        sz = cfg.sizing
        for i, yr in enumerate(sz.years):
            tam_v = sz.tam_vietnam_bvnd[i]
            sam_v = sz.sam_hcmc_bvnd[i]
            cat_v = sz.sam_prime_catchment_bvnd[i]
            som_v = sz.som_pick_bvnd[i]
            writer.writerow(
                [
                    yr,
                    f"{tam_v:.2f}",
                    f"{sam_v:.2f}",
                    f"{cat_v:.2f}",
                    f"{som_v:.2f}",
                    f"{(som_v / sam_v) * 100.0:.3f}",
                    f"{(som_v / cat_v) * 100.0:.2f}",
                ]
            )

    # 3. Customer Personas & Mutual Demand CSV
    personas_csv = output_dir / "customer_personas_and_demand.csv"
    with personas_csv.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "priority_rank",
                "priority_tier",
                "segment_name",
                "persona_title",
                "customer_share_pct",
                "baseline_customers",
                "visits_per_month",
                "baseline_visits_monthly",
                "visit_share_pct",
                "total_spend_per_visit_kvnd",
                "court_hire_per_visit_kvnd",
                "paddle_balls_per_visit_kvnd",
                "coaching_per_visit_kvnd",
                "fnb_per_visit_kvnd",
                "ancillary_per_visit_kvnd",
                "baseline_monthly_revenue_mvnd",
                "baseline_marketing_budget_mvnd",
                "marketing_cost_per_visit_vnd",
                "marketing_cost_per_customer_vnd",
                "main_case_need",
                "demographics",
                "schedule_habit",
                "pain_points",
                "derived_demand",
            ]
        )
        for p in sorted(cfg.personas, key=lambda x: x.priority_rank):
            writer.writerow(
                [
                    p.id,
                    p.priority_rank,
                    p.priority_tier,
                    p.segment_name,
                    p.persona_title,
                    f"{p.customer_share_pct:.1f}",
                    p.baseline_customers,
                    f"{p.visits_per_month:.2f}",
                    p.baseline_visits_monthly,
                    f"{p.visit_share_pct:.2f}",
                    f"{p.total_spend_per_visit_kvnd:.2f}",
                    f"{p.court_hire_per_visit_kvnd:.2f}",
                    f"{p.paddle_balls_per_visit_kvnd:.2f}",
                    f"{p.coaching_per_visit_kvnd:.2f}",
                    f"{p.fnb_per_visit_kvnd:.2f}",
                    f"{p.ancillary_per_visit_kvnd:.2f}",
                    f"{p.baseline_monthly_revenue_mvnd:.2f}",
                    f"{p.baseline_marketing_budget_mvnd:.2f}",
                    f"{p.marketing_cost_per_visit_vnd:.2f}",
                    f"{p.marketing_cost_per_customer_vnd:.2f}",
                    p.main_case_need,
                    p.demographics,
                    p.schedule_habit,
                    p.pain_points,
                    p.derived_demand,
                ]
            )

    # 4. Concise Executive REPORT.md for Section 1 (Our Market) & Section 2 (Our Customers)
    report_md = output_dir / "REPORT.md"
    pb = analysis.pick_baseline

    comp_rows_md = []
    for c in cfg.competitors:
        rev_b = c.annual_revenue_mvnd / 1000.0
        comp_rows_md.append(
            f"| **{c.name}** | {c.district} | `{c.courts} courts` | `{c.offpeak_price_kvnd:.0f}k / {c.peak_price_kvnd:.0f}k` (`{c.blended_price_kvnd:.0f}k`) | "
            f"`{c.facebook_followers_k:.1f}k / {c.tiktok_followers_k:.1f}k / {c.instagram_followers_k:.1f}k` | **`{c.social_followers_k:.1f}k`** | "
            f"**`{c.annual_revenue_mvnd:,.0f}M`** (`{rev_b:.2f}B`) |"
        )
    comp_table_str = "\n".join(comp_rows_md)

    pillar_rows_md = []
    for pl in cfg.mutual_demand_pillars:
        pillar_rows_md.append(
            f"| **{pl.pillar_name}** | `{pl.segments_served}` | {pl.customer_friction} | {pl.mutual_need_statement} |"
        )
    pillar_table_str = "\n".join(pillar_rows_md)

    report_content = f"""# Section 1 (`Our Market`) & Section 2 (`Our Customers`) - Market Research & Customer Demand Report

**Company**: PICK! Pickleball Club (Ho Chi Minh City)  
**Scope**: Vietnam & Ho Chi Minh City (HCMC) Commercial Pickleball Market Sizing (`TAM - SAM - SOM`), Current HCMC Competitive Positioning, 4 Customer Personas, and Derived Mutual Demand  
**Internal Evidence Vault**: Complete primary/secondary sources, URLs, and multi-platform social reach breakdowns are archived in `MarketResearch/RESEARCH_SOURCES.md` and parameterized in `configs/market.toml`.

---

## 1. Our Market: Vietnam & HCMC Pickleball Ecosystem & Competitive Positioning

### 1.1 The Vietnam & HCMC Pickleball Boom (`2024 - 2026` Institutionalization)
Vietnam has evolved from an emerging recreational trend in early 2024 into Asia-Pacific's fastest-growing commercial pickleball market:
- **#1 Public Awareness in APAC (`88.0%`) & `16 Million` Frequent Players**: Regional consumer tracking reports **`88.0%` national awareness** and **`37.0%` population trial** in Vietnam, supported by Global Top 10 DUPR rating growth velocity and landmark international events hosted locally, including the **2026 Pickleball World Cup in Da Nang** (`~5,000` athletes from `>80` countries) and the **PPA Tour Asia Ho Chi Minh City Open**.
- **Trillion-VND Consumer Spend Velocity**: Domestic e-commerce equipment and court apparel sales surged **`+1,282%` YoY**, crossing **`1,000B VND/yr` (`~$40M USD`)** across Shopee, TikTok Shop, and Lazada, confirming sustained consumer commitment.
- **Ho Chi Minh City (`HCMC`) as the Commercial Epicenter**: HCMC operates **`~1,050` active courts across `~165` commercial venues** in 2026 (projected to expand to **`~1,650` courts across `~240` venues by `2029`**), with dense clustering across `Thao Dien / Thu Duc`, `Binh Thanh`, and `Phu My Hung / District 7`.

### 1.2 Market Sizing (`TAM - SAM - SOM`, `2026` Current vs. `2029` Industry Forecast)

| Market Tier | Scope & Revenue Components | `2026` Current Market | `2029` Projected Market | `2026-2029` Growth (`CAGR`) | Strategic Relevance to PICK! |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **TAM** (`Vietnam National Ecosystem`) | All commercial court hire (`980B`), equipment retail (`820B`), coaching (`360B`) & courtside F&B/events (`290B`) across Vietnam (`~3,200` to `~5,400` courts) | **`2,450.0B VND/yr`** | **`5,180.0B VND/yr`** | **`+28.35%` CAGR** (`2.11x`) | Macro tailwind driving continuous inflow of first-time `Trend Explorers` and `Families` |
| **SAM** (`HCMC Commercial Venue Market`) | Commercial court hire (`655B`), coaching (`215B`), F&B (`165B`) & memberships/rentals (`85B`) across HCMC's `1,050` to `1,650` courts | **`1,120.0B VND/yr`** | **`2,145.0B VND/yr`** | **`+24.18%` CAGR** (`1.92x`) | Direct addressable city-wide venue and community market in Ho Chi Minh City |
| **Prime Catchment SAM** (`Core HCMC Hubs`) | Prime urban HCMC catchment (`Thao Dien / Thu Duc`, `Binh Thanh`, `District 7` & `<3 km` local radius) | **`142.0B VND/yr`** | **`241.0B VND/yr`** | **`+19.28%` CAGR** (`1.70x`) | Immediate geographic catchment where PICK! competes for regular players |
| **SOM** (`PICK! Market Capture`) | Current `6 courts` baseline (`7.68B/yr` = `0.69%` of HCMC SAM) vs. obtainable multi-venue Year 3 share (`1.55%` of HCMC SAM) | **`7.68B VND/yr`** (`0.69%` of SAM) | **`33.23B VND/yr`** (`1.55%` of SAM) | **`+62.95%` CAGR** (`4.33x` Obtainable) | Demonstrates substantial headroom within HCMC's expanding commercial market |

### 1.3 HCMC Competitive Positioning: High Brand Attention vs. Capacity & Off-Peak Bottlenecks
Per **Case Page 2**, **4 new pickleball venues opened within `3 km` of PICK! in the past year, 2 of them offering lower prices at peak times (`< 260k VND/hr`)**. Benchmarking **PICK! (`6 courts`)** against **9 real HCMC pickleball clubs** across **Peak Court Price (`X`)**, **Combined Multi-Platform Social Reach (`Y`: Facebook + TikTok + Instagram)**, and **Estimated Annual Revenue (`Bubble Size`)** reveals PICK!'s exact market position today:

| Club / Venue Name | HCMC District / Hub | Capacity | Off-Peak / Peak (`Blended kVND/hr`) | Platform Reach (`FB / TikTok / IG`) | Combined Reach (`Followers`) | Estimated Annual Revenue (`M VND/yr`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{comp_table_str}

- **Key Diagnostic Insights from `01_hcmc_market_positioning_bubble_chart.png`**:
  1. **PICK! Commands #1 Online Attention in HCMC (`{pb.social_followers_k:.1f}k` Followers)**: PICK!'s `42,000` TikTok and Instagram follower base is **$2.26\\times$ the HCMC peer average (`{analysis.avg_peer_followers_k:.1f}k` combined followers across Facebook, TikTok, and Instagram)**. Lack of brand awareness is not PICK!'s problem.
  2. **Why PICK!'s Revenue Bubble (`7.68B VND/yr`) Trails 10-to-14 Court Rivals (`9.60B - 12.90B VND/yr`)**:
     - **60% Empty Weekday Daytime Courts**: At a static `160k VND/hr` weekday daytime rate (`06:00-17:00`), PICK! achieves only `40.0%` daytime utilization while HCMC competitors offer `80k-130k VND/hr` off-peak slots to capture students and flexible daytime groups.
     - **Price-Undercut Exposure at Peak (`< 260k VND/hr`)**: Seven of the nine surveyed HCMC peers price peak hours below `260k VND/hr` (`150k-250k VND/hr`), mirroring the Case finding that new local rivals undercut PICK! during evening peak hours.
     - **Zero Membership Lock-In & 6-Court Peak Ceiling**: Because PICK! has `1,150` free registered accounts (`0` paid memberships) and its `6 courts` sit `95%` full on weekday evenings and `90%` full on weekends, overflow peak groups face zero switching cost to try cheaper nearby courts.

---

## 2. Our Customers: 4 HCMC Personas, Unit Economics & Derived Mutual Demand

### 2.1 Reconciliation of Customer Segments, Per-Visit Spend & Marketing Acquisition Cost
Grounded simultaneously in **Case Page 2**, **Case Exhibit 1 (`640M VND/month`)**, **Case Exhibit 2 (`25.00M VND/month` marketing)**, and **Case Exhibit 4**, PICK!'s `3,600` monthly unique customers (`8.5714%` monthly active conversion of its `42,000` followers) and `7,596` monthly visits decompose into four behavioral customer profiles:

| Priority Rank & Tier | Customer Segment & Profile | Share of Customers (`Customers/Month`) | Visits / Customer / Month (`Visits/Month`) | Court Hire / Visit | Paddle & Balls / Visit | Coaching / Visit | Food & Drinks / Visit | Total Spend / Visit (`Exhibit 4`) | Baseline Monthly Revenue | Baseline Marketing Spend (`Marketing Cost / Visit`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PRIORITY #1** (`Core Volume Segment`) | **Social Players** (*After-Work Young Professionals*) | **`35.0%`** (`1,260`) | **`2.0`** (`2,520` = `33.2%`) | `57.38k` | `1.60k` | `9.52k` | **`11.50k`** | **`80.00k VND`** | `201.60M VND` | `8.50M VND` (**`3,373 VND/visit`**) |
| **PRIORITY #2** (`High-Churn Trial Segment`) | **Trend Explorers** (*University Students & First-Timers*) | **`30.0%`** (`1,080`) | **`1.1`** (`1,188` = `15.6%`) | `49.01k` | **`5.20k`** | `4.80k` | `5.99k` | **`65.00k VND`** | `77.22M VND` | `7.50M VND` (**`6,313 VND/visit`**) |
| **Tier 2 Anchor** (`High-Frequency Base`) | **Competitive Players** (*Advanced Racket-Sport Regulars*) | **`20.0%`** (`720`) | **`4.5`** (`3,240` = `42.7%`) | **`62.90k`** | `0.80k` | `21.30k` | `10.00k` | **`95.00k VND`** | **`307.80M VND`** | `5.00M VND` (**`1,543 VND/visit`**) |
| **Tier 3 Niche** (`High-Ancillary Coaching`) | **Families** (*Weekend Parents & Junior Beginners*) | **`15.0%`** (`540`) | **`1.2`** (`648` = `8.5%`) | `46.42k` | `3.40k` | **`25.15k`** | `10.03k` | **`85.00k VND`** | `55.08M VND` | `4.00M VND` (**`6,173 VND/visit`**) |
| **Total / Average** | **All 4 Segments Reconciled** | **`100.0%`** (`3,600`) | **`2.11`** (`7,596` $\\approx$ `7,600`) | **`436.70M`** (`Exhibit 1: 435M`) | **`15.00M`** (`Exhibit 1: 15M`) | **`115.00M`** (`Exhibit 1: 115M`) | **`75.00M`** (`Exhibit 1: 75M`) | **`84.48k VND`** | **`641.70M VND`** (`Exhibit 1: 640M`) | **`25.00M VND`** (**`3,291 VND/visit`**) |

### 2.2 Why `Social Players (35%)` & `Trend Explorers (30%)` Are Our #1 and #2 Priority Segments
1. **They Represent `65.0%` of PICK!'s Unique Customer Base (`2,340` of `3,600` Customers/Month) Yet Generate Only `48.8%` of Visits (`3,708` of `7,596`)**:
   - **Trend Explorers (`30.0%` of customers = `1,080/mo`)** incur an implied marketing cost of **`6,313 VND / visit`** - **$4.09\\times$ higher than Competitive Players (`1,543 VND/visit`)** - because viral TikTok/Instagram reels bring them in once (`1.1 visits/mo`), after which `~89%` churn without booking a second visit.
   - **Social Players (`35.0%` of customers = `1,260/mo`)** represent the single largest customer pool and spend the most on courtside Food & Drinks (**`11.50k VND/visit`**), but average only `2.0 visits/mo` and have zero switching cost when cheaper `< 260k VND/hr` local rivals try to poach evening friend groups.
2. **Complementary Role of `Competitive Players (20%)` and `Families (15%)`**:
   - **Competitive Players (`20.0%` of customers = `720/mo`)** already visit `4.5x/mo` (`42.7%` of all visits) at the lowest marketing cost per visit (`1,543 VND/visit`), acting as the high-frequency anchor.
   - **Families (`15.0%` of customers = `540/mo`)** generate the highest ancillary spend per visit (**`38.58k VND/visit`**, led by `25.15k VND` in coaching), anchoring weekend morning/afternoon coaching demand.

### 2.3 Derived Mutual Demand Across All Four Customer Personas

| Derived Mutual Demand Pillar | Segments Served | Root Customer Friction Identified | Derived Customer Need Statement |
| :--- | :---: | :--- | :--- |
{pillar_table_str}

---

## 3. Generated Deliverables (`MarketResearch/outputs/`)
- **Visual 1 (`Our Market` - HCMC Competitive Positioning Bubble Chart & TAM-SAM-SOM Sizing)**: `MarketResearch/outputs/01_hcmc_market_positioning_bubble_chart.png`
- **Visual 2 (`Our Customers` - 4-Persona Comparison Table, Per-Visit Spend Breakdown & Priority Share Analysis)**: `MarketResearch/outputs/02_customer_personas_and_mutual_demand.png`
- **CSV 1 (HCMC Competitor Positioning & Multi-Platform Reach)**: `MarketResearch/outputs/hcmc_competitor_positioning.csv`
- **CSV 2 (Vietnam TAM, HCMC SAM & Obtainable SOM Sizing `2024-2029`)**: `MarketResearch/outputs/tam_sam_som_market_sizing.csv`
- **CSV 3 (4 Customer Personas, Unit Economics & Derived Demand)**: `MarketResearch/outputs/customer_personas_and_demand.csv`
- **Internal Research & Citation Vault**: `MarketResearch/RESEARCH_SOURCES.md`
"""
    report_md.write_text(report_content, encoding="ascii")
    return comp_csv, sizing_csv, personas_csv, report_md
