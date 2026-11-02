"""Configuration loader for `MarketResearch/` (`configs/market.toml`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class CompetitorRecord:
    id: str
    name: str
    category: str
    district: str
    courts: int
    offpeak_price_kvnd: float
    peak_price_kvnd: float
    blended_price_kvnd: float
    daytime_util_pct: float
    peak_util_pct: float
    blended_util_pct: float
    facebook_followers_k: float
    tiktok_followers_k: float
    instagram_followers_k: float
    social_followers_k: float
    annual_revenue_mvnd: float
    positioning_note: str


@dataclass(frozen=True)
class CustomerPersonaRecord:
    id: str
    segment_name: str
    persona_title: str
    priority_tier: str
    priority_rank: int
    customer_share_pct: float
    baseline_customers: int
    visits_per_month: float
    baseline_visits_monthly: int
    visit_share_pct: float
    total_spend_per_visit_kvnd: float
    court_hire_per_visit_kvnd: float
    paddle_balls_per_visit_kvnd: float
    coaching_per_visit_kvnd: float
    fnb_per_visit_kvnd: float
    ancillary_per_visit_kvnd: float
    baseline_monthly_revenue_mvnd: float
    baseline_marketing_budget_mvnd: float
    marketing_cost_per_visit_vnd: float
    marketing_cost_per_customer_vnd: float
    demographics: str
    schedule_habit: str
    main_case_need: str
    pain_points: str
    derived_demand: str


@dataclass(frozen=True)
class MutualDemandPillar:
    pillar_id: str
    pillar_name: str
    segments_served: str
    customer_friction: str
    mutual_need_statement: str


@dataclass(frozen=True)
class MarketSizingConfig:
    years: list[int]
    tam_vietnam_bvnd: list[float]
    tam_cagr_2026_2029_pct: float
    sam_hcmc_bvnd: list[float]
    sam_cagr_2026_2029_pct: float
    sam_prime_catchment_bvnd: list[float]
    catchment_cagr_2026_2029_pct: float
    som_pick_bvnd: list[float]
    som_cagr_2026_2029_pct: float
    tam_2026_breakdown: dict[str, float]
    sam_2026_breakdown: dict[str, float]


@dataclass(frozen=True)
class MarketResearchConfig:
    country: str
    primary_city: str
    base_year: int
    target_year: int
    vietnam_awareness_pct: float
    vietnam_trial_pct: float
    vietnam_frequent_players_millions: float
    vietnam_ecommerce_gear_2025_bvnd: float
    vietnam_ecommerce_yoy_growth_pct: float
    hcmc_active_courts_2026: int
    hcmc_active_venues_2026: int
    hcmc_active_courts_2029: int
    hcmc_active_venues_2029: int
    local_radius_km: float
    new_local_rivals_past_year: int
    cheaper_peak_local_rivals: int
    sizing: MarketSizingConfig
    competitors: list[CompetitorRecord]
    personas: list[CustomerPersonaRecord]
    mutual_demand_pillars: list[MutualDemandPillar]


def load_market_research_config(config_path: str | Path) -> MarketResearchConfig:
    """Load `configs/market.toml` from a file path or central `configs/` directory."""
    p = Path(config_path)
    if p.is_dir():
        toml_file = p / "market.toml"
    else:
        toml_file = p

    with toml_file.open("rb") as f:
        raw = tomllib.load(f)

    ov = raw["market_overview"]
    sz = raw["market_sizing"]

    sizing = MarketSizingConfig(
        years=[int(x) for x in sz["years"]],
        tam_vietnam_bvnd=[float(x) for x in sz["tam_vietnam_bvnd"]],
        tam_cagr_2026_2029_pct=float(sz["tam_cagr_2026_2029_pct"]),
        sam_hcmc_bvnd=[float(x) for x in sz["sam_hcmc_bvnd"]],
        sam_cagr_2026_2029_pct=float(sz["sam_cagr_2026_2029_pct"]),
        sam_prime_catchment_bvnd=[float(x) for x in sz["sam_prime_catchment_bvnd"]],
        catchment_cagr_2026_2029_pct=float(sz["catchment_cagr_2026_2029_pct"]),
        som_pick_bvnd=[float(x) for x in sz["som_pick_bvnd"]],
        som_cagr_2026_2029_pct=float(sz["som_cagr_2026_2029_pct"]),
        tam_2026_breakdown={k: float(v) for k, v in sz["tam_2026_breakdown"].items()},
        sam_2026_breakdown={k: float(v) for k, v in sz["sam_2026_breakdown"].items()},
    )

    competitors = [
        CompetitorRecord(
            id=str(c["id"]),
            name=str(c["name"]),
            category=str(c["category"]),
            district=str(c["district"]),
            courts=int(c["courts"]),
            offpeak_price_kvnd=float(c["offpeak_price_kvnd"]),
            peak_price_kvnd=float(c["peak_price_kvnd"]),
            blended_price_kvnd=float(c["blended_price_kvnd"]),
            daytime_util_pct=float(c["daytime_util_pct"]),
            peak_util_pct=float(c["peak_util_pct"]),
            blended_util_pct=float(c["blended_util_pct"]),
            facebook_followers_k=float(c["facebook_followers_k"]),
            tiktok_followers_k=float(c["tiktok_followers_k"]),
            instagram_followers_k=float(c["instagram_followers_k"]),
            social_followers_k=float(c["social_followers_k"]),
            annual_revenue_mvnd=float(c["annual_revenue_mvnd"]),
            positioning_note=str(c["positioning_note"]),
        )
        for c in raw["competitors"]
    ]

    personas = [
        CustomerPersonaRecord(
            id=str(pr["id"]),
            segment_name=str(pr["segment_name"]),
            persona_title=str(pr["persona_title"]),
            priority_tier=str(pr["priority_tier"]),
            priority_rank=int(pr["priority_rank"]),
            customer_share_pct=float(pr["customer_share_pct"]),
            baseline_customers=int(pr["baseline_customers"]),
            visits_per_month=float(pr["visits_per_month"]),
            baseline_visits_monthly=int(pr["baseline_visits_monthly"]),
            visit_share_pct=float(pr["visit_share_pct"]),
            total_spend_per_visit_kvnd=float(pr["total_spend_per_visit_kvnd"]),
            court_hire_per_visit_kvnd=float(pr["court_hire_per_visit_kvnd"]),
            paddle_balls_per_visit_kvnd=float(pr["paddle_balls_per_visit_kvnd"]),
            coaching_per_visit_kvnd=float(pr["coaching_per_visit_kvnd"]),
            fnb_per_visit_kvnd=float(pr["fnb_per_visit_kvnd"]),
            ancillary_per_visit_kvnd=float(pr["ancillary_per_visit_kvnd"]),
            baseline_monthly_revenue_mvnd=float(pr["baseline_monthly_revenue_mvnd"]),
            baseline_marketing_budget_mvnd=float(pr["baseline_marketing_budget_mvnd"]),
            marketing_cost_per_visit_vnd=float(pr["marketing_cost_per_visit_vnd"]),
            marketing_cost_per_customer_vnd=float(pr["marketing_cost_per_customer_vnd"]),
            demographics=str(pr["demographics"]),
            schedule_habit=str(pr["schedule_habit"]),
            main_case_need=str(pr["main_case_need"]),
            pain_points=str(pr["pain_points"]),
            derived_demand=str(pr["derived_demand"]),
        )
        for pr in raw["personas"]
    ]

    pillars = [
        MutualDemandPillar(
            pillar_id=str(pl["pillar_id"]),
            pillar_name=str(pl["pillar_name"]),
            segments_served=str(pl["segments_served"]),
            customer_friction=str(pl["customer_friction"]),
            mutual_need_statement=str(pl["mutual_need_statement"]),
        )
        for pl in raw["mutual_demand_pillars"]
    ]

    return MarketResearchConfig(
        country=str(ov["country"]),
        primary_city=str(ov["primary_city"]),
        base_year=int(ov["base_year"]),
        target_year=int(ov["target_year"]),
        vietnam_awareness_pct=float(ov["vietnam_awareness_pct"]),
        vietnam_trial_pct=float(ov["vietnam_trial_pct"]),
        vietnam_frequent_players_millions=float(ov["vietnam_frequent_players_millions"]),
        vietnam_ecommerce_gear_2025_bvnd=float(ov["vietnam_ecommerce_gear_2025_bvnd"]),
        vietnam_ecommerce_yoy_growth_pct=float(ov["vietnam_ecommerce_yoy_growth_pct"]),
        hcmc_active_courts_2026=int(ov["hcmc_active_courts_2026"]),
        hcmc_active_venues_2026=int(ov["hcmc_active_venues_2026"]),
        hcmc_active_courts_2029=int(ov["hcmc_active_courts_2029"]),
        hcmc_active_venues_2029=int(ov["hcmc_active_venues_2029"]),
        local_radius_km=float(ov["local_radius_km"]),
        new_local_rivals_past_year=int(ov["new_local_rivals_past_year"]),
        cheaper_peak_local_rivals=int(ov["cheaper_peak_local_rivals"]),
        sizing=sizing,
        competitors=competitors,
        personas=personas,
        mutual_demand_pillars=pillars,
    )
