"""Configuration loader for `Marketing/` reading from the central `CaseComp/configs/` folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
import numpy as np


@dataclass(frozen=True)
class DistParam:
    """Truncated Normal distribution parameter specification."""

    dist: str
    mean: float
    std: float
    min: float
    max: float
    unit: str = ""
    source: str = ""

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample from the truncated normal distribution using numpy Generator."""
        if self.std <= 0.0 or self.min >= self.max:
            return np.full(size, self.mean, dtype=float)
        draws = rng.normal(loc=self.mean, scale=self.std, size=size)
        return np.clip(draws, self.min, self.max)


def _parse_dist(d: dict) -> DistParam:
    return DistParam(
        dist=str(d.get("dist", "truncated_normal")),
        mean=float(d["mean"]),
        std=float(d.get("std", 0.0)),
        min=float(d.get("min", d["mean"])),
        max=float(d.get("max", d["mean"])),
        unit=str(d.get("unit", "")),
        source=str(d.get("source", "")),
    )


@dataclass(frozen=True)
class SegmentMarketingConfig:
    key: str
    label: str
    main_need: str
    share_dist: DistParam
    visits_per_month_dist: DistParam
    monthly_visits_dist: DistParam
    baseline_marketing_budget_mvnd_dist: DistParam
    marketing_vnd_per_visit_dist: DistParam
    marketing_elasticity_dist: DistParam
    spend_per_visit_kvnd_dist: DistParam
    court_hire_spend_per_visit_kvnd_dist: DistParam
    paddle_balls_spend_per_visit_kvnd_dist: DistParam
    coaching_spend_per_visit_kvnd_dist: DistParam
    fnb_spend_per_visit_kvnd_dist: DistParam
    subscription_propensity_dist: DistParam


@dataclass(frozen=True)
class SocialFunnelConfig:
    initial_followers_dist: DistParam
    follower_to_customer_conversion_dist: DistParam
    monthly_new_follow_rate_dist: DistParam
    monthly_unfollow_rate_dist: DistParam


@dataclass(frozen=True)
class MarketingAppConfig:
    # Simulation settings
    random_seed: int
    n_simulations: int
    train_fraction: float
    projection_months: int
    model_start_month: str

    # Funnel & Segment distributions
    unique_customers_dist: DistParam
    player_visits_anchor_dist: DistParam
    social_funnel: SocialFunnelConfig
    segments: tuple[SegmentMarketingConfig, ...]

    # Baseline marketing & COGS shares
    baseline_total_marketing_mvnd_dist: DistParam
    coach_fees_share_dist: DistParam
    fnb_supplies_share_dist: DistParam
    payment_fee_pct_dist: DistParam

    # External trend & Venue 2 scaling parameters
    trend_monthly_change_dist: DistParam
    trend_floor: float
    trend_ceiling: float
    venue2_open_month_index: int
    venue2_courts_added: int
    venue2_fill_up_months_dist: DistParam

    # Membership parameters
    membership_launch_month_index: int
    membership_fee_mvnd: float
    membership_churn_dist: DistParam
    membership_new_sub_dist: DistParam
    membership_ceiling_dist: DistParam

    # Quarterly Seasonal Tournament parameters
    tournament_enabled: bool
    tournament_start_month_index: int
    tournament_calendar_months: tuple[int, ...]
    tournament_follower_boost_mult: float
    tournament_participants_6_courts_dist: DistParam
    tournament_participants_20_courts_dist: DistParam
    tournament_fnb_per_participant_kvnd_dist: DistParam
    tournament_paddle_per_participant_kvnd_dist: DistParam
    tournament_coaching_per_participant_kvnd_dist: DistParam


def load_marketing_config(config_path: str | Path) -> MarketingAppConfig:
    """Load central TOML configurations for the Marketing Funnel & Segment ML Optimizer."""
    p = Path(config_path)
    config_dir = p if p.is_dir() else p.parent

    with (config_dir / "pricing.toml").open("rb") as f:
        pricing_raw = tomllib.load(f)
    with (config_dir / "customers.toml").open("rb") as f:
        cust_raw = tomllib.load(f)
    with (config_dir / "opex.toml").open("rb") as f:
        opex_raw = tomllib.load(f)
    with (config_dir / "cogs.toml").open("rb") as f:
        cogs_raw = tomllib.load(f)
    with (config_dir / "capex.toml").open("rb") as f:
        capex_raw = tomllib.load(f)
    with (config_dir / "externals.toml").open("rb") as f:
        ext_raw = tomllib.load(f)
    with (config_dir / "randomness.toml").open("rb") as f:
        rand_raw = tomllib.load(f)

    segment_keys = ("competitive", "social", "trend_explorers", "families")
    segments_list: list[SegmentMarketingConfig] = []
    for k in segment_keys:
        sd = cust_raw["segments"][k]
        segments_list.append(
            SegmentMarketingConfig(
                key=k,
                label=str(sd["label"]),
                main_need=str(sd["main_need"]),
                share_dist=_parse_dist(sd["share"]),
                visits_per_month_dist=_parse_dist(sd["visits_per_month"]),
                monthly_visits_dist=_parse_dist(sd["monthly_visits_dist"]),
                baseline_marketing_budget_mvnd_dist=_parse_dist(sd["baseline_marketing_budget_mvnd_dist"]),
                marketing_vnd_per_visit_dist=_parse_dist(sd["marketing_vnd_per_visit_dist"]),
                marketing_elasticity_dist=_parse_dist(sd["marketing_elasticity_exponent"]),
                spend_per_visit_kvnd_dist=_parse_dist(sd["spend_per_visit_kvnd"]),
                court_hire_spend_per_visit_kvnd_dist=_parse_dist(sd["court_hire_spend_per_visit_kvnd"]),
                paddle_balls_spend_per_visit_kvnd_dist=_parse_dist(sd["paddle_balls_spend_per_visit_kvnd"]),
                coaching_spend_per_visit_kvnd_dist=_parse_dist(sd["coaching_spend_per_visit_kvnd"]),
                fnb_spend_per_visit_kvnd_dist=_parse_dist(sd["fnb_spend_per_visit_kvnd"]),
                subscription_propensity_dist=_parse_dist(sd["subscription_propensity_weight"]),
            )
        )

    sm_raw = cust_raw["social_media"]
    social_funnel = SocialFunnelConfig(
        initial_followers_dist=_parse_dist(sm_raw["initial_followers"]),
        follower_to_customer_conversion_dist=_parse_dist(sm_raw["follower_to_customer_conversion_rate"]),
        monthly_new_follow_rate_dist=_parse_dist(sm_raw["monthly_new_follow_rate"]),
        monthly_unfollow_rate_dist=_parse_dist(sm_raw["monthly_unfollow_rate"]),
    )

    mc_raw = rand_raw["monte_carlo"]
    comp_raw = pricing_raw["company"]
    vs_raw = capex_raw["venue_scaling"]
    mem_raw = cust_raw["membership_program"]
    qt_raw = cust_raw["quarterly_tournaments"]
    pt_raw = ext_raw["pickleball_trend"]

    return MarketingAppConfig(
        random_seed=int(mc_raw["random_seed"]) + 101,
        n_simulations=int(mc_raw["n_simulations"]),
        train_fraction=float(mc_raw["train_fraction"]),
        projection_months=int(comp_raw["projection_months"]),
        model_start_month=str(comp_raw["model_start_month"]),
        unique_customers_dist=_parse_dist(cust_raw["unique_customers_monthly"]),
        player_visits_anchor_dist=_parse_dist(cust_raw["player_visits_per_month_anchor"]),
        social_funnel=social_funnel,
        segments=tuple(segments_list),
        baseline_total_marketing_mvnd_dist=_parse_dist(opex_raw["sga"]["marketing_monthly_mvnd"]),
        coach_fees_share_dist=_parse_dist(cogs_raw["coach_fees_share_of_coaching_rev"]),
        fnb_supplies_share_dist=_parse_dist(cogs_raw["fnb_supplies_share_of_fnb_rev"]),
        payment_fee_pct_dist=_parse_dist(cogs_raw["payment_fee_pct_of_revenue"]),
        trend_monthly_change_dist=_parse_dist(pt_raw["monthly_interest_change_rate"]),
        trend_floor=float(pt_raw["min_cumulative_trend_floor"]),
        trend_ceiling=float(pt_raw["max_cumulative_trend_ceiling"]),
        venue2_open_month_index=int(vs_raw["open_month_index"]),
        venue2_courts_added=int(vs_raw["new_venue_courts"]),
        venue2_fill_up_months_dist=_parse_dist(vs_raw["fill_up_duration_months_dist"]),
        membership_launch_month_index=int(mem_raw["launch_month_index"]),
        membership_fee_mvnd=float(mem_raw["monthly_fee_mvnd"]),
        membership_churn_dist=_parse_dist(mem_raw["monthly_churn_rate"]),
        membership_new_sub_dist=_parse_dist(mem_raw["monthly_new_subscriber_rate"]),
        membership_ceiling_dist=_parse_dist(mem_raw["addressable_member_ceiling"]),
        tournament_enabled=bool(qt_raw["enabled"]),
        tournament_start_month_index=int(qt_raw["start_month_index"]),
        tournament_calendar_months=tuple(int(x) for x in qt_raw["quarter_end_calendar_months"]),
        tournament_follower_boost_mult=float(qt_raw["tournament_month_follower_boost_mult"]),
        tournament_participants_6_courts_dist=_parse_dist(qt_raw["participants_6_courts"]),
        tournament_participants_20_courts_dist=_parse_dist(qt_raw["participants_20_courts"]),
        tournament_fnb_per_participant_kvnd_dist=_parse_dist(qt_raw["ancillary_fnb_per_participant_kvnd"]),
        tournament_paddle_per_participant_kvnd_dist=_parse_dist(qt_raw["ancillary_paddle_balls_per_participant_kvnd"]),
        tournament_coaching_per_participant_kvnd_dist=_parse_dist(qt_raw["ancillary_coaching_clinic_per_participant_kvnd"]),
    )
