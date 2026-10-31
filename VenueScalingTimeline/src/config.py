"""Typed configuration loader for VenueScalingTimeline (reading from central CaseComp/configs/capex.toml, opex.toml, pricing.toml, randomness.toml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
import numpy as np


@dataclass(frozen=True)
class DistParam:
    dist: str
    mean: float
    std: float
    min: float
    max: float
    unit: str = ""
    source: str = ""

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...] = 1) -> np.ndarray:
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
class VenueScalingMetaConfig:
    enabled: bool
    project_name: str
    company: str
    existing_courts: int
    new_venue_courts: int
    total_expanded_courts: int
    capacity_ratio_new_vs_existing: float
    capacity_ratio_total_vs_existing: float
    build_start_date: str
    build_start_month_index: int
    build_start_month_label: str
    build_duration_months: int
    build_duration_weeks: float
    build_end_month_index: int
    build_end_month_label: str
    open_month_index: int
    open_month_label: str
    total_timeline_horizon_weeks: float
    total_fitout_capex_mvnd: float
    even_monthly_fitout_capex_mvnd: float
    even_weekly_fitout_capex_mvnd: float
    investor_total_new_money_mvnd: float
    investor_reserve_mvnd: float
    investor_signing_tranche_month_index: int
    investor_signing_tranche_mvnd: float
    investor_equity_injection_month_index: int
    investor_expansion_tranche_mvnd: float
    new_venue_lease_monthly_mvnd: float
    lease_start_month_index: int
    case_static_ref_new_club_rev_mvnd: float
    fill_up_duration_months_dist: DistParam
    initial_open_utilization_fraction_dist: DistParam
    network_customer_catchment_expansion_dist: DistParam


@dataclass(frozen=True)
class Venue2OpExConfig:
    lease_start_month_index: int
    operations_open_month_index: int
    venue_lease_dist: DistParam
    electricity_lighting_water_dist: DistParam
    maintenance_and_insurance_dist: DistParam
    staff_wages_dist: DistParam


@dataclass(frozen=True)
class VenueStageConfig:
    id: str
    name: str
    short_name: str
    color: str
    light_color: str
    target_weeks: float
    budget_share_pct: float
    stage_capex_mvnd: float
    summary: str


@dataclass(frozen=True)
class VenueTaskConfig:
    id: str
    stage_id: str
    name: str
    core_deliverable: str
    track: str
    start_week: float
    duration_weeks: float
    dependencies: tuple[str, ...]
    is_critical_path: bool
    is_parallel_track: bool
    owner: str
    capex_mvnd: float
    citation_keys: tuple[str, ...]
    estimation_rationale: str


@dataclass(frozen=True)
class VenueMilestoneConfig:
    id: str
    name: str
    week: float
    stage_id: str
    description: str


@dataclass(frozen=True)
class VenueCitationConfig:
    key: str
    title: str
    authors: str
    url: str
    relevance: str


@dataclass(frozen=True)
class VenueScalingConfig:
    configs_dir: Path
    meta: VenueScalingMetaConfig
    venue2_opex: Venue2OpExConfig
    stages: dict[str, VenueStageConfig]
    tasks: tuple[VenueTaskConfig, ...]
    milestones: tuple[VenueMilestoneConfig, ...]
    citations: dict[str, VenueCitationConfig]
    random_seed: int
    n_simulations: int
    ci_low_percentile: float
    ci_high_percentile: float
    venue1_ancillary_rev_mvnd: float


def load_venue_scaling_config(config_path: Path | None = None) -> VenueScalingConfig:
    """Load VenueScalingTimeline configuration from central `CaseComp/configs/`."""
    if config_path is None:
        configs_dir = Path(__file__).resolve().parents[2] / "configs"
    else:
        resolved = config_path.resolve()
        configs_dir = resolved if resolved.is_dir() else resolved.parent

    with (configs_dir / "capex.toml").open("rb") as f:
        capex_raw = tomllib.load(f)
    with (configs_dir / "opex.toml").open("rb") as f:
        opex_raw = tomllib.load(f)
    with (configs_dir / "pricing.toml").open("rb") as f:
        pricing_raw = tomllib.load(f)
    with (configs_dir / "randomness.toml").open("rb") as f:
        rand_raw = tomllib.load(f)

    vs = capex_raw["venue_scaling"]
    meta = VenueScalingMetaConfig(
        enabled=bool(vs["enabled"]),
        project_name=str(vs["project_name"]),
        company=str(vs["company"]),
        existing_courts=int(vs["existing_courts"]),
        new_venue_courts=int(vs["new_venue_courts"]),
        total_expanded_courts=int(vs["total_expanded_courts"]),
        capacity_ratio_new_vs_existing=float(vs["capacity_ratio_new_vs_existing"]),
        capacity_ratio_total_vs_existing=float(vs["capacity_ratio_total_vs_existing"]),
        build_start_date=str(vs["build_start_date"]),
        build_start_month_index=int(vs["build_start_month_index"]),
        build_start_month_label=str(vs["build_start_month_label"]),
        build_duration_months=int(vs["build_duration_months"]),
        build_duration_weeks=float(vs["build_duration_weeks"]),
        build_end_month_index=int(vs["build_end_month_index"]),
        build_end_month_label=str(vs["build_end_month_label"]),
        open_month_index=int(vs["open_month_index"]),
        open_month_label=str(vs["open_month_label"]),
        total_timeline_horizon_weeks=float(vs["total_timeline_horizon_weeks"]),
        total_fitout_capex_mvnd=float(vs["total_fitout_capex_mvnd"]),
        even_monthly_fitout_capex_mvnd=float(vs["even_monthly_fitout_capex_mvnd"]),
        even_weekly_fitout_capex_mvnd=float(vs["even_weekly_fitout_capex_mvnd"]),
        investor_total_new_money_mvnd=float(vs["investor_total_new_money_mvnd"]),
        investor_reserve_mvnd=float(vs["investor_reserve_mvnd"]),
        investor_signing_tranche_month_index=int(vs.get("investor_signing_tranche_month_index", 1)),
        investor_signing_tranche_mvnd=float(vs.get("investor_signing_tranche_mvnd", 500.0)),
        investor_equity_injection_month_index=int(vs["investor_equity_injection_month_index"]),
        investor_expansion_tranche_mvnd=float(vs.get("investor_expansion_tranche_mvnd", 7500.0)),
        new_venue_lease_monthly_mvnd=float(vs["new_venue_lease_monthly_mvnd"]),
        lease_start_month_index=int(vs["lease_start_month_index"]),
        case_static_ref_new_club_rev_mvnd=float(vs["case_static_ref_new_club_rev_mvnd"]),
        fill_up_duration_months_dist=_parse_dist(vs["fill_up_duration_months_dist"]),
        initial_open_utilization_fraction_dist=_parse_dist(vs["initial_open_utilization_fraction_dist"]),
        network_customer_catchment_expansion_dist=_parse_dist(vs["network_customer_catchment_expansion_dist"]),
    )

    v2o = opex_raw["venue2_opex_and_sga"]
    venue2_opex = Venue2OpExConfig(
        lease_start_month_index=int(v2o["lease_start_month_index"]),
        operations_open_month_index=int(v2o["operations_open_month_index"]),
        venue_lease_dist=_parse_dist(v2o["venue_lease_monthly_mvnd"]),
        electricity_lighting_water_dist=_parse_dist(v2o["electricity_lighting_water_monthly_mvnd"]),
        maintenance_and_insurance_dist=_parse_dist(v2o["maintenance_and_insurance_monthly_mvnd"]),
        staff_wages_dist=_parse_dist(v2o["staff_wages_monthly_mvnd"]),
    )

    stages: dict[str, VenueStageConfig] = {}
    for k, v in capex_raw["venue_stages"].items():
        stages[k] = VenueStageConfig(
            id=str(v["id"]),
            name=str(v["name"]),
            short_name=str(v["short_name"]),
            color=str(v["color"]),
            light_color=str(v["light_color"]),
            target_weeks=float(v["target_weeks"]),
            budget_share_pct=float(v["budget_share_pct"]),
            stage_capex_mvnd=float(v["stage_capex_mvnd"]),
            summary=str(v["summary"]),
        )

    tasks: list[VenueTaskConfig] = []
    for t in capex_raw["venue_tasks"]:
        tasks.append(
            VenueTaskConfig(
                id=str(t["id"]),
                stage_id=str(t["stage_id"]),
                name=str(t["name"]),
                core_deliverable=str(t["core_deliverable"]),
                track=str(t["track"]),
                start_week=float(t["start_week"]),
                duration_weeks=float(t["duration_weeks"]),
                dependencies=tuple(str(d) for d in t.get("dependencies", [])),
                is_critical_path=bool(t["is_critical_path"]),
                is_parallel_track=bool(t["is_parallel_track"]),
                owner=str(t["owner"]),
                capex_mvnd=float(t["capex_mvnd"]),
                citation_keys=tuple(str(c) for c in t.get("citation_keys", [])),
                estimation_rationale=str(t["estimation_rationale"]),
            )
        )

    milestones: list[VenueMilestoneConfig] = []
    for m in capex_raw["venue_milestones"]:
        milestones.append(
            VenueMilestoneConfig(
                id=str(m["id"]),
                name=str(m["name"]),
                week=float(m["week"]),
                stage_id=str(m["stage_id"]),
                description=str(m["description"]),
            )
        )

    citations: dict[str, VenueCitationConfig] = {}
    for c in capex_raw["venue_citations"]:
        key = str(c["key"])
        citations[key] = VenueCitationConfig(
            key=key,
            title=str(c["title"]),
            authors=str(c["authors"]),
            url=str(c["url"]),
            relevance=str(c["relevance"]),
        )

    rev_lines = pricing_raw["baseline_revenue_lines"]
    v1_ancillary = (
        float(rev_lines["paddle_rental_and_balls_mvnd"])
        + float(rev_lines["coaching_mvnd"])
        + float(rev_lines["food_and_drinks_mvnd"])
    )

    mc = rand_raw["monte_carlo"]
    return VenueScalingConfig(
        configs_dir=configs_dir,
        meta=meta,
        venue2_opex=venue2_opex,
        stages=stages,
        tasks=tuple(tasks),
        milestones=tuple(milestones),
        citations=citations,
        random_seed=int(mc["random_seed"]),
        n_simulations=int(mc["n_simulations"]),
        ci_low_percentile=float(mc["ci_low_percentile"]),
        ci_high_percentile=float(mc["ci_high_percentile"]),
        venue1_ancillary_rev_mvnd=v1_ancillary,
    )
