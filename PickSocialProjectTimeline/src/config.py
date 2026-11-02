"""Typed configuration loader for PickSocialProjectTimeline (reading from central CaseComp/configs/opex.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DistParamConfig:
    dist: str
    mean: float
    std: float
    min: float
    max: float
    unit: str
    source: str


@dataclass(frozen=True)
class ProjectMetaConfig:
    name: str
    company: str
    start_date: str
    core_build_months: int
    core_build_weeks: float
    total_horizon_weeks: float
    team_size_fte: float
    sprint_length_weeks: float


@dataclass(frozen=True)
class CostRatesConfig:
    tech_lead_weekly_mvnd: float
    fullstack_dev_weekly_mvnd: float
    qa_devops_weekly_mvnd: float
    ui_ux_fe_weekly_mvnd: float
    employer_oncost_pct: float
    stage3_milestone_holdback_pct: float
    upfront_lump_sum_benchmark_pct: float
    baseline_monthly_ocf_mvnd: float
    adaptive_monthly_ocf_mvnd: float
    task_buffer_utilization_dist: DistParamConfig


@dataclass(frozen=True)
class PlatformMonthlyOpExConfig:
    activation_month_index: int
    post_hypercare_month_index: int
    cloud_hosting_db_redis: DistParamConfig
    sms_otp_notifications: DistParamConfig
    observability_cicd_security_saas: DistParamConfig
    ongoing_maintenance_patching_retainer: DistParamConfig
    school_university_community_ops: DistParamConfig

    @property
    def launch_months_5_6_mean_mvnd(self) -> float:
        return (
            self.cloud_hosting_db_redis.mean
            + self.sms_otp_notifications.mean
            + self.observability_cicd_security_saas.mean
            + self.school_university_community_ops.mean
        )

    @property
    def steady_state_month7_plus_mean_mvnd(self) -> float:
        return self.launch_months_5_6_mean_mvnd + self.ongoing_maintenance_patching_retainer.mean


@dataclass(frozen=True)
class StageConfig:
    id: str
    name: str
    short_name: str
    color: str
    light_color: str
    target_weeks: float
    effort_share_pct: float
    summary: str


@dataclass(frozen=True)
class TaskConfig:
    id: str
    stage_id: str
    name: str
    core_feature: str
    track: str
    start_week: float
    duration_weeks: float
    dependencies: tuple[str, ...]
    is_critical_path: bool
    is_simultaneous_qa: bool
    owner: str
    citation_keys: tuple[str, ...]
    estimation_rationale: str
    role_allocation: str
    person_weeks: float
    blended_weekly_rate_mvnd: float
    direct_infra_cost_mvnd: float
    contingency_buffer_pct: float
    cost_dist_mean_mvnd: float
    cost_dist_std_mvnd: float
    cost_dist_min_mvnd: float
    cost_dist_max_mvnd: float
    cost_citation_keys: tuple[str, ...]
    cost_rationale: str


@dataclass(frozen=True)
class MilestoneConfig:
    id: str
    name: str
    week: float
    stage_id: str
    description: str


@dataclass(frozen=True)
class CitationConfig:
    key: str
    title: str
    authors: str
    publication: str
    url: str
    benchmark_used: str


@dataclass(frozen=True)
class TimelineConfig:
    project: ProjectMetaConfig
    cost_rates: CostRatesConfig
    platform_monthly_opex: PlatformMonthlyOpExConfig
    stages: dict[str, StageConfig]
    tasks: tuple[TaskConfig, ...]
    milestones: tuple[MilestoneConfig, ...]
    citations: dict[str, CitationConfig]


def _load_dist(d: dict) -> DistParamConfig:
    return DistParamConfig(
        dist=str(d.get("dist", "truncated_normal")),
        mean=float(d["mean"]),
        std=float(d["std"]),
        min=float(d["min"]),
        max=float(d["max"]),
        unit=str(d.get("unit", "M VND")),
        source=str(d.get("source", "")),
    )


def load_timeline_config(configs_dir: Path) -> TimelineConfig:
    """Load and validate timeline, task cost distributions, and platform OpEx from central CaseComp/configs/opex.toml."""
    configs_dir = configs_dir.resolve()
    opex_path = configs_dir / "opex.toml" if configs_dir.is_dir() else configs_dir

    with opex_path.open("rb") as f:
        raw = tomllib.load(f)

    proj_raw = raw["platform_project"]
    project = ProjectMetaConfig(
        name=str(proj_raw["name"]),
        company=str(proj_raw["company"]),
        start_date=str(proj_raw["start_date"]),
        core_build_months=int(proj_raw["core_build_months"]),
        core_build_weeks=float(proj_raw["core_build_weeks"]),
        total_horizon_weeks=float(proj_raw["total_horizon_weeks"]),
        team_size_fte=float(proj_raw["team_size_fte"]),
        sprint_length_weeks=float(proj_raw["sprint_length_weeks"]),
    )

    cr_raw = raw["platform_cost_rates"]
    cost_rates = CostRatesConfig(
        tech_lead_weekly_mvnd=float(cr_raw["tech_lead_weekly_mvnd"]),
        fullstack_dev_weekly_mvnd=float(cr_raw["fullstack_dev_weekly_mvnd"]),
        qa_devops_weekly_mvnd=float(cr_raw["qa_devops_weekly_mvnd"]),
        ui_ux_fe_weekly_mvnd=float(cr_raw["ui_ux_fe_weekly_mvnd"]),
        employer_oncost_pct=float(cr_raw["employer_oncost_pct"]),
        stage3_milestone_holdback_pct=float(cr_raw["stage3_milestone_holdback_pct"]),
        upfront_lump_sum_benchmark_pct=float(cr_raw["upfront_lump_sum_benchmark_pct"]),
        baseline_monthly_ocf_mvnd=float(cr_raw["baseline_monthly_ocf_mvnd"]),
        adaptive_monthly_ocf_mvnd=float(cr_raw["adaptive_monthly_ocf_mvnd"]),
        task_buffer_utilization_dist=_load_dist(cr_raw["task_buffer_utilization_dist"]),
    )

    pm_meta = raw.get("platform_Monthly_opex", raw.get("platform_monthly_opex", {}))
    pm_raw = raw["platform_monthly_opex"]
    platform_monthly_opex = PlatformMonthlyOpExConfig(
        activation_month_index=int(pm_meta.get("activation_month_index", 5)),
        post_hypercare_month_index=int(pm_meta.get("post_hypercare_month_index", 7)),
        cloud_hosting_db_redis=_load_dist(pm_raw["cloud_hosting_db_redis_mvnd"]),
        sms_otp_notifications=_load_dist(pm_raw["sms_otp_notifications_mvnd"]),
        observability_cicd_security_saas=_load_dist(pm_raw["observability_cicd_security_saas_mvnd"]),
        ongoing_maintenance_patching_retainer=_load_dist(pm_raw["ongoing_maintenance_patching_retainer_mvnd"]),
        school_university_community_ops=_load_dist(pm_raw["school_university_community_ops_mvnd"]),
    )

    stages: dict[str, StageConfig] = {}
    for s_key, s_val in raw["stages"].items():
        stages[s_key] = StageConfig(
            id=str(s_val["id"]),
            name=str(s_val["name"]),
            short_name=str(s_val["short_name"]),
            color=str(s_val["color"]),
            light_color=str(s_val["light_color"]),
            target_weeks=float(s_val["target_weeks"]),
            effort_share_pct=float(s_val["effort_share_pct"]),
            summary=str(s_val["summary"]),
        )

    tasks: list[TaskConfig] = []
    for t_val in raw["tasks"]:
        tasks.append(
            TaskConfig(
                id=str(t_val["id"]),
                stage_id=str(t_val["stage_id"]),
                name=str(t_val["name"]),
                core_feature=str(t_val["core_feature"]),
                track=str(t_val["track"]),
                start_week=float(t_val["start_week"]),
                duration_weeks=float(t_val["duration_weeks"]),
                dependencies=tuple(str(d) for d in t_val.get("dependencies", [])),
                is_critical_path=bool(t_val["is_critical_path"]),
                is_simultaneous_qa=bool(t_val["is_simultaneous_qa"]),
                owner=str(t_val["owner"]),
                citation_keys=tuple(str(c) for c in t_val.get("citation_keys", [])),
                estimation_rationale=str(t_val["estimation_rationale"]),
                role_allocation=str(t_val["role_allocation"]),
                person_weeks=float(t_val["person_weeks"]),
                blended_weekly_rate_mvnd=float(t_val["blended_weekly_rate_mvnd"]),
                direct_infra_cost_mvnd=float(t_val["direct_infra_cost_mvnd"]),
                contingency_buffer_pct=float(t_val["contingency_buffer_pct"]),
                cost_dist_mean_mvnd=float(t_val["cost_dist_mean_mvnd"]),
                cost_dist_std_mvnd=float(t_val["cost_dist_std_mvnd"]),
                cost_dist_min_mvnd=float(t_val["cost_dist_min_mvnd"]),
                cost_dist_max_mvnd=float(t_val["cost_dist_max_mvnd"]),
                cost_citation_keys=tuple(str(c) for c in t_val.get("cost_citation_keys", [])),
                cost_rationale=str(t_val["cost_rationale"]),
            )
        )

    milestones: list[MilestoneConfig] = []
    for m_val in raw["milestones"]:
        milestones.append(
            MilestoneConfig(
                id=str(m_val["id"]),
                name=str(m_val["name"]),
                week=float(m_val["week"]),
                stage_id=str(m_val["stage_id"]),
                description=str(m_val["description"]),
            )
        )

    citations: dict[str, CitationConfig] = {}
    for c_val in raw["citations"]:
        key = str(c_val["key"])
        citations[key] = CitationConfig(
            key=key,
            title=str(c_val["title"]),
            authors=str(c_val["authors"]),
            publication=str(c_val["publication"]),
            url=str(c_val["url"]),
            benchmark_used=str(c_val["benchmark_used"]),
        )

    return TimelineConfig(
        project=project,
        cost_rates=cost_rates,
        platform_monthly_opex=platform_monthly_opex,
        stages=stages,
        tasks=tuple(tasks),
        milestones=tuple(milestones),
        citations=citations,
    )
