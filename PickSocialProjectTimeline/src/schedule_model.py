"""Work Breakdown Structure (WBS), task-level cost & contingency buffer estimation, recurring platform OpEx, and leveled cost allocation engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .config import TimelineConfig


@dataclass(frozen=True)
class ScheduledTask:
    id: str
    stage_id: str
    stage_name: str
    stage_short_name: str
    stage_color: str
    stage_light_color: str
    name: str
    core_feature: str
    track: str
    start_week: float
    end_week: float
    duration_weeks: float
    duration_days: int
    start_date: date
    end_date: date
    dependencies: tuple[str, ...]
    is_critical_path: bool
    is_simultaneous_qa: bool
    owner: str
    citation_keys: tuple[str, ...]
    estimation_rationale: str
    role_allocation: str
    person_weeks: float
    blended_weekly_rate_mvnd: float
    base_labor_cost_mvnd: float
    direct_infra_cost_mvnd: float
    base_total_cost_mvnd: float
    contingency_buffer_pct: float
    contingency_buffer_mvnd: float
    buffered_total_cost_mvnd: float
    cost_dist_mean_mvnd: float
    cost_dist_std_mvnd: float
    sprint_drawdown_mvnd: float
    milestone_holdback_mvnd: float
    cost_citation_keys: tuple[str, ...]
    cost_rationale: str


@dataclass(frozen=True)
class ScheduledMilestone:
    id: str
    name: str
    week: float
    target_date: date
    stage_id: str
    stage_color: str
    description: str


@dataclass(frozen=True)
class StageSummary:
    stage_id: str
    name: str
    short_name: str
    color: str
    light_color: str
    start_week: float
    end_week: float
    span_weeks: float
    sum_task_weeks: float
    sum_person_weeks: float
    effort_share_pct: float
    start_date: date
    end_date: date
    task_count: int
    summary: str
    base_labor_cost_mvnd: float
    direct_infra_cost_mvnd: float
    base_total_cost_mvnd: float
    contingency_buffer_mvnd: float
    buffered_total_cost_mvnd: float
    effective_buffer_pct: float


@dataclass(frozen=True)
class TimelineResults:
    project_start_date: date
    core_build_end_date: date
    full_horizon_end_date: date
    tasks: tuple[ScheduledTask, ...]
    milestones: tuple[ScheduledMilestone, ...]
    stage_summaries: tuple[StageSummary, ...]
    wbs_df: pd.DataFrame
    weekly_concurrency_df: pd.DataFrame
    weekly_cost_allocation_df: pd.DataFrame
    monthly_cost_allocation_df: pd.DataFrame
    platform_opex_summary_df: pd.DataFrame
    core_build_base_cost_mvnd: float
    core_build_buffer_mvnd: float
    core_build_buffered_cost_mvnd: float
    total_base_labor_cost_mvnd: float
    total_direct_infra_cost_mvnd: float
    total_base_cost_mvnd: float
    total_buffer_mvnd: float
    total_buffered_cost_mvnd: float
    weighted_buffer_pct: float
    launch_monthly_platform_opex_mvnd: float
    steady_state_monthly_platform_opex_mvnd: float


def _overlap_weeks(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return length of overlap between interval [a_start, a_end] and [b_start, b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_project_schedule(cfg: TimelineConfig) -> TimelineResults:
    """Compute calendar dates (starting Jan 2027), stage spans, task costs, recurring platform OpEx, and leveled cost allocation."""
    base_date = date.fromisoformat(cfg.project.start_date)
    holdback_frac = cfg.cost_rates.stage3_milestone_holdback_pct / 100.0

    scheduled_tasks: list[ScheduledTask] = []
    for t in cfg.tasks:
        st = cfg.stages[t.stage_id]
        end_wk = t.start_week + t.duration_weeks
        s_date = base_date + timedelta(days=int(round(t.start_week * 7)))
        e_date = base_date + timedelta(days=int(round(end_wk * 7)) - 1)
        dur_days = int(round(t.duration_weeks * 7))

        labor_mvnd = round(t.person_weeks * t.blended_weekly_rate_mvnd, 2)
        infra_mvnd = round(t.direct_infra_cost_mvnd, 2)
        base_tot_mvnd = round(labor_mvnd + infra_mvnd, 2)
        buf_mvnd = round(base_tot_mvnd * (t.contingency_buffer_pct / 100.0), 2)
        buffered_tot_mvnd = round(base_tot_mvnd + buf_mvnd, 2)

        if t.stage_id == "S3":
            hb_mvnd = round(base_tot_mvnd * holdback_frac, 2)
            draw_mvnd = round(base_tot_mvnd - hb_mvnd, 2)
        else:
            hb_mvnd = 0.0
            draw_mvnd = base_tot_mvnd

        scheduled_tasks.append(
            ScheduledTask(
                id=t.id,
                stage_id=t.stage_id,
                stage_name=st.name,
                stage_short_name=st.short_name,
                stage_color=st.color,
                stage_light_color=st.light_color,
                name=t.name,
                core_feature=t.core_feature,
                track=t.track,
                start_week=t.start_week,
                end_week=end_wk,
                duration_weeks=t.duration_weeks,
                duration_days=dur_days,
                start_date=s_date,
                end_date=e_date,
                dependencies=t.dependencies,
                is_critical_path=t.is_critical_path,
                is_simultaneous_qa=t.is_simultaneous_qa,
                owner=t.owner,
                citation_keys=t.citation_keys,
                estimation_rationale=t.estimation_rationale,
                role_allocation=t.role_allocation,
                person_weeks=t.person_weeks,
                blended_weekly_rate_mvnd=t.blended_weekly_rate_mvnd,
                base_labor_cost_mvnd=labor_mvnd,
                direct_infra_cost_mvnd=infra_mvnd,
                base_total_cost_mvnd=base_tot_mvnd,
                contingency_buffer_pct=t.contingency_buffer_pct,
                contingency_buffer_mvnd=buf_mvnd,
                buffered_total_cost_mvnd=buffered_tot_mvnd,
                cost_dist_mean_mvnd=t.cost_dist_mean_mvnd,
                cost_dist_std_mvnd=t.cost_dist_std_mvnd,
                sprint_drawdown_mvnd=draw_mvnd,
                milestone_holdback_mvnd=hb_mvnd,
                cost_citation_keys=t.cost_citation_keys,
                cost_rationale=t.cost_rationale,
            )
        )

    scheduled_milestones: list[ScheduledMilestone] = []
    for m in cfg.milestones:
        st = cfg.stages[m.stage_id]
        m_date = base_date + timedelta(days=int(round(m.week * 7)) - 1)
        scheduled_milestones.append(
            ScheduledMilestone(
                id=m.id,
                name=m.name,
                week=m.week,
                target_date=m_date,
                stage_id=m.stage_id,
                stage_color=st.color,
                description=m.description,
            )
        )

    stage_summaries: list[StageSummary] = []
    for s_id, st in cfg.stages.items():
        s_tasks = [t for t in scheduled_tasks if t.stage_id == s_id]
        min_w = min(t.start_week for t in s_tasks)
        max_w = max(t.end_week for t in s_tasks)
        span_w = max_w - min_w
        sum_w = sum(t.duration_weeks for t in s_tasks)
        sum_pw = round(sum(t.person_weeks for t in s_tasks), 3)
        s_date = base_date + timedelta(days=int(round(min_w * 7)))
        e_date = base_date + timedelta(days=int(round(max_w * 7)) - 1)

        s_labor = round(sum(t.base_labor_cost_mvnd for t in s_tasks), 2)
        s_infra = round(sum(t.direct_infra_cost_mvnd for t in s_tasks), 2)
        s_base = round(sum(t.base_total_cost_mvnd for t in s_tasks), 2)
        s_buf = round(sum(t.contingency_buffer_mvnd for t in s_tasks), 2)
        s_buffered = round(sum(t.buffered_total_cost_mvnd for t in s_tasks), 2)
        s_buf_pct = round((s_buf / s_base) * 100.0, 2) if s_base > 0 else 0.0

        stage_summaries.append(
            StageSummary(
                stage_id=s_id,
                name=st.name,
                short_name=st.short_name,
                color=st.color,
                light_color=st.light_color,
                start_week=min_w,
                end_week=max_w,
                span_weeks=span_w,
                sum_task_weeks=sum_w,
                sum_person_weeks=sum_pw,
                effort_share_pct=st.effort_share_pct,
                start_date=s_date,
                end_date=e_date,
                task_count=len(s_tasks),
                summary=st.summary,
                base_labor_cost_mvnd=s_labor,
                direct_infra_cost_mvnd=s_infra,
                base_total_cost_mvnd=s_base,
                contingency_buffer_mvnd=s_buf,
                buffered_total_cost_mvnd=s_buffered,
                effective_buffer_pct=s_buf_pct,
            )
        )

    # Project-wide cost totals
    total_base_labor = round(sum(t.base_labor_cost_mvnd for t in scheduled_tasks), 2)
    total_direct_infra = round(sum(t.direct_infra_cost_mvnd for t in scheduled_tasks), 2)
    total_base_cost = round(sum(t.base_total_cost_mvnd for t in scheduled_tasks), 2)
    total_buffer = round(sum(t.contingency_buffer_mvnd for t in scheduled_tasks), 2)
    total_buffered_cost = round(sum(t.buffered_total_cost_mvnd for t in scheduled_tasks), 2)
    weighted_buf_pct = round((total_buffer / total_base_cost) * 100.0, 2)

    core_tasks = [t for t in scheduled_tasks if t.stage_id in ("S1", "S2", "S3", "S4")]
    core_base_cost = round(sum(t.base_total_cost_mvnd for t in core_tasks), 2)
    core_buffer = round(sum(t.contingency_buffer_mvnd for t in core_tasks), 2)
    core_buffered_cost = round(sum(t.buffered_total_cost_mvnd for t in core_tasks), 2)

    # Build WBS DataFrame (including full cost, buffer & distribution parameters)
    wbs_rows = []
    for t in scheduled_tasks:
        wbs_rows.append(
            {
                "task_id": t.id,
                "stage_id": t.stage_id,
                "stage_name": t.stage_short_name,
                "task_name": t.name,
                "core_feature_deliverable": t.core_feature,
                "engineering_track": t.track,
                "start_week": f"W{t.start_week + 1:.1f}",
                "end_week": f"W{t.end_week:.1f}",
                "start_week_num": t.start_week,
                "end_week_num": t.end_week,
                "duration_weeks": t.duration_weeks,
                "duration_days": t.duration_days,
                "start_date": t.start_date.isoformat(),
                "end_date": t.end_date.isoformat(),
                "dependencies": ", ".join(t.dependencies) if t.dependencies else "None (Kickoff)",
                "critical_path": "Yes" if t.is_critical_path else "Parallel Track",
                "simultaneous_qa_track": "Yes (Continuous QA)" if t.is_simultaneous_qa else "No",
                "owner": t.owner,
                "role_allocation": t.role_allocation,
                "person_weeks": t.person_weeks,
                "blended_weekly_rate_mvnd": t.blended_weekly_rate_mvnd,
                "base_labor_cost_mvnd": t.base_labor_cost_mvnd,
                "direct_infra_cost_mvnd": t.direct_infra_cost_mvnd,
                "base_total_cost_mvnd": t.base_total_cost_mvnd,
                "contingency_buffer_pct": t.contingency_buffer_pct,
                "contingency_buffer_mvnd": t.contingency_buffer_mvnd,
                "buffered_total_cost_mvnd": t.buffered_total_cost_mvnd,
                "cost_dist_mean_mvnd": t.cost_dist_mean_mvnd,
                "cost_dist_std_mvnd": t.cost_dist_std_mvnd,
                "sprint_drawdown_mvnd": t.sprint_drawdown_mvnd,
                "milestone_holdback_mvnd": t.milestone_holdback_mvnd,
                "schedule_citations": ", ".join(t.citation_keys),
                "estimation_rationale": t.estimation_rationale,
                "cost_citations": ", ".join(t.cost_citation_keys),
                "cost_rationale": t.cost_rationale,
            }
        )
    wbs_df = pd.DataFrame(wbs_rows)

    # Recurring Monthly Platform Operating Expenses Summary Table
    pm = cfg.platform_monthly_opex
    opex_items = [
        (
            "POPEX_1",
            "Cloud Infrastructure, Managed PostgreSQL/Redis & Cloudflare CDN",
            "Month 5+ (May-27 onward)",
            pm.cloud_hosting_db_redis,
        ),
        (
            "POPEX_2",
            "SMS OTP Verification & Open-Play Push Notifications",
            "Month 5+ (May-27 onward)",
            pm.sms_otp_notifications,
        ),
        (
            "POPEX_3",
            "Observability (Sentry APM), Telemetry (PostHog), CI/CD & SSL/Domains",
            "Month 5+ (May-27 onward)",
            pm.observability_cicd_security_saas,
        ),
        (
            "POPEX_4",
            "School/University Partnerships & Beginner League Community Ops",
            "Month 5+ (May-27 onward)",
            pm.school_university_community_ops,
        ),
        (
            "POPEX_5",
            "Ongoing Engineering Maintenance, Security Patching & ML Retraining Retainer",
            "Month 7+ (Jul-27 onward, post-hypercare)",
            pm.ongoing_maintenance_patching_retainer,
        ),
    ]
    popex_rows = []
    for code, name, window, d in opex_items:
        popex_rows.append(
            {
                "item_code": code,
                "expense_line": name,
                "active_window": window,
                "dist_type": d.dist,
                "monthly_mean_mvnd": d.mean,
                "monthly_std_mvnd": d.std,
                "monthly_min_mvnd": d.min,
                "monthly_max_mvnd": d.max,
                "annual_steady_state_mvnd": round(d.mean * 12.0, 2),
                "source": d.source,
            }
        )
    platform_opex_summary_df = pd.DataFrame(popex_rows)

    # Compute week-by-week concurrency AND weekly cost allocation across the 24-week lifecycle
    n_weeks = int(round(cfg.project.total_horizon_weeks))
    total_s3_holdback = round(sum(t.milestone_holdback_mvnd for t in scheduled_tasks), 2)
    weekly_even_reserve = total_buffer / float(n_weeks)
    launch_popex_wk = pm.launch_months_5_6_mean_mvnd / 4.0

    month_names = [
        ("M1", "Jan-27", "Month 1: Jan 2027 (Planning & Infra)"),
        ("M2", "Feb-27", "Month 2: Feb 2027 (Booking & ML Core)"),
        ("M3", "Mar-27", "Month 3: Mar 2027 (Payments & Social)"),
        ("M4", "Apr-27", "Month 4: Apr 2027 (Integration & UAT)"),
        ("M5", "May-27", "Month 5: May 2027 (Legacy Migration)"),
        ("M6", "Jun-27", "Month 6: Jun 2027 (Hypercare & Continuous Ops)"),
    ]

    wk_records = []
    wk_cost_records = []

    for w_idx in range(n_weeks):
        w_start = float(w_idx)
        w_end = float(w_idx + 1)
        w_mid = w_idx + 0.5
        w_start_dt = base_date + timedelta(days=w_idx * 7)
        m_idx = min(w_idx // 4, 5)
        m_code, m_short, _ = month_names[m_idx]

        active = [t for t in scheduled_tasks if t.start_week <= w_mid <= t.end_week]
        dev_tracks = sum(1 for t in active if not t.is_simultaneous_qa and t.stage_id in ("S1", "S2", "S3"))
        qa_tracks = sum(1 for t in active if t.is_simultaneous_qa or t.stage_id == "S4")
        mig_ops_tracks = sum(1 for t in active if t.stage_id in ("S5", "S6") and not t.is_simultaneous_qa)
        wk_records.append(
            {
                "week_num": w_idx + 1,
                "week_label": f"W{w_idx + 1}",
                "month_code": m_code,
                "month_label": m_short,
                "week_start_date": w_start_dt.isoformat(),
                "active_total_tasks": len(active),
                "active_arch_infra_dev_tracks": dev_tracks,
                "active_qa_testing_tracks": qa_tracks,
                "active_migration_ops_tracks": mig_ops_tracks,
            }
        )

        # Cost accruals in week [w_start, w_end]
        sprint_active_base_wk = 0.0
        for t in scheduled_tasks:
            ov = _overlap_weeks(t.start_week, t.end_week, w_start, w_end)
            if ov > 0.0:
                frac = ov / t.duration_weeks
                sprint_active_base_wk += t.sprint_drawdown_mvnd * frac

        # Holdback release schedule across the 3 acceptance gates (M4 UAT, M5 Cutover, M6 Final Hypercare):
        if w_idx in (14, 15):
            hb_release_wk = 0.075 * total_s3_holdback
        elif w_idx in (18, 19):
            hb_release_wk = 0.225 * total_s3_holdback
        elif w_idx in (22, 23):
            hb_release_wk = 0.200 * total_s3_holdback
        else:
            hb_release_wk = 0.0

        # Recurring platform monthly OpEx starts in Month 5 (W17..W24, i.e. w_idx >= 16)
        rec_opex_wk = launch_popex_wk if w_idx >= 16 else 0.0

        committed_dev_base_wk = sprint_active_base_wk + hb_release_wk
        buffered_dev_wk = committed_dev_base_wk + weekly_even_reserve

        wk_cost_records.append(
            {
                "week_num": w_idx + 1,
                "week_label": f"W{w_idx + 1}",
                "month_code": m_code,
                "month_label": m_short,
                "week_start_date": w_start_dt.isoformat(),
                "leveled_sprint_active_base_mvnd": round(sprint_active_base_wk, 2),
                "leveled_milestone_holdback_release_mvnd": round(hb_release_wk, 2),
                "leveled_committed_dev_base_mvnd": round(committed_dev_base_wk, 2),
                "leveled_contingency_reserve_mvnd": round(weekly_even_reserve, 2),
                "leveled_buffered_dev_total_mvnd": round(buffered_dev_wk, 2),
                "recurring_platform_opex_mvnd": round(rec_opex_wk, 2),
                "total_platform_outflow_buffered_mvnd": round(buffered_dev_wk + rec_opex_wk, 2),
            }
        )

    weekly_concurrency_df = pd.DataFrame(wk_records)
    weekly_cost_allocation_df = pd.DataFrame(wk_cost_records)

    # Build Monthly Cost Allocation DataFrame (M1: Jan-27 .. M6: Jun-27)
    monthly_rows = []
    cum_committed_base = 0.0
    cum_buffered_dev = 0.0

    for m_idx, (m_code, m_short, m_full) in enumerate(month_names):
        sub = weekly_cost_allocation_df[weekly_cost_allocation_df["month_code"] == m_code]
        w_first = int(sub["week_num"].min())
        w_last = int(sub["week_num"].max())

        sprint_base_m = round(float(sub["leveled_sprint_active_base_mvnd"].sum()), 2)
        hb_rel_m = round(float(sub["leveled_milestone_holdback_release_mvnd"].sum()), 2)
        committed_base_m = round(sprint_base_m + hb_rel_m, 2)
        reserve_m = round(float(sub["leveled_contingency_reserve_mvnd"].sum()), 2)
        buf_dev_m = round(committed_base_m + reserve_m, 2)
        rec_opex_m = round(float(sub["recurring_platform_opex_mvnd"].sum()), 2)

        cum_committed_base = round(cum_committed_base + committed_base_m, 2)
        cum_buffered_dev = round(cum_buffered_dev + buf_dev_m, 2)

        if m_idx == 0:
            payment_mech = "Bi-weekly Sprint 1-2 actuals (S1/S2); 0% upfront lump-sum"
        elif m_idx in (1, 2):
            payment_mech = "78% Stage 3 bi-weekly sprint drawdowns; 22% held back for M4/M5/M6 gates"
        elif m_idx == 3:
            payment_mech = "Stage 4 integration/UAT actuals + 15% Stage 3 holdback released at M4 UAT sign-off"
        elif m_idx == 4:
            payment_mech = "Stage 5 migration/cutover + 45% holdback at M5 + live Platform OpEx (16.5M/mo)"
        else:
            payment_mech = "Stage 6 hypercare/ML calibration + 40% holdback at M6 + live Platform OpEx (16.5M/mo)"

        monthly_rows.append(
            {
                "month_code": m_code,
                "month_label": m_short,
                "month_title": m_full,
                "weeks_window": f"W{w_first}-W{w_last}",
                "leveled_sprint_active_base_mvnd": sprint_base_m,
                "leveled_milestone_holdback_release_mvnd": hb_rel_m,
                "leveled_committed_base_mvnd": committed_base_m,
                "leveled_contingency_reserve_mvnd": reserve_m,
                "leveled_max_buffered_total_mvnd": buf_dev_m,
                "recurring_platform_opex_mvnd": rec_opex_m,
                "total_platform_cost_with_opex_base_mvnd": round(committed_base_m + rec_opex_m, 2),
                "total_platform_cost_with_opex_buffered_mvnd": round(buf_dev_m + rec_opex_m, 2),
                "cumulative_committed_base_mvnd": cum_committed_base,
                "cumulative_max_buffered_mvnd": cum_buffered_dev,
                "disbursement_mechanism": payment_mech,
            }
        )

    monthly_cost_allocation_df = pd.DataFrame(monthly_rows)

    core_end = base_date + timedelta(days=int(round(cfg.project.core_build_weeks * 7)) - 1)
    full_end = base_date + timedelta(days=int(round(cfg.project.total_horizon_weeks * 7)) - 1)

    return TimelineResults(
        project_start_date=base_date,
        core_build_end_date=core_end,
        full_horizon_end_date=full_end,
        tasks=tuple(scheduled_tasks),
        milestones=tuple(scheduled_milestones),
        stage_summaries=tuple(stage_summaries),
        wbs_df=wbs_df,
        weekly_concurrency_df=weekly_concurrency_df,
        weekly_cost_allocation_df=weekly_cost_allocation_df,
        monthly_cost_allocation_df=monthly_cost_allocation_df,
        platform_opex_summary_df=platform_opex_summary_df,
        core_build_base_cost_mvnd=core_base_cost,
        core_build_buffer_mvnd=core_buffer,
        core_build_buffered_cost_mvnd=core_buffered_cost,
        total_base_labor_cost_mvnd=total_base_labor,
        total_direct_infra_cost_mvnd=total_direct_infra,
        total_base_cost_mvnd=total_base_cost,
        total_buffer_mvnd=total_buffer,
        total_buffered_cost_mvnd=total_buffered_cost,
        weighted_buffer_pct=weighted_buf_pct,
        launch_monthly_platform_opex_mvnd=round(pm.launch_months_5_6_mean_mvnd, 2),
        steady_state_monthly_platform_opex_mvnd=round(pm.steady_state_month7_plus_mean_mvnd, 2),
    )
