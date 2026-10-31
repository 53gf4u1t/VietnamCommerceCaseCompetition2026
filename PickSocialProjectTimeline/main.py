"""Entry point for PickSocialProjectTimeline (invoked via root `uv run main`)."""

from __future__ import annotations

from pathlib import Path
import time

from .src.config import load_timeline_config
from .src.reporter import export_timeline_outputs
from .src.schedule_model import build_project_schedule
from .src.visualizer import generate_all_timeline_charts


def run_project_timeline(
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run the Path C Social & Booking Platform Timeline, Cost Estimation & Allocation pipeline."""
    t0 = time.perf_counter()
    base_dir = Path(__file__).resolve().parent
    cfg_path = config_path if config_path is not None else (base_dir.parent / "configs")
    out_dir = output_dir if output_dir is not None else (base_dir / "outputs")

    print("=" * 86)
    print("PICK! SOCIAL & BOOKING PLATFORM (PATH C) — TIMELINE, COST & CASH-FLOW ALLOCATION")
    print("=" * 86)

    print(f"[1/4] Loading timeline & cost configuration from {cfg_path.name}...")
    cfg = load_timeline_config(cfg_path)
    print(f"      Project Start: {cfg.project.start_date} | Core Build Target: {cfg.project.core_build_months} Months ({cfg.project.core_build_weeks:.0f} Weeks)")

    print("[2/4] Computing WBS schedule, task cost buffers, and cash-flow-leveled allocation...")
    res = build_project_schedule(cfg)
    print(f"      Scheduled {len(res.tasks)} tasks across {len(res.stage_summaries)} stages & {len(res.milestones)} milestones")
    print(f"      Core Product Complete Date (M4): {res.core_build_end_date.isoformat()} (Enters Migration Phase)")
    print(
        f"      4-Month Core Build (S1–S4): {res.core_build_base_cost_mvnd:.2f}M Base + "
        f"{res.core_build_buffer_mvnd:.2f}M Buffer = {res.core_build_buffered_cost_mvnd:.2f}M VND"
    )
    print(
        f"      6-Month Full Lifecycle (S1–S6): {res.total_base_cost_mvnd:.2f}M Base + "
        f"{res.total_buffer_mvnd:.2f}M Buffer (+{res.weighted_buffer_pct:.2f}%) = {res.total_buffered_cost_mvnd:.2f}M VND"
    )

    print("[3/4] Rendering Gantt chart, concurrency profile, and cost/cash-flow allocation charts...")
    chart_paths = generate_all_timeline_charts(cfg, res, out_dir)
    for cp in chart_paths:
        print(f"      Saved chart: {cp.name}")

    print("[4/4] Exporting WBS CSV, cost allocation CSVs, and cited methodology REPORT.md...")
    exports = export_timeline_outputs(cfg, res, chart_paths, out_dir)
    for k, p in exports.items():
        print(f"      Saved {k}: {p.name}")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Outputs written to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_project_timeline()
