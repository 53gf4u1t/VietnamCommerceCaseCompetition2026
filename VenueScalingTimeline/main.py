"""Entry point for VenueScalingTimeline (invoked via root `uv run main`)."""

from __future__ import annotations

from pathlib import Path
import time

try:
    from VenueScalingTimeline.src.config import load_venue_scaling_config
    from VenueScalingTimeline.src.schedule_model import build_venue_scaling_schedule
    from VenueScalingTimeline.src.visualizer import generate_all_venue_scaling_charts
    from VenueScalingTimeline.src.reporter import export_venue_scaling_outputs
except ImportError:
    from src.config import load_venue_scaling_config
    from src.schedule_model import build_venue_scaling_schedule
    from src.visualizer import generate_all_venue_scaling_charts
    from src.reporter import export_venue_scaling_outputs


def run_venue_scaling_timeline(
    config_path: Path | None = None,
    output_dir: Path | None = None,
    adaptive_output_dir: Path | None = None,
) -> Path:
    """Run the Sub-Solution Path A (14-Court Second Club Venue Scaling starting Q4 2027) pipeline."""
    t0 = time.perf_counter()
    base_dir = Path(__file__).resolve().parent
    cfg_path = config_path if config_path is not None else (base_dir.parent / "configs")
    out_dir = output_dir if output_dir is not None else (base_dir / "outputs")

    print("=" * 86)
    print("PICK! VENUE SCALING TIMELINE (PATH A: 6 TO 20 COURTS, Q4 2027 KICKOFF)")
    print("=" * 86)

    print(f"[1/4] Loading venue scaling & CapEx configuration from {cfg_path.name}...")
    cfg = load_venue_scaling_config(cfg_path)
    meta = cfg.meta
    print(
        f"      Build Window: {meta.build_start_month_label} to {meta.build_end_month_label} "
        f"({meta.build_duration_months} Months / {meta.build_duration_weeks:.0f} Weeks) | "
        f"Opens: {meta.open_month_label} (Month {meta.open_month_index})"
    )

    print("[2/4] Computing WBS schedule, even 7.0B VND CapEx split (1,750M/mo), and 6-to-9 month fill-up ramp...")
    res = build_venue_scaling_schedule(cfg, adaptive_output_dir=adaptive_output_dir)
    print(
        f"      Even Monthly Fit-Out CapEx: {meta.even_monthly_fitout_capex_mvnd:,.2f}M VND/mo across "
        f"{meta.build_duration_months} months = {meta.total_fitout_capex_mvnd:,.2f}M VND (+{meta.investor_reserve_mvnd:,.0f}M Reserve)"
    )
    print(
        f"      Steady-State Venue 2 (14 Courts) Revenue (OUR Model): "
        f"{res.venue2_steady_court_hire_mvnd:,.2f}M Court Hire + {res.venue2_steady_ancillary_mvnd:,.2f}M Ancillary = "
        f"{res.venue2_steady_total_rev_mvnd:,.2f}M VND/mo"
    )

    print("[3/4] Rendering Gantt chart, 20-court fill-up ramp, and even CapEx allocation charts...")
    chart_paths = generate_all_venue_scaling_charts(cfg, res, out_dir)
    for cp in chart_paths:
        print(f"      Saved chart: {cp.name}")

    print("[4/4] Exporting WBS CSV, 60-month CapEx & fill-up ramp CSV, and cited REPORT.md...")
    exports = export_venue_scaling_outputs(cfg, res, chart_paths, out_dir)
    for k, p in exports.items():
        print(f"      Saved {k}: {p.name}")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Outputs written to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_venue_scaling_timeline()
