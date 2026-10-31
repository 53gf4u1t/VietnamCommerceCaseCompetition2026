"""Main entry point for PICK! Adaptive Court Pricing simulation and ML pipeline (using central CaseComp/configs/)."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from AdaptivePricing.src.config import load_config
    from AdaptivePricing.src.simulator import run_simulations
    from AdaptivePricing.src.model import train_and_evaluate
    from AdaptivePricing.src.visualizer import (
        plot_assumptions_distributions,
        plot_demand_curve_variations,
        plot_daily_revenue_comparison,
        plot_monthly_calendar_heatmap,
        plot_occupancy_and_30pct_stress_test,
        plot_ml_model_diagnostics,
    )
    from AdaptivePricing.src.reporter import export_tables_and_report
except ImportError:
    from src.config import load_config
    from src.simulator import run_simulations
    from src.model import train_and_evaluate
    from src.visualizer import (
        plot_assumptions_distributions,
        plot_demand_curve_variations,
        plot_daily_revenue_comparison,
        plot_monthly_calendar_heatmap,
        plot_occupancy_and_30pct_stress_test,
        plot_ml_model_diagnostics,
    )
    from src.reporter import export_tables_and_report


def run_adaptive_pricing(
    config_path: Path | None = None,
    configs_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run the AdaptivePricing simulation & ML optimization pipeline and return its output_dir."""
    base_dir = Path(__file__).resolve().parent
    cfg_dir = config_path or configs_dir or (base_dir.parent / "configs")
    out_dir = output_dir or (base_dir / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AdaptivePricing 1/4] Loading central configurations from {cfg_dir.relative_to(base_dir.parent)}/...")
    cfg = load_config(cfg_dir)

    print(
        f"[AdaptivePricing 2/4] Running {cfg.simulation.n_simulations} monthly Monte Carlo simulations "
        f"({cfg.simulation.n_simulations * cfg.simulation.days_per_month * cfg.simulation.hours_per_day:,} court-hours)..."
    )
    sim = run_simulations(cfg)

    print("[AdaptivePricing 3/4] Training Gradient Boosting ML Pricing Model & evaluating technical/economic metrics...")
    _, eval_res = train_and_evaluate(cfg, sim)

    print(f"[AdaptivePricing 4/4] Generating visual package and data tables in {out_dir}...")
    c1 = plot_assumptions_distributions(cfg, sim, out_dir)
    c2 = plot_demand_curve_variations(cfg, sim, eval_res, out_dir)
    c3 = plot_daily_revenue_comparison(cfg, eval_res, out_dir)
    c4 = plot_monthly_calendar_heatmap(cfg, eval_res, out_dir)
    c5 = plot_occupancy_and_30pct_stress_test(cfg, eval_res, out_dir)
    c6 = plot_ml_model_diagnostics(cfg, eval_res, out_dir)

    daily_csv, sched_csv, report_md, ml_metrics_md = export_tables_and_report(
        cfg, sim, eval_res, [c1, c2, c3, c4, c5, c6], out_dir
    )

    base_rev_m = eval_res.mean_baseline_monthly_rev_kvnd / 1000.0
    ml_rev_m = eval_res.mean_ml_monthly_rev_kvnd / 1000.0
    lift_pct = (ml_rev_m - base_rev_m) / base_rev_m * 100.0
    m = eval_res.ml_metrics

    print("\n" + "=" * 86)
    print("PICK! ADAPTIVE COURT PRICING — EXECUTION & ML TECHNICAL SUMMARY")
    print("=" * 86)
    print(f"  Simulated Monthly Booked Hours : {sim.monthly_booked_hours.mean():.1f} ± {sim.monthly_booked_hours.std():.1f} hrs (Target: 1,900)")
    print(f"  Baseline Monthly Court Revenue : {base_rev_m:.2f}M VND (Case Exhibit 1: 435.00M VND)")
    print(f"  ML Adaptive Court Revenue      : {ml_rev_m:.2f}M VND (+{ml_rev_m - base_rev_m:.2f}M VND / +{lift_pct:.2f}%)")
    print(f"  Weekday Daytime Utilization    : {eval_res.baseline_wd_daytime_util*100:.1f}% to {eval_res.ml_wd_daytime_util*100:.1f}%")
    print(f"  -30% Demand Shock Revenue      : Baseline {eval_res.stress30_baseline_monthly_rev_kvnd/1000:.2f}M vs ML {eval_res.stress30_ml_monthly_rev_kvnd/1000:.2f}M VND")
    print("-" * 86)
    print(f"  ML Out-of-Sample R² (Train/Test): {m.train_r2:.4f} / {m.test_r2:.4f} (Explained Var: {m.test_explained_var:.4f})")
    print(f"  ML Out-of-Sample RMSE / MAE     : RMSE = {m.test_rmse_kvnd:.2f}k VND | MAE = {m.test_mae_kvnd:.2f}k VND | MAPE = {m.test_mape_pct:.2f}%")
    print(f"  Oracle Revenue Efficiency       : {m.oracle_capture_ratio_pct:.2f}% of Perfect-Foresight Oracle ({m.uplift_capture_vs_oracle_pct:.1f}% of Max Lift)")
    print("-" * 86)
    print(f"  Generated Charts : {c1.name}, {c2.name}, {c3.name}, {c4.name}, {c5.name}, {c6.name}")
    print(f"  Generated Tables : {daily_csv.name}, {sched_csv.name}")
    print(f"  Markdown Reports : {report_md.name}, {ml_metrics_md.name}")
    print("=" * 86)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PICK! Adaptive Court Pricing — Monte Carlo Simulation & ML Optimization"
    )
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=base_dir.parent / "configs",
        help="Path to central CaseComp/configs directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "outputs",
        help="Directory to write generated charts, CSVs, and report",
    )
    args = parser.parse_args()
    run_adaptive_pricing(configs_dir=args.configs_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
