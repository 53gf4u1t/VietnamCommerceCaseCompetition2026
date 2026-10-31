"""Main orchestration module for `Marketing/` (invoked by `CaseComp/main.py`)."""

from __future__ import annotations

from pathlib import Path

from .src.config import load_marketing_config
from .src.model import train_and_evaluate_marketing_ml
from .src.reporter import export_marketing_outputs
from .src.simulator import run_marketing_simulation
from .src.visualizer import (
    plot_funnel_and_12_distributions,
    plot_marketing_ml_diagnostics,
    plot_ml_segment_budgets_and_visits,
    plot_segment_ancillary_revenue_breakdown,
)


def run_marketing(
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Execute the complete Marketing Funnel, 4-Segment ML Marketing Budget Predictor,
    and 12-Distribution Ancillary Revenue Engine, saving all CSVs, charts, and
    reports into `Marketing/outputs/`.
    """
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading central Marketing & Customer Segment distributions from `configs/`...")
    cfg = load_marketing_config(config_path)

    print(
        f"2. Simulating {cfg.n_simulations} Monte Carlo trajectories x {cfg.projection_months} months "
        f"(Follower-to-Customer ~8.57% conversion + 4 segments + 12 ancillary spending distributions)..."
    )
    sim = run_marketing_simulation(cfg)

    print("3. Training & evaluating 4-Segment HistGradientBoostingRegressor Marketing Budget Predictor (80/20 split)...")
    ml_res = train_and_evaluate_marketing_ml(cfg, sim)
    m = ml_res.ml_metrics
    print(
        f"   -> Held-Out Test R2 = {m.test_r2:.4f} | RMSE = {m.test_rmse_mvnd:.4f} M VND | "
        f"MAPE = {m.test_mape_pct:.2f}% | Oracle Capture = {m.oracle_net_contrib_capture_pct:.2f}%"
    )

    print("4. Rendering presentation-ready Marketing Funnel, ML Budget & 12-Ancillary charts...")
    p1 = plot_funnel_and_12_distributions(cfg, sim, out_dir)
    p2 = plot_ml_segment_budgets_and_visits(sim, ml_res, out_dir)
    p3 = plot_segment_ancillary_revenue_breakdown(sim, ml_res, out_dir)
    p4 = plot_marketing_ml_diagnostics(sim, ml_res, out_dir)
    print(f"   -> Saved: {p1.name}, {p2.name}, {p3.name}, {p4.name}")

    print("5. Exporting 60-month segment marketing, visits & 12-distribution ancillary revenue CSVs and reports...")
    c1, c2, r1, r2 = export_marketing_outputs(cfg, sim, ml_res, out_dir)
    print(f"   -> Saved: {c1.name}, {c2.name}, {r1.name}, {r2.name}")

    return out_dir
