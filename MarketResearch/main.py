"""Main orchestration module for `MarketResearch/` (Section 1: Our Market & Section 2: Our Customers)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from theme import set_theme
from MarketResearch.src.config import load_market_research_config
from MarketResearch.src.model import run_market_research_analysis
from MarketResearch.src.reporter import export_market_research_outputs
from MarketResearch.src.visualizer import (
    plot_customer_personas_and_mutual_demand,
    plot_hcmc_market_positioning_bubble_chart,
)


def run_market_research(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Execute the complete MarketResearch module (Vietnam/HCMC TAM-SAM-SOM sizing,
    12-club competitive positioning bubble chart, and 4 customer personas & mutual
    demand analysis), saving all PNGs, CSVs, and REPORT.md into `MarketResearch/outputs/`.
    """
    cfg_path = Path(config_path) if config_path else ROOT_DIR / "configs" / "market.toml"
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading Vietnam & HCMC Market Sizing, Competitor Survey & Customer Personas from `configs/market.toml`...")
    cfg = load_market_research_config(cfg_path)

    print("2. Running Competitive Positioning, TAM-SAM-SOM Share & Customer Mutual Demand Analysis...")
    analysis = run_market_research_analysis(cfg)
    print(
        f"   -> 2026 TAM = {analysis.tam_2026_bvnd:,.1f}B VND | HCMC SAM = {analysis.sam_2026_bvnd:,.1f}B VND | "
        f"PICK! Base SOM = {analysis.som_2026_bvnd:.2f}B VND ({analysis.som_share_of_sam_2026_pct:.2f}% SAM)"
    )
    print(
        f"   -> 2029 Y3 TAM = {analysis.tam_2029_bvnd:,.1f}B VND | HCMC SAM = {analysis.sam_2029_bvnd:,.1f}B VND | "
        f"PICK! Y3 SOM = {analysis.som_2029_bvnd:.2f}B VND ({analysis.som_share_of_sam_2029_pct:.2f}% SAM, "
        f"{analysis.som_share_of_catchment_2029_pct:.1f}% Prime Catchment)"
    )

    print("3. Rendering publication-grade Market Positioning & Customer Persona charts...")
    p1 = plot_hcmc_market_positioning_bubble_chart(cfg, analysis, out_dir)
    p2 = plot_customer_personas_and_mutual_demand(cfg, analysis, out_dir)
    print(f"   -> Saved: {p1.name}, {p2.name}")

    print("4. Exporting Competitor Positioning, TAM-SAM-SOM Sizing & Customer Persona CSVs and REPORT.md...")
    c1, c2, c3, r1 = export_market_research_outputs(cfg, analysis, out_dir)
    print(f"   -> Saved: {c1.name}, {c2.name}, {c3.name}, {r1.name}")

    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the PICK! MarketResearch analysis & visualization suite.")
    parser.add_argument(
        "--theme",
        type=str,
        default="Latte",
        help="Catppuccin color scheme ('Mocha' or 'Latte'). Default: Latte.",
    )
    args = parser.parse_args(argv)
    active_theme = set_theme(args.theme)
    print(f"[MarketResearch] Active Theme: Catppuccin {active_theme}")
    run_market_research()


if __name__ == "__main__":
    main()
