"""Unified single entry point (`uv run main`) for the PICK! Case Competition pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from AdaptivePricing.main import run_adaptive_pricing
from FinancialModel.main import run_financial_model
from Marketing.main import run_marketing
from PickSocialProjectTimeline.main import run_project_timeline
from VenueScalingTimeline.main import run_venue_scaling_timeline

CENTRAL_CONFIGS_DIR = ROOT_DIR / "configs"


def main() -> None:
    """
    Execute the complete 5-stage CaseComp pipeline via a single command (`uv run main`):
      1. AdaptivePricing           — Loads `configs/pricing.toml` & `configs/randomness.toml`,
                                     runs Monte Carlo simulation & ML court pricing optimization,
                                     writing required reports & schedules into AdaptivePricing/outputs/.
      2. PickSocialProjectTimeline — Loads `configs/opex.toml`, builds the Path C Custom Platform
                                     4-month (16-week) core build + migration & continuous ops Gantt
                                     chart, leveled pay-as-needed cost allocation, and recurring
                                     monthly platform OpEx schedule starting Jan 2027 in
                                     PickSocialProjectTimeline/outputs/.
      3. VenueScalingTimeline      — Loads `configs/capex.toml` & `configs/opex.toml`, schedules the
                                     Path A 14-court second club expansion (`6` -> `20` courts) from
                                     start of Q3 2027 (`Month 7 / Jul-27`), with a 4-month build
                                     (`Jul-27` to `Oct-27`), evenly split `1,750M VND/month` fit-out CapEx,
                                     `330M VND/month` 5-year lease, and 6-to-9 month fill-up ramp in
                                     VenueScalingTimeline/outputs/.
      4. Marketing                 — Loads `configs/customers.toml`, `configs/opex.toml` & `configs/externals.toml`,
                                     models the TikTok/IG follower-to-customer funnel (`~8.57%` conversion),
                                     4-segment marketing cost per visit (`x, y, z, w` VND/visit), trains a
                                     4-segment `HistGradientBoostingRegressor` Marketing Budget Predictor,
                                     and projects the 12 Segment-by-Ancillary revenue distributions
                                     (`Paddle/Balls`, `Coaching`, `Food & Drinks`) in Marketing/outputs/.
      5. FinancialModel            — Loads all central distribution configs from `configs/*.toml`
                                     (including `externals.toml`), verifies all 4 upstream `outputs/`
                                     folders, simulates 300 Monte Carlo 60-month (`Jan-27` to `Dec-31`)
                                     3-statement trajectories, and outputs all tables, charts, and
                                     monthly cash balance schedules in FinancialModel/outputs/.
    """
    print("=" * 86)
    print("VIETNAM COMMERCE CASE COMPETITION 2026 — STAGE 1/5: AdaptivePricing (Central configs/)")
    print("=" * 86)
    ap_output_dir = run_adaptive_pricing(
        config_path=CENTRAL_CONFIGS_DIR,
        output_dir=ROOT_DIR / "AdaptivePricing" / "outputs",
    )

    print("\n" + "=" * 86)
    print("VIETNAM COMMERCE CASE COMPETITION 2026 — STAGE 2/5: PickSocialProjectTimeline (Central configs/)")
    print("=" * 86)
    tl_output_dir = run_project_timeline(
        config_path=CENTRAL_CONFIGS_DIR,
        output_dir=ROOT_DIR / "PickSocialProjectTimeline" / "outputs",
    )

    print("\n" + "=" * 86)
    print("VIETNAM COMMERCE CASE COMPETITION 2026 — STAGE 3/5: VenueScalingTimeline (Central configs/)")
    print("=" * 86)
    vs_output_dir = run_venue_scaling_timeline(
        config_path=CENTRAL_CONFIGS_DIR,
        output_dir=ROOT_DIR / "VenueScalingTimeline" / "outputs",
    )

    print("\n" + "=" * 86)
    print("VIETNAM COMMERCE CASE COMPETITION 2026 — STAGE 4/5: Marketing (Central configs/)")
    print("=" * 86)
    mkt_output_dir = run_marketing(
        config_path=CENTRAL_CONFIGS_DIR,
        output_dir=ROOT_DIR / "Marketing" / "outputs",
    )

    print("\n" + "=" * 86)
    print("VIETNAM COMMERCE CASE COMPETITION 2026 — STAGE 5/5: FinancialModel (Central configs/)")
    print("=" * 86)
    run_financial_model(
        config_path=CENTRAL_CONFIGS_DIR,
        output_dir=ROOT_DIR / "FinancialModel" / "outputs",
        adaptive_output_dir=ap_output_dir,
        timeline_output_dir=tl_output_dir,
        venue_scaling_output_dir=vs_output_dir,
        marketing_output_dir=mkt_output_dir,
    )


if __name__ == "__main__":
    main()
