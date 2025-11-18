#!/bin/bash
#
# Quick validation baseline runner for DeepSky ATC
#
# Runs short-haul baseline simulation for fast conflict validation.
# Completes in ~30 minutes with domestic routes only.
#
# Usage:
#   chmod +x scripts/quick_validate.sh
#   ./scripts/quick_validate.sh
#

echo ""
echo "================================================================================"
echo " DeepSky ATC - Quick Validation Baseline"
echo " Short-haul routes only (<800nm) for fast conflict testing"
echo "================================================================================"
echo ""

# Run short-haul baseline
echo "Step 1: Running short-haul baseline simulation..."
echo "  - 100 domestic routes per scenario"
echo "  - 2-hour simulation (7200 seconds)"
echo "  - 5 staffing scenarios"
echo "  - Expected runtime: ~30 minutes"
echo ""

python3 scripts/run_baseline_short_haul.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ ERROR: Short-haul baseline failed!"
    exit 1
fi

echo ""
echo "Step 2: Generating analysis and visualizations..."
echo ""

# Note: analyze_baseline.py needs --input parameter support
# For now, just show where results are
echo "✓ Short-haul baseline complete!"
echo ""
echo "Results saved to:"
echo "  - data/baseline_short_haul/scenario_*.json"
echo "  - data/baseline_short_haul/baseline_summary.csv"
echo "  - docs/reports/short_haul/baseline_comparison.md"
echo ""
echo "To analyze results:"
echo "  python scripts/analyze_baseline.py  # (update to support --input flag)"
echo ""
echo "To run full 4-hour baseline:"
echo "  python scripts/run_baseline.py"
echo ""
echo "================================================================================"
echo " Quick Validation Complete!"
echo "================================================================================"
echo ""
