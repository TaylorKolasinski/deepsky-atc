# DeepSky ATC Baseline Systems

DeepSky ATC provides **two baseline configurations** optimized for different use cases:

1. **SHORT-HAUL BASELINE** - Quick validation and conflict testing
2. **FULL BASELINE** - Comprehensive realistic analysis

---

## 1. Short-Haul Baseline (Quick Validation)

**Purpose**: Fast conflict validation and testing during development

### Configuration
- **File**: `data/baseline_config_short_haul.json`
- **Script**: `scripts/run_baseline_short_haul.py`
- **Quick Runner**: `scripts/quick_validate.sh`

### Parameters
```json
{
  "route_filter": {
    "max_distance_nm": 800,
    "description": "Domestic routes only (1-2 hour flights)"
  },
  "simulation_parameters": {
    "num_routes_per_scenario": 100,
    "simulation_duration": 7200,
    "departure_interval_range": [20, 40]
  }
}
```

### Key Features
- ✅ **Domestic routes only** (<800nm): JFK-BOS, JFK-DCA, JFK-ATL, etc.
- ✅ **High aircraft density**: 100 routes per scenario
- ✅ **Tight departure spacing**: 20-40 seconds between flights
- ✅ **2-hour simulation**: Enough for short routes to complete
- ✅ **Fast execution**: ~30 minutes total runtime

### Expected Results
```
Completed flights per scenario: 30-50
Peak concurrent aircraft: 20-40
Conflicts: YES - high density triggers conflicts
Safety score variation: Clear degradation visible
Run time: 5-10 minutes per scenario
```

### Usage
```bash
# Quick validation (recommended for testing)
./scripts/quick_validate.sh

# Or run directly
python scripts/run_baseline_short_haul.py
```

### When to Use
- ✅ Daily development and testing
- ✅ Validating conflict detection works
- ✅ Quick sanity checks before commits
- ✅ Debugging simulation issues
- ✅ Rapid iteration on Phase 3 AI controller

### Sample Routes
```
KBOS → KJFK: 157 nm (45 min)
KJFK → KDCA: 202 nm (60 min)
KATL → KJFK: 660 nm (90 min)
KJFK → KORD: 628 nm (85 min)
KSFO → KJFK: 2242 nm (FILTERED OUT - too long)
```

---

## 2. Full Baseline (Final Results)

**Purpose**: Comprehensive realistic operational analysis

### Configuration
- **File**: `data/baseline_config.json`
- **Script**: `scripts/run_baseline.py`

### Parameters
```json
{
  "route_filter": {
    "max_distance_nm": null,
    "description": "All routes - mixed short/medium/long haul"
  },
  "simulation_parameters": {
    "num_routes_per_scenario": 50,
    "simulation_duration": 14400,
    "departure_interval_range": [30, 90]
  }
}
```

### Key Features
- ✅ **All routes**: Realistic mix of domestic and international
- ✅ **4-hour simulation**: Captures full operational cycle
- ✅ **Realistic density**: 50 routes per scenario
- ✅ **Natural spacing**: 30-90 seconds between departures
- ✅ **Production quality**: For demos, blog posts, LinkedIn

### Expected Results
```
Completed flights per scenario: 15-30
Peak concurrent aircraft: 10-25
Conflicts: YES - realistic operational conflicts
Safety score variation: Full degradation visible
Route mix: Short (30%), Medium (40%), Long (30%)
Run time: 20-30 minutes per scenario
```

### Usage
```bash
# Full baseline (for final results)
python scripts/run_baseline.py
```

### When to Use
- ✅ Final baseline for Phase 3 evaluation
- ✅ Blog posts and LinkedIn content
- ✅ Demos and presentations
- ✅ Comprehensive performance analysis
- ✅ Realistic operational scenarios

### Sample Routes
```
KBOS → KJFK: 157 nm (45 min) ✓ Completes
KATL → KJFK: 660 nm (90 min) ✓ Completes
KJFK → LFPG: 3150 nm (7 hours) - Active, not complete
KJFK → OTHH: 5815 nm (13 hours) - Just departed
Mixed operational profile - realistic!
```

---

## Comparison Table

| Feature | Short-Haul | Full |
|---------|------------|------|
| **Route Distance** | <800nm only | All distances |
| **Aircraft per Scenario** | 100 | 50 |
| **Simulation Duration** | 2 hours | 4 hours |
| **Departure Spacing** | 20-40s | 30-90s |
| **Total Runtime** | ~30 min | ~2 hours |
| **Completed Flights** | 30-50 | 15-30 |
| **Peak Concurrent** | 20-40 | 10-25 |
| **Conflicts Expected** | YES | YES |
| **Use Case** | Quick validation | Final analysis |
| **When to Run** | Daily testing | Before demos |
| **Output Directory** | `data/baseline_short_haul/` | `data/baseline/` |

---

## Workflow Recommendation

### During Development (Daily)
```bash
# Quick validation with short-haul
./scripts/quick_validate.sh

# Verify:
# - Conflicts detected ✓
# - Flights completing ✓
# - Safety scores varying ✓
# - ~30 minutes runtime ✓
```

### Before Demos/Posts (Weekly)
```bash
# Full baseline for comprehensive results
python scripts/run_baseline.py

# Verify:
# - Realistic operational profile ✓
# - Mixed route distances ✓
# - Production-quality data ✓
# - ~2 hours runtime (run overnight) ✓
```

---

## Output Structure

### Short-Haul Baseline
```
data/baseline_short_haul/
├── scenario_A_full_staffing.json
├── scenario_B_normal_operations.json
├── scenario_C_reduced_staffing.json
├── scenario_D_government_shutdown.json
├── scenario_E_no_control.json
├── baseline_summary.csv
└── output/
    └── simulation_*.json

docs/reports/short_haul/
└── baseline_comparison.md
```

### Full Baseline
```
data/baseline/
├── scenario_A_full_staffing.json
├── scenario_B_normal_operations.json
├── scenario_C_reduced_staffing.json
├── scenario_D_government_shutdown.json
├── scenario_E_no_control.json
├── baseline_summary.csv
└── output/
    └── simulation_*.json

docs/reports/
└── baseline_comparison.md
```

---

## Distance Calculation

Both baselines use `FlightRoute.calculate_great_circle_distance()` to filter and analyze routes:

```python
# Added to src/route_generator.py
def calculate_great_circle_distance(self) -> float:
    """
    Calculate great-circle distance between departure and arrival.
    Uses haversine formula for accuracy.
    """
    # Implementation uses haversine_distance from src/physics.py
    return distance_nm
```

### Usage Example
```python
from src.route_generator import load_routes_from_data

# Load all routes
all_routes = load_routes_from_data()

# Filter for short-haul
short_haul = [r for r in all_routes
              if r.calculate_great_circle_distance() <= 800]

print(f"Short-haul routes: {len(short_haul)}")
```

---

## Key Insights

### Why Two Baselines?

**SHORT-HAUL** solves the "no conflicts, no completions" problem:
- ✅ Routes complete within simulation window
- ✅ High aircraft density creates conflicts
- ✅ Fast iteration during development
- ❌ Not realistic operational mix

**FULL** provides realistic operational analysis:
- ✅ Mixed route distances (like real JFK)
- ✅ Natural operational patterns
- ✅ Production-quality metrics
- ❌ Slower runtime (2 hours)

### Best Practice
1. Use **short-haul** during active development
2. Run **full baseline** before major milestones
3. Use **short-haul** for conflict validation
4. Use **full baseline** for demos and posts

---

## Troubleshooting

### Short-Haul: "Not enough short-haul routes"
```bash
# Check available short routes
python -c "
from src.route_generator import load_routes_from_data
routes = load_routes_from_data()
short = [r for r in routes if r.calculate_great_circle_distance() <= 800]
print(f'Short-haul routes available: {len(short)}')
"

# If <50 routes, lower the filter threshold:
# Edit data/baseline_config_short_haul.json
# Change "max_distance_nm": 800 to 1000
```

### Full Baseline: "Taking too long"
```bash
# Reduce aircraft count for faster testing:
# Edit data/baseline_config.json
# Change "num_routes_per_scenario": 50 to 25

# Or reduce duration:
# Change "simulation_duration": 14400 to 7200
```

### No Conflicts Detected
```bash
# Use short-haul baseline instead
./scripts/quick_validate.sh

# Short-haul generates conflicts via:
# - Higher aircraft density (100 vs 50)
# - Tighter spacing (20-40s vs 30-90s)
# - Routes complete faster (more concurrent traffic)
```

---

## Future Enhancements

### Potential Additions
1. **Medium-haul baseline** (800-2000nm): Balance of both
2. **Stress test baseline** (200 aircraft): Maximum density
3. **International baseline** (>3000nm only): Long-haul focus
4. **Custom scenarios**: User-defined route filters

### Phase 3 Integration
The AI controller will be evaluated against **both** baselines:
- **Short-haul**: Quick validation, rapid iteration
- **Full baseline**: Final performance metrics, demos

---

## Quick Reference

```bash
# Quick validation TODAY (30 min)
./scripts/quick_validate.sh

# Full baseline LATER (2 hours)
python scripts/run_baseline.py

# Test single scenario
python scripts/test_baseline_single.py

# Analyze results
python scripts/analyze_baseline.py
```

---

**Created**: 2025-11-18
**Version**: 1.0
**Status**: ✅ Both baselines ready for use
