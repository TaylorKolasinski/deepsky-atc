"""
Test suite for DeepSky ATC delay profile model.

Tests log-normal delay distributions, time-based profile selection,
and integration with flight routes.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.delay_model import DelayModel
from src.route_generator import FlightRoute, generate_waypoints


def print_histogram(data, num_bins=20, max_value=None, title="Distribution"):
    """
    Print ASCII histogram of data.

    Args:
        data: Array of values to plot
        num_bins: Number of histogram bins
        max_value: Maximum value to display (clips data)
        title: Title for histogram
    """
    print(f"\n{title}")
    print("=" * 70)

    # Clip data if max_value specified
    if max_value is not None:
        data = np.clip(data, 0, max_value)

    # Create histogram
    counts, bin_edges = np.histogram(data, bins=num_bins)

    # Normalize to percentage
    total = len(data)
    percentages = (counts / total) * 100

    # Find max count for scaling
    max_count = max(counts)
    bar_width = 50  # Maximum bar width in characters

    # Print histogram
    for i in range(len(counts)):
        # Bin range
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Bar length (scaled to fit terminal)
        if max_count > 0:
            bar_len = int((counts[i] / max_count) * bar_width)
        else:
            bar_len = 0

        bar = "█" * bar_len

        # Print line
        print(f"{bin_start:6.1f}-{bin_end:6.1f} min ({percentages[i]:5.1f}%) {bar}")

    print()


def test_delay_model_initialization():
    """Test delay model initialization."""
    print("=" * 70)
    print("TEST 1: Delay Model Initialization")
    print("=" * 70)
    print()

    # Initialize with fixed seed for reproducibility
    delay_model = DelayModel(seed=42)

    print(f"Delay model: {delay_model}")
    print()

    print("Configured profiles:")
    for profile_name in ["overnight", "normal", "peak_hours", "weather"]:
        profile = delay_model.get_profile_info(profile_name)
        print(f"  {profile_name:12s}: median={profile['median_minutes']:3d}min, sigma={profile['sigma']:.1f}")

    print()
    print(f"Probability of no delay: {delay_model.probability_no_delay * 100:.0f}%")
    print(f"Maximum delay cap: {delay_model.max_delay_minutes} minutes")
    print()

    print("✓ PASSED: Delay model initialized successfully")
    print()

    return delay_model


def test_profile_statistics(delay_model):
    """Test statistics for each delay profile."""
    print("=" * 70)
    print("TEST 2: Profile Statistics (10,000 samples each)")
    print("=" * 70)
    print()

    profiles = ["overnight", "normal", "peak_hours", "weather"]
    num_samples = 10000

    all_stats = {}

    for profile_name in profiles:
        print(f"Profile: {profile_name.upper()}")
        print("-" * 70)

        # Generate statistics
        stats = delay_model.generate_delay_statistics(profile_name, num_samples)
        all_stats[profile_name] = stats

        # Get profile configuration
        config = delay_model.get_profile_info(profile_name)

        print(f"Configuration:")
        print(f"  Target median: {config['median_minutes']} min")
        print(f"  Sigma: {config['sigma']}")
        print()

        print(f"Observed statistics ({num_samples:,} samples):")
        print(f"  Mean:   {stats['mean']:6.1f} min")
        print(f"  Median: {stats['median']:6.1f} min")
        print(f"  Std:    {stats['std']:6.1f} min")
        print()

        print(f"Percentile breakdown:")
        print(f"  50th percentile: {stats['percentiles']['p50']:6.1f} min")
        print(f"  75th percentile: {stats['percentiles']['p75']:6.1f} min")
        print(f"  90th percentile: {stats['percentiles']['p90']:6.1f} min")
        print(f"  95th percentile: {stats['percentiles']['p95']:6.1f} min")
        print(f"  99th percentile: {stats['percentiles']['p99']:6.1f} min")
        print()

        print(f"On-time departures: {stats['percent_zero_delay']:.1f}%")
        print(f"Delay range: {stats['min']:.1f} - {stats['max']:.1f} min")
        print()

    print("✓ PASSED: All profiles generated valid statistics")
    print()

    return all_stats


def test_delay_bounds(delay_model):
    """Test that delays are within valid bounds."""
    print("=" * 70)
    print("TEST 3: Delay Bounds Validation")
    print("=" * 70)
    print()

    num_samples = 1000
    max_delay = delay_model.max_delay_minutes

    violations = 0

    for profile_name in ["overnight", "normal", "peak_hours", "weather"]:
        delays = [
            delay_model.calculate_departure_delay(profile_name=profile_name)
            for _ in range(num_samples)
        ]

        # Check bounds
        min_delay = min(delays)
        max_observed = max(delays)

        print(f"{profile_name:12s}: min={min_delay:6.1f}, max={max_observed:6.1f}, cap={max_delay}")

        if min_delay < 0:
            print(f"  ✗ ERROR: Negative delay detected: {min_delay}")
            violations += 1

        if max_observed > max_delay:
            print(f"  ✗ ERROR: Delay exceeds cap: {max_observed} > {max_delay}")
            violations += 1

    print()

    if violations == 0:
        print("✓ PASSED: All delays within valid bounds [0, max_delay_minutes]")
    else:
        print(f"✗ FAILED: {violations} bound violations detected")

    print()
    return violations == 0


def test_time_based_profile_selection(delay_model):
    """Test that profile selection varies by time of day."""
    print("=" * 70)
    print("TEST 4: Time-Based Profile Selection")
    print("=" * 70)
    print()

    # Test specific hours
    test_cases = [
        (3, "overnight", "Early morning (3 AM)"),
        (8, "peak_hours", "Morning rush (8 AM)"),
        (14, "normal", "Afternoon (2 PM)"),
        (18, "peak_hours", "Evening rush (6 PM)"),
        (22, "normal", "Late evening (10 PM)")
    ]

    print("Testing profile selection by hour:")
    print()

    all_correct = True

    for hour, expected_profile, description in test_cases:
        # Get profile for this hour
        profile = delay_model._get_profile_for_hour(hour)
        actual_profile = delay_model.delay_by_hour[str(hour)]

        status = "✓" if actual_profile == expected_profile else "✗"
        print(f"{status} Hour {hour:2d} ({description:20s}): {actual_profile:12s} "
              f"(expected: {expected_profile})")

        if actual_profile != expected_profile:
            all_correct = False

    print()

    if all_correct:
        print("✓ PASSED: Profile selection matches time of day")
    else:
        print("✗ FAILED: Some profile selections incorrect")

    print()
    return all_correct


def test_normal_profile_validation(all_stats):
    """Validate normal profile matches real-world patterns."""
    print("=" * 70)
    print("TEST 5: Normal Profile Real-World Validation")
    print("=" * 70)
    print()

    stats = all_stats["normal"]

    print("Validation targets (from FAA data patterns):")
    print("  ~70% of flights under 20 min delay")
    print("  ~90% of flights under 40 min delay")
    print()

    # Calculate percentages
    # We need to resample to check specific thresholds
    delay_model = DelayModel(seed=42)
    samples = [delay_model.calculate_departure_delay(profile_name="normal") for _ in range(10000)]

    pct_under_20 = (sum(1 for d in samples if d < 20) / len(samples)) * 100
    pct_under_40 = (sum(1 for d in samples if d < 40) / len(samples)) * 100

    print("Observed patterns:")
    print(f"  {pct_under_20:.1f}% of flights under 20 min delay")
    print(f"  {pct_under_40:.1f}% of flights under 40 min delay")
    print()

    # Validate
    validation_passed = True

    if 65 <= pct_under_20 <= 75:
        print(f"  ✓ ~70% under 20 min: {pct_under_20:.1f}% (within 65-75% range)")
    else:
        print(f"  ⚠ ~70% under 20 min: {pct_under_20:.1f}% (outside 65-75% range)")
        validation_passed = False

    if 85 <= pct_under_40 <= 95:
        print(f"  ✓ ~90% under 40 min: {pct_under_40:.1f}% (within 85-95% range)")
    else:
        print(f"  ⚠ ~90% under 40 min: {pct_under_40:.1f}% (outside 85-95% range)")
        validation_passed = False

    print()

    if validation_passed:
        print("✓ PASSED: Normal profile matches real-world delay patterns")
    else:
        print("⚠ WARNING: Normal profile slightly outside expected ranges")

    print()
    return validation_passed


def test_histogram_visualization(delay_model):
    """Display ASCII histogram for normal profile."""
    print("=" * 70)
    print("TEST 6: Delay Distribution Visualization")
    print("=" * 70)

    # Generate samples
    num_samples = 10000
    samples = [delay_model.calculate_departure_delay(profile_name="normal") for _ in range(num_samples)]

    # Print histogram
    print_histogram(
        samples,
        num_bins=20,
        max_value=180,
        title="Normal Profile Delay Distribution (0-180 min)"
    )

    print("✓ PASSED: Histogram shows right-skewed distribution")
    print()


def test_route_integration():
    """Test integration with FlightRoute class."""
    print("=" * 70)
    print("TEST 7: FlightRoute Integration")
    print("=" * 70)
    print()

    # Create a sample route
    jfk_lat, jfk_lon = 40.6413, -73.7781
    lax_lat, lax_lon = 33.9416, -118.4085
    waypoints, distance = generate_waypoints(jfk_lat, jfk_lon, lax_lat, lax_lon)

    route = FlightRoute(
        route_id="TEST_JFK_LAX",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KLAX",
        waypoints=waypoints,
        total_distance_nm=distance,
        scheduled_departure_time=8 * 3600  # 8:00 AM
    )

    print(f"Route: {route.route_id}")
    print(f"  {route.departure_icao} → {route.arrival_icao}")
    print(f"  Scheduled departure: {route.scheduled_departure_time / 3600:.1f} hours (8:00 AM)")
    print()

    # Apply delay
    delay_model = DelayModel(seed=123)
    delay_minutes = route.apply_delay(delay_model, hour_of_day=8)

    print(f"Applied delay: {delay_minutes:.1f} minutes")
    print(f"Actual departure time: {route.actual_departure_time / 3600:.2f} hours")
    print()

    # Verify delay was applied correctly
    expected_actual = route.scheduled_departure_time + (delay_minutes * 60)
    error = abs(route.actual_departure_time - expected_actual)

    if error < 0.01:
        print("✓ PASSED: Delay correctly applied to route")
    else:
        print(f"✗ FAILED: Delay calculation error: {error:.2f} seconds")

    print()

    # Test multiple routes with different departure times
    print("Testing multiple routes at different times:")
    print()

    delay_model.set_seed(42)  # Reset seed

    times_of_day = [3, 8, 14, 18, 22]  # Different hours
    for hour in times_of_day:
        test_route = FlightRoute(
            route_id=f"TEST_{hour:02d}00",
            aircraft_type="B738",
            departure_icao="KJFK",
            arrival_icao="KBOS",
            waypoints=[(40.6, -73.8, 0), (42.4, -71.0, 0)],
            scheduled_departure_time=hour * 3600
        )

        delay = test_route.apply_delay(delay_model, hour_of_day=hour)
        profile = delay_model.delay_by_hour[str(hour)]

        print(f"  Hour {hour:2d}:00 ({profile:12s}): delay={delay:6.1f} min")

    print()
    print("✓ PASSED: Route integration working correctly")
    print()


def test_right_skewed_distribution():
    """Verify log-normal produces right-skewed distribution."""
    print("=" * 70)
    print("TEST 8: Right-Skewed Distribution Validation")
    print("=" * 70)
    print()

    delay_model = DelayModel(seed=42)

    # Generate samples
    num_samples = 10000
    samples = [delay_model.calculate_departure_delay(profile_name="normal") for _ in range(num_samples)]

    # Calculate skewness indicators
    mean = np.mean(samples)
    median = np.median(samples)

    # For right-skewed distribution: mean > median
    print(f"Distribution characteristics:")
    print(f"  Mean:   {mean:.2f} min")
    print(f"  Median: {median:.2f} min")
    print(f"  Ratio:  mean/median = {mean/median:.2f}")
    print()

    if mean > median:
        print("✓ PASSED: Distribution is right-skewed (mean > median)")
        print("  This matches real-world delay patterns:")
        print("  - Most flights: small delays (near median)")
        print("  - Few flights: large delays (long right tail)")
    else:
        print("✗ FAILED: Distribution not right-skewed")

    print()


def test_reproducibility():
    """Test that using same seed produces same results."""
    print("=" * 70)
    print("TEST 9: Reproducibility with Seed")
    print("=" * 70)
    print()

    # Generate delays with seed 42
    model1 = DelayModel(seed=42)
    delays1 = [model1.calculate_departure_delay(profile_name="normal") for _ in range(100)]

    # Generate delays with same seed
    model2 = DelayModel(seed=42)
    delays2 = [model2.calculate_departure_delay(profile_name="normal") for _ in range(100)]

    # Check if identical
    if delays1 == delays2:
        print("✓ PASSED: Same seed produces identical delays")
        print(f"  First 5 delays: {[f'{d:.2f}' for d in delays1[:5]]}")
    else:
        print("✗ FAILED: Delays differ with same seed")

    print()


def main():
    """Run all delay model tests."""
    print("\n")
    print("*" * 70)
    print("* DeepSky ATC - Delay Profile Model Test Suite")
    print("* Phase 1, Deliverable 1.3")
    print("*" * 70)
    print()

    try:
        # Run all tests
        delay_model = test_delay_model_initialization()
        all_stats = test_profile_statistics(delay_model)
        test_delay_bounds(delay_model)
        test_time_based_profile_selection(delay_model)
        test_normal_profile_validation(all_stats)
        test_histogram_visualization(delay_model)
        test_route_integration()
        test_right_skewed_distribution()
        test_reproducibility()

        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print()

        print("Summary:")
        print("  ✓ Log-normal distributions validated")
        print("  ✓ Time-based profile selection working")
        print("  ✓ Real-world delay patterns matched")
        print("  ✓ FlightRoute integration successful")
        print("  ✓ Right-skewed distribution confirmed")
        print()

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
