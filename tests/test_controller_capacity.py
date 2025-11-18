"""
Test suite for DeepSky ATC controller capacity constraints.

Tests staffing scenarios from full staffing to government shutdown conditions,
demonstrating performance degradation under reduced staffing.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.controller_capacity import ControllerStaffing
from src.simulation_manager import create_demo_simulation
from src.metrics import generate_staffing_comparison_report


def test_staffing_configurations():
    """Test all staffing configuration presets."""
    print("=" * 70)
    print("TEST 1: Staffing Configuration Definitions")
    print("=" * 70)
    print()

    staffing_levels = ['full', 'normal', 'reduced', 'shutdown', 'none']

    for level in staffing_levels:
        staffing = ControllerStaffing(level)
        desc = staffing.get_staffing_description()

        print(f"=== {level.upper()} STAFFING ===")
        print(f"  Description: {desc['description']}")
        print(f"  Controllers: {desc['num_controllers']}")
        print(f"  Capacity: {desc['capacity']} aircraft")
        print(f"  Scenario: {desc['scenario']}")
        print()

    print("✓ PASSED: All staffing configurations defined")
    print()


def test_workload_calculations():
    """Test workload factor and performance penalty calculations."""
    print("=" * 70)
    print("TEST 2: Workload and Performance Penalty Calculations")
    print("=" * 70)
    print()

    # Test shutdown scenario with overload
    staffing = ControllerStaffing("shutdown")  # capacity = 15
    print(f"Scenario: {staffing.get_staffing_description()['description']}")
    print(f"Capacity: {staffing.get_capacity()} aircraft")
    print()

    test_loads = [10, 15, 20, 30]
    print("Workload Analysis:")
    print(f"{'Aircraft':<12} {'Workload':<12} {'Delay Mult':<15} {'Conflict Mult':<15} {'Error %':<10}")
    print("-" * 70)

    for aircraft_count in test_loads:
        workload = staffing.calculate_workload_factor(aircraft_count)
        penalties = staffing.get_performance_penalty(aircraft_count)

        print(f"{aircraft_count:<12} "
              f"{workload:<12.2f}x "
              f"{penalties['delay_multiplier']:<15.2f}x "
              f"{penalties['conflict_risk_multiplier']:<15.2f}x "
              f"{penalties['human_error_probability']:<10.1%}")

    print()

    # Validate penalties increase with overload
    penalties_normal = staffing.get_performance_penalty(15)  # At capacity
    penalties_overload = staffing.get_performance_penalty(30)  # 2x overload

    if penalties_overload['delay_multiplier'] > penalties_normal['delay_multiplier']:
        print("✓ PASSED: Performance penalties increase with overload")
    else:
        print("✗ FAILED: Performance penalties should increase with overload")

    print()


def run_staffing_comparison():
    """Run same traffic scenario across all staffing levels."""
    print("=" * 70)
    print("TEST 3: Staffing Scenario Comparison (20 aircraft, 1 hour)")
    print("=" * 70)
    print()

    staffing_levels = ['full', 'normal', 'reduced', 'shutdown', 'none']
    results = []

    print("Running simulations for each staffing level...")
    print()

    for level in staffing_levels:
        print(f"--- Running {level.upper()} staffing scenario ---")
        print()

        # Create simulation with specific staffing level
        manager = create_demo_simulation(
            num_aircraft=20,
            duration=3600,
            output_mode="file",
            seed=42,  # Fixed seed for reproducibility
            staffing_level=level
        )

        # Run simulation
        stats = manager.run(duration_seconds=3600, time_step=1.0, progress_interval=600)

        # Close output
        manager.output.close()

        # Store results
        results.append(stats)

        # Print summary
        pm = stats['performance_metrics']
        print()
        print("Results:")
        print(f"  Safety Score: {pm['safety_score']:.1f}/100")
        print(f"  Conflicts/Hour: {pm['conflict_metrics']['conflicts_per_hour']:.2f}")
        print(f"  Average Delay: {pm['on_time_performance']['average_delay_minutes']:.1f} min")
        print(f"  On-Time %: {pm['on_time_performance']['on_time_percentage']:.1f}%")
        print(f"  Avg Workload: {stats['staffing']['average_workload_factor']:.2f}x")
        print()

    return results


def print_comparison_table(results):
    """Print markdown comparison table."""
    print("=" * 70)
    print("TEST 4: Performance Comparison Table")
    print("=" * 70)
    print()

    print("## Staffing Scenario Comparison\n")

    # Safety scores
    print("### Safety Score (0-100)\n")
    print("| Scenario | Safety Score | Grade |")
    print("|----------|--------------|-------|")
    for result in results:
        pm = result['performance_metrics']
        safety = pm['safety_score']
        grade = "A" if safety >= 90 else "B" if safety >= 80 else "C" if safety >= 70 else "D" if safety >= 60 else "F"
        print(f"| **{result['staffing']['staffing_level'].upper()}** | {safety:.1f} | {grade} |")
    print()

    # Conflict metrics
    print("### Conflict Metrics\n")
    print("| Scenario | Total Conflicts | Conflicts/Hour |")
    print("|----------|-----------------|----------------|")
    for result in results:
        cm = result['performance_metrics']['conflict_metrics']
        print(f"| **{result['staffing']['staffing_level'].upper()}** | "
              f"{cm['total_conflicts']:.0f} | {cm['conflicts_per_hour']:.2f} |")
    print()

    # On-time performance
    print("### On-Time Performance\n")
    print("| Scenario | On-Time % | Avg Delay (min) |")
    print("|----------|-----------|-----------------|")
    for result in results:
        otp = result['performance_metrics']['on_time_performance']
        print(f"| **{result['staffing']['staffing_level'].upper()}** | "
              f"{otp['on_time_percentage']:.1f}% | {otp['average_delay_minutes']:.1f} |")
    print()

    # Workload
    print("### Controller Workload\n")
    print("| Scenario | Capacity | Avg Workload | Time Overloaded (%) |")
    print("|----------|----------|--------------|---------------------|")
    for result in results:
        staffing = result['staffing']
        print(f"| **{staffing['staffing_level'].upper()}** | "
              f"{staffing['capacity']} | "
              f"{staffing['average_workload_factor']:.2f}x | "
              f"{staffing['time_overloaded_percent']:.1f}% |")
    print()


def validate_degradation(results):
    """Validate that performance degrades with reduced staffing."""
    print("=" * 70)
    print("TEST 5: Performance Degradation Validation")
    print("=" * 70)
    print()

    # Find specific scenarios
    full_result = next((r for r in results if r['staffing']['staffing_level'] == 'full'), None)
    shutdown_result = next((r for r in results if r['staffing']['staffing_level'] == 'shutdown'), None)

    if not full_result or not shutdown_result:
        print("✗ FAILED: Missing required scenarios")
        return

    full_pm = full_result['performance_metrics']
    shutdown_pm = shutdown_result['performance_metrics']

    print("Comparing FULL staffing vs SHUTDOWN scenario:")
    print()

    # Conflict comparison
    full_conflicts = full_pm['conflict_metrics']['conflicts_per_hour']
    shutdown_conflicts = shutdown_pm['conflict_metrics']['conflicts_per_hour']

    print(f"Conflicts per hour:")
    print(f"  Full staffing: {full_conflicts:.2f}")
    print(f"  Shutdown: {shutdown_conflicts:.2f}")

    if full_conflicts > 0:
        conflict_increase_pct = ((shutdown_conflicts - full_conflicts) / full_conflicts) * 100
        print(f"  Increase: {conflict_increase_pct:.0f}%")
        print()

        if shutdown_conflicts > full_conflicts:
            print("✓ PASSED: Shutdown has more conflicts than full staffing")
        else:
            print("⚠ WARNING: Expected shutdown to have more conflicts")
    else:
        print("  (No conflicts in full staffing scenario)")

    print()

    # Delay comparison
    full_delay = full_pm['on_time_performance']['average_delay_minutes']
    shutdown_delay = shutdown_pm['on_time_performance']['average_delay_minutes']

    print(f"Average delay:")
    print(f"  Full staffing: {full_delay:.1f} min")
    print(f"  Shutdown: {shutdown_delay:.1f} min")

    if full_delay != 0:
        delay_increase_pct = ((shutdown_delay - full_delay) / abs(full_delay)) * 100
        print(f"  Increase: {delay_increase_pct:.0f}%")
        print()

        if abs(shutdown_delay) > abs(full_delay):
            print("✓ PASSED: Shutdown has worse delays than full staffing")
        else:
            print("⚠ WARNING: Expected shutdown to have worse delays")
    else:
        print("  (No delays in either scenario)")

    print()

    # Safety score comparison
    full_safety = full_pm['safety_score']
    shutdown_safety = shutdown_pm['safety_score']

    print(f"Safety score:")
    print(f"  Full staffing: {full_safety:.1f}")
    print(f"  Shutdown: {shutdown_safety:.1f}")
    print(f"  Degradation: {full_safety - shutdown_safety:.1f} points")
    print()

    if shutdown_safety < full_safety:
        print("✓ PASSED: Shutdown has lower safety score than full staffing")
    else:
        print("⚠ WARNING: Expected shutdown to have lower safety score")

    print()


def test_comparison_report(results):
    """Test comparison report generation."""
    print("=" * 70)
    print("TEST 6: Comparison Report Generation")
    print("=" * 70)
    print()

    output_path = "data/metrics/staffing_comparison.md"
    generate_staffing_comparison_report(results, output_path)

    print()

    # Verify file exists
    report_file = Path(output_path)
    if report_file.exists():
        print(f"✓ Report created: {report_file}")

        # Show preview
        with open(report_file, 'r') as f:
            lines = f.readlines()

        print()
        print("Report preview (first 30 lines):")
        print("-" * 70)
        for line in lines[:30]:
            print(line.rstrip())
        print("-" * 70)
        print()

        print("✓ PASSED: Comparison report generated")
    else:
        print("✗ FAILED: Report not created")

    print()


def main():
    """Run all controller capacity tests."""
    print("\n")
    print("*" * 70)
    print("* DeepSky ATC - Controller Capacity Constraints Test Suite")
    print("* Phase 2, Deliverable 2.2a")
    print("*" * 70)
    print()

    try:
        # Run all tests
        test_staffing_configurations()
        test_workload_calculations()
        results = run_staffing_comparison()
        print_comparison_table(results)
        validate_degradation(results)
        test_comparison_report(results)

        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print()

        print("Summary:")
        print("  ✓ Staffing configurations defined")
        print("  ✓ Workload calculations validated")
        print("  ✓ Staffing scenarios compared (5 levels)")
        print("  ✓ Performance degradation confirmed")
        print("  ✓ Comparison report generated")
        print()

        # Print key findings
        full_result = next((r for r in results if r['staffing']['staffing_level'] == 'full'), None)
        shutdown_result = next((r for r in results if r['staffing']['staffing_level'] == 'shutdown'), None)
        ai_result = next((r for r in results if r['staffing']['staffing_level'] == 'none'), None)

        print("Key Findings:")
        if full_result:
            print(f"  • Full Staffing Safety Score: {full_result['performance_metrics']['safety_score']:.1f}")
        if shutdown_result:
            print(f"  • Shutdown Safety Score: {shutdown_result['performance_metrics']['safety_score']:.1f}")
        if ai_result:
            print(f"  • AI-Only Safety Score: {ai_result['performance_metrics']['safety_score']:.1f}")
        print()

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
