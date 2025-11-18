"""
Short-haul baseline simulation runner for DeepSky ATC.

Runs quick validation baseline with DOMESTIC ROUTES ONLY (<800nm).
Designed for fast conflict validation and testing - completes in ~30 minutes.

Usage:
    python scripts/run_baseline_short_haul.py
"""

import sys
from pathlib import Path
import json
import random
import csv
from datetime import datetime

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.airspace import Airspace
from src.delay_model import DelayModel
from src.simulation_output import SimulationOutput
from src.simulation_manager import SimulationManager
from src.controller_capacity import ControllerStaffing
from src.route_generator import load_routes_from_data
from src.metrics import generate_staffing_comparison_report


def load_config():
    """Load short-haul baseline configuration."""
    config_path = project_root / "data" / "baseline_config_short_haul.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def filter_short_haul_routes(all_routes, max_distance_nm):
    """
    Filter routes to only include short-haul domestic flights.

    Args:
        all_routes: List of all available routes
        max_distance_nm: Maximum route distance in nautical miles

    Returns:
        List of short-haul routes (<= max_distance_nm)
    """
    print("=" * 80)
    print("FILTERING FOR SHORT-HAUL ROUTES")
    print("=" * 80)
    print()
    print(f"Filter criteria: Distance <= {max_distance_nm} nm")
    print(f"Total routes available: {len(all_routes)}")
    print()

    print("Calculating distances for all routes...")
    short_haul_routes = []

    for i, route in enumerate(all_routes):
        # Calculate great-circle distance
        distance_nm = route.calculate_great_circle_distance()

        if distance_nm <= max_distance_nm:
            short_haul_routes.append(route)

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_routes)} routes... "
                  f"({len(short_haul_routes)} short-haul found)")

    print()
    print(f"✓ Filtered: {len(short_haul_routes)} short-haul routes (<= {max_distance_nm} nm)")
    print(f"  Filtered out: {len(all_routes) - len(short_haul_routes)} long-haul routes")
    print()

    # Show distance distribution
    if short_haul_routes:
        distances = [r.calculate_great_circle_distance() for r in short_haul_routes]
        print("Distance distribution:")
        print(f"  Min: {min(distances):.0f} nm")
        print(f"  Max: {max(distances):.0f} nm")
        print(f"  Average: {sum(distances) / len(distances):.0f} nm")
        print(f"  Median: {sorted(distances)[len(distances) // 2]:.0f} nm")
        print()

        # Show sample routes
        print("Sample short-haul routes:")
        for i, route in enumerate(sorted(short_haul_routes, key=lambda r: r.calculate_great_circle_distance())[:10]):
            dist = route.calculate_great_circle_distance()
            print(f"  {i+1}. {route.departure_icao} → {route.arrival_icao}: {dist:.0f} nm ({route.aircraft_type})")
        print()

    return short_haul_routes


def run_scenario(scenario_config, config, routes):
    """
    Run single short-haul baseline scenario.

    Args:
        scenario_config: Scenario configuration dict
        config: Global configuration dict
        routes: List of short-haul routes

    Returns:
        Dictionary with simulation results
    """
    print("=" * 80)
    print(f"SCENARIO {scenario_config['id']}: {scenario_config['name'].upper()}")
    print("=" * 80)
    print()
    print(f"Description: {scenario_config['description']}")
    print(f"Staffing Level: {scenario_config['staffing_level']}")
    print(f"Delay Profile: {scenario_config['delay_profile']}")
    print()

    # Simulation parameters
    num_routes = config['simulation_parameters']['num_routes_per_scenario']
    duration = config['simulation_parameters']['simulation_duration']
    seed = config['simulation_parameters']['random_seed']
    time_step = config['simulation_parameters']['time_step']
    progress_interval = config['simulation_parameters']['progress_interval']
    departure_min, departure_max = config['simulation_parameters']['departure_interval_range']

    # Set random seed for reproducibility (unique per scenario)
    scenario_seed = seed + ord(scenario_config['id'])
    random.seed(scenario_seed)

    print(f"Simulation parameters:")
    print(f"  Aircraft: {num_routes}")
    print(f"  Duration: {duration}s ({duration / 60:.0f} minutes)")
    print(f"  Departure spacing: {departure_min}-{departure_max} seconds")
    print(f"  Seed: {scenario_seed}")
    print()

    # Select random routes
    if len(routes) >= num_routes:
        selected_routes = random.sample(routes, num_routes)
    else:
        selected_routes = routes
        print(f"⚠ WARNING: Only {len(routes)} short-haul routes available (requested {num_routes})")

    print(f"✓ Selected {len(selected_routes)} routes")
    print()

    # Show route distance distribution for this scenario
    distances = [r.calculate_great_circle_distance() for r in selected_routes]
    print("Selected route distances:")
    print(f"  Min: {min(distances):.0f} nm")
    print(f"  Max: {max(distances):.0f} nm")
    print(f"  Average: {sum(distances) / len(distances):.0f} nm")
    print()

    # Create components
    print("Setting up simulation components...")
    airspace = Airspace()
    delay_model = DelayModel(seed=scenario_seed)
    controller_staffing = ControllerStaffing(scenario_config['staffing_level'])

    # Create output (file mode)
    output_dir = project_root / config['output_directories']['baseline_results']
    output = SimulationOutput(
        output_dir=str(output_dir / "output"),
        output_mode="file",
        center_lat=airspace.center_lat,
        center_lon=airspace.center_lon
    )

    # Create simulation manager
    manager = SimulationManager(airspace, delay_model, output, controller_staffing)

    staffing_desc = controller_staffing.get_staffing_description()
    print(f"  Staffing: {staffing_desc['description']}")
    print(f"  Controllers: {staffing_desc['num_controllers']}")
    print(f"  Capacity: {staffing_desc['capacity']} aircraft")
    print()

    # Add aircraft with staggered departures
    print("=" * 80)
    print("ADDING AIRCRAFT TO SIMULATION")
    print("=" * 80)
    print()

    departure_time = 0.0
    added_count = 0

    for i, route in enumerate(selected_routes):
        aircraft_id = manager.add_aircraft(route, departure_time)
        added_count += 1

        # Show first 5 and periodic updates
        if i < 5 or (i + 1) % 20 == 0 or i == len(selected_routes) - 1:
            dist = route.calculate_great_circle_distance()
            print(f"  [{i + 1}/{len(selected_routes)}] Added {aircraft_id}: "
                  f"{route.departure_icao}→{route.arrival_icao} "
                  f"({dist:.0f}nm, departs t={departure_time:.0f}s)")

        departure_interval = random.uniform(departure_min, departure_max)
        departure_time += departure_interval

    print()
    print(f"✓ Successfully added {added_count} aircraft")
    print(f"  Departure window: {(departure_time - departure_interval) / 60:.1f} minutes")
    print()

    # Run simulation
    print("=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    print()
    print(f"Duration: {duration}s ({duration / 60:.0f} minutes)")
    print(f"Progress updates every {progress_interval}s")
    print()
    print("Starting simulation...")
    print()

    stats = manager.run(
        duration_seconds=duration,
        time_step=time_step,
        progress_interval=progress_interval
    )

    # Close output
    manager.output.close()

    print()
    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print()

    # Print detailed results
    pm = stats['performance_metrics']
    staffing_info = stats['staffing']

    print("FLIGHT STATISTICS:")
    print(f"  Total aircraft spawned: {stats['total_flights']}")
    print(f"  Flights completed: {stats['completed_flights']}")
    print(f"  Still active: {stats['active_flights']}")
    print(f"  Still waiting: {stats['waiting_flights']}")
    print()

    # Celebrate if flights completed!
    if stats['completed_flights'] > 0:
        print(f"✓ SUCCESS: {stats['completed_flights']} flights completed!")
        print(f"  Completion rate: {stats['completed_flights'] / stats['total_flights'] * 100:.1f}%")
        print()

    print("PERFORMANCE METRICS:")
    print(f"  Safety Score: {pm['safety_score']:.1f}/100")
    print()
    print(f"  Conflicts:")
    print(f"    Total: {pm['conflict_metrics']['total_conflicts']:.0f}")
    print(f"    Per Hour: {pm['conflict_metrics']['conflicts_per_hour']:.2f}")
    print(f"    Critical: {pm['conflict_metrics']['critical_conflicts']:.0f}")
    print(f"    Warning: {pm['conflict_metrics']['warning_conflicts']:.0f}")
    print(f"    Near: {pm['conflict_metrics']['near_conflicts']:.0f}")
    print()

    # Celebrate if conflicts detected!
    if pm['conflict_metrics']['total_conflicts'] > 0:
        print(f"✓ CONFLICTS DETECTED: {pm['conflict_metrics']['total_conflicts']:.0f} total")
        print(f"  This is good - it means aircraft are actually interacting!")
        print()

    print(f"  On-Time Performance:")
    print(f"    Completed: {pm['on_time_performance']['total_flights_completed']}")
    print(f"    On-Time %: {pm['on_time_performance']['on_time_percentage']:.1f}%")
    print(f"    Avg Delay: {pm['on_time_performance']['average_delay_minutes']:.1f} min")
    print()
    print(f"  Throughput:")
    print(f"    Flights/Hour: {pm['throughput']['flights_per_hour']:.2f}")
    print(f"    Peak Concurrent: {int(pm['throughput']['peak_concurrent_aircraft'])} aircraft")
    print(f"    Avg Concurrent: {pm['throughput']['average_concurrent_aircraft']:.2f} aircraft")
    print()
    print(f"  Controller Workload:")
    print(f"    Avg: {staffing_info['average_workload_factor']:.2f}x")
    print(f"    Peak: {staffing_info['peak_workload_factor']:.2f}x")
    print(f"    Overloaded: {staffing_info['time_overloaded_percent']:.1f}%")
    print()

    # Add scenario metadata
    result = {
        'scenario_id': scenario_config['id'],
        'scenario_name': scenario_config['name'],
        'staffing_level': scenario_config['staffing_level'],
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'num_routes': len(selected_routes),
            'duration': duration,
            'seed': scenario_seed,
            'delay_profile': scenario_config['delay_profile'],
            'max_route_distance': max(distances),
            'min_route_distance': min(distances),
            'avg_route_distance': sum(distances) / len(distances)
        },
        'results': stats
    }

    return result


def save_scenario_result(result, config):
    """Save individual scenario result to JSON."""
    output_dir = project_root / config['output_directories']['baseline_results']
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"scenario_{result['scenario_id']}_{result['scenario_name'].lower().replace(' ', '_')}.json"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Saved: {filepath}")
    print()


def generate_summary_csv(all_results, config):
    """Generate summary statistics CSV."""
    output_dir = project_root / config['output_directories']['baseline_results']
    csv_path = output_dir / "baseline_summary.csv"

    headers = [
        'Scenario ID',
        'Scenario Name',
        'Staffing Level',
        'Controllers',
        'Capacity',
        'Safety Score',
        'Total Conflicts',
        'Conflicts/Hour',
        'Critical Conflicts',
        'Flights Completed',
        'On-Time %',
        'Avg Delay (min)',
        'Peak Concurrent',
        'Avg Workload',
        'Peak Workload'
    ]

    rows = []
    for result in all_results:
        pm = result['results']['performance_metrics']
        staffing = result['results']['staffing']

        row = [
            result['scenario_id'],
            result['scenario_name'],
            result['staffing_level'],
            staffing['num_controllers'],
            staffing['capacity'],
            f"{pm['safety_score']:.1f}",
            f"{pm['conflict_metrics']['total_conflicts']:.0f}",
            f"{pm['conflict_metrics']['conflicts_per_hour']:.2f}",
            f"{pm['conflict_metrics']['critical_conflicts']:.0f}",
            pm['on_time_performance']['total_flights_completed'],
            f"{pm['on_time_performance']['on_time_percentage']:.1f}",
            f"{pm['on_time_performance']['average_delay_minutes']:.1f}",
            f"{int(pm['throughput']['peak_concurrent_aircraft'])}",
            f"{staffing['average_workload_factor']:.2f}",
            f"{staffing['peak_workload_factor']:.2f}"
        ]
        rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"✓ Summary CSV saved: {csv_path}")
    print()


def print_final_summary(all_results):
    """Print final comparison summary."""
    print("=" * 80)
    print("SHORT-HAUL BASELINE COMPARISON")
    print("=" * 80)
    print()

    print("| Scenario | Safety | Conflicts | Completed | Peak Concurrent |")
    print("|----------|--------|-----------|-----------|-----------------|")

    for result in all_results:
        pm = result['results']['performance_metrics']
        print(f"| {result['scenario_id']}: {result['scenario_name']:<15} | "
              f"{pm['safety_score']:6.1f} | "
              f"{pm['conflict_metrics']['total_conflicts']:9.0f} | "
              f"{pm['on_time_performance']['total_flights_completed']:9} | "
              f"{int(pm['throughput']['peak_concurrent_aircraft']):15} |")

    print()

    # Calculate shutdown degradation
    full_result = next((r for r in all_results if r['scenario_id'] == 'A'), None)
    shutdown_result = next((r for r in all_results if r['scenario_id'] == 'D'), None)

    if full_result and shutdown_result:
        print("KEY FINDINGS (SHORT-HAUL BASELINE):")
        print()

        full_pm = full_result['results']['performance_metrics']
        shutdown_pm = shutdown_result['results']['performance_metrics']

        # Safety
        safety_drop = full_pm['safety_score'] - shutdown_pm['safety_score']
        print(f"1. Safety Impact:")
        print(f"   Full staffing: {full_pm['safety_score']:.1f}")
        print(f"   Shutdown: {shutdown_pm['safety_score']:.1f}")
        print(f"   Drop: {safety_drop:.1f} points")
        print()

        # Conflicts
        full_conflicts = full_pm['conflict_metrics']['conflicts_per_hour']
        shutdown_conflicts = shutdown_pm['conflict_metrics']['conflicts_per_hour']
        if full_conflicts > 0:
            conflict_increase = ((shutdown_conflicts - full_conflicts) / full_conflicts) * 100
            print(f"2. Conflict Impact:")
            print(f"   Full: {full_conflicts:.2f} conflicts/hour")
            print(f"   Shutdown: {shutdown_conflicts:.2f} conflicts/hour")
            print(f"   Increase: {conflict_increase:+.0f}%")
            print()

        # Completions
        full_completed = full_pm['on_time_performance']['total_flights_completed']
        shutdown_completed = shutdown_pm['on_time_performance']['total_flights_completed']
        print(f"3. Throughput:")
        print(f"   Full: {full_completed} flights completed")
        print(f"   Shutdown: {shutdown_completed} flights completed")
        print()


def main():
    """Run all short-haul baseline scenarios."""
    print("\n")
    print("*" * 80)
    print("* DeepSky ATC - Short-Haul Baseline Simulation (Quick Validation)")
    print("* Domestic routes only (<800nm) for fast conflict testing")
    print("*" * 80)
    print()

    # Load configuration
    print("=" * 80)
    print("LOADING CONFIGURATION")
    print("=" * 80)
    print()
    config = load_config()
    print(f"✓ Configuration: {config['name']}")
    print(f"  Purpose: {config['purpose']}")
    print(f"  Scenarios: {len(config['scenarios'])}")
    print(f"  Aircraft per scenario: {config['simulation_parameters']['num_routes_per_scenario']}")
    print(f"  Simulation duration: {config['simulation_parameters']['simulation_duration']}s ({config['simulation_parameters']['simulation_duration'] / 60:.0f} min)")
    print(f"  Max route distance: {config['route_filter']['max_distance_nm']} nm")
    print()

    # Load routes
    print("=" * 80)
    print("LOADING FLIGHT ROUTES")
    print("=" * 80)
    print()
    all_routes = load_routes_from_data()

    if len(all_routes) == 0:
        print("✗ ERROR: No routes available!")
        print("  Run: python src/data_acquisition.py")
        return 1

    print(f"✓ Loaded {len(all_routes)} total routes")
    print()

    # Filter for short-haul
    short_haul_routes = filter_short_haul_routes(
        all_routes,
        config['route_filter']['max_distance_nm']
    )

    if len(short_haul_routes) < config['route_filter']['min_routes']:
        print(f"✗ ERROR: Not enough short-haul routes!")
        print(f"  Found: {len(short_haul_routes)}")
        print(f"  Required: {config['route_filter']['min_routes']}")
        return 1

    # Run each scenario
    print("=" * 80)
    print(f"RUNNING {len(config['scenarios'])} SHORT-HAUL SCENARIOS")
    print("=" * 80)
    print()

    all_results = []

    for scenario_config in config['scenarios']:
        try:
            result = run_scenario(scenario_config, config, short_haul_routes)
            save_scenario_result(result, config)
            all_results.append(result)

        except Exception as e:
            print()
            print(f"✗ ERROR in scenario {scenario_config['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary outputs
    print("=" * 80)
    print("GENERATING SUMMARY OUTPUTS")
    print("=" * 80)
    print()

    generate_summary_csv(all_results, config)

    # Comparison report
    comparison_results = [r['results'] for r in all_results]
    report_dir = project_root / config['output_directories']['reports']
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "baseline_comparison.md"

    generate_staffing_comparison_report(comparison_results, str(report_path))

    # Print final summary
    print()
    print_final_summary(all_results)

    print("=" * 80)
    print("SHORT-HAUL BASELINE COMPLETE")
    print("=" * 80)
    print()
    print("Outputs:")
    print(f"  Results: {config['output_directories']['baseline_results']}")
    print(f"  CSV: {config['output_directories']['baseline_results']}/baseline_summary.csv")
    print(f"  Report: {config['output_directories']['reports']}/baseline_comparison.md")
    print()
    print("Next steps:")
    print("  1. Review results for conflicts and completions")
    print("  2. Run analysis: python scripts/analyze_baseline.py --input data/baseline_short_haul/")
    print("  3. For full baseline: python scripts/run_baseline.py")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
