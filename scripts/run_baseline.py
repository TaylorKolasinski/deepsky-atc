"""
Baseline simulation runner for DeepSky ATC.

Runs comprehensive baseline scenarios across all staffing levels to establish
"no AI control" performance benchmarks for Phase 3 evaluation.

Usage:
    python scripts/run_baseline.py
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
    """Load baseline configuration."""
    config_path = project_root / "data" / "baseline_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def run_scenario(scenario_config, config, routes):
    """
    Run single baseline scenario.

    Args:
        scenario_config: Scenario configuration dict
        config: Global configuration dict
        routes: List of available routes

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

    print(f"DEBUG: Total routes available: {len(routes)}")
    print(f"DEBUG: Requesting {num_routes} routes for scenario")
    print()

    # Select random routes
    if len(routes) >= num_routes:
        selected_routes = random.sample(routes, num_routes)
    else:
        selected_routes = routes
        print(f"⚠ WARNING: Only {len(routes)} routes available (requested {num_routes})")

    print(f"✓ Selected {len(selected_routes)} routes")

    # Show sample routes
    print()
    print("Sample routes:")
    for i, route in enumerate(selected_routes[:3]):
        print(f"  {i+1}. {route.departure_icao} → {route.arrival_icao} "
              f"({route.total_distance_nm:.0f} nm, {route.aircraft_type})")
    if len(selected_routes) > 3:
        print(f"  ... and {len(selected_routes) - 3} more routes")
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
    print(f"  Controllers: {staffing_desc['num_controllers']}")
    print(f"  Capacity: {staffing_desc['capacity']} aircraft")
    print()

    # Add aircraft with staggered departures
    print("=" * 80)
    print("ADDING AIRCRAFT TO SIMULATION")
    print("=" * 80)
    print()
    print(f"Adding {len(selected_routes)} aircraft with staggered departures...")
    print(f"Departure interval: {departure_min}-{departure_max} seconds")
    print()

    departure_time = 0.0
    added_count = 0

    for i, route in enumerate(selected_routes):
        # Add aircraft
        aircraft_id = manager.add_aircraft(route, departure_time)
        added_count += 1

        # Show first few and periodic updates
        if i < 3 or (i + 1) % 10 == 0 or i == len(selected_routes) - 1:
            print(f"  [{i + 1}/{len(selected_routes)}] Added {aircraft_id}: "
                  f"{route.departure_icao}→{route.arrival_icao} "
                  f"(departs at t={departure_time:.0f}s)")

        # Calculate next departure time (random interval)
        departure_interval = random.uniform(departure_min, departure_max)
        departure_time += departure_interval

    print()
    print(f"✓ Successfully added {added_count} aircraft to simulation")
    print(f"  Total aircraft in manager: {manager.total_aircraft_spawned}")
    print(f"  Departure window: {(departure_time - departure_interval) / 60:.1f} minutes")
    print(f"  Last departure: t={departure_time - departure_interval:.0f}s")
    print()

    # Run simulation
    print("=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    print()
    print(f"Simulation duration: {duration}s ({duration / 60:.0f} minutes)")
    print(f"Time step: {time_step}s")
    print(f"Progress updates every: {progress_interval}s")
    print()
    print("Starting simulation loop...")
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

    # Print summary with debugging info
    pm = stats['performance_metrics']
    staffing_info = stats['staffing']

    print("Flight Statistics:")
    print(f"  Total aircraft spawned: {stats['total_flights']}")
    print(f"  Flights completed: {stats['completed_flights']}")
    print(f"  Still active: {stats['active_flights']}")
    print(f"  Still waiting: {stats['waiting_flights']}")
    print()

    # Warning if no flights completed
    if stats['completed_flights'] == 0:
        print("⚠ WARNING: No flights completed!")
        print("  This likely means:")
        print("  - Routes are too long for 1-hour simulation")
        print("  - Most JFK routes are international (2-4+ hour flights)")
        print("  - Consider: longer simulation or shorter domestic routes")
        print()

    print("=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print()
    print(f"Safety Score: {pm['safety_score']:.1f}/100")
    print()
    print(f"Conflicts:")
    print(f"  Total: {pm['conflict_metrics']['total_conflicts']:.0f}")
    print(f"  Per Hour: {pm['conflict_metrics']['conflicts_per_hour']:.2f}")
    print(f"  Critical: {pm['conflict_metrics']['critical_conflicts']:.0f}")
    print(f"  Warning: {pm['conflict_metrics']['warning_conflicts']:.0f}")
    print(f"  Near: {pm['conflict_metrics']['near_conflicts']:.0f}")
    print()
    print(f"On-Time Performance:")
    print(f"  Completed: {pm['on_time_performance']['total_flights_completed']}")
    print(f"  On-Time %: {pm['on_time_performance']['on_time_percentage']:.1f}%")
    print(f"  Avg Delay: {pm['on_time_performance']['average_delay_minutes']:.1f} min")
    print()
    print(f"Throughput:")
    print(f"  Flights/Hour: {pm['throughput']['flights_per_hour']:.2f}")
    print(f"  Peak Concurrent: {int(pm['throughput']['peak_concurrent_aircraft'])} aircraft")
    print(f"  Avg Concurrent: {pm['throughput']['average_concurrent_aircraft']:.2f} aircraft")
    print()
    print(f"Controller Workload:")
    print(f"  Avg: {staffing_info['average_workload_factor']:.2f}x")
    print(f"  Peak: {staffing_info['peak_workload_factor']:.2f}x")
    print(f"  Overloaded: {staffing_info['time_overloaded_percent']:.1f}%")
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
            'delay_profile': scenario_config['delay_profile']
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

    # CSV headers
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
        'Avg Workload',
        'Peak Workload',
        'Time Overloaded %'
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
            f"{staffing['average_workload_factor']:.2f}",
            f"{staffing['peak_workload_factor']:.2f}",
            f"{staffing['time_overloaded_percent']:.1f}"
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
    print("FINAL BASELINE COMPARISON")
    print("=" * 80)
    print()

    print("| Scenario | Safety | Conflicts/Hr | Avg Delay | On-Time % |")
    print("|----------|--------|--------------|-----------|-----------|")

    for result in all_results:
        pm = result['results']['performance_metrics']
        print(f"| {result['scenario_id']}: {result['scenario_name']:<15} | "
              f"{pm['safety_score']:6.1f} | "
              f"{pm['conflict_metrics']['conflicts_per_hour']:12.2f} | "
              f"{pm['on_time_performance']['average_delay_minutes']:9.1f} | "
              f"{pm['on_time_performance']['on_time_percentage']:9.1f}% |")

    print()

    # Calculate shutdown degradation
    full_result = next((r for r in all_results if r['scenario_id'] == 'A'), None)
    shutdown_result = next((r for r in all_results if r['scenario_id'] == 'D'), None)

    if full_result and shutdown_result:
        print("KEY FINDINGS:")
        print()

        full_pm = full_result['results']['performance_metrics']
        shutdown_pm = shutdown_result['results']['performance_metrics']

        # Safety score degradation
        safety_drop = full_pm['safety_score'] - shutdown_pm['safety_score']
        print(f"1. Government Shutdown Impact:")
        print(f"   Safety score dropped {safety_drop:.1f} points "
              f"({full_pm['safety_score']:.1f} → {shutdown_pm['safety_score']:.1f})")

        # Conflict increase
        full_conflicts = full_pm['conflict_metrics']['conflicts_per_hour']
        shutdown_conflicts = shutdown_pm['conflict_metrics']['conflicts_per_hour']

        if full_conflicts > 0:
            conflict_increase = ((shutdown_conflicts - full_conflicts) / full_conflicts) * 100
            print(f"   Conflicts increased {conflict_increase:.0f}% "
                  f"({full_conflicts:.2f} → {shutdown_conflicts:.2f} per hour)")
        else:
            print(f"   Conflicts: {full_conflicts:.2f} → {shutdown_conflicts:.2f} per hour")

        # Delay increase
        full_delay = full_pm['on_time_performance']['average_delay_minutes']
        shutdown_delay = shutdown_pm['on_time_performance']['average_delay_minutes']

        if full_delay != 0:
            delay_increase = ((shutdown_delay - full_delay) / abs(full_delay)) * 100
            print(f"   Delays increased {delay_increase:.0f}% "
                  f"({full_delay:.1f} → {shutdown_delay:.1f} minutes)")
        else:
            print(f"   Delays: {full_delay:.1f} → {shutdown_delay:.1f} minutes")

        print()
        print("2. Phase 3 AI Controller Targets:")
        print(f"   MUST BEAT: Shutdown scenario (safety {shutdown_pm['safety_score']:.1f})")
        print(f"   SHOULD MATCH: Normal ops (safety ~{full_pm['safety_score']:.1f})")
        print(f"   STRETCH GOAL: Exceed full staffing performance")
        print()


def main():
    """Run all baseline scenarios."""
    print("\n")
    print("*" * 80)
    print("* DeepSky ATC - Baseline Simulation Runner")
    print("* Phase 2, Deliverable 2.3")
    print("*" * 80)
    print()

    # Load configuration
    print("=" * 80)
    print("LOADING CONFIGURATION")
    print("=" * 80)
    print()
    config = load_config()
    print(f"✓ Configuration loaded")
    print(f"  Scenarios: {len(config['scenarios'])}")
    print(f"  Aircraft per scenario: {config['simulation_parameters']['num_routes_per_scenario']}")
    print(f"  Simulation duration: {config['simulation_parameters']['simulation_duration']}s")
    print()

    # Load routes
    print("=" * 80)
    print("LOADING FLIGHT ROUTES")
    print("=" * 80)
    print()
    print("Loading routes from data/jfk_routes.csv...")
    all_routes = load_routes_from_data()

    if len(all_routes) == 0:
        print()
        print("✗ ERROR: No routes available!")
        print("  Run data acquisition first:")
        print("  python src/data_acquisition.py")
        return 1

    print(f"✓ Successfully loaded {len(all_routes)} routes")
    print()

    # Show route statistics
    print("Route statistics:")
    total_dist = sum(r.total_distance_nm for r in all_routes)
    avg_dist = total_dist / len(all_routes) if all_routes else 0
    print(f"  Average distance: {avg_dist:.0f} nm")

    # Sample routes
    print()
    print("Sample routes:")
    for i, route in enumerate(all_routes[:5]):
        print(f"  {i+1}. {route.departure_icao} → {route.arrival_icao}: "
              f"{route.total_distance_nm:.0f} nm ({route.aircraft_type})")
    if len(all_routes) > 5:
        print(f"  ... and {len(all_routes) - 5} more")
    print()

    # Run each scenario
    print("=" * 80)
    print(f"RUNNING {len(config['scenarios'])} BASELINE SCENARIOS")
    print("=" * 80)
    print()

    all_results = []

    for scenario_config in config['scenarios']:
        try:
            result = run_scenario(scenario_config, config, all_routes)
            save_scenario_result(result, config)
            all_results.append(result)

        except Exception as e:
            print()
            print("=" * 80)
            print(f"✗ ERROR in scenario {scenario_config['id']}")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # Generate summary outputs
    print("=" * 80)
    print("GENERATING SUMMARY OUTPUTS")
    print("=" * 80)
    print()

    # Summary CSV
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
    print("BASELINE SIMULATIONS COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run: python scripts/analyze_baseline.py")
    print("  2. Review: docs/reports/baseline_comparison.md")
    print("  3. Check: data/baseline/baseline_summary.csv")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
