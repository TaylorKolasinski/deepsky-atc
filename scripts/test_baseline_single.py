"""
Test baseline simulation with single scenario (Government Shutdown).

Quick test to verify aircraft are loading and simulating correctly.

Usage:
    python scripts/test_baseline_single.py
"""

import sys
from pathlib import Path
import json
import random

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.airspace import Airspace
from src.delay_model import DelayModel
from src.simulation_output import SimulationOutput
from src.simulation_manager import SimulationManager
from src.controller_capacity import ControllerStaffing
from src.route_generator import load_routes_from_data


def main():
    """Run single scenario test (Government Shutdown)."""
    print("\n")
    print("*" * 80)
    print("* DeepSky ATC - Baseline Test (Single Scenario)")
    print("* Testing: Government Shutdown Scenario")
    print("*" * 80)
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
    print(f"  Total routes: {len(all_routes)}")
    print(f"  Average distance: {avg_dist:.0f} nm")
    print()

    # Sample routes
    print("Sample routes:")
    for i, route in enumerate(all_routes[:5]):
        print(f"  {i+1}. {route.departure_icao} → {route.arrival_icao}: "
              f"{route.total_distance_nm:.0f} nm ({route.aircraft_type})")
    if len(all_routes) > 5:
        print(f"  ... and {len(all_routes) - 5} more")
    print()

    # Select routes
    print("=" * 80)
    print("SELECTING ROUTES FOR SIMULATION")
    print("=" * 80)
    print()

    num_routes = 50
    seed = 42
    random.seed(seed)

    print(f"Requesting: {num_routes} routes")
    print(f"Random seed: {seed}")
    print()

    if len(all_routes) >= num_routes:
        selected_routes = random.sample(all_routes, num_routes)
    else:
        selected_routes = all_routes
        print(f"⚠ WARNING: Only {len(all_routes)} routes available")

    print(f"✓ Selected {len(selected_routes)} routes")
    print()

    print("Selected routes:")
    for i, route in enumerate(selected_routes[:10]):
        print(f"  {i+1}. {route.departure_icao} → {route.arrival_icao}: "
              f"{route.total_distance_nm:.0f} nm")
    if len(selected_routes) > 10:
        print(f"  ... and {len(selected_routes) - 10} more")
    print()

    # Setup simulation
    print("=" * 80)
    print("SETTING UP SIMULATION")
    print("=" * 80)
    print()

    print("Creating simulation components...")
    airspace = Airspace()
    delay_model = DelayModel(seed=seed)
    controller_staffing = ControllerStaffing("shutdown")

    output = SimulationOutput(
        output_dir="data/baseline/output_test",
        output_mode="file",
        center_lat=airspace.center_lat,
        center_lon=airspace.center_lon
    )

    manager = SimulationManager(airspace, delay_model, output, controller_staffing)

    staffing_desc = controller_staffing.get_staffing_description()
    print(f"✓ Simulation components created")
    print(f"  Staffing: {staffing_desc['description']}")
    print(f"  Controllers: {staffing_desc['num_controllers']}")
    print(f"  Capacity: {staffing_desc['capacity']} aircraft")
    print()

    # Add aircraft
    print("=" * 80)
    print("ADDING AIRCRAFT TO SIMULATION")
    print("=" * 80)
    print()

    departure_min, departure_max = 30, 90
    print(f"Adding {len(selected_routes)} aircraft...")
    print(f"Departure interval: {departure_min}-{departure_max} seconds")
    print()

    departure_time = 0.0
    added_count = 0

    for i, route in enumerate(selected_routes):
        aircraft_id = manager.add_aircraft(route, departure_time)
        added_count += 1

        # Show first 5 and last
        if i < 5 or i == len(selected_routes) - 1:
            print(f"  [{i + 1}/{len(selected_routes)}] Added {aircraft_id}: "
                  f"{route.departure_icao}→{route.arrival_icao} "
                  f"(departs at t={departure_time:.0f}s, "
                  f"distance={route.total_distance_nm:.0f}nm)")

        departure_interval = random.uniform(departure_min, departure_max)
        departure_time += departure_interval

    print()
    print(f"✓ Successfully added {added_count} aircraft")
    print(f"  Total in manager: {manager.total_aircraft_spawned}")
    print(f"  Departure window: {(departure_time - departure_interval) / 60:.1f} minutes")
    print()

    # Run simulation
    print("=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    print()

    duration = 3600
    time_step = 1.0
    progress_interval = 300

    print(f"Duration: {duration}s ({duration / 60:.0f} minutes)")
    print(f"Time step: {time_step}s")
    print(f"Progress updates every: {progress_interval}s")
    print()
    print("Starting simulation...")
    print()

    stats = manager.run(
        duration_seconds=duration,
        time_step=time_step,
        progress_interval=progress_interval
    )

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
    print(f"  Still waiting to depart: {stats['waiting_flights']}")
    print()

    if stats['completed_flights'] == 0:
        print("⚠ WARNING: No flights completed during 1-hour simulation!")
        print()
        print("  Analysis:")
        print(f"  - Average route distance: {avg_dist:.0f} nm")
        print(f"  - Estimated flight time at 450 knots: {avg_dist / 450:.1f} hours")
        print(f"  - Simulation duration: {duration / 3600:.1f} hours")
        print()
        print("  This is EXPECTED because:")
        print("  - Most JFK routes are international (Europe, Asia, South America)")
        print("  - International flights take 2-6 hours")
        print("  - We only simulated 1 hour")
        print()
        print("  Solutions for Phase 3:")
        print("  1. Increase simulation duration to 4-6 hours")
        print("  2. Filter for domestic routes (<1000 nm)")
        print("  3. Increase aircraft density (more concurrent traffic)")
        print()

    print("PERFORMANCE METRICS:")
    print(f"  Safety Score: {pm['safety_score']:.1f}/100")
    print()
    print(f"  Conflicts:")
    print(f"    Total: {pm['conflict_metrics']['total_conflicts']:.0f}")
    print(f"    Per Hour: {pm['conflict_metrics']['conflicts_per_hour']:.2f}")
    print(f"    Critical: {pm['conflict_metrics']['critical_conflicts']:.0f}")
    print()
    print(f"  Throughput:")
    print(f"    Peak Concurrent: {int(pm['throughput']['peak_concurrent_aircraft'])} aircraft")
    print(f"    Avg Concurrent: {pm['throughput']['average_concurrent_aircraft']:.2f} aircraft")
    print()
    print(f"  Controller Workload:")
    print(f"    Avg: {staffing_info['average_workload_factor']:.2f}x")
    print(f"    Peak: {staffing_info['peak_workload_factor']:.2f}x")
    print(f"    Time Overloaded: {staffing_info['time_overloaded_percent']:.1f}%")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()

    if stats['total_flights'] == 50 and stats['total_flights'] == added_count:
        print("✓ SUCCESS: All aircraft properly added and tracked")
    else:
        print(f"⚠ MISMATCH: Added {added_count} but manager reports {stats['total_flights']}")

    print()
    print("Next steps:")
    print("  - If this looks correct, run full baseline:")
    print("    python scripts/run_baseline.py")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
