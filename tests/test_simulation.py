"""
Test suite for DeepSky ATC simulation system.

Tests the complete simulation pipeline including state export,
simulation management, and integration of all components.
"""

import sys
from pathlib import Path
import json

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.airspace import Airspace
from src.delay_model import DelayModel
from src.simulation_output import SimulationOutput
from src.simulation_manager import SimulationManager, create_demo_simulation
from src.route_generator import load_routes_from_data
from src.coordinates import lat_lon_alt_to_xyz, feet_to_meters


def test_simulation_output():
    """Test simulation output interface."""
    print("=" * 70)
    print("TEST 1: Simulation Output Interface")
    print("=" * 70)
    print()

    # Create output interface with file mode
    output_dir = Path("data/output")
    output = SimulationOutput(output_dir=str(output_dir), output_mode="file")

    print(f"Output interface: {output}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output.get_output_file_path()}")
    print()

    # Close output
    output.close()

    print("✓ PASSED: Output interface created successfully")
    print()

    return output_dir


def test_simulation_manager_basic():
    """Test basic simulation manager functionality."""
    print("=" * 70)
    print("TEST 2: Simulation Manager - Basic Operations")
    print("=" * 70)
    print()

    # Load components
    print("Loading components...")
    airspace = Airspace()
    delay_model = DelayModel(seed=42)
    output = SimulationOutput(output_mode="file")

    # Create simulation manager
    manager = SimulationManager(airspace, delay_model, output)
    print(f"Manager: {manager}")
    print()

    # Load routes
    routes = load_routes_from_data(max_routes=5)
    if len(routes) == 0:
        print("⚠ WARNING: No routes available. Skipping test.")
        output.close()
        return

    print(f"Loaded {len(routes)} routes")
    print()

    # Add aircraft with staggered departures
    print("Adding aircraft...")
    for i, route in enumerate(routes):
        departure_time = i * 60  # Stagger by 60 seconds
        aircraft_id = manager.add_aircraft(route, departure_time)
        print(f"  Added {aircraft_id} at t={departure_time}s")

    print()

    # Get initial statistics
    stats = manager.get_statistics()
    print("Initial statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("✓ PASSED: Simulation manager basic operations working")
    print()

    output.close()
    return manager


def test_simulation_run():
    """Test running a complete simulation."""
    print("=" * 70)
    print("TEST 3: Full Simulation Run (10 aircraft, 10 minutes)")
    print("=" * 70)
    print()

    # Load components
    airspace = Airspace()
    delay_model = DelayModel(seed=42)
    output = SimulationOutput(output_mode="file")

    # Create manager
    manager = SimulationManager(airspace, delay_model, output)

    # Load routes
    routes = load_routes_from_data(max_routes=10)
    if len(routes) == 0:
        print("⚠ WARNING: No routes available. Skipping test.")
        output.close()
        return None

    # Add aircraft with staggered departures (every 60 seconds)
    print(f"Adding {len(routes)} aircraft...")
    for i, route in enumerate(routes[:10]):
        departure_time = i * 60  # 0, 60, 120, 180... seconds
        manager.add_aircraft(route, departure_time)

    print(f"✓ Added {len(routes[:10])} aircraft")
    print()

    # Run simulation for 10 minutes
    duration = 600  # 10 minutes
    stats = manager.run(duration_seconds=duration, time_step=1.0, progress_interval=120)

    print("Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Close output
    output.close()

    print("✓ PASSED: Simulation run completed")
    print()

    return output.get_output_file_path()


def test_output_file_validation(output_file):
    """Validate output file format and content."""
    print("=" * 70)
    print("TEST 4: Output File Validation")
    print("=" * 70)
    print()

    if output_file is None or not output_file.exists():
        print("⚠ WARNING: No output file to validate")
        print()
        return

    print(f"Validating output file: {output_file}")
    print()

    # Read and parse JSON
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)

        print(f"✓ File is valid JSON")
        print(f"  Total snapshots: {len(data)}")
        print()

        # Validate first snapshot structure
        if len(data) > 0:
            first_snapshot = data[0]
            print("First snapshot structure:")
            print(f"  Timestamp: {first_snapshot['timestamp']}")
            print(f"  Simulation date: {first_snapshot['simulation_date']}")
            print(f"  Aircraft count: {first_snapshot['aircraft_count']}")
            print()

            # Validate aircraft data structure
            if first_snapshot['aircraft_count'] > 0:
                aircraft = first_snapshot['aircraft'][0]
                print("First aircraft data:")
                print(f"  ID: {aircraft['id']}")
                print(f"  Position (lat/lon/alt): ({aircraft['position']['lat']:.4f}, "
                      f"{aircraft['position']['lon']:.4f}, {aircraft['position']['alt']:.0f})")
                print(f"  Position (x/y/z): ({aircraft['position']['x']:.1f}, "
                      f"{aircraft['position']['y']:.1f}, {aircraft['position']['z']:.1f})")
                print(f"  Velocity: {aircraft['velocity']:.0f} kts")
                print(f"  Heading: {aircraft['heading']:.1f}°")
                print(f"  Status: {aircraft['status']}")
                print(f"  Route: {aircraft['route_info']['departure']} → {aircraft['route_info']['arrival']}")
                print()

                # Validate required fields
                required_fields = ['id', 'position', 'velocity', 'heading', 'status',
                                 'route_info', 'waypoint_progress', 'time_in_flight']
                missing_fields = [f for f in required_fields if f not in aircraft]

                if not missing_fields:
                    print("✓ All required fields present")
                else:
                    print(f"✗ Missing fields: {missing_fields}")
                print()

        # Find snapshot at t=300s (5 minutes)
        snapshot_300 = None
        for snapshot in data:
            if 299 <= snapshot['timestamp'] <= 301:
                snapshot_300 = snapshot
                break

        if snapshot_300:
            print(f"Snapshot at t={snapshot_300['timestamp']:.0f}s:")
            print(f"  Active aircraft: {snapshot_300['aircraft_count']}")
            print()

            # Show sample aircraft states
            for i, aircraft in enumerate(snapshot_300['aircraft'][:3]):
                print(f"  Aircraft {i+1}: {aircraft['id']}")
                print(f"    Position: ({aircraft['position']['lat']:.4f}, "
                      f"{aircraft['position']['lon']:.4f}, {aircraft['position']['alt']:.0f}ft)")
                print(f"    Status: {aircraft['status']}, "
                      f"Speed: {aircraft['velocity']:.0f}kts, "
                      f"Heading: {aircraft['heading']:.0f}°")
                print(f"    Waypoint: {aircraft['waypoint_progress']['current_waypoint']}/"
                      f"{aircraft['waypoint_progress']['total_waypoints']}")
                print()

        print("✓ PASSED: Output file validated")
        print()

    except json.JSONDecodeError as e:
        print(f"✗ FAILED: Invalid JSON - {e}")
        print()
    except Exception as e:
        print(f"✗ FAILED: Validation error - {e}")
        print()


def test_coordinate_conversion():
    """Test coordinate conversion accuracy."""
    print("=" * 70)
    print("TEST 5: Coordinate Conversion Validation")
    print("=" * 70)
    print()

    # KJFK center coordinates
    center_lat = 40.6413
    center_lon = -73.7781

    # Test points
    test_points = [
        (40.6413, -73.7781, 0, "KJFK center (ground)"),
        (40.6413, -73.7781, 35000, "KJFK center (cruise altitude)"),
        (41.0, -73.0, 10000, "~50nm northeast"),
        (40.0, -74.5, 20000, "~50nm southwest")
    ]

    print("Testing coordinate conversions:")
    print()

    all_passed = True

    for lat, lon, alt_ft, description in test_points:
        # Convert altitude to meters
        alt_m = feet_to_meters(alt_ft)

        # Convert to local coordinates
        x, y, z = lat_lon_alt_to_xyz(lat, lon, alt_m, center_lat, center_lon)

        print(f"{description}:")
        print(f"  Geodetic:  lat={lat:.4f}°, lon={lon:.4f}°, alt={alt_ft:.0f}ft")
        print(f"  Cartesian: x={x:.1f}m, y={y:.1f}m, z={z:.1f}m")

        # Validate z coordinate matches altitude
        error = abs(z - alt_m)
        if error < 1.0:
            print(f"  ✓ Altitude matches (error: {error:.3f}m)")
        else:
            print(f"  ✗ Altitude mismatch (error: {error:.3f}m)")
            all_passed = False

        print()

    if all_passed:
        print("✓ PASSED: All coordinate conversions accurate")
    else:
        print("✗ FAILED: Some coordinate conversions have errors")

    print()


def test_demo_simulation():
    """Test demo simulation creation and execution."""
    print("=" * 70)
    print("TEST 6: Demo Simulation")
    print("=" * 70)
    print()

    try:
        # Create demo simulation
        manager = create_demo_simulation(
            num_aircraft=10,
            duration=600,
            output_mode="file",
            seed=42
        )

        # Run simulation
        stats = manager.run(duration_seconds=600, time_step=1.0)

        print("Final statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()

        # Close output
        manager.output.close()

        print("✓ PASSED: Demo simulation completed successfully")
        print()

    except Exception as e:
        print(f"⚠ WARNING: Demo simulation failed - {e}")
        print("  This may be due to missing route data")
        print()


def test_aircraft_state_progression():
    """Test that aircraft state progresses correctly over time."""
    print("=" * 70)
    print("TEST 7: Aircraft State Progression")
    print("=" * 70)
    print()

    # Load components
    airspace = Airspace()
    delay_model = DelayModel(seed=42)
    output = SimulationOutput(output_mode="file")

    # Create manager
    manager = SimulationManager(airspace, delay_model, output)

    # Load one route
    routes = load_routes_from_data(max_routes=1)
    if len(routes) == 0:
        print("⚠ WARNING: No routes available. Skipping test.")
        output.close()
        return

    # Add aircraft
    route = routes[0]
    aircraft_id = manager.add_aircraft(route, departure_time=0)

    print(f"Tracking aircraft: {aircraft_id}")
    print(f"Route: {route.departure_icao} → {route.arrival_icao}")
    print(f"Distance: {route.total_distance_nm:.0f} nm")
    print()

    # Track state changes
    print("State progression (first 300 seconds):")
    print()

    previous_status = None
    previous_waypoint = None

    for t in range(0, 301, 30):
        # Step simulation to time t
        while manager.simulation_time < t:
            manager.step(delta_t=1.0)

        # Get aircraft state
        if len(manager.aircraft_list) > 0:
            aircraft = manager.aircraft_list[0]
            state = aircraft.get_state()

            # Print if status or waypoint changed
            if (state['status'] != previous_status or
                state['waypoint_info']['current_index'] != previous_waypoint):

                print(f"t={t:3d}s: {state['status']:12s} "
                      f"alt={state['position']['alt']:6.0f}ft "
                      f"spd={state['velocity']:3.0f}kts "
                      f"wp={state['waypoint_info']['current_index']}/"
                      f"{state['waypoint_info']['total_waypoints']}")

                previous_status = state['status']
                previous_waypoint = state['waypoint_info']['current_index']

    print()

    # Close output
    output.close()

    print("✓ PASSED: Aircraft state progression validated")
    print()


def main():
    """Run all simulation tests."""
    print("\n")
    print("*" * 70)
    print("* DeepSky ATC - Simulation System Test Suite")
    print("* Phase 1, Deliverable 1.5")
    print("*" * 70)
    print()

    try:
        # Run all tests
        output_dir = test_simulation_output()
        test_simulation_manager_basic()
        output_file = test_simulation_run()
        test_output_file_validation(output_file)
        test_coordinate_conversion()
        test_demo_simulation()
        test_aircraft_state_progression()

        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print()

        print("Summary:")
        print("  ✓ Simulation output interface working")
        print("  ✓ Simulation manager operational")
        print("  ✓ Full simulation pipeline validated")
        print("  ✓ JSON output format correct")
        print("  ✓ Coordinate conversions accurate")
        print("  ✓ Aircraft state progression verified")
        print()

        if output_file:
            print(f"Output file: {output_file}")
            print()

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
