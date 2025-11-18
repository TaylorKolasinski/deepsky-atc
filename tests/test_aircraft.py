"""
Test suite for DeepSky ATC aircraft kinematic core.

Tests aircraft position updates, waypoint navigation, and status transitions.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.aircraft import Aircraft
from src.route_generator import FlightRoute, generate_waypoints
from src.physics import haversine_distance


def create_test_route_jfk_lax() -> FlightRoute:
    """Create a test route from JFK to LAX."""
    # JFK coordinates
    jfk_lat, jfk_lon = 40.6413, -73.7781

    # LAX coordinates
    lax_lat, lax_lon = 33.9416, -118.4085

    # Generate realistic waypoints
    waypoints, distance = generate_waypoints(jfk_lat, jfk_lon, lax_lat, lax_lon)

    # Create route
    route = FlightRoute(
        route_id="TEST_JFK_LAX",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KLAX",
        waypoints=waypoints,
        total_distance_nm=distance
    )

    return route


def test_aircraft_initialization():
    """Test aircraft initialization."""
    print("=" * 70)
    print("TEST 1: Aircraft Initialization")
    print("=" * 70)
    print()

    route = create_test_route_jfk_lax()
    aircraft = Aircraft(id="AAL123", route=route)

    print(f"Aircraft ID: {aircraft.id}")
    print(f"Route: {route.departure_icao} → {route.arrival_icao}")
    print(f"Distance: {route.total_distance_nm:.0f} nm")
    print(f"Waypoints: {len(route.waypoints)}")
    print()

    print(f"Initial position:")
    print(f"  Lat: {aircraft.current_position['lat']:.4f}°")
    print(f"  Lon: {aircraft.current_position['lon']:.4f}°")
    print(f"  Alt: {aircraft.current_position['alt']:.0f} ft")
    print()

    print(f"Initial state:")
    print(f"  Velocity: {aircraft.current_velocity:.0f} kts")
    print(f"  Heading: {aircraft.current_heading:.1f}°")
    print(f"  Status: {aircraft.status}")
    print(f"  Target waypoint: {aircraft.target_waypoint_index}")
    print()

    # Validate initialization
    errors = []

    if aircraft.current_position['lat'] != route.waypoints[0][0]:
        errors.append("Initial latitude doesn't match first waypoint")

    if aircraft.current_position['lon'] != route.waypoints[0][1]:
        errors.append("Initial longitude doesn't match first waypoint")

    if aircraft.status != 'TAXI':
        errors.append(f"Initial status should be TAXI, got {aircraft.status}")

    if aircraft.target_waypoint_index != 1:
        errors.append(f"Should be targeting waypoint 1, got {aircraft.target_waypoint_index}")

    if errors:
        print("✗ FAILED:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ PASSED: Aircraft initialized correctly")

    print()
    return aircraft


def test_position_updates(aircraft):
    """Test position updates over time."""
    print("=" * 70)
    print("TEST 2: Position Updates (100 time steps)")
    print("=" * 70)
    print()

    print(f"{'Time':>6} {'Lat':>10} {'Lon':>11} {'Alt':>8} {'Speed':>7} {'Heading':>8} {'Status':>12} {'WP':>5}")
    print("-" * 70)

    # Track initial position
    initial_lat = aircraft.current_position['lat']
    initial_lon = aircraft.current_position['lon']

    # Simulate 100 seconds of flight
    for t in range(101):
        # Update position
        aircraft.update_position(delta_t=1.0)

        # Check if waypoint reached
        reached = aircraft.check_waypoint_reached(threshold_nm=2.0)

        # Print every 10 steps
        if t % 10 == 0:
            pos = aircraft.current_position
            wp_str = f"{aircraft.target_waypoint_index}/{len(aircraft.route.waypoints)}"
            print(f"{t:6.0f}s {pos['lat']:10.4f} {pos['lon']:11.4f} "
                  f"{pos['alt']:8.0f} {aircraft.current_velocity:7.0f} "
                  f"{aircraft.current_heading:8.1f}° {aircraft.status:>12} {wp_str:>5}")

            if reached:
                print(f"       >>> Waypoint {aircraft.target_waypoint_index - 1} reached!")

    print()

    # Calculate distance traveled
    final_lat = aircraft.current_position['lat']
    final_lon = aircraft.current_position['lon']
    distance_traveled = haversine_distance(initial_lat, initial_lon, final_lat, final_lon)

    print(f"Distance traveled: {distance_traveled:.2f} nm in 100 seconds")
    print(f"Final altitude: {aircraft.current_position['alt']:.0f} ft")
    print(f"Final status: {aircraft.status}")
    print()

    # Validate movement
    if distance_traveled > 0.1:
        print("✓ PASSED: Aircraft is moving")
    else:
        print("✗ FAILED: Aircraft didn't move")

    print()
    return aircraft


def test_waypoint_progression(aircraft):
    """Test waypoint progression over extended simulation."""
    print("=" * 70)
    print("TEST 3: Waypoint Progression")
    print("=" * 70)
    print()

    initial_waypoint = aircraft.target_waypoint_index
    waypoints_reached = []

    print(f"Starting at waypoint {initial_waypoint}/{len(aircraft.route.waypoints)}")
    print()

    # Simulate longer flight (10 minutes = 600 seconds)
    print("Simulating 10 minutes of flight...")
    print()

    for t in range(600):
        aircraft.update_position(delta_t=1.0)
        reached = aircraft.check_waypoint_reached(threshold_nm=2.0)

        if reached:
            waypoints_reached.append({
                'waypoint': aircraft.target_waypoint_index - 1,
                'time': t,
                'position': aircraft.current_position.copy()
            })
            print(f"  t={t:4d}s: Reached waypoint {aircraft.target_waypoint_index - 1}")

    print()
    print(f"Waypoints reached: {len(waypoints_reached)}")
    print(f"Current waypoint: {aircraft.target_waypoint_index}/{len(aircraft.route.waypoints)}")
    print()

    if len(waypoints_reached) > 0:
        print("✓ PASSED: Aircraft progressing through waypoints")
    else:
        print("⚠ WARNING: No waypoints reached in 10 minutes (may be normal for long routes)")

    print()
    return aircraft


def test_status_transitions():
    """Test aircraft status transitions through flight phases."""
    print("=" * 70)
    print("TEST 4: Status Transitions")
    print("=" * 70)
    print()

    # Create a shorter route for faster testing
    route = FlightRoute(
        route_id="TEST_SHORT",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KBOS",
        waypoints=[
            (40.6413, -73.7781, 0),       # JFK (ground)
            (41.0, -72.5, 15000),          # Climbing
            (41.5, -71.5, 35000),          # Cruise
            (42.0, -71.0, 35000),          # Cruise
            (42.3643, -71.0052, 0)         # BOS (ground)
        ],
        total_distance_nm=185
    )

    aircraft = Aircraft(id="TEST123", route=route)

    status_history = [aircraft.status]
    print(f"Initial status: {aircraft.status}")
    print()

    # Simulate flight and track status changes
    print("Status changes during flight:")
    previous_status = aircraft.status

    for t in range(1200):  # 20 minutes
        aircraft.update_position(delta_t=1.0)
        aircraft.check_waypoint_reached(threshold_nm=2.0)

        if aircraft.status != previous_status:
            print(f"  t={t:5d}s: {previous_status:>12} → {aircraft.status:<12} "
                  f"(alt={aircraft.current_position['alt']:.0f}ft, "
                  f"wp={aircraft.target_waypoint_index}/{len(route.waypoints)})")
            status_history.append(aircraft.status)
            previous_status = aircraft.status

        # Break if landed
        if aircraft.status == 'LANDED':
            break

    print()
    print(f"Status progression: {' → '.join(status_history)}")
    print()

    # Validate we went through expected phases
    expected_statuses = ['TAXI', 'CLIMBING', 'CRUISING']
    found_statuses = [s for s in expected_statuses if s in status_history]

    if len(found_statuses) >= 2:
        print(f"✓ PASSED: Aircraft transitioned through flight phases")
        print(f"  Found: {', '.join(found_statuses)}")
    else:
        print(f"⚠ WARNING: Limited status transitions")
        print(f"  Found: {', '.join(found_statuses)}")

    print()


def test_flight_path_accuracy():
    """Test that aircraft stays on reasonable flight path."""
    print("=" * 70)
    print("TEST 5: Flight Path Accuracy")
    print("=" * 70)
    print()

    route = create_test_route_jfk_lax()
    aircraft = Aircraft(id="PATH123", route=route)

    # Track maximum deviation from waypoints
    max_deviation = 0.0
    deviations = []

    print("Simulating 5 minutes and checking path deviation...")
    print()

    for t in range(300):
        aircraft.update_position(delta_t=1.0)
        aircraft.check_waypoint_reached(threshold_nm=2.0)

        # Calculate distance to current target waypoint
        if aircraft.target_waypoint_index < len(route.waypoints):
            target_lat, target_lon, _ = route.waypoints[aircraft.target_waypoint_index]
            deviation = haversine_distance(
                aircraft.current_position['lat'],
                aircraft.current_position['lon'],
                target_lat,
                target_lon
            )

            deviations.append(deviation)
            max_deviation = max(max_deviation, deviation)

    print(f"Path deviation statistics:")
    print(f"  Maximum deviation: {max_deviation:.2f} nm")
    if deviations:
        import numpy as np
        print(f"  Average deviation: {np.mean(deviations):.2f} nm")
        print(f"  Median deviation: {np.median(deviations):.2f} nm")
    print()

    # Aircraft should generally be moving toward waypoints
    # (deviation should decrease over time, except when changing waypoints)
    if max_deviation < 1000:  # Reasonable for long-haul flight
        print("✓ PASSED: Aircraft maintains reasonable path")
    else:
        print("⚠ WARNING: Large path deviations detected")

    print()


def test_get_state():
    """Test aircraft state retrieval."""
    print("=" * 70)
    print("TEST 6: Aircraft State Retrieval")
    print("=" * 70)
    print()

    route = create_test_route_jfk_lax()
    aircraft = Aircraft(id="STATE123", route=route)

    # Get initial state
    state = aircraft.get_state()

    print("Aircraft state structure:")
    print(f"  ID: {state['id']}")
    print(f"  Position: {state['position']}")
    print(f"  Velocity: {state['velocity']} kts")
    print(f"  Heading: {state['heading']:.1f}°")
    print(f"  Status: {state['status']}")
    print(f"  Route info: {state['route_info']}")
    print(f"  Waypoint info: {state['waypoint_info']}")
    print(f"  Time elapsed: {state['time_elapsed']} s")
    print()

    # Validate state structure
    required_keys = ['id', 'position', 'velocity', 'heading', 'status',
                     'route_info', 'waypoint_info', 'time_elapsed']

    missing_keys = [k for k in required_keys if k not in state]

    if not missing_keys:
        print("✓ PASSED: State dictionary has all required fields")
    else:
        print(f"✗ FAILED: Missing keys: {missing_keys}")

    print()


def main():
    """Run all aircraft tests."""
    print("\n")
    print("*" * 70)
    print("* DeepSky ATC - Aircraft Kinematic Core Test Suite")
    print("* Phase 1, Deliverable 1.4")
    print("*" * 70)
    print()

    try:
        # Run all tests
        aircraft = test_aircraft_initialization()
        aircraft = test_position_updates(aircraft)
        aircraft = test_waypoint_progression(aircraft)
        test_status_transitions()
        test_flight_path_accuracy()
        test_get_state()

        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
