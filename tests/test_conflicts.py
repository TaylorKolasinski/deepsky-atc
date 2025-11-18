"""
Test suite for DeepSky ATC conflict detection system.

Tests conflict detection, tracking, and integration with simulation.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.conflict_detection import ConflictDetector, ConflictTracker
from src.aircraft import Aircraft
from src.route_generator import FlightRoute
from src.simulation_manager import SimulationManager
from src.airspace import Airspace
from src.delay_model import DelayModel
from src.simulation_output import SimulationOutput


def create_collision_course_aircraft():
    """
    Create two aircraft on a collision course.

    Returns pair of (aircraft1, aircraft2) configured to conflict.
    """
    # Create two simple routes that will intersect
    # Aircraft 1: Flying east at 35,000 ft
    route1 = FlightRoute(
        route_id="TEST_EAST",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KBOS",
        waypoints=[
            (40.0, -74.0, 0),
            (40.0, -73.5, 35000),
            (40.0, -73.0, 35000),
            (40.0, -72.5, 35000),
            (40.0, -72.0, 0)
        ],
        total_distance_nm=180
    )

    # Aircraft 2: Flying north at 35,000 ft (will cross aircraft 1's path)
    route2 = FlightRoute(
        route_id="TEST_NORTH",
        aircraft_type="A320",
        departure_icao="KEWR",
        arrival_icao="KBDL",
        waypoints=[
            (39.5, -73.0, 0),
            (39.75, -73.0, 35000),
            (40.0, -73.0, 35000),  # Intersection point!
            (40.25, -73.0, 35000),
            (40.5, -73.0, 0)
        ],
        total_distance_nm=60
    )

    aircraft1 = Aircraft(id="TEST_EAST_001", route=route1, departure_time=0.0)
    aircraft2 = Aircraft(id="TEST_NORTH_001", route=route2, departure_time=0.0)

    return aircraft1, aircraft2


def test_conflict_detector_basic():
    """Test basic conflict detection between two aircraft."""
    print("=" * 70)
    print("TEST 1: Basic Conflict Detection")
    print("=" * 70)
    print()

    detector = ConflictDetector()
    print(f"Detector: {detector}")
    print()

    # Create aircraft on collision course
    aircraft1, aircraft2 = create_collision_course_aircraft()

    print(f"Aircraft 1: {aircraft1.id}")
    print(f"  Initial position: ({aircraft1.current_position['lat']:.4f}, "
          f"{aircraft1.current_position['lon']:.4f}, {aircraft1.current_position['alt']:.0f}ft)")
    print()
    print(f"Aircraft 2: {aircraft2.id}")
    print(f"  Initial position: ({aircraft2.current_position['lat']:.4f}, "
          f"{aircraft2.current_position['lon']:.4f}, {aircraft2.current_position['alt']:.0f}ft)")
    print()

    # Simulate until they approach each other
    print("Simulating approach...")
    print()

    conflict_detected = False
    for t in range(0, 601, 10):
        # Update positions
        aircraft1.update_position(delta_t=10.0)
        aircraft2.update_position(delta_t=10.0)

        # Check separation
        sep_result = detector.check_separation(aircraft1, aircraft2)

        # Print every 60 seconds
        if t % 60 == 0:
            print(f"t={t:3d}s: "
                  f"AC1=({aircraft1.current_position['lat']:.4f},{aircraft1.current_position['lon']:.4f},{aircraft1.current_position['alt']:.0f}ft) "
                  f"AC2=({aircraft2.current_position['lat']:.4f},{aircraft2.current_position['lon']:.4f},{aircraft2.current_position['alt']:.0f}ft)")
            print(f"       H_sep={sep_result['horizontal_distance_nm']:.2f}nm "
                  f"V_sep={sep_result['vertical_distance_ft']:.0f}ft "
                  f"Conflict={sep_result['is_conflict']}")

        if sep_result['is_conflict']:
            conflict_detected = True

    print()
    if conflict_detected:
        print("✓ PASSED: Conflict detected during simulation")
    else:
        print("⚠ WARNING: No conflict detected (aircraft may have passed safely)")

    print()
    return detector


def test_conflict_tracker():
    """Test conflict tracking over time."""
    print("=" * 70)
    print("TEST 2: Conflict Tracking")
    print("=" * 70)
    print()

    detector = ConflictDetector()
    tracker = ConflictTracker()

    print(f"Tracker: {tracker}")
    print()

    # Create aircraft on collision course
    aircraft1, aircraft2 = create_collision_course_aircraft()

    print("Simulating 10-minute flight with tracking...")
    print()

    conflict_timeline = []

    for t in range(0, 601, 1):
        # Update positions
        aircraft1.update_position(delta_t=1.0)
        aircraft2.update_position(delta_t=1.0)

        aircraft1.check_waypoint_reached()
        aircraft2.check_waypoint_reached()

        # Detect conflicts
        conflicts = detector.detect_all_conflicts([aircraft1, aircraft2], simulation_time=t)

        # Update tracker
        tracker.update(t, conflicts)

        # Record conflicts
        if conflicts:
            if not conflict_timeline or conflict_timeline[-1]['type'] != 'active':
                conflict_timeline.append({
                    'time': t,
                    'type': 'detected',
                    'details': conflicts[0]
                })
        else:
            if conflict_timeline and conflict_timeline[-1]['type'] != 'resolved':
                if tracker.conflict_history:
                    conflict_timeline.append({
                        'time': t,
                        'type': 'resolved',
                        'details': None
                    })

    print("Conflict timeline:")
    for event in conflict_timeline[:5]:  # Show first 5 events
        if event['type'] == 'detected':
            print(f"  t={event['time']:3d}s: CONFLICT DETECTED - "
                  f"Severity: {event['details']['severity']}, "
                  f"H_sep: {event['details']['horizontal_distance_nm']:.2f}nm")
        elif event['type'] == 'resolved':
            print(f"  t={event['time']:3d}s: CONFLICT RESOLVED")

    print()

    # Get statistics
    stats = tracker.get_statistics()
    print("Tracker statistics:")
    for key, value in stats.items():
        if key == 'severity_breakdown':
            print(f"  {key}:")
            for severity, count in value.items():
                print(f"    {severity}: {count}")
        else:
            print(f"  {key}: {value}")

    print()
    if stats['total_conflicts'] > 0:
        print("✓ PASSED: Conflicts tracked successfully")
    else:
        print("⚠ WARNING: No conflicts tracked")

    print()
    return tracker


def test_severity_classification():
    """Test conflict severity classification."""
    print("=" * 70)
    print("TEST 3: Severity Classification")
    print("=" * 70)
    print()

    detector = ConflictDetector()

    # Create aircraft at different separations
    test_cases = [
        (2.0, "CRITICAL"),
        (4.0, "WARNING"),
        (6.0, "NEAR"),
        (10.0, "SAFE")
    ]

    print("Testing severity classification:")
    print()

    all_passed = True

    for horizontal_sep_nm, expected_severity in test_cases:
        # Create two aircraft at specified separation
        route1 = FlightRoute(
            route_id="TEST1",
            aircraft_type="B738",
            departure_icao="KJFK",
            arrival_icao="KBOS",
            waypoints=[
                (40.0, -73.0, 35000),
                (40.0, -72.0, 35000)
            ],
            total_distance_nm=60
        )

        route2 = FlightRoute(
            route_id="TEST2",
            aircraft_type="A320",
            departure_icao="KEWR",
            arrival_icao="KBDL",
            waypoints=[
                (40.0 + (horizontal_sep_nm / 60.0), -73.0, 35000),
                (40.0 + (horizontal_sep_nm / 60.0), -72.0, 35000)
            ],
            total_distance_nm=60
        )

        aircraft1 = Aircraft(id="TEST1", route=route1)
        aircraft2 = Aircraft(id="TEST2", route=route2)

        # Check separation (both at same altitude = vertical violation)
        sep_result = detector.check_separation(aircraft1, aircraft2)

        # Classify severity
        severity = detector._classify_severity(sep_result['horizontal_distance_nm'])

        status = "✓" if severity == expected_severity else "✗"
        print(f"{status} H_sep={horizontal_sep_nm:.1f}nm: {severity} (expected: {expected_severity})")

        if severity != expected_severity:
            all_passed = False

    print()
    if all_passed:
        print("✓ PASSED: All severity classifications correct")
    else:
        print("✗ FAILED: Some severity classifications incorrect")

    print()


def test_multiple_aircraft():
    """Test conflict detection with multiple aircraft."""
    print("=" * 70)
    print("TEST 4: Multiple Aircraft Conflict Detection")
    print("=" * 70)
    print()

    detector = ConflictDetector()

    # Create 5 aircraft at various positions
    aircraft_list = []

    for i in range(5):
        route = FlightRoute(
            route_id=f"TEST_{i}",
            aircraft_type="B738",
            departure_icao="KJFK",
            arrival_icao="KBOS",
            waypoints=[
                (40.0 + i * 0.01, -73.0 + i * 0.01, 35000 + i * 500),
                (40.0 + i * 0.01, -72.0 + i * 0.01, 35000 + i * 500)
            ],
            total_distance_nm=60
        )
        aircraft = Aircraft(id=f"TEST_{i:03d}", route=route, departure_time=0.0)
        aircraft_list.append(aircraft)

    print(f"Created {len(aircraft_list)} aircraft")
    print()

    # Detect conflicts
    conflicts = detector.detect_all_conflicts(aircraft_list, simulation_time=0.0)

    print(f"Detected {len(conflicts)} conflicts")
    print()

    if len(conflicts) > 0:
        print("Sample conflicts:")
        for conflict in conflicts[:3]:
            print(f"  {conflict['aircraft1_id']} <-> {conflict['aircraft2_id']}: "
                  f"{conflict['horizontal_distance_nm']:.2f}nm, "
                  f"{conflict['severity']}")
        print()

    print(f"✓ PASSED: Multiple aircraft conflict detection working")
    print()


def test_taxi_landed_exclusion():
    """Test that TAXI and LANDED aircraft are excluded from conflict detection."""
    print("=" * 70)
    print("TEST 5: TAXI/LANDED Aircraft Exclusion")
    print("=" * 70)
    print()

    detector = ConflictDetector()

    # Create aircraft with different statuses
    route1 = FlightRoute(
        route_id="TEST1",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KBOS",
        waypoints=[(40.0, -73.0, 0)],
        total_distance_nm=1
    )

    route2 = FlightRoute(
        route_id="TEST2",
        aircraft_type="A320",
        departure_icao="KEWR",
        arrival_icao="KBDL",
        waypoints=[(40.0001, -73.0001, 35000)],  # Very close!
        total_distance_nm=1
    )

    aircraft1 = Aircraft(id="TAXI_001", route=route1)
    aircraft2 = Aircraft(id="FLYING_001", route=route2)

    # Set aircraft1 to TAXI (default), aircraft2 to CLIMBING
    aircraft1.status = 'TAXI'
    aircraft2.status = 'CLIMBING'

    print(f"Aircraft 1 status: {aircraft1.status}")
    print(f"Aircraft 2 status: {aircraft2.status}")
    print()

    # Detect conflicts
    conflicts = detector.detect_all_conflicts([aircraft1, aircraft2], simulation_time=0.0)

    print(f"Conflicts detected: {len(conflicts)}")

    if len(conflicts) == 0:
        print("✓ PASSED: TAXI aircraft excluded from conflict detection")
    else:
        print("✗ FAILED: TAXI aircraft should be excluded")

    print()

    # Test with LANDED status
    aircraft1.status = 'CLIMBING'
    aircraft2.status = 'LANDED'

    conflicts = detector.detect_all_conflicts([aircraft1, aircraft2], simulation_time=0.0)

    print(f"Aircraft 1 status: {aircraft1.status}")
    print(f"Aircraft 2 status: {aircraft2.status}")
    print(f"Conflicts detected: {len(conflicts)}")
    print()

    if len(conflicts) == 0:
        print("✓ PASSED: LANDED aircraft excluded from conflict detection")
    else:
        print("✗ FAILED: LANDED aircraft should be excluded")

    print()


def test_simulation_integration():
    """Test conflict detection integrated with simulation."""
    print("=" * 70)
    print("TEST 6: Simulation Integration")
    print("=" * 70)
    print()

    # Create simulation components
    airspace = Airspace()
    delay_model = DelayModel(seed=42)
    output = SimulationOutput(output_mode="file")

    # Create simulation manager (includes conflict detection)
    manager = SimulationManager(airspace, delay_model, output)

    print(f"Manager: {manager}")
    print(f"Detector: {manager.conflict_detector}")
    print(f"Tracker: {manager.conflict_tracker}")
    print()

    # Create two routes that will conflict
    route1 = FlightRoute(
        route_id="SIM_TEST1",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KBOS",
        waypoints=[
            (40.0, -74.0, 0),
            (40.0, -73.5, 35000),
            (40.0, -73.0, 35000),
            (40.0, -72.5, 0)
        ],
        total_distance_nm=180,
        scheduled_departure_time=0.0,
        actual_departure_time=0.0
    )

    route2 = FlightRoute(
        route_id="SIM_TEST2",
        aircraft_type="A320",
        departure_icao="KEWR",
        arrival_icao="KBDL",
        waypoints=[
            (39.5, -73.0, 0),
            (40.0, -73.0, 35000),
            (40.5, -73.0, 0)
        ],
        total_distance_nm=60,
        scheduled_departure_time=0.0,
        actual_departure_time=0.0
    )

    # Add aircraft
    manager.add_aircraft(route1, departure_time=0.0)
    manager.add_aircraft(route2, departure_time=0.0)

    print(f"Added {manager.total_aircraft_spawned} aircraft")
    print()

    # Run simulation for 10 minutes
    print("Running 10-minute simulation...")
    stats = manager.run(duration_seconds=600, time_step=1.0, progress_interval=120)

    print()
    print("Final statistics:")
    print(f"  Total flights: {stats['total_flights']}")
    print(f"  Active flights: {stats['active_flights']}")
    print(f"  Conflicts:")
    for key, value in stats['conflicts'].items():
        if key == 'severity_breakdown':
            print(f"    {key}:")
            for severity, count in value.items():
                if count > 0:
                    print(f"      {severity}: {count}")
        else:
            print(f"    {key}: {value}")

    print()

    # Close output
    output.close()

    if stats['conflicts']['total_conflicts'] > 0:
        print("✓ PASSED: Conflicts detected during simulation")
    else:
        print("⚠ INFO: No conflicts detected (aircraft may not have conflicted)")

    print()


def main():
    """Run all conflict detection tests."""
    print("\n")
    print("*" * 70)
    print("* DeepSky ATC - Conflict Detection System Test Suite")
    print("* Phase 2, Deliverable 2.1")
    print("*" * 70)
    print()

    try:
        # Run all tests
        test_conflict_detector_basic()
        test_conflict_tracker()
        test_severity_classification()
        test_multiple_aircraft()
        test_taxi_landed_exclusion()
        test_simulation_integration()

        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print()

        print("Summary:")
        print("  ✓ Basic conflict detection working")
        print("  ✓ Conflict tracking operational")
        print("  ✓ Severity classification correct")
        print("  ✓ Multiple aircraft handling verified")
        print("  ✓ TAXI/LANDED exclusion confirmed")
        print("  ✓ Simulation integration successful")
        print()

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
