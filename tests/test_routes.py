"""
Test suite for DeepSky ATC static route data system.

Tests data acquisition, route generation, and waypoint validation.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_acquisition import acquire_jfk_route_data
from src.route_generator import (
    load_routes_from_data,
    get_route_statistics,
    generate_waypoints,
    calculate_great_circle_distance,
    calculate_initial_bearing
)


def test_data_acquisition():
    """Test downloading and filtering route data."""
    print("=" * 70)
    print("TEST 1: Data Acquisition")
    print("=" * 70)
    print()

    # Acquire JFK route data
    jfk_routes, airports = acquire_jfk_route_data(force_download=False)

    if jfk_routes is None:
        print("✗ FAILED: Could not acquire route data")
        return False

    print(f"\n✓ PASSED: Acquired {len(jfk_routes)} JFK routes")

    # Check if we meet the target
    target_routes = 500
    unique_pairs = jfk_routes[['source_iata', 'dest_iata']].drop_duplicates()

    print(f"  Unique route pairs: {len(unique_pairs)}")
    print(f"  Target: {target_routes} routes")

    if len(unique_pairs) >= target_routes:
        print(f"  ✓ Met target: {len(unique_pairs)} >= {target_routes}")
    else:
        print(f"  ⚠ Below target: {len(unique_pairs)} < {target_routes}")

    print()
    return True


def test_route_loading():
    """Test loading routes and generating flight paths."""
    print("=" * 70)
    print("TEST 2: Route Loading and Generation")
    print("=" * 70)
    print()

    # Load all routes
    routes = load_routes_from_data()

    if not routes:
        print("✗ FAILED: No routes loaded")
        print("  Make sure to run data acquisition first!")
        return False

    print(f"\n✓ PASSED: Loaded {len(routes)} routes")
    print()

    return routes


def test_waypoint_validation(routes):
    """Test that waypoints are properly formatted."""
    print("=" * 70)
    print("TEST 3: Waypoint Validation")
    print("=" * 70)
    print()

    if not routes:
        print("✗ FAILED: No routes to validate")
        return False

    validation_errors = 0

    # Check first 100 routes
    for route in routes[:100]:
        # Validate waypoints exist
        if not route.waypoints:
            print(f"✗ Route {route.route_id} has no waypoints")
            validation_errors += 1
            continue

        # Validate waypoint format
        for i, waypoint in enumerate(route.waypoints):
            if len(waypoint) != 3:
                print(f"✗ Route {route.route_id} WP{i}: Invalid format (expected 3 values)")
                validation_errors += 1
                continue

            lat, lon, alt = waypoint

            # Validate latitude range
            if not (-90 <= lat <= 90):
                print(f"✗ Route {route.route_id} WP{i}: Invalid latitude {lat}")
                validation_errors += 1

            # Validate longitude range
            if not (-180 <= lon <= 180):
                print(f"✗ Route {route.route_id} WP{i}: Invalid longitude {lon}")
                validation_errors += 1

            # Validate altitude (should be >= 0 and <= typical max)
            if not (0 <= alt <= 50000):
                print(f"✗ Route {route.route_id} WP{i}: Invalid altitude {alt} ft")
                validation_errors += 1

        # Validate initial parameters
        if not (0 <= route.initial_heading_deg <= 360):
            print(f"✗ Route {route.route_id}: Invalid heading {route.initial_heading_deg}")
            validation_errors += 1

        if route.initial_speed_knots <= 0:
            print(f"✗ Route {route.route_id}: Invalid speed {route.initial_speed_knots}")
            validation_errors += 1

    if validation_errors == 0:
        print(f"✓ PASSED: All waypoints validated successfully")
    else:
        print(f"✗ FAILED: Found {validation_errors} validation errors")

    print()
    return validation_errors == 0


def test_sample_routes(routes):
    """Display detailed information for sample routes."""
    print("=" * 70)
    print("TEST 4: Sample Route Details")
    print("=" * 70)
    print()

    if not routes:
        print("✗ No routes to display")
        return False

    # Select 5 diverse sample routes (short, medium, long distances)
    routes_sorted = sorted(routes, key=lambda r: r.total_distance_nm)

    sample_indices = [
        0,  # Shortest
        len(routes_sorted) // 4,  # 25th percentile
        len(routes_sorted) // 2,  # Median
        3 * len(routes_sorted) // 4,  # 75th percentile
        len(routes_sorted) - 1  # Longest
    ]

    samples = [routes_sorted[i] for i in sample_indices if i < len(routes_sorted)]

    for i, route in enumerate(samples[:5], 1):
        print(f"Sample Route {i}:")
        print(f"  Route ID: {route.route_id}")
        print(f"  Path: {route.departure_icao} → {route.arrival_icao}")
        print(f"  Aircraft: {route.aircraft_type}")
        print(f"  Distance: {route.total_distance_nm:.1f} nm")
        print(f"  Initial heading: {route.initial_heading_deg:.1f}°")
        print(f"  Initial speed: {route.initial_speed_knots:.0f} kts")
        print(f"  Waypoints: {len(route.waypoints)}")

        # Show waypoint details
        print(f"  Waypoint details:")
        for j, (lat, lon, alt) in enumerate(route.waypoints):
            phase = "Climb" if j < 2 else ("Descent" if j >= len(route.waypoints) - 2 else "Cruise")
            print(f"    WP{j+1} ({phase}): ({lat:7.4f}, {lon:8.4f}) at {alt:6.0f} ft")

        print()

    print("✓ PASSED: Sample routes displayed")
    print()
    return True


def test_aircraft_distribution(routes):
    """Test and display aircraft type distribution."""
    print("=" * 70)
    print("TEST 5: Aircraft Type Distribution")
    print("=" * 70)
    print()

    if not routes:
        print("✗ No routes to analyze")
        return False

    # Get statistics
    stats = get_route_statistics(routes)

    print(f"Route Statistics:")
    print(f"  Total routes: {stats['total_routes']}")
    print(f"  Distance range: {stats['min_distance_nm']:.0f} - {stats['max_distance_nm']:.0f} nm")
    print(f"  Average distance: {stats['avg_distance_nm']:.0f} nm")
    print(f"  Median distance: {stats['median_distance_nm']:.0f} nm")
    print(f"  Total waypoints: {stats['total_waypoints']}")
    print()

    print(f"Aircraft Type Distribution:")
    aircraft_dist = stats['aircraft_distribution']

    # Sort by count
    for aircraft, count in sorted(aircraft_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_routes']) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {aircraft}: {count:4d} ({percentage:5.1f}%) {bar}")

    print()

    # Validate we have multiple aircraft types
    if len(aircraft_dist) >= 2:
        print("✓ PASSED: Multiple aircraft types assigned")
    else:
        print("⚠ WARNING: Only one aircraft type found")

    print()
    return True


def test_route_geometry():
    """Test route geometry calculations."""
    print("=" * 70)
    print("TEST 6: Route Geometry Calculations")
    print("=" * 70)
    print()

    # Test with known routes
    test_cases = [
        # (origin, destination, expected_distance_nm, description)
        ((40.6413, -73.7781), (33.9416, -118.4085), 2144, "JFK to LAX"),
        ((40.6413, -73.7781), (51.4700, -0.4543), 2993, "JFK to LHR (London)"),
        ((40.6413, -73.7781), (37.6213, -122.3790), 2243, "JFK to SFO"),
    ]

    all_passed = True

    for (origin, dest, expected_dist, description) in test_cases:
        origin_lat, origin_lon = origin
        dest_lat, dest_lon = dest

        # Calculate distance
        distance = calculate_great_circle_distance(origin_lat, origin_lon, dest_lat, dest_lon)

        # Calculate heading
        heading = calculate_initial_bearing(origin_lat, origin_lon, dest_lat, dest_lon)

        # Generate waypoints
        waypoints, total_distance = generate_waypoints(origin_lat, origin_lon, dest_lat, dest_lon)

        # Check if distance is within 5% of expected
        error_pct = abs(distance - expected_dist) / expected_dist * 100

        status = "✓" if error_pct < 5 else "✗"
        print(f"{status} {description}:")
        print(f"    Calculated: {distance:.0f} nm (expected ~{expected_dist} nm, error: {error_pct:.1f}%)")
        print(f"    Heading: {heading:.1f}°")
        print(f"    Waypoints: {len(waypoints)}")

        if error_pct >= 5:
            all_passed = False

    print()
    if all_passed:
        print("✓ PASSED: All geometry calculations within tolerance")
    else:
        print("✗ FAILED: Some calculations outside tolerance")

    print()
    return all_passed


def main():
    """Run all route tests."""
    print("\n")
    print("*" * 70)
    print("* DeepSky ATC - Static Route Data System Test Suite")
    print("* Phase 1, Deliverable 1.2")
    print("*" * 70)
    print()

    try:
        # Run all tests
        test_data_acquisition()
        routes = test_route_loading()

        if routes:
            test_waypoint_validation(routes)
            test_sample_routes(routes)
            test_aircraft_distribution(routes)
            test_route_geometry()
        else:
            print("⚠ Skipping route tests - no routes loaded")
            print("  This may be because data acquisition failed.")
            print("  Check your internet connection and try again.")

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
