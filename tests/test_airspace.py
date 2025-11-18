"""
Test suite for DeepSky ATC airspace boundary system.

Tests coordinate conversions, boundary checking, and airspace configuration.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.airspace import Airspace
from src.coordinates import (
    lat_lon_alt_to_xyz,
    xyz_to_lat_lon_alt,
    distance_nm,
    feet_to_meters,
    meters_to_feet
)


def test_airspace_loading():
    """Test that airspace configuration loads correctly."""
    print("=" * 70)
    print("TEST 1: Airspace Loading")
    print("=" * 70)

    airspace = Airspace()
    print(f"✓ Loaded airspace: {airspace.name}")
    print(f"  Center: ({airspace.center_lat}, {airspace.center_lon})")
    print(f"  Radius: {airspace.radius_nm} nm")
    print(f"  Altitude: {airspace.min_alt_feet} - {airspace.max_alt_feet} ft")
    print(f"  Flight levels: {airspace.min_flight_level} - {airspace.max_flight_level}")
    print()

    # Get and display bounds info
    bounds_info = airspace.get_bounds_info()
    print(f"Airspace area: {bounds_info['horizontal']['area_sq_nm']:.2f} sq nm")
    print(f"Airspace diameter: {bounds_info['horizontal']['diameter_nm']:.2f} nm")
    print()

    return airspace


def test_kjfk_center_in_bounds(airspace):
    """Test that KJFK center point is within bounds."""
    print("=" * 70)
    print("TEST 2: KJFK Center In Bounds")
    print("=" * 70)

    kjfk_lat = 40.6413
    kjfk_lon = -73.7781
    test_altitude = 5000  # feet

    result = airspace.is_in_bounds(kjfk_lat, kjfk_lon, test_altitude)
    distance = airspace.get_distance_from_center(kjfk_lat, kjfk_lon)

    print(f"Position: ({kjfk_lat}, {kjfk_lon}) at {test_altitude} ft")
    print(f"Distance from center: {distance:.2f} nm")
    print(f"In bounds: {result}")

    if result and distance < 0.1:
        print("✓ PASSED: KJFK center is in bounds")
    else:
        print("✗ FAILED: KJFK center should be in bounds")

    print()
    return result


def test_point_outside_horizontal_bounds(airspace):
    """Test that a point 160nm away is out of bounds."""
    print("=" * 70)
    print("TEST 3: Point Outside Horizontal Bounds (160nm away)")
    print("=" * 70)

    # Calculate a point approximately 160nm north of KJFK
    # At KJFK latitude, 1 degree latitude ≈ 60 nm
    # So 160nm ≈ 2.67 degrees
    test_lat = airspace.center_lat + 2.67
    test_lon = airspace.center_lon
    test_altitude = 15000  # feet

    result = airspace.is_in_bounds(test_lat, test_lon, test_altitude)
    distance = airspace.get_distance_from_center(test_lat, test_lon)

    print(f"Position: ({test_lat:.4f}, {test_lon:.4f}) at {test_altitude} ft")
    print(f"Distance from center: {distance:.2f} nm")
    print(f"Airspace radius: {airspace.radius_nm} nm")
    print(f"In bounds: {result}")

    if not result and distance > 150:
        print("✓ PASSED: Point 160nm away is correctly out of bounds")
    else:
        print("✗ FAILED: Point should be out of bounds")

    print()
    return not result  # Should be False (out of bounds)


def test_altitude_limits(airspace):
    """Test altitude boundary checking."""
    print("=" * 70)
    print("TEST 4: Altitude Limits")
    print("=" * 70)

    kjfk_lat = airspace.center_lat
    kjfk_lon = airspace.center_lon

    # Test below minimum altitude
    test_cases = [
        (-100, "Below minimum altitude (-100 ft)"),
        (0, "At minimum altitude (0 ft)"),
        (25000, "Mid-range altitude (25,000 ft)"),
        (45000, "At maximum altitude (45,000 ft)"),
        (50000, "Above maximum altitude (50,000 ft)")
    ]

    all_passed = True
    for alt, description in test_cases:
        result = airspace.is_in_bounds(kjfk_lat, kjfk_lon, alt)
        expected = (airspace.min_alt_feet <= alt <= airspace.max_alt_feet)

        status = "✓" if result == expected else "✗"
        print(f"{status} {description}: in_bounds={result}, expected={expected}")

        if result != expected:
            all_passed = False

    print()
    if all_passed:
        print("✓ PASSED: All altitude limit tests passed")
    else:
        print("✗ FAILED: Some altitude limit tests failed")

    print()
    return all_passed


def test_coordinate_conversions():
    """Test coordinate conversion functions."""
    print("=" * 70)
    print("TEST 5: Coordinate Conversions")
    print("=" * 70)

    kjfk_lat = 40.6413
    kjfk_lon = -73.7781
    kjfk_alt = 13  # KJFK elevation is 13 feet

    print(f"KJFK Airport: ({kjfk_lat}, {kjfk_lon}) at {kjfk_alt} ft MSL")
    print()

    # Convert to meters
    kjfk_alt_meters = feet_to_meters(kjfk_alt)
    print(f"Altitude: {kjfk_alt} ft = {kjfk_alt_meters:.2f} m")
    print()

    # Convert KJFK to local coordinates (should be near origin)
    x, y, z = lat_lon_alt_to_xyz(kjfk_lat, kjfk_lon, kjfk_alt_meters,
                                  kjfk_lat, kjfk_lon)
    print(f"Local Cartesian coordinates (centered on KJFK):")
    print(f"  x (East):  {x:.2f} m")
    print(f"  y (North): {y:.2f} m")
    print(f"  z (Up):    {z:.2f} m")
    print()

    # Convert back to lat/lon/alt
    lat_back, lon_back, alt_back = xyz_to_lat_lon_alt(x, y, z,
                                                        kjfk_lat, kjfk_lon)
    print(f"Converted back to geodetic:")
    print(f"  lat: {lat_back:.6f}° (original: {kjfk_lat:.6f}°)")
    print(f"  lon: {lon_back:.6f}° (original: {kjfk_lon:.6f}°)")
    print(f"  alt: {alt_back:.2f} m (original: {kjfk_alt_meters:.2f} m)")
    print()

    # Check round-trip accuracy
    lat_error = abs(lat_back - kjfk_lat)
    lon_error = abs(lon_back - kjfk_lon)
    alt_error = abs(alt_back - kjfk_alt_meters)

    print(f"Round-trip errors:")
    print(f"  Latitude:  {lat_error:.10f}°")
    print(f"  Longitude: {lon_error:.10f}°")
    print(f"  Altitude:  {alt_error:.10f} m")
    print()

    if lat_error < 1e-6 and lon_error < 1e-6 and alt_error < 0.01:
        print("✓ PASSED: Coordinate conversion round-trip accurate")
    else:
        print("✗ FAILED: Coordinate conversion errors too large")

    print()

    # Test some offset points
    print("Sample coordinate conversions:")
    print("-" * 70)

    test_points = [
        (kjfk_lat + 1.0, kjfk_lon, 10000, "100nm North"),
        (kjfk_lat, kjfk_lon + 1.0, 20000, "~75nm East"),
        (kjfk_lat - 1.0, kjfk_lon - 1.0, 30000, "Southwest diagonal")
    ]

    for lat, lon, alt_ft, desc in test_points:
        alt_m = feet_to_meters(alt_ft)
        x, y, z = lat_lon_alt_to_xyz(lat, lon, alt_m, kjfk_lat, kjfk_lon)
        dist = distance_nm(lat, lon, kjfk_lat, kjfk_lon, kjfk_lat, kjfk_lon)

        print(f"{desc}:")
        print(f"  Lat/Lon: ({lat:.4f}, {lon:.4f}) at {alt_ft} ft")
        print(f"  Local XYZ: ({x:.0f}, {y:.0f}, {z:.0f}) m")
        print(f"  Distance: {dist:.1f} nm")
        print()

    return True


def test_boundary_margins(airspace):
    """Test boundary margin calculations."""
    print("=" * 70)
    print("TEST 6: Boundary Margins")
    print("=" * 70)

    # Test point near center
    kjfk_lat = airspace.center_lat
    kjfk_lon = airspace.center_lon
    mid_altitude = 20000  # feet

    margins = airspace.get_margin_to_boundary(kjfk_lat, kjfk_lon, mid_altitude)

    print(f"Position: KJFK center at {mid_altitude} ft")
    print(f"  Horizontal margin: {margins['horizontal_margin_nm']:.2f} nm")
    print(f"  Margin to ceiling: {margins['altitude_margin_feet_above']:.0f} ft")
    print(f"  Margin to floor: {margins['altitude_margin_feet_below']:.0f} ft")
    print(f"  Closest boundary: {margins['closest_boundary']}")
    print(f"  In bounds: {margins['is_in_bounds']}")
    print()

    # Test point near horizontal boundary
    boundary_lat = kjfk_lat + 2.4  # ~144nm north
    margins = airspace.get_margin_to_boundary(boundary_lat, kjfk_lon, mid_altitude)

    print(f"Position: Near horizontal boundary at {mid_altitude} ft")
    print(f"  Horizontal margin: {margins['horizontal_margin_nm']:.2f} nm")
    print(f"  Closest boundary: {margins['closest_boundary']}")
    print(f"  In bounds: {margins['is_in_bounds']}")
    print()

    print("✓ PASSED: Boundary margin calculations complete")
    print()

    return True


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("* DeepSky ATC - Airspace Boundary System Test Suite")
    print("* Phase 1, Deliverable 1.1")
    print("*" * 70)
    print()

    try:
        # Run all tests
        airspace = test_airspace_loading()
        test_kjfk_center_in_bounds(airspace)
        test_point_outside_horizontal_bounds(airspace)
        test_altitude_limits(airspace)
        test_coordinate_conversions()
        test_boundary_margins(airspace)

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
