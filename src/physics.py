"""
Physics and navigation utilities for DeepSky ATC.

Provides functions for calculating distances, bearings, and position updates
for aircraft navigation.
"""

import numpy as np
from typing import Tuple


# Constants
EARTH_RADIUS_NM = 3440.065  # Earth radius in nautical miles
FEET_PER_NAUTICAL_MILE = 6076.12  # Conversion factor
KNOTS_TO_FPM = 101.269  # Knots to feet per minute (1 knot = 1.68781 ft/s * 60)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula.

    This is the shortest distance over the earth's surface, giving an 'as-the-crow-flies'
    distance between the points (ignoring any hills, valleys, etc.).

    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees

    Returns:
        Distance in nautical miles

    Example:
        >>> # Distance from JFK to LAX
        >>> dist = haversine_distance(40.6413, -73.7781, 33.9416, -118.4085)
        >>> print(f"{dist:.1f} nm")
        2146.0 nm
    """
    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in nautical miles
    distance_nm = EARTH_RADIUS_NM * c

    return distance_nm


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing (forward azimuth) from point 1 to point 2.

    The bearing is the compass direction to follow from the start point to reach
    the end point. Note that along a great circle, the bearing may change as you
    move along the path.

    Args:
        lat1: Latitude of starting point in decimal degrees
        lon1: Longitude of starting point in decimal degrees
        lat2: Latitude of destination point in decimal degrees
        lon2: Longitude of destination point in decimal degrees

    Returns:
        Initial bearing in degrees (0-360), where:
        - 0° = North
        - 90° = East
        - 180° = South
        - 270° = West

    Example:
        >>> # Bearing from JFK to LAX (roughly west)
        >>> bearing = calculate_bearing(40.6413, -73.7781, 33.9416, -118.4085)
        >>> print(f"{bearing:.1f}°")
        273.8°
    """
    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate bearing
    dlon = lon2_rad - lon1_rad

    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

    bearing_rad = np.arctan2(x, y)

    # Convert to degrees and normalize to 0-360
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360

    return bearing_deg


def update_position_by_bearing(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_nm: float
) -> Tuple[float, float]:
    """
    Calculate new position from a starting point, bearing, and distance.

    Uses spherical geometry to compute the destination point given a starting point,
    an initial bearing, and a distance to travel.

    Args:
        lat: Starting latitude in decimal degrees
        lon: Starting longitude in decimal degrees
        bearing_deg: Bearing/heading to travel in degrees (0-360)
        distance_nm: Distance to travel in nautical miles

    Returns:
        Tuple of (new_lat, new_lon) in decimal degrees

    Example:
        >>> # Move 100nm north from JFK
        >>> new_lat, new_lon = update_position_by_bearing(40.6413, -73.7781, 0, 100)
        >>> print(f"New position: ({new_lat:.4f}, {new_lon:.4f})")
        New position: (42.3079, -73.7781)
    """
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing_deg)

    # Angular distance in radians
    angular_distance = distance_nm / EARTH_RADIUS_NM

    # Calculate new position
    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(angular_distance) +
        np.cos(lat_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    )

    new_lon_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat_rad),
        np.cos(angular_distance) - np.sin(lat_rad) * np.sin(new_lat_rad)
    )

    # Convert back to degrees
    new_lat = np.degrees(new_lat_rad)
    new_lon = np.degrees(new_lon_rad)

    # Normalize longitude to -180 to 180
    new_lon = ((new_lon + 180) % 360) - 180

    return new_lat, new_lon


def interpolate_heading(current_heading: float, target_heading: float, turn_rate_deg: float) -> float:
    """
    Smoothly interpolate heading change with a maximum turn rate.

    Aircraft cannot instantly change heading - they need to bank and turn.
    This function limits the rate of heading change to simulate realistic turns.

    Args:
        current_heading: Current heading in degrees (0-360)
        target_heading: Desired heading in degrees (0-360)
        turn_rate_deg: Maximum turn rate in degrees per update

    Returns:
        New heading in degrees (0-360)

    Example:
        >>> # Turn from 0° to 90° with max 5° per update
        >>> new_heading = interpolate_heading(0, 90, 5)
        >>> print(f"{new_heading:.1f}°")
        5.0°
    """
    # Normalize headings to 0-360
    current_heading = current_heading % 360
    target_heading = target_heading % 360

    # Calculate shortest turn direction
    diff = target_heading - current_heading

    # Normalize difference to -180 to 180
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    # Limit turn rate
    if abs(diff) <= turn_rate_deg:
        # Can reach target in one step
        new_heading = target_heading
    else:
        # Turn at maximum rate in correct direction
        new_heading = current_heading + (turn_rate_deg if diff > 0 else -turn_rate_deg)

    # Normalize result to 0-360
    new_heading = new_heading % 360

    return new_heading


def calculate_climb_rate(current_alt_ft: float, target_alt_ft: float, max_rate_fpm: float) -> float:
    """
    Calculate vertical speed to reach target altitude.

    Args:
        current_alt_ft: Current altitude in feet
        target_alt_ft: Target altitude in feet
        max_rate_fpm: Maximum climb/descent rate in feet per minute

    Returns:
        Vertical speed in feet per minute (positive = climb, negative = descent)

    Example:
        >>> # Climbing from 10,000 to 35,000 ft
        >>> rate = calculate_climb_rate(10000, 35000, 2000)
        >>> print(f"{rate:.0f} fpm")
        2000 fpm
    """
    diff = target_alt_ft - current_alt_ft

    if abs(diff) < 100:  # Within 100 feet, consider arrived
        return 0.0

    if diff > 0:
        # Climbing
        return min(diff, max_rate_fpm)
    else:
        # Descending
        return max(diff, -max_rate_fpm)


def knots_to_meters_per_second(knots: float) -> float:
    """Convert speed from knots to meters per second."""
    return knots * 0.514444


def meters_per_second_to_knots(mps: float) -> float:
    """Convert speed from meters per second to knots."""
    return mps / 0.514444


def feet_per_minute_to_feet_per_second(fpm: float) -> float:
    """Convert vertical speed from feet per minute to feet per second."""
    return fpm / 60.0


if __name__ == "__main__":
    """Test physics functions when module is executed directly."""
    print("Testing physics module...")
    print()

    # Test haversine distance
    print("Test 1: Haversine Distance")
    jfk_lat, jfk_lon = 40.6413, -73.7781
    lax_lat, lax_lon = 33.9416, -118.4085
    dist = haversine_distance(jfk_lat, jfk_lon, lax_lat, lax_lon)
    print(f"  JFK to LAX: {dist:.1f} nm")
    print()

    # Test bearing calculation
    print("Test 2: Bearing Calculation")
    bearing = calculate_bearing(jfk_lat, jfk_lon, lax_lat, lax_lon)
    print(f"  JFK to LAX bearing: {bearing:.1f}°")
    print()

    # Test position update
    print("Test 3: Position Update")
    new_lat, new_lon = update_position_by_bearing(jfk_lat, jfk_lon, 0, 100)
    print(f"  100nm north from JFK: ({new_lat:.4f}, {new_lon:.4f})")
    print()

    # Test heading interpolation
    print("Test 4: Heading Interpolation")
    new_heading = interpolate_heading(0, 90, 5)
    print(f"  Turn from 0° toward 90° (max 5°/step): {new_heading:.1f}°")
    print()

    print("All physics tests complete!")
