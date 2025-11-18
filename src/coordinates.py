"""
Coordinate transformation utilities for DeepSky ATC.

Provides conversions between geodetic (lat/lon/alt) and local Cartesian (x/y/z)
coordinates using a flat-earth approximation (local tangent plane).

Coordinate system:
- x-axis: Points East (meters)
- y-axis: Points North (meters)
- z-axis: Points Up (meters altitude)
"""

import numpy as np

# Constants
EARTH_RADIUS_METERS = 6371000.0  # meters
NAUTICAL_MILE_METERS = 1852.0     # meters


def lat_lon_alt_to_xyz(lat: float, lon: float, alt: float,
                       center_lat: float, center_lon: float) -> tuple[float, float, float]:
    """
    Convert geodetic coordinates to local Cartesian coordinates.

    Uses flat-earth approximation (local tangent plane) centered at the reference point.
    Suitable for regions up to a few hundred kilometers from the center.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        alt: Altitude in meters above sea level
        center_lat: Reference latitude in decimal degrees (e.g., KJFK)
        center_lon: Reference longitude in decimal degrees (e.g., KJFK)

    Returns:
        Tuple (x, y, z) in meters:
            x: East offset from center (positive = east, negative = west)
            y: North offset from center (positive = north, negative = south)
            z: Altitude in meters (same as input altitude)

    Example:
        >>> # KJFK center should map to origin
        >>> x, y, z = lat_lon_alt_to_xyz(40.6413, -73.7781, 1000, 40.6413, -73.7781)
        >>> print(f"x={x:.1f}, y={y:.1f}, z={z:.1f}")
        x=0.0, y=0.0, z=1000.0
    """
    # Convert degrees to radians for trigonometry
    center_lat_rad = np.radians(center_lat)

    # Convert degrees to radians for angle calculations
    deg_to_rad = np.pi / 180.0

    # Calculate offsets in degrees
    delta_lat = lat - center_lat
    delta_lon = lon - center_lon

    # Convert to meters using flat-earth approximation
    # y (North): Simple scaling by Earth's radius
    y = delta_lat * EARTH_RADIUS_METERS * deg_to_rad

    # x (East): Scale by Earth's radius and cosine of latitude
    # (longitude lines converge at poles)
    x = delta_lon * EARTH_RADIUS_METERS * np.cos(center_lat_rad) * deg_to_rad

    # z (Up): Altitude remains the same
    z = alt

    return (x, y, z)


def xyz_to_lat_lon_alt(x: float, y: float, z: float,
                       center_lat: float, center_lon: float) -> tuple[float, float, float]:
    """
    Convert local Cartesian coordinates to geodetic coordinates.

    Inverse of lat_lon_alt_to_xyz using flat-earth approximation.

    Args:
        x: East offset from center in meters
        y: North offset from center in meters
        z: Altitude in meters above sea level
        center_lat: Reference latitude in decimal degrees (e.g., KJFK)
        center_lon: Reference longitude in decimal degrees (e.g., KJFK)

    Returns:
        Tuple (lat, lon, alt) in decimal degrees and meters:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            alt: Altitude in meters above sea level

    Example:
        >>> # Origin should map back to KJFK center
        >>> lat, lon, alt = xyz_to_lat_lon_alt(0, 0, 1000, 40.6413, -73.7781)
        >>> print(f"lat={lat:.4f}, lon={lon:.4f}, alt={alt:.1f}")
        lat=40.6413, lon=-73.7781, alt=1000.0
    """
    # Convert degrees to radians for trigonometry
    center_lat_rad = np.radians(center_lat)

    # Convert radians to degrees
    rad_to_deg = 180.0 / np.pi

    # Convert meters back to degrees
    # Latitude: Simple inverse scaling
    delta_lat = y / (EARTH_RADIUS_METERS * np.pi / 180.0)
    lat = center_lat + delta_lat

    # Longitude: Inverse scaling with latitude correction
    delta_lon = x / (EARTH_RADIUS_METERS * np.cos(center_lat_rad) * np.pi / 180.0)
    lon = center_lon + delta_lon

    # Altitude remains the same
    alt = z

    return (lat, lon, alt)


def distance_nm(lat1: float, lon1: float, lat2: float, lon2: float,
                center_lat: float, center_lon: float) -> float:
    """
    Calculate horizontal distance between two points in nautical miles.

    Uses the local Cartesian coordinate system for distance calculation.

    Args:
        lat1, lon1: First point (decimal degrees)
        lat2, lon2: Second point (decimal degrees)
        center_lat, center_lon: Reference point for coordinate system

    Returns:
        Distance in nautical miles

    Example:
        >>> # Distance from KJFK to a point 100nm east
        >>> dist = distance_nm(40.6413, -72.0, 40.6413, -73.7781, 40.6413, -73.7781)
        >>> print(f"{dist:.1f} nm")
        100.0 nm
    """
    # Convert both points to local Cartesian
    x1, y1, _ = lat_lon_alt_to_xyz(lat1, lon1, 0, center_lat, center_lon)
    x2, y2, _ = lat_lon_alt_to_xyz(lat2, lon2, 0, center_lat, center_lon)

    # Calculate Euclidean distance
    distance_meters = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Convert to nautical miles
    distance_nautical_miles = distance_meters / NAUTICAL_MILE_METERS

    return distance_nautical_miles


def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet * 0.3048


def meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters / 0.3048
