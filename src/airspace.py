"""
Airspace boundary management for DeepSky ATC.

Defines the controlled airspace region and provides methods to check
if aircraft positions are within authorized boundaries.
"""

import json
from pathlib import Path
from typing import Dict, Any

from src.coordinates import distance_nm, feet_to_meters


class Airspace:
    """
    Represents a defined airspace region with geographic and altitude boundaries.

    The airspace is defined as a cylindrical region:
    - Horizontal: Circular area with specified radius from center point
    - Vertical: Altitude range from minimum to maximum flight level

    Attributes:
        name: Descriptive name of the airspace
        center_lat: Center latitude in decimal degrees
        center_lon: Center longitude in decimal degrees
        radius_nm: Horizontal radius in nautical miles
        min_alt_feet: Minimum altitude in feet
        max_alt_feet: Maximum altitude in feet
    """

    def __init__(self, config_path: str = None):
        """
        Initialize airspace from configuration file.

        Args:
            config_path: Path to airspace configuration JSON file.
                        If None, uses default path: data/airspace_config.json

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid or missing required fields
        """
        if config_path is None:
            # Default to data/airspace_config.json relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "data" / "airspace_config.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Airspace configuration not found: {config_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Parse required fields
        try:
            self.name = self.config["name"]
            self.center_lat = self.config["center"]["lat"]
            self.center_lon = self.config["center"]["lon"]
            self.radius_nm = self.config["boundaries"]["radius_nm"]
            self.min_alt_feet = self.config["altitude"]["min_feet"]
            self.max_alt_feet = self.config["altitude"]["max_feet"]
        except KeyError as e:
            raise ValueError(f"Invalid airspace config: missing field {e}")

        # Store additional metadata
        self.center_description = self.config["center"].get("description", "")
        self.min_flight_level = self.config["altitude"].get("min_flight_level", "")
        self.max_flight_level = self.config["altitude"].get("max_flight_level", "")

    def is_in_bounds(self, lat: float, lon: float, alt_feet: float) -> bool:
        """
        Check if a position is within the airspace boundaries.

        Tests both horizontal (distance from center) and vertical (altitude)
        constraints. The airspace is treated as a cylinder.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            alt_feet: Altitude in feet above sea level

        Returns:
            True if position is within boundaries, False otherwise

        Example:
            >>> airspace = Airspace()
            >>> # KJFK center should be in bounds
            >>> airspace.is_in_bounds(40.6413, -73.7781, 5000)
            True
            >>> # Point 200nm away should be out of bounds
            >>> airspace.is_in_bounds(45.0, -73.7781, 5000)
            False
            >>> # Altitude too high should be out of bounds
            >>> airspace.is_in_bounds(40.6413, -73.7781, 50000)
            False
        """
        # Check altitude bounds
        if alt_feet < self.min_alt_feet or alt_feet > self.max_alt_feet:
            return False

        # Check horizontal bounds (distance from center)
        distance = distance_nm(
            lat, lon,
            self.center_lat, self.center_lon,
            self.center_lat, self.center_lon
        )

        if distance > self.radius_nm:
            return False

        return True

    def get_bounds_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the airspace boundaries.

        Returns:
            Dictionary containing:
                - name: Airspace name
                - center: Dict with lat, lon, and description
                - horizontal: Dict with radius and distance units
                - vertical: Dict with altitude limits and flight levels
                - area: Calculated area in square nautical miles

        Example:
            >>> airspace = Airspace()
            >>> info = airspace.get_bounds_info()
            >>> print(f"Airspace: {info['name']}")
            >>> print(f"Radius: {info['horizontal']['radius_nm']} nm")
            >>> print(f"Altitude: {info['vertical']['min_feet']}-{info['vertical']['max_feet']} ft")
        """
        import math

        # Calculate area (circle)
        area_sq_nm = math.pi * (self.radius_nm ** 2)

        return {
            "name": self.name,
            "center": {
                "lat": self.center_lat,
                "lon": self.center_lon,
                "description": self.center_description
            },
            "horizontal": {
                "radius_nm": self.radius_nm,
                "radius_km": self.radius_nm * 1.852,
                "diameter_nm": self.radius_nm * 2,
                "area_sq_nm": area_sq_nm
            },
            "vertical": {
                "min_feet": self.min_alt_feet,
                "max_feet": self.max_alt_feet,
                "range_feet": self.max_alt_feet - self.min_alt_feet,
                "min_flight_level": self.min_flight_level,
                "max_flight_level": self.max_flight_level
            },
            "volume": {
                "cylinder_volume_cubic_nm": area_sq_nm * (self.max_alt_feet / 6076.12)  # convert ft to nm
            }
        }

    def get_distance_from_center(self, lat: float, lon: float) -> float:
        """
        Calculate horizontal distance from airspace center.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            Distance in nautical miles
        """
        return distance_nm(
            lat, lon,
            self.center_lat, self.center_lon,
            self.center_lat, self.center_lon
        )

    def get_margin_to_boundary(self, lat: float, lon: float, alt_feet: float) -> Dict[str, float]:
        """
        Calculate distances to airspace boundaries from a given position.

        Useful for proximity warnings and buffer zone management.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            alt_feet: Altitude in feet

        Returns:
            Dictionary with:
                - horizontal_margin_nm: Distance to horizontal boundary (negative if outside)
                - altitude_margin_feet_above: Distance to upper altitude limit
                - altitude_margin_feet_below: Distance to lower altitude limit
                - closest_boundary: Name of the closest boundary
        """
        # Horizontal margin
        distance = self.get_distance_from_center(lat, lon)
        horizontal_margin = self.radius_nm - distance

        # Vertical margins
        margin_above = self.max_alt_feet - alt_feet
        margin_below = alt_feet - self.min_alt_feet

        # Determine closest boundary
        margins = {
            "horizontal": horizontal_margin * 6076.12,  # convert to feet for comparison
            "ceiling": margin_above,
            "floor": margin_below
        }
        closest_boundary = min(margins.items(), key=lambda x: x[1])[0]

        return {
            "horizontal_margin_nm": horizontal_margin,
            "altitude_margin_feet_above": margin_above,
            "altitude_margin_feet_below": margin_below,
            "closest_boundary": closest_boundary,
            "is_in_bounds": self.is_in_bounds(lat, lon, alt_feet)
        }

    def __repr__(self) -> str:
        """String representation of the airspace."""
        return (
            f"Airspace(name='{self.name}', "
            f"center=({self.center_lat:.4f}, {self.center_lon:.4f}), "
            f"radius={self.radius_nm}nm, "
            f"alt={self.min_alt_feet}-{self.max_alt_feet}ft)"
        )
