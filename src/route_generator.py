"""
Flight route generation module for DeepSky ATC.

Creates realistic flight routes with waypoints, climb/cruise/descent phases,
and appropriate aircraft assignments.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from src.physics import haversine_distance


# Constants for flight planning
CRUISE_ALTITUDE_FT = 35000  # Typical cruise altitude
CLIMB_DISTANCE_NM = 100     # Distance to reach cruise altitude
DESCENT_DISTANCE_NM = 100   # Distance for descent to landing
INITIAL_SPEED_KTS = 250     # Initial speed (typical tower departure speed)
CRUISE_SPEED_KTS = 450      # Typical cruise speed
EARTH_RADIUS_NM = 3440.065  # Earth radius in nautical miles


@dataclass
class FlightRoute:
    """
    Represents a flight route with waypoints and flight parameters.

    Attributes:
        route_id: Unique identifier for this route
        aircraft_type: ICAO aircraft type code (e.g., B738, A320, B77W)
        departure_icao: Departure airport ICAO code
        arrival_icao: Arrival airport ICAO code
        waypoints: List of (lat, lon, altitude_ft) tuples defining the flight path
        initial_speed_knots: Initial airspeed at departure
        initial_heading_deg: Initial heading in degrees (0-360)
        initial_altitude_ft: Initial altitude at departure (usually 0 at gate)
        total_distance_nm: Total route distance in nautical miles
        scheduled_departure_time: Scheduled departure time in seconds (default 0.0)
        actual_departure_time: Actual departure time in seconds (None until delay applied)
    """
    route_id: str
    aircraft_type: str
    departure_icao: str
    arrival_icao: str
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    initial_speed_knots: float = INITIAL_SPEED_KTS
    initial_heading_deg: float = 0.0
    initial_altitude_ft: float = 0.0
    total_distance_nm: float = 0.0
    scheduled_departure_time: float = 0.0
    actual_departure_time: Optional[float] = None

    def __repr__(self) -> str:
        """String representation of the route."""
        delay_str = ""
        if self.actual_departure_time is not None:
            delay_min = (self.actual_departure_time - self.scheduled_departure_time) / 60.0
            delay_str = f", delay={delay_min:.1f}min"

        return (
            f"FlightRoute(id='{self.route_id}', {self.departure_icao}→{self.arrival_icao}, "
            f"{self.aircraft_type}, {len(self.waypoints)} waypoints, {self.total_distance_nm:.0f}nm{delay_str})"
        )

    def apply_delay(self, delay_model, hour_of_day: Optional[int] = None) -> float:
        """
        Apply departure delay to this route using a delay model.

        Args:
            delay_model: DelayModel instance to generate delays
            hour_of_day: Hour of day (0-23) for time-based profile selection.
                        If None, uses scheduled_departure_time to determine hour.

        Returns:
            Delay amount in minutes

        Example:
            >>> from src.delay_model import DelayModel
            >>> delay_model = DelayModel(seed=42)
            >>> route = FlightRoute(...)
            >>> delay_min = route.apply_delay(delay_model, hour_of_day=8)
            >>> print(f"Delayed by {delay_min:.1f} minutes")
        """
        # Calculate actual departure time
        self.actual_departure_time = delay_model.get_actual_departure_time(
            scheduled_time=self.scheduled_departure_time,
            hour_of_day=hour_of_day
        )

        # Calculate and return delay in minutes
        delay_minutes = (self.actual_departure_time - self.scheduled_departure_time) / 60.0

        return delay_minutes

    def calculate_great_circle_distance(self) -> float:
        """
        Calculate great-circle distance between departure and arrival airports.

        Uses haversine formula to compute the shortest distance over Earth's surface
        between the first and last waypoints of this route.

        Returns:
            Distance in nautical miles

        Example:
            >>> route = FlightRoute(...)
            >>> distance_nm = route.calculate_great_circle_distance()
            >>> print(f"Great circle distance: {distance_nm:.0f} nm")
        """
        if not self.waypoints or len(self.waypoints) < 2:
            return 0.0

        # Get first and last waypoints (departure and arrival)
        departure_lat, departure_lon, _ = self.waypoints[0]
        arrival_lat, arrival_lon, _ = self.waypoints[-1]

        # Calculate haversine distance
        distance_nm = haversine_distance(
            departure_lat, departure_lon,
            arrival_lat, arrival_lon
        )

        return distance_nm

    def get_route_summary(self) -> dict:
        """Get summary information about this route."""
        summary = {
            'route_id': self.route_id,
            'departure': self.departure_icao,
            'arrival': self.arrival_icao,
            'aircraft_type': self.aircraft_type,
            'distance_nm': self.total_distance_nm,
            'num_waypoints': len(self.waypoints),
            'initial_heading': self.initial_heading_deg,
            'initial_speed': self.initial_speed_knots,
            'max_altitude': max((wp[2] for wp in self.waypoints), default=0),
            'scheduled_departure_time': self.scheduled_departure_time,
            'actual_departure_time': self.actual_departure_time
        }

        # Add delay information if available
        if self.actual_departure_time is not None:
            summary['delay_minutes'] = (self.actual_departure_time - self.scheduled_departure_time) / 60.0

        return summary


def calculate_great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula.

    Args:
        lat1, lon1: First point (decimal degrees)
        lat2, lon2: Second point (decimal degrees)

    Returns:
        Distance in nautical miles
    """
    # Convert to radians
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


def calculate_initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing (heading) from point 1 to point 2.

    Args:
        lat1, lon1: Starting point (decimal degrees)
        lat2, lon2: Ending point (decimal degrees)

    Returns:
        Initial bearing in degrees (0-360)
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate bearing
    dlon = lon2_rad - lon1_rad
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

    bearing_rad = np.arctan2(x, y)
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360

    return bearing_deg


def interpolate_great_circle(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    fraction: float
) -> Tuple[float, float]:
    """
    Interpolate a point along a great circle route.

    Args:
        lat1, lon1: Starting point (decimal degrees)
        lat2, lon2: Ending point (decimal degrees)
        fraction: Position along route (0.0 = start, 1.0 = end)

    Returns:
        Tuple of (lat, lon) at the specified fraction along the route
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate angular distance
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    angular_distance = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Interpolate
    a_frac = np.sin((1 - fraction) * angular_distance) / np.sin(angular_distance)
    b_frac = np.sin(fraction * angular_distance) / np.sin(angular_distance)

    x = a_frac * np.cos(lat1_rad) * np.cos(lon1_rad) + b_frac * np.cos(lat2_rad) * np.cos(lon2_rad)
    y = a_frac * np.cos(lat1_rad) * np.sin(lon1_rad) + b_frac * np.cos(lat2_rad) * np.sin(lon2_rad)
    z = a_frac * np.sin(lat1_rad) + b_frac * np.sin(lat2_rad)

    lat_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon_rad = np.arctan2(y, x)

    return np.degrees(lat_rad), np.degrees(lon_rad)


def generate_waypoints(
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    num_waypoints: Optional[int] = None
) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Generate realistic flight path waypoints with climb, cruise, and descent phases.

    Creates waypoints along a great circle route with realistic altitude profile:
    - Climb phase: 0 → cruise altitude over ~100nm
    - Cruise phase: Constant altitude for middle segment
    - Descent phase: Cruise altitude → 0 over ~100nm

    Args:
        origin_lat, origin_lon: Departure airport coordinates (decimal degrees)
        dest_lat, dest_lon: Arrival airport coordinates (decimal degrees)
        num_waypoints: Number of waypoints to generate (None = auto based on distance)

    Returns:
        Tuple of (waypoints_list, total_distance_nm) where waypoints_list contains
        (lat, lon, altitude_ft) tuples
    """
    # Calculate total distance
    total_distance_nm = calculate_great_circle_distance(origin_lat, origin_lon, dest_lat, dest_lon)

    # Determine number of waypoints based on distance
    if num_waypoints is None:
        if total_distance_nm < 200:
            num_waypoints = 5  # Short flight: fewer waypoints
        elif total_distance_nm < 1000:
            num_waypoints = 7  # Medium flight
        else:
            num_waypoints = 10  # Long flight: more waypoints

    # Generate waypoints along great circle
    waypoints = []

    for i in range(num_waypoints):
        # Calculate fraction along route
        fraction = i / (num_waypoints - 1) if num_waypoints > 1 else 0.5

        # Interpolate position
        lat, lon = interpolate_great_circle(origin_lat, origin_lon, dest_lat, dest_lon, fraction)

        # Calculate altitude based on flight phase
        distance_from_origin = fraction * total_distance_nm
        distance_to_dest = total_distance_nm - distance_from_origin

        if distance_from_origin < CLIMB_DISTANCE_NM and total_distance_nm > CLIMB_DISTANCE_NM:
            # Climb phase: linear climb from 0 to cruise altitude
            altitude_ft = (distance_from_origin / CLIMB_DISTANCE_NM) * CRUISE_ALTITUDE_FT
        elif distance_to_dest < DESCENT_DISTANCE_NM and total_distance_nm > DESCENT_DISTANCE_NM:
            # Descent phase: linear descent from cruise to 0
            altitude_ft = (distance_to_dest / DESCENT_DISTANCE_NM) * CRUISE_ALTITUDE_FT
        else:
            # Cruise phase: constant altitude
            # For very short flights, use reduced cruise altitude
            if total_distance_nm < 200:
                altitude_ft = min(CRUISE_ALTITUDE_FT, 15000)  # Lower altitude for short flights
            else:
                altitude_ft = CRUISE_ALTITUDE_FT

        waypoints.append((lat, lon, altitude_ft))

    return waypoints, total_distance_nm


def select_aircraft_type(distance_nm: float) -> str:
    """
    Select appropriate aircraft type based on route distance.

    Args:
        distance_nm: Route distance in nautical miles

    Returns:
        ICAO aircraft type code
    """
    if distance_nm < 500:
        # Short-haul: Boeing 737-800
        return "B738"
    elif distance_nm < 3000:
        # Medium-haul: Airbus A320
        return "A320"
    else:
        # Long-haul: Boeing 777-300ER
        return "B77W"


def load_routes_from_data(
    jfk_routes_file: Optional[Path] = None,
    airports_file: Optional[Path] = None,
    max_routes: Optional[int] = None
) -> List[FlightRoute]:
    """
    Load flight routes from filtered JFK route data.

    Reads the filtered routes CSV, looks up airport coordinates,
    generates waypoints, and creates FlightRoute objects.

    Args:
        jfk_routes_file: Path to JFK routes CSV (default: data/jfk_routes.csv)
        airports_file: Path to airports CSV (default: data/airports_raw.csv)
        max_routes: Maximum number of routes to load (None = all)

    Returns:
        List of FlightRoute objects
    """
    # Default file paths
    if jfk_routes_file is None:
        jfk_routes_file = Path(__file__).parent.parent / "data" / "jfk_routes.csv"
    if airports_file is None:
        airports_file = Path(__file__).parent.parent / "data" / "airports_raw.csv"

    print(f"Loading routes from {jfk_routes_file}...")

    # Load data
    try:
        routes_df = pd.read_csv(jfk_routes_file)
        airports_df = pd.read_csv(airports_file)
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print("  Run data acquisition first!")
        return []

    print(f"✓ Loaded {len(routes_df)} routes")

    # Create airport lookup dictionary
    airport_lookup = {}
    for _, row in airports_df.iterrows():
        iata = row['iata']
        if pd.notna(iata):
            airport_lookup[iata] = {
                'icao': row['icao'] if pd.notna(row['icao']) else iata,
                'lat': row['latitude'],
                'lon': row['longitude'],
                'name': row['name']
            }

    # Get unique route pairs (deduplicate by source-destination)
    unique_routes = routes_df[['source_iata', 'dest_iata']].drop_duplicates()

    if max_routes is not None:
        unique_routes = unique_routes.head(max_routes)

    print(f"Generating {len(unique_routes)} unique routes...")

    # Generate FlightRoute objects
    flight_routes = []
    skipped = 0

    for idx, (_, row) in enumerate(unique_routes.iterrows()):
        source_iata = row['source_iata']
        dest_iata = row['dest_iata']

        # Look up airport coordinates
        if source_iata not in airport_lookup or dest_iata not in airport_lookup:
            skipped += 1
            continue

        source = airport_lookup[source_iata]
        dest = airport_lookup[dest_iata]

        # Generate waypoints
        waypoints, distance_nm = generate_waypoints(
            source['lat'], source['lon'],
            dest['lat'], dest['lon']
        )

        # Select aircraft type based on distance
        aircraft_type = select_aircraft_type(distance_nm)

        # Calculate initial heading
        initial_heading = calculate_initial_bearing(
            source['lat'], source['lon'],
            dest['lat'], dest['lon']
        )

        # Create route ID
        route_id = f"{source_iata}_{dest_iata}_{idx:04d}"

        # Create FlightRoute object
        route = FlightRoute(
            route_id=route_id,
            aircraft_type=aircraft_type,
            departure_icao=source['icao'],
            arrival_icao=dest['icao'],
            waypoints=waypoints,
            initial_speed_knots=INITIAL_SPEED_KTS,
            initial_heading_deg=initial_heading,
            initial_altitude_ft=0.0,
            total_distance_nm=distance_nm
        )

        flight_routes.append(route)

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Generated {idx + 1}/{len(unique_routes)} routes...")

    print(f"✓ Generated {len(flight_routes)} flight routes")
    if skipped > 0:
        print(f"  Skipped {skipped} routes (missing airport data)")

    return flight_routes


def get_route_statistics(routes: List[FlightRoute]) -> dict:
    """
    Calculate statistics about a list of routes.

    Args:
        routes: List of FlightRoute objects

    Returns:
        Dictionary with statistics
    """
    if not routes:
        return {}

    distances = [r.total_distance_nm for r in routes]
    aircraft_types = [r.aircraft_type for r in routes]

    # Count aircraft types
    aircraft_counts = {}
    for ac_type in aircraft_types:
        aircraft_counts[ac_type] = aircraft_counts.get(ac_type, 0) + 1

    return {
        'total_routes': len(routes),
        'min_distance_nm': min(distances),
        'max_distance_nm': max(distances),
        'avg_distance_nm': np.mean(distances),
        'median_distance_nm': np.median(distances),
        'aircraft_distribution': aircraft_counts,
        'total_waypoints': sum(len(r.waypoints) for r in routes)
    }


if __name__ == "__main__":
    """Test route generation when module is executed directly."""
    print("Testing route generation...")

    # Test waypoint generation for a sample route
    # JFK to LAX (approximately)
    jfk_lat, jfk_lon = 40.6413, -73.7781
    lax_lat, lax_lon = 33.9416, -118.4085

    waypoints, distance = generate_waypoints(jfk_lat, jfk_lon, lax_lat, lax_lon)

    print(f"\nSample route: JFK → LAX")
    print(f"Distance: {distance:.1f} nm")
    print(f"Waypoints: {len(waypoints)}")
    print("\nWaypoint details:")
    for i, (lat, lon, alt) in enumerate(waypoints):
        print(f"  WP{i+1}: ({lat:.4f}, {lon:.4f}) at {alt:.0f} ft")
