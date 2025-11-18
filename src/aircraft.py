"""
Aircraft kinematic model for DeepSky ATC.

Represents a single aircraft in the simulation with position, velocity,
heading, and status tracking. Provides methods for position updates and
waypoint navigation.
"""

from typing import Dict, Optional
import numpy as np

from src.route_generator import FlightRoute
from src.physics import (
    haversine_distance,
    calculate_bearing,
    update_position_by_bearing,
    interpolate_heading,
    calculate_climb_rate,
    feet_per_minute_to_feet_per_second
)


# Performance parameters (realistic jet aircraft values)
CLIMB_RATE_FPM = 2000          # Feet per minute (typical climb rate)
DESCENT_RATE_FPM = 1500        # Feet per minute (typical descent rate)
CRUISE_SPEED_KNOTS = 450       # Knots (typical jet cruise speed)
CLIMB_SPEED_KNOTS = 250        # Knots (speed during climb phase)
DESCENT_SPEED_KNOTS = 250      # Knots (speed during descent phase)
TAXI_SPEED_KNOTS = 20          # Knots (ground taxi speed)
MAX_TURN_RATE_DEG_PER_SEC = 3  # Degrees per second (standard rate turn)


class Aircraft:
    """
    Represents a single aircraft in the ATC simulation.

    Tracks position, velocity, heading, and flight status. Provides methods
    for updating position based on kinematics and managing waypoint navigation.

    Attributes:
        id: Unique aircraft identifier (e.g., "AAL123")
        route: FlightRoute object defining the planned flight path
        current_position: Dict with 'lat', 'lon', 'alt' (degrees, feet)
        current_velocity: Current airspeed in knots
        current_heading: Current heading in degrees (0-360)
        target_waypoint_index: Index of next waypoint in route
        status: Current flight phase ('TAXI', 'CLIMBING', 'CRUISING', 'DESCENDING', 'LANDING', 'LANDED')
        time_elapsed: Seconds since departure
    """

    def __init__(self, id: str, route: FlightRoute, departure_time: float = 0.0):
        """
        Initialize aircraft at departure airport.

        Args:
            id: Unique aircraft identifier
            route: FlightRoute object with waypoints and parameters
            departure_time: Simulation time of departure (seconds)
        """
        self.id = id
        self.route = route
        self.time_elapsed = departure_time

        # Initialize at first waypoint (departure airport)
        if not route.waypoints:
            raise ValueError(f"Route for {id} has no waypoints")

        start_lat, start_lon, start_alt = route.waypoints[0]

        self.current_position = {
            'lat': start_lat,
            'lon': start_lon,
            'alt': start_alt
        }

        # Set initial velocity (taxi speed on ground)
        self.current_velocity = TAXI_SPEED_KNOTS

        # Calculate initial heading toward first waypoint
        # (or second waypoint if we need to move from first)
        if len(route.waypoints) > 1:
            next_lat, next_lon, _ = route.waypoints[1]
            self.current_heading = calculate_bearing(
                start_lat, start_lon,
                next_lat, next_lon
            )
        else:
            self.current_heading = route.initial_heading_deg

        # Start at waypoint 0, targeting waypoint 1
        self.target_waypoint_index = 1 if len(route.waypoints) > 1 else 0

        # Initial status
        self.status = 'TAXI'

    def calculate_heading_to_waypoint(self, target_lat: float, target_lon: float) -> float:
        """
        Calculate bearing from current position to target waypoint.

        Args:
            target_lat: Target waypoint latitude in decimal degrees
            target_lon: Target waypoint longitude in decimal degrees

        Returns:
            Bearing in degrees (0-360)
        """
        return calculate_bearing(
            self.current_position['lat'],
            self.current_position['lon'],
            target_lat,
            target_lon
        )

    def _determine_target_speed(self) -> float:
        """
        Determine appropriate target speed based on current flight phase.

        Returns:
            Target speed in knots
        """
        if self.status == 'TAXI':
            return TAXI_SPEED_KNOTS
        elif self.status == 'CLIMBING':
            return CLIMB_SPEED_KNOTS
        elif self.status == 'DESCENDING' or self.status == 'LANDING':
            return DESCENT_SPEED_KNOTS
        elif self.status == 'CRUISING':
            return CRUISE_SPEED_KNOTS
        elif self.status == 'LANDED':
            return 0.0
        else:
            return CRUISE_SPEED_KNOTS

    def _update_flight_status(self, target_alt: float) -> None:
        """
        Update flight status based on current and target altitude.

        Args:
            target_alt: Target altitude in feet
        """
        current_alt = self.current_position['alt']
        alt_diff = target_alt - current_alt

        # If we've reached the final waypoint
        if self.target_waypoint_index >= len(self.route.waypoints):
            if current_alt < 100:
                self.status = 'LANDED'
            else:
                self.status = 'LANDING'
        # Determine phase based on altitude change
        elif abs(alt_diff) < 500:  # Within 500 feet of target = cruising
            self.status = 'CRUISING'
        elif alt_diff > 500:  # Need to climb
            self.status = 'CLIMBING'
        elif alt_diff < -500:  # Need to descend
            self.status = 'DESCENDING'

    def update_position(self, delta_t: float = 1.0) -> Dict[str, float]:
        """
        Update aircraft position based on current velocity and heading.

        Uses kinematic equations to calculate new position after delta_t seconds.
        Updates position, altitude, heading toward waypoint, and flight status.

        Args:
            delta_t: Time step in seconds (default 1.0)

        Returns:
            Dictionary with updated position {'lat', 'lon', 'alt'}
        """
        # Don't update if landed
        if self.status == 'LANDED':
            return self.current_position.copy()

        # Get target waypoint
        if self.target_waypoint_index >= len(self.route.waypoints):
            # No more waypoints, maintain current state
            return self.current_position.copy()

        target_lat, target_lon, target_alt = self.route.waypoints[self.target_waypoint_index]

        # Calculate desired heading to waypoint
        target_heading = self.calculate_heading_to_waypoint(target_lat, target_lon)

        # Update heading (aircraft can't turn instantly)
        max_turn_this_step = MAX_TURN_RATE_DEG_PER_SEC * delta_t
        self.current_heading = interpolate_heading(
            self.current_heading,
            target_heading,
            max_turn_this_step
        )

        # Update flight status based on target altitude
        self._update_flight_status(target_alt)

        # Determine target speed and smoothly adjust current speed
        target_speed = self._determine_target_speed()
        speed_change_rate = 5.0  # knots per second
        max_speed_change = speed_change_rate * delta_t

        if abs(target_speed - self.current_velocity) < max_speed_change:
            self.current_velocity = target_speed
        elif target_speed > self.current_velocity:
            self.current_velocity += max_speed_change
        else:
            self.current_velocity -= max_speed_change

        # Calculate horizontal distance traveled
        # Convert knots to nautical miles per second
        speed_nm_per_sec = self.current_velocity / 3600.0
        distance_traveled_nm = speed_nm_per_sec * delta_t

        # Update horizontal position based on heading and distance
        new_lat, new_lon = update_position_by_bearing(
            self.current_position['lat'],
            self.current_position['lon'],
            self.current_heading,
            distance_traveled_nm
        )

        # Update altitude (vertical motion)
        current_alt = self.current_position['alt']

        # Determine climb/descent rate
        if self.status == 'CLIMBING':
            max_rate_fpm = CLIMB_RATE_FPM
        elif self.status == 'DESCENDING' or self.status == 'LANDING':
            max_rate_fpm = DESCENT_RATE_FPM
        else:
            max_rate_fpm = 500  # Small adjustments during cruise

        # Calculate vertical speed needed
        vertical_speed_fpm = calculate_climb_rate(current_alt, target_alt, max_rate_fpm)

        # Convert to feet per second and update altitude
        vertical_speed_fps = feet_per_minute_to_feet_per_second(vertical_speed_fpm)
        alt_change = vertical_speed_fps * delta_t
        new_alt = current_alt + alt_change

        # Ensure altitude doesn't go negative
        new_alt = max(0.0, new_alt)

        # Update position
        self.current_position = {
            'lat': new_lat,
            'lon': new_lon,
            'alt': new_alt
        }

        # Increment time
        self.time_elapsed += delta_t

        return self.current_position.copy()

    def check_waypoint_reached(self, threshold_nm: float = 2.0) -> bool:
        """
        Check if aircraft has reached the current target waypoint.

        If within threshold distance, advances to next waypoint and updates
        heading and status accordingly.

        Args:
            threshold_nm: Distance threshold in nautical miles (default 2.0)

        Returns:
            True if waypoint was reached and we advanced, False otherwise
        """
        # Check if we have a valid target waypoint
        if self.target_waypoint_index >= len(self.route.waypoints):
            return False

        # Calculate distance to target waypoint
        target_lat, target_lon, target_alt = self.route.waypoints[self.target_waypoint_index]

        distance_to_waypoint = haversine_distance(
            self.current_position['lat'],
            self.current_position['lon'],
            target_lat,
            target_lon
        )

        # Check if within threshold
        if distance_to_waypoint <= threshold_nm:
            # Waypoint reached! Advance to next
            self.target_waypoint_index += 1

            # Update heading to new target (if there is one)
            if self.target_waypoint_index < len(self.route.waypoints):
                next_lat, next_lon, next_alt = self.route.waypoints[self.target_waypoint_index]
                self.current_heading = self.calculate_heading_to_waypoint(next_lat, next_lon)

                # Update status based on new target altitude
                self._update_flight_status(next_alt)
            else:
                # No more waypoints - we're landing/landed
                if self.current_position['alt'] < 100:
                    self.status = 'LANDED'
                else:
                    self.status = 'LANDING'

            return True

        return False

    def get_state(self) -> Dict:
        """
        Get complete current state of the aircraft.

        Returns:
            Dictionary containing all aircraft state information:
            - id: Aircraft identifier
            - position: Current lat/lon/alt
            - velocity: Current airspeed in knots
            - heading: Current heading in degrees
            - status: Current flight phase
            - route_info: Route identification
            - waypoint_info: Current waypoint progress
            - time: Time elapsed since departure
        """
        # Get target waypoint info
        if self.target_waypoint_index < len(self.route.waypoints):
            target_wp = self.route.waypoints[self.target_waypoint_index]
            target_lat, target_lon, target_alt = target_wp
            distance_to_target = haversine_distance(
                self.current_position['lat'],
                self.current_position['lon'],
                target_lat,
                target_lon
            )
        else:
            target_lat, target_lon, target_alt = None, None, None
            distance_to_target = 0.0

        return {
            'id': self.id,
            'position': self.current_position.copy(),
            'velocity': self.current_velocity,
            'heading': self.current_heading,
            'status': self.status,
            'route_info': {
                'route_id': self.route.route_id,
                'departure': self.route.departure_icao,
                'arrival': self.route.arrival_icao,
                'aircraft_type': self.route.aircraft_type
            },
            'waypoint_info': {
                'current_index': self.target_waypoint_index,
                'total_waypoints': len(self.route.waypoints),
                'target_position': {
                    'lat': target_lat,
                    'lon': target_lon,
                    'alt': target_alt
                } if target_lat is not None else None,
                'distance_to_target_nm': distance_to_target
            },
            'time_elapsed': self.time_elapsed
        }

    def __repr__(self) -> str:
        """String representation of the aircraft."""
        return (
            f"Aircraft(id='{self.id}', {self.route.departure_icao}→{self.route.arrival_icao}, "
            f"pos=({self.current_position['lat']:.4f}, {self.current_position['lon']:.4f}, "
            f"{self.current_position['alt']:.0f}ft), "
            f"hdg={self.current_heading:.0f}°, spd={self.current_velocity:.0f}kts, "
            f"status={self.status}, wp={self.target_waypoint_index}/{len(self.route.waypoints)})"
        )


if __name__ == "__main__":
    """Test aircraft module when executed directly."""
    print("Testing Aircraft class...")
    print()

    # Create a simple test route
    from src.route_generator import FlightRoute

    test_route = FlightRoute(
        route_id="TEST001",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KBOS",
        waypoints=[
            (40.6413, -73.7781, 0),      # JFK
            (41.0, -72.5, 15000),         # Waypoint 1 (climbing)
            (41.5, -71.5, 35000),         # Waypoint 2 (cruise)
            (42.0, -71.0, 35000),         # Waypoint 3 (cruise)
            (42.3643, -71.0052, 0)        # BOS
        ],
        total_distance_nm=185
    )

    # Create aircraft
    aircraft = Aircraft(id="TEST001", route=test_route)

    print(f"Initial state: {aircraft}")
    print()

    # Simulate a few time steps
    print("Simulating 10 seconds of flight:")
    for i in range(10):
        aircraft.update_position(delta_t=1.0)
        aircraft.check_waypoint_reached(threshold_nm=2.0)

        if i % 2 == 0:
            print(f"  t={i}s: {aircraft}")

    print()
    print("Aircraft state test complete!")
