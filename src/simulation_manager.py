"""
Simulation management for DeepSky ATC.

Orchestrates the ATC simulation by managing aircraft, advancing time,
and coordinating with output interfaces.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import random

from src.airspace import Airspace
from src.delay_model import DelayModel
from src.simulation_output import SimulationOutput
from src.aircraft import Aircraft
from src.route_generator import FlightRoute, load_routes_from_data
from src.conflict_detection import ConflictDetector, ConflictTracker


class SimulationManager:
    """
    Manages the ATC simulation lifecycle.

    Coordinates aircraft creation, position updates, waypoint tracking,
    conflict detection, and state export. Provides high-level interface
    for running simulations.

    Attributes:
        airspace: Airspace boundary configuration
        delay_model: Delay model for departure times
        output: Output interface for state export
        conflict_detector: Conflict detection engine
        conflict_tracker: Conflict history and statistics
        aircraft_list: List of active Aircraft objects
        simulation_time: Current simulation time in seconds
        total_aircraft_spawned: Total number of aircraft created
        completed_flights: Number of aircraft that have landed
    """

    def __init__(
        self,
        airspace: Airspace,
        delay_model: DelayModel,
        output: SimulationOutput
    ):
        """
        Initialize simulation manager.

        Args:
            airspace: Airspace configuration
            delay_model: Delay model for departure delays
            output: Output interface for state export
        """
        self.airspace = airspace
        self.delay_model = delay_model
        self.output = output

        # Conflict detection
        self.conflict_detector = ConflictDetector()
        self.conflict_tracker = ConflictTracker()

        # Simulation state
        self.aircraft_list: List[Aircraft] = []
        self.simulation_time: float = 0.0
        self.total_aircraft_spawned: int = 0
        self.completed_flights: int = 0

    def add_aircraft(
        self,
        route: FlightRoute,
        departure_time: float
    ) -> str:
        """
        Add aircraft to simulation.

        Creates an Aircraft instance from the route, applies departure delay,
        and adds to active aircraft list.

        Args:
            route: FlightRoute defining the flight path
            departure_time: Scheduled departure time in seconds

        Returns:
            Aircraft ID

        Example:
            >>> manager = SimulationManager(airspace, delay_model, output)
            >>> route = FlightRoute(...)
            >>> aircraft_id = manager.add_aircraft(route, departure_time=0)
            >>> print(f"Added aircraft: {aircraft_id}")
        """
        # Set scheduled departure time on route
        route.scheduled_departure_time = departure_time

        # Apply departure delay
        # Extract hour from departure_time for time-based profile selection
        hour_of_day = int((departure_time / 3600) % 24)
        delay_minutes = route.apply_delay(self.delay_model, hour_of_day=hour_of_day)

        # Create unique aircraft ID
        aircraft_id = f"{route.departure_icao}{route.arrival_icao}_{self.total_aircraft_spawned:04d}"

        # Create aircraft instance
        # Aircraft will start at actual_departure_time
        aircraft = Aircraft(
            id=aircraft_id,
            route=route,
            departure_time=route.actual_departure_time
        )

        # Add to active list (will be inactive until departure time)
        self.aircraft_list.append(aircraft)
        self.total_aircraft_spawned += 1

        return aircraft_id

    def step(self, delta_t: float = 1.0) -> int:
        """
        Advance simulation by one time step.

        Updates all active aircraft positions, checks waypoint progress,
        removes landed aircraft, and exports current state.

        Args:
            delta_t: Time step in seconds (default 1.0)

        Returns:
            Number of active aircraft after step

        Example:
            >>> manager = SimulationManager(airspace, delay_model, output)
            >>> active_count = manager.step(delta_t=1.0)
            >>> print(f"{active_count} aircraft in flight")
        """
        # Advance simulation time
        self.simulation_time += delta_t

        # Filter active aircraft (those that have departed and not landed)
        active_aircraft = []

        for aircraft in self.aircraft_list:
            # Check if aircraft has departed
            if self.simulation_time < aircraft.time_elapsed:
                # Not yet departed - keep in list but don't update
                active_aircraft.append(aircraft)
                continue

            # Aircraft is in flight - update position
            aircraft.update_position(delta_t=delta_t)
            aircraft.check_waypoint_reached(threshold_nm=2.0)

            # Check if landed
            if aircraft.status == 'LANDED':
                self.completed_flights += 1
                # Don't add to active list (remove from simulation)
            else:
                active_aircraft.append(aircraft)

        # Update aircraft list
        self.aircraft_list = active_aircraft

        # Only export aircraft that are actually in flight (not waiting to depart)
        in_flight_aircraft = [
            ac for ac in self.aircraft_list
            if self.simulation_time >= ac.time_elapsed
        ]

        # Detect conflicts
        conflicts = self.conflict_detector.detect_all_conflicts(
            in_flight_aircraft,
            self.simulation_time
        )

        # Update conflict tracker
        self.conflict_tracker.update(self.simulation_time, conflicts)

        # Export current state (including conflicts)
        self.output.export_state(
            self.simulation_time,
            in_flight_aircraft,
            conflicts
        )

        return len(in_flight_aircraft)

    def run(
        self,
        duration_seconds: float,
        time_step: float = 1.0,
        progress_interval: float = 60.0
    ) -> Dict[str, Any]:
        """
        Run simulation for specified duration.

        Repeatedly calls step() to advance simulation, printing progress
        at regular intervals.

        Args:
            duration_seconds: Total simulation duration in seconds
            time_step: Time step for each iteration in seconds (default 1.0)
            progress_interval: Print progress every N seconds (default 60.0)

        Returns:
            Dictionary with final simulation statistics

        Example:
            >>> manager = SimulationManager(airspace, delay_model, output)
            >>> # Add aircraft...
            >>> stats = manager.run(duration_seconds=3600, time_step=1.0)
            >>> print(f"Completed {stats['completed_flights']} flights")
        """
        print(f"\nStarting simulation...")
        print(f"  Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
        print(f"  Time step: {time_step} seconds")
        print(f"  Total aircraft: {self.total_aircraft_spawned}")
        print()

        num_steps = int(duration_seconds / time_step)
        next_progress_time = progress_interval

        for step_num in range(num_steps):
            # Advance simulation
            active_count = self.step(delta_t=time_step)

            # Print progress
            if self.simulation_time >= next_progress_time:
                print(f"  t={self.simulation_time:6.0f}s: "
                      f"{active_count:3d} active, "
                      f"{self.completed_flights:3d} completed, "
                      f"{self.total_aircraft_spawned - self.completed_flights - active_count:3d} waiting")
                next_progress_time += progress_interval

        print()
        print("Simulation complete!")
        print()

        # Return final statistics
        return self.get_statistics()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current simulation statistics.

        Returns:
            Dictionary with simulation statistics:
            - total_flights: Total aircraft spawned
            - completed_flights: Number of landed aircraft
            - active_flights: Number currently in flight
            - waiting_flights: Number waiting to depart
            - total_simulation_time: Current simulation time
            - conflicts: Conflict statistics from tracker

        Example:
            >>> manager = SimulationManager(airspace, delay_model, output)
            >>> stats = manager.get_statistics()
            >>> print(f"Active: {stats['active_flights']}")
            >>> print(f"Conflicts: {stats['conflicts']['total_conflicts']}")
        """
        # Count aircraft in different states
        active_count = 0
        waiting_count = 0

        for aircraft in self.aircraft_list:
            if self.simulation_time >= aircraft.time_elapsed:
                active_count += 1
            else:
                waiting_count += 1

        # Get conflict statistics
        conflict_stats = self.conflict_tracker.get_statistics()

        return {
            'total_flights': self.total_aircraft_spawned,
            'completed_flights': self.completed_flights,
            'active_flights': active_count,
            'waiting_flights': waiting_count,
            'total_simulation_time': self.simulation_time,
            'conflicts': conflict_stats
        }

    def __repr__(self) -> str:
        """String representation of simulation manager."""
        stats = self.get_statistics()
        return (
            f"SimulationManager(t={self.simulation_time:.0f}s, "
            f"active={stats['active_flights']}, "
            f"completed={stats['completed_flights']}, "
            f"total={stats['total_flights']})"
        )


def create_demo_simulation(
    num_aircraft: int = 20,
    duration: float = 3600,
    output_mode: str = "file",
    seed: Optional[int] = None
) -> SimulationManager:
    """
    Create a demo simulation with pre-configured routes and settings.

    Convenience function for quick testing and demonstration. Loads
    real routes from data, creates aircraft with staggered departures,
    and sets up output interface.

    Args:
        num_aircraft: Number of aircraft to create (default 20)
        duration: Simulation duration in seconds (default 3600 = 1 hour)
        output_mode: Output mode - "file", "stdout", or "both" (default "file")
        seed: Random seed for reproducibility (default None)

    Returns:
        Configured SimulationManager ready to run

    Example:
        >>> # Quick demo simulation
        >>> manager = create_demo_simulation(num_aircraft=10, duration=600)
        >>> stats = manager.run(duration_seconds=600)
        >>> print(f"Completed {stats['completed_flights']} flights")
    """
    print("=" * 70)
    print("Creating Demo Simulation")
    print("=" * 70)
    print()

    # Set random seed if specified
    if seed is not None:
        random.seed(seed)

    # Load airspace
    print("Loading airspace configuration...")
    airspace = Airspace()
    print(f"  ✓ Loaded: {airspace.name}")
    print(f"  Center: ({airspace.center_lat:.4f}, {airspace.center_lon:.4f})")
    print(f"  Radius: {airspace.radius_nm} nm")
    print()

    # Load delay model
    print("Loading delay model...")
    delay_model = DelayModel(seed=seed)
    print(f"  ✓ Loaded: {delay_model}")
    print()

    # Create output interface
    print(f"Creating output interface (mode: {output_mode})...")
    output = SimulationOutput(
        output_dir="data/output",
        output_mode=output_mode,
        center_lat=airspace.center_lat,
        center_lon=airspace.center_lon
    )
    print(f"  ✓ Created: {output}")
    print()

    # Load routes
    print("Loading flight routes...")
    all_routes = load_routes_from_data(max_routes=num_aircraft)

    if len(all_routes) == 0:
        raise RuntimeError("No routes available. Run data acquisition first!")

    # Select random routes
    if len(all_routes) > num_aircraft:
        selected_routes = random.sample(all_routes, num_aircraft)
    else:
        selected_routes = all_routes

    print(f"  ✓ Selected {len(selected_routes)} routes")
    print()

    # Create simulation manager
    print("Creating simulation manager...")
    manager = SimulationManager(airspace, delay_model, output)
    print(f"  ✓ Created: {manager}")
    print()

    # Add aircraft with staggered departures
    print(f"Adding {len(selected_routes)} aircraft...")

    # Stagger departures over first 30 minutes
    departure_window = min(1800, duration / 2)  # 30 minutes or half duration
    departure_interval = departure_window / len(selected_routes)

    for i, route in enumerate(selected_routes):
        departure_time = i * departure_interval
        aircraft_id = manager.add_aircraft(route, departure_time)

        if (i + 1) % 5 == 0 or i == len(selected_routes) - 1:
            print(f"  Added {i + 1}/{len(selected_routes)} aircraft...")

    print()
    print(f"✓ Demo simulation ready!")
    print(f"  Aircraft: {num_aircraft}")
    print(f"  Departure window: {departure_window / 60:.1f} minutes")
    print(f"  Simulation duration: {duration / 60:.1f} minutes")
    print()

    return manager


if __name__ == "__main__":
    """Test simulation manager when module is executed directly."""
    print("Testing SimulationManager...")
    print()

    try:
        # Create demo simulation
        manager = create_demo_simulation(
            num_aircraft=5,
            duration=300,
            output_mode="stdout",
            seed=42
        )

        # Run simulation
        stats = manager.run(duration_seconds=300, time_step=1.0)

        print("Final Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run data acquisition first!")

    print()
    print("SimulationManager test complete!")
