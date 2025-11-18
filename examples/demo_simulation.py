"""
Demo simulation showing aircraft in flight.

Creates a simple simulation with a few aircraft and shows sample output.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.airspace import Airspace
from src.delay_model import DelayModel
from src.simulation_output import SimulationOutput
from src.simulation_manager import SimulationManager
from src.route_generator import FlightRoute, generate_waypoints


def main():
    """Run a simple demo simulation."""
    print("=" * 70)
    print("DeepSky ATC - Demo Simulation")
    print("=" * 70)
    print()

    # Create components
    print("Setting up simulation...")
    airspace = Airspace()
    delay_model = DelayModel(seed=42)
    output = SimulationOutput(output_mode="both")  # Both file and stdout

    # Create simulation manager
    manager = SimulationManager(airspace, delay_model, output)

    # Create a simple route (JFK to BOS - short flight)
    jfk_lat, jfk_lon = 40.6413, -73.7781
    bos_lat, bos_lon = 42.3643, -71.0052
    waypoints, distance = generate_waypoints(jfk_lat, jfk_lon, bos_lat, bos_lon)

    route = FlightRoute(
        route_id="DEMO_JFK_BOS",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KBOS",
        waypoints=waypoints,
        total_distance_nm=distance
    )

    print(f"Route: {route.departure_icao} → {route.arrival_icao}")
    print(f"Distance: {distance:.0f} nm")
    print(f"Waypoints: {len(waypoints)}")
    print()

    # Add aircraft with NO delay (depart immediately)
    route.scheduled_departure_time = 0.0
    route.actual_departure_time = 0.0  # Override delay

    aircraft_id = manager.add_aircraft(route, departure_time=0.0)
    # Manually set actual departure time to avoid delay
    manager.aircraft_list[0].time_elapsed = 0.0

    print(f"Added aircraft: {aircraft_id}")
    print()

    # Run simulation for 60 seconds
    print("Running simulation for 60 seconds...")
    print("Sample snapshots:")
    print()

    for t in range(0, 61, 10):
        # Step to time t
        while manager.simulation_time < t:
            manager.step(delta_t=1.0)

        # Get aircraft state
        if len(manager.aircraft_list) > 0:
            aircraft = manager.aircraft_list[0]
            state = aircraft.get_state()

            print(f"t={t:3d}s: {state['status']:12s} "
                  f"pos=({state['position']['lat']:.4f}, {state['position']['lon']:.4f}) "
                  f"alt={state['position']['alt']:6.0f}ft "
                  f"spd={state['velocity']:3.0f}kts "
                  f"hdg={state['heading']:3.0f}°")

    print()

    # Close output
    output.close()

    # Show statistics
    stats = manager.get_statistics()
    print("Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print(f"Output saved to: {output.get_output_file_path()}")
    print()


if __name__ == "__main__":
    main()
