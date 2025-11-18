"""
Simulation output interface for DeepSky ATC.

Exports simulation state to JSON format for visualization, analysis, and streaming
to external systems (e.g., Unreal Engine via gRPC).
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

from src.coordinates import lat_lon_alt_to_xyz, feet_to_meters


class SimulationOutput:
    """
    Manages simulation state export to files or stdout.

    Provides flexible output modes for real-time visualization, logging,
    and post-simulation analysis. Exports complete aircraft state including
    both geodetic (lat/lon/alt) and local Cartesian (x/y/z) coordinates.

    Attributes:
        output_dir: Directory for output files
        output_mode: One of "file", "stdout", "both"
        output_file: Path to current output file
        file_handle: Open file handle for writing
        center_lat: Reference latitude for coordinate conversion
        center_lon: Reference longitude for coordinate conversion
    """

    def __init__(
        self,
        output_dir: str = "data/output",
        output_mode: str = "file",
        center_lat: float = 40.6413,  # KJFK
        center_lon: float = -73.7781   # KJFK
    ):
        """
        Initialize simulation output interface.

        Args:
            output_dir: Directory to write output files (created if doesn't exist)
            output_mode: Output mode - "file", "stdout", or "both"
            center_lat: Reference latitude for local coordinate conversion
            center_lon: Reference longitude for local coordinate conversion

        Raises:
            ValueError: If output_mode is invalid
        """
        if output_mode not in ["file", "stdout", "both"]:
            raise ValueError(f"Invalid output_mode: {output_mode}. Must be 'file', 'stdout', or 'both'")

        self.output_dir = Path(output_dir)
        self.output_mode = output_mode
        self.output_file = None
        self.file_handle = None
        self.center_lat = center_lat
        self.center_lon = center_lon

        # Create output directory if using file output
        if self.output_mode in ["file", "both"]:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = self.output_dir / f"simulation_{timestamp}.json"

            # Open file for writing
            self.file_handle = open(self.output_file, 'w')

            # Write opening bracket for JSON array
            self.file_handle.write("[\n")
            self.first_write = True

    def export_state(
        self,
        simulation_time: float,
        aircraft_list: List,
        conflicts: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export current simulation state.

        Generates a complete snapshot of the simulation including all aircraft
        states with both geodetic and local Cartesian coordinates, plus any
        active conflicts.

        Args:
            simulation_time: Current simulation time in seconds
            aircraft_list: List of Aircraft objects
            conflicts: List of conflict dictionaries (optional)

        Returns:
            Dictionary containing simulation state

        Example:
            >>> output = SimulationOutput(output_mode="stdout")
            >>> state = output.export_state(100.0, [aircraft1, aircraft2], conflicts=[])
            >>> print(f"Active aircraft: {state['aircraft_count']}")
            >>> print(f"Active conflicts: {state['conflict_count']}")
        """
        if conflicts is None:
            conflicts = []
        # Generate simulation timestamp
        # Use simulation_time to create a datetime (assume starts at epoch or specific date)
        base_date = datetime(2024, 1, 15, 0, 0, 0)  # Arbitrary simulation start date
        current_datetime = base_date.timestamp() + simulation_time
        simulation_date = datetime.fromtimestamp(current_datetime).isoformat()

        # Build aircraft state array
        aircraft_states = []

        for aircraft in aircraft_list:
            # Get aircraft state
            state = aircraft.get_state()

            # Get position in lat/lon/alt
            lat = state['position']['lat']
            lon = state['position']['lon']
            alt_ft = state['position']['alt']

            # Convert altitude to meters for coordinate conversion
            alt_m = feet_to_meters(alt_ft)

            # Convert to local Cartesian coordinates (x, y, z in meters)
            x, y, z = lat_lon_alt_to_xyz(
                lat, lon, alt_m,
                self.center_lat, self.center_lon
            )

            # Build aircraft state entry
            aircraft_state = {
                'id': state['id'],
                'position': {
                    'lat': float(lat),
                    'lon': float(lon),
                    'alt': float(alt_ft),  # feet
                    'x': float(x),         # meters (East)
                    'y': float(y),         # meters (North)
                    'z': float(z)          # meters (Up/altitude)
                },
                'velocity': float(state['velocity']),  # knots
                'heading': float(state['heading']),    # degrees
                'status': state['status'],
                'route_info': {
                    'departure': state['route_info']['departure'],
                    'arrival': state['route_info']['arrival'],
                    'aircraft_type': state['route_info']['aircraft_type']
                },
                'waypoint_progress': {
                    'current_waypoint': state['waypoint_info']['current_index'],
                    'total_waypoints': state['waypoint_info']['total_waypoints']
                },
                'time_in_flight': float(state['time_elapsed'])  # seconds
            }

            aircraft_states.append(aircraft_state)

        # Build complete state snapshot
        state_snapshot = {
            'timestamp': float(simulation_time),
            'simulation_date': simulation_date,
            'aircraft_count': len(aircraft_list),
            'aircraft': aircraft_states,
            'conflict_count': len(conflicts),
            'conflicts': conflicts
        }

        # Output based on mode
        if self.output_mode in ["file", "both"]:
            self._write_to_file(state_snapshot)

        if self.output_mode in ["stdout", "both"]:
            self._write_to_stdout(state_snapshot)

        return state_snapshot

    def _write_to_file(self, state_snapshot: Dict[str, Any]) -> None:
        """
        Write state snapshot to JSON file.

        Args:
            state_snapshot: State dictionary to write
        """
        if self.file_handle is None:
            return

        # Add comma if not first entry
        if not self.first_write:
            self.file_handle.write(",\n")
        else:
            self.first_write = False

        # Write JSON (indented for readability)
        json_str = json.dumps(state_snapshot, indent=2)
        self.file_handle.write(json_str)
        self.file_handle.flush()  # Ensure data is written

    def _write_to_stdout(self, state_snapshot: Dict[str, Any]) -> None:
        """
        Write state snapshot to stdout.

        Args:
            state_snapshot: State dictionary to write
        """
        # Print compact JSON to stdout
        json_str = json.dumps(state_snapshot)
        print(json_str)
        sys.stdout.flush()

    def close(self) -> None:
        """
        Close output file and clean up resources.

        Should be called when simulation is complete to ensure proper
        JSON file structure and resource cleanup.
        """
        if self.file_handle is not None:
            # Write closing bracket for JSON array
            self.file_handle.write("\n]")
            self.file_handle.close()
            self.file_handle = None

            print(f"\n✓ Simulation output written to: {self.output_file}")

    def get_output_file_path(self) -> Optional[Path]:
        """
        Get path to current output file.

        Returns:
            Path to output file, or None if not using file output
        """
        return self.output_file

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures file is closed."""
        self.close()

    def __repr__(self) -> str:
        """String representation of output interface."""
        return f"SimulationOutput(mode={self.output_mode}, file={self.output_file})"


if __name__ == "__main__":
    """Test simulation output when module is executed directly."""
    print("Testing SimulationOutput...")
    print()

    # Create mock aircraft state for testing
    class MockAircraft:
        def __init__(self, id):
            self.id = id

        def get_state(self):
            return {
                'id': self.id,
                'position': {'lat': 40.6413, 'lon': -73.7781, 'alt': 5000},
                'velocity': 250,
                'heading': 90,
                'status': 'CLIMBING',
                'route_info': {
                    'departure': 'KJFK',
                    'arrival': 'KLAX',
                    'aircraft_type': 'B738'
                },
                'waypoint_info': {
                    'current_index': 2,
                    'total_waypoints': 10
                },
                'time_elapsed': 300
            }

    # Test stdout output
    print("Test 1: stdout output")
    with SimulationOutput(output_mode="stdout") as output:
        aircraft = [MockAircraft("TEST001"), MockAircraft("TEST002")]
        state = output.export_state(100.0, aircraft)
        print(f"Exported state with {state['aircraft_count']} aircraft")

    print()
    print("SimulationOutput test complete!")
