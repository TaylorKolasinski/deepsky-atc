"""
DeepSky ATC - RL Observation Builder
Utilities for building and normalizing observations for the RL environment.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to [-1, 1] range.

    Args:
        value: The value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value

    Returns:
        Normalized value in [-1, 1]
    """
    if max_val == min_val:
        return 0.0
    normalized = 2 * (value - min_val) / (max_val - min_val) - 1
    return np.clip(normalized, -1.0, 1.0)


def normalize_position(pos: np.ndarray, bounds: Dict[str, List[float]]) -> np.ndarray:
    """
    Normalize position coordinates to [-1, 1] range.

    Args:
        pos: Position array [x, y, z]
        bounds: Dictionary with 'x', 'y', 'z' bounds

    Returns:
        Normalized position array
    """
    normalized = np.zeros(3)
    normalized[0] = normalize_value(pos[0], bounds['x'][0], bounds['x'][1])
    normalized[1] = normalize_value(pos[1], bounds['y'][0], bounds['y'][1])
    normalized[2] = normalize_value(pos[2], bounds['z'][0], bounds['z'][1])
    return normalized


def normalize_velocity(vel: np.ndarray, bounds: List[float]) -> np.ndarray:
    """
    Normalize velocity components to [-1, 1] range.

    Args:
        vel: Velocity array [vx, vy, vz]
        bounds: Min and max velocity values

    Returns:
        Normalized velocity array
    """
    normalized = np.zeros(3)
    for i in range(3):
        normalized[i] = normalize_value(vel[i], bounds[0], bounds[1])
    return normalized


def normalize_heading(heading: float) -> float:
    """
    Normalize heading (0-360 degrees) to [-1, 1].

    Args:
        heading: Heading in degrees [0, 360]

    Returns:
        Normalized heading in [-1, 1]
    """
    return normalize_value(heading, 0.0, 360.0)


def build_density_grid(
    aircraft_list: List[Dict],
    grid_size: Tuple[int, int, int] = (10, 10, 10),
    bounds: Optional[Dict[str, List[float]]] = None
) -> np.ndarray:
    """
    Build a 3D density grid showing aircraft distribution in airspace.

    Args:
        aircraft_list: List of aircraft dictionaries with 'position' key
        grid_size: Tuple of (x_bins, y_bins, z_bins)
        bounds: Spatial bounds for the grid

    Returns:
        3D numpy array with aircraft counts per cell, normalized
    """
    if bounds is None:
        bounds = {
            'x': [-50000, 50000],
            'y': [-50000, 50000],
            'z': [0, 15000]
        }

    # Initialize grid
    grid = np.zeros(grid_size)

    if len(aircraft_list) == 0:
        return grid

    # Calculate bin edges
    x_edges = np.linspace(bounds['x'][0], bounds['x'][1], grid_size[0] + 1)
    y_edges = np.linspace(bounds['y'][0], bounds['y'][1], grid_size[1] + 1)
    z_edges = np.linspace(bounds['z'][0], bounds['z'][1], grid_size[2] + 1)

    # Populate grid
    for aircraft in aircraft_list:
        pos = aircraft.get('position', [0, 0, 0])

        # Find bin indices
        x_idx = np.searchsorted(x_edges, pos[0], side='right') - 1
        y_idx = np.searchsorted(y_edges, pos[1], side='right') - 1
        z_idx = np.searchsorted(z_edges, pos[2], side='right') - 1

        # Clip to valid range
        x_idx = np.clip(x_idx, 0, grid_size[0] - 1)
        y_idx = np.clip(y_idx, 0, grid_size[1] - 1)
        z_idx = np.clip(z_idx, 0, grid_size[2] - 1)

        grid[x_idx, y_idx, z_idx] += 1

    # Normalize by maximum count (to keep values in reasonable range)
    max_count = np.max(grid) if np.max(grid) > 0 else 1
    grid = grid / max_count

    return grid


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate 3D Euclidean distance between two positions.

    Args:
        pos1: First position [x, y, z]
        pos2: Second position [x, y, z]

    Returns:
        Distance in meters
    """
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def predict_conflicts(
    aircraft_list: List[Dict],
    lookahead_seconds: int = 60,
    critical_distance: float = 5000.0,
    warning_distance: float = 10000.0
) -> List[Dict]:
    """
    Predict potential conflicts in the near future based on current trajectories.

    Args:
        aircraft_list: List of aircraft with position and velocity
        lookahead_seconds: How far ahead to predict
        critical_distance: Distance threshold for critical conflicts (meters)
        warning_distance: Distance threshold for warnings (meters)

    Returns:
        List of predicted conflicts with structure:
        {
            'aircraft1_idx': int,
            'aircraft2_idx': int,
            'min_distance': float,
            'time_to_conflict': float,
            'severity': 'critical' or 'warning'
        }
    """
    conflicts = []
    n_aircraft = len(aircraft_list)

    for i in range(n_aircraft):
        for j in range(i + 1, n_aircraft):
            aircraft1 = aircraft_list[i]
            aircraft2 = aircraft_list[j]

            # Get current state
            pos1 = np.array(aircraft1.get('position', [0, 0, 0]))
            pos2 = np.array(aircraft2.get('position', [0, 0, 0]))
            vel1 = np.array(aircraft1.get('velocity', [0, 0, 0]))
            vel2 = np.array(aircraft2.get('velocity', [0, 0, 0]))

            # Predict minimum distance over lookahead period
            min_distance = float('inf')
            time_to_conflict = -1

            # Sample at 1-second intervals
            for t in range(0, lookahead_seconds + 1):
                future_pos1 = pos1 + vel1 * t
                future_pos2 = pos2 + vel2 * t
                distance = calculate_distance(future_pos1, future_pos2)

                if distance < min_distance:
                    min_distance = distance
                    time_to_conflict = t

            # Check if it's a conflict
            if min_distance < warning_distance:
                severity = 'critical' if min_distance < critical_distance else 'warning'
                conflicts.append({
                    'aircraft1_idx': i,
                    'aircraft2_idx': j,
                    'min_distance': min_distance,
                    'time_to_conflict': time_to_conflict,
                    'severity': severity
                })

    logger.debug(f"Predicted {len(conflicts)} conflicts in next {lookahead_seconds}s")
    return conflicts


def flatten_observation(obs_dict: Dict) -> np.ndarray:
    """
    Flatten a structured observation dictionary into a single vector for RL.

    Args:
        obs_dict: Dictionary containing:
            - aircraft_states: (max_aircraft, state_dim) array
            - conflict_info: flattened conflict data
            - airspace_density: (10, 10, 10) grid
            - global_state: global info vector

    Returns:
        Flattened 1D numpy array
    """
    components = []

    # Flatten aircraft states
    if 'aircraft_states' in obs_dict:
        aircraft_flat = obs_dict['aircraft_states'].flatten()
        components.append(aircraft_flat)

    # Flatten conflict info
    if 'conflict_info' in obs_dict:
        conflict_flat = obs_dict['conflict_info'].flatten()
        components.append(conflict_flat)

    # Flatten density grid
    if 'airspace_density' in obs_dict:
        density_flat = obs_dict['airspace_density'].flatten()
        components.append(density_flat)

    # Add global state
    if 'global_state' in obs_dict:
        global_flat = obs_dict['global_state'].flatten()
        components.append(global_flat)

    # Concatenate all components
    if len(components) == 0:
        return np.array([])

    flattened = np.concatenate(components)
    logger.debug(f"Flattened observation size: {len(flattened)}")

    return flattened.astype(np.float32)


def encode_status_onehot(status: str) -> np.ndarray:
    """
    Encode aircraft status as one-hot vector.

    Args:
        status: One of ['TAXI', 'CLIMBING', 'CRUISING', 'DESCENDING', 'LANDING']

    Returns:
        One-hot encoded vector of length 5
    """
    statuses = ['TAXI', 'CLIMBING', 'CRUISING', 'DESCENDING', 'LANDING']
    onehot = np.zeros(len(statuses))

    if status in statuses:
        idx = statuses.index(status)
        onehot[idx] = 1.0

    return onehot


def build_aircraft_state_vector(
    aircraft: Dict,
    bounds: Dict[str, List[float]],
    normalize: bool = True
) -> np.ndarray:
    """
    Build a state vector for a single aircraft.

    Args:
        aircraft: Aircraft dictionary with position, velocity, heading, status, route_info
        bounds: Normalization bounds
        normalize: Whether to normalize values

    Returns:
        State vector: [pos(3), vel(3), heading(1), status(5), route_info(3)] = 15 dims
    """
    state_vector = []

    # Position (3)
    pos = np.array(aircraft.get('position', [0, 0, 0]))
    if normalize:
        pos = normalize_position(pos, bounds)
    state_vector.extend(pos)

    # Velocity (3)
    vel = np.array(aircraft.get('velocity', [0, 0, 0]))
    if normalize:
        vel = normalize_velocity(vel, bounds['velocity'])
    state_vector.extend(vel)

    # Heading (1)
    heading = aircraft.get('heading', 0)
    if normalize:
        heading = normalize_heading(heading)
    state_vector.append(heading)

    # Status one-hot (5)
    status = aircraft.get('status', 'CRUISING')
    status_onehot = encode_status_onehot(status)
    state_vector.extend(status_onehot)

    # Route info (3): departure_time, scheduled_arrival, distance_remaining
    route_info = aircraft.get('route_info', [0, 0, 0])
    if normalize:
        # Normalize times to [0, 1] assuming max episode is 3600 seconds
        route_info = [
            route_info[0] / 3600.0,
            route_info[1] / 3600.0,
            normalize_value(route_info[2], 0, 100000)  # max distance
        ]
    state_vector.extend(route_info)

    return np.array(state_vector, dtype=np.float32)
