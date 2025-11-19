"""
DeepSky ATC - Reinforcement Learning Environment
Gym-compatible RL environment for training AI air traffic controllers.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.simulation_manager import SimulationManager
from src.airspace import Airspace
from src.delay_model import DelayModel
from src.simulation_output import SimulationOutput
from src.route_generator import load_routes_from_data, FlightRoute
from src.reward_functions import create_reward_function, RewardFunction
from src.observation_builder import (
    build_aircraft_state_vector,
    build_density_grid,
    predict_conflicts,
    flatten_observation,
    normalize_value
)

logger = logging.getLogger(__name__)


class ATCEnvironment(gym.Env):
    """
    Gym-compatible environment for ATC reinforcement learning.

    Provides complex observation space with aircraft states, conflict predictions,
    and airspace density. Supports continuous action space for heading, altitude,
    and speed control.

    Attributes:
        simulation_manager: SimulationManager instance
        reward_function: RewardFunction instance for calculating rewards
        max_aircraft: Maximum number of aircraft supported
        episode_length: Maximum episode duration in seconds
        config: Environment configuration dictionary
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        simulation_manager: Optional[SimulationManager] = None,
        reward_function: str = "safety_first",
        max_aircraft: int = 60,
        config_path: Optional[str] = None
    ):
        """
        Initialize ATC environment.

        Args:
            simulation_manager: SimulationManager instance (or None to create default)
            reward_function: Name of reward function to use
            max_aircraft: Maximum number of aircraft to support
            config_path: Path to RL configuration JSON file
        """
        super(ATCEnvironment, self).__init__()

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'data' / 'rl_config.json'

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Environment parameters
        self.max_aircraft = self.config['environment']['max_aircraft']
        self.episode_length = self.config['environment']['episode_length_seconds']
        self.action_step_size = self.config['environment']['action_step_size']
        self.verbose = self.config['environment'].get('verbose_logging', False)

        # Create simulation manager if not provided
        if simulation_manager is None:
            airspace = Airspace()
            delay_model = DelayModel()
            output = SimulationOutput()
            simulation_manager = SimulationManager(airspace, delay_model, output)

        self.simulation_manager = simulation_manager

        # Load reward function
        reward_config = self.config['reward_functions'].get(
            reward_function,
            self.config['reward_functions']['safety_first']
        )
        self.reward_function: RewardFunction = create_reward_function(
            reward_function,
            reward_config
        )

        # Define observation and action spaces
        self._setup_spaces()

        # Episode state
        self.current_step = 0
        self.episode_conflicts = []
        self.episode_completions = []
        self.previous_conflicts = []

        logger.info(
            f"Initialized ATCEnvironment with {self.max_aircraft} max aircraft, "
            f"reward={reward_function}, episode_length={self.episode_length}s"
        )

    def _setup_spaces(self):
        """
        Setup observation and action spaces for the environment.
        """
        # Calculate observation space dimensions
        aircraft_state_dim = 15  # pos(3) + vel(3) + heading(1) + status(5) + route(3)
        max_aircraft = self.max_aircraft

        # Observation components:
        # - Aircraft states: max_aircraft * 15
        # - Conflict info: 100 conflicts * 4 (aircraft1, aircraft2, distance, severity)
        # - Airspace density: 10 * 10 * 10 = 1000
        # - Global state: 3 (sim_time, active_aircraft, workload)

        aircraft_obs_dim = max_aircraft * aircraft_state_dim
        conflict_obs_dim = 100 * 4  # Store up to 100 conflicts
        density_obs_dim = 10 * 10 * 10
        global_obs_dim = 3

        total_obs_dim = (
            aircraft_obs_dim +
            conflict_obs_dim +
            density_obs_dim +
            global_obs_dim
        )

        # Observation space: normalized to [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_obs_dim,),
            dtype=np.float32
        )

        # Action space: continuous control for each aircraft
        # For each aircraft: [heading_change, altitude_change, speed_change]
        action_bounds = self.config['action_space']

        # Build low and high bounds for all aircraft
        single_aircraft_low = np.array([
            action_bounds['heading_change_range'][0],
            action_bounds['altitude_change_range'][0],
            action_bounds['speed_change_range'][0]
        ])
        single_aircraft_high = np.array([
            action_bounds['heading_change_range'][1],
            action_bounds['altitude_change_range'][1],
            action_bounds['speed_change_range'][1]
        ])

        action_low = np.tile(single_aircraft_low, max_aircraft)
        action_high = np.tile(single_aircraft_high, max_aircraft)

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )

        if self.verbose:
            logger.info(f"Observation space: {self.observation_space.shape}")
            logger.info(f"Action space: {self.action_space.shape}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Creates a new random scenario with 50-100 aircraft and random routes.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset simulation manager
        self.simulation_manager.aircraft_list = []
        self.simulation_manager.simulation_time = 0.0
        self.simulation_manager.total_aircraft_spawned = 0
        self.simulation_manager.completed_flights = 0
        # Reset conflict tracker manually
        self.simulation_manager.conflict_tracker.active_conflicts = {}
        self.simulation_manager.conflict_tracker.conflict_history = []
        self.simulation_manager.conflict_tracker.conflict_id_counter = 0

        # Reset episode state
        self.current_step = 0
        self.episode_conflicts = []
        self.episode_completions = []
        self.previous_conflicts = []

        # Load available routes
        routes = load_routes_from_data()

        # Generate random scenario with 50-100 aircraft
        num_aircraft = np.random.randint(
            self.config['environment']['random_scenario_aircraft_range'][0],
            self.config['environment']['random_scenario_aircraft_range'][1]
        )

        # Schedule aircraft with random departure times over first 30 minutes
        for i in range(num_aircraft):
            # Pick random route
            route = np.random.choice(routes)

            # Random departure time in first 1800 seconds (30 minutes)
            departure_time = np.random.uniform(0, 1800)

            # Add aircraft to simulation
            self.simulation_manager.add_aircraft(route, departure_time)

        logger.info(f"Reset environment with {num_aircraft} aircraft")

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action array with shape (max_aircraft * 3,)
                   Each aircraft has [heading_change, altitude_change, speed_change]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply actions to aircraft
        self._apply_actions(action)

        # Store previous conflicts for comparison
        self.previous_conflicts = self.simulation_manager.conflict_tracker.get_active_conflicts()

        # Run simulation for action_step_size seconds (10 timesteps of 1 second)
        new_completions = []
        for _ in range(self.action_step_size):
            # Get aircraft that are about to complete
            pre_step_active = len([
                ac for ac in self.simulation_manager.aircraft_list
                if ac.status != 'LANDED'
            ])

            # Step simulation
            self.simulation_manager.step(delta_t=1.0)

            # Check for new completions
            post_step_active = len([
                ac for ac in self.simulation_manager.aircraft_list
                if ac.status != 'LANDED'
            ])

            if post_step_active < pre_step_active:
                # Aircraft completed - record it
                # Note: This is approximate, ideally we'd track the actual aircraft
                num_new_completions = pre_step_active - post_step_active
                for _ in range(num_new_completions):
                    new_completions.append({
                        'scheduled_arrival': self.simulation_manager.simulation_time,
                        'actual_arrival': self.simulation_manager.simulation_time
                    })

        # Get current conflicts
        current_conflicts = self.simulation_manager.conflict_tracker.get_active_conflicts()

        # Calculate reward
        env_state = self._get_env_state()
        reward = self.reward_function.calculate(
            env_state=env_state,
            actions_taken=action,
            conflicts=current_conflicts,
            completions=new_completions,
            timestep=int(self.simulation_manager.simulation_time)
        )

        # Update episode tracking
        self.episode_conflicts.extend(current_conflicts)
        self.episode_completions.extend(new_completions)
        self.current_step += 1

        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()

        if self.verbose:
            logger.debug(
                f"Step {self.current_step}: "
                f"reward={reward:.2f}, "
                f"active_aircraft={len(self.simulation_manager.aircraft_list)}, "
                f"conflicts={len(current_conflicts)}"
            )

        return observation, reward, terminated, truncated, info

    def _apply_actions(self, action: np.ndarray):
        """
        Apply actions to active aircraft.

        Args:
            action: Flattened action array (max_aircraft * 3)
        """
        # Reshape to (max_aircraft, 3)
        action = action.reshape(self.max_aircraft, 3)

        # Get active aircraft
        active_aircraft = [
            ac for ac in self.simulation_manager.aircraft_list
            if self.simulation_manager.simulation_time >= ac.time_elapsed
            and ac.status != 'LANDED'
        ]

        # Apply actions to each active aircraft
        for i, aircraft in enumerate(active_aircraft):
            if i >= self.max_aircraft:
                break  # Exceeds max aircraft limit

            heading_change, altitude_change, speed_change = action[i]

            # Apply heading change (clamp to valid range)
            new_heading = aircraft.current_heading + heading_change
            aircraft.current_heading = new_heading % 360.0

            # Apply altitude change (clamp to valid range)
            new_altitude = aircraft.current_position['alt'] + altitude_change
            aircraft.current_position['alt'] = np.clip(new_altitude, 0, 50000)

            # Apply speed change (clamp to valid range)
            new_speed = aircraft.current_velocity + speed_change
            aircraft.current_velocity = np.clip(new_speed, 0, 600)

            if self.verbose:
                logger.debug(
                    f"Applied action to {aircraft.id}: "
                    f"heading_Δ={heading_change:.1f}°, "
                    f"alt_Δ={altitude_change:.0f}ft, "
                    f"speed_Δ={speed_change:.1f}kts"
                )

    def _get_observation(self) -> np.ndarray:
        """
        Build observation from current simulation state.

        Returns:
            Flattened observation array
        """
        bounds = self.config['observation_space']['bounds']

        # Get active aircraft
        active_aircraft = [
            ac for ac in self.simulation_manager.aircraft_list
            if self.simulation_manager.simulation_time >= ac.time_elapsed
            and ac.status != 'LANDED'
        ]

        # Build aircraft states (max_aircraft * 15)
        aircraft_states = np.zeros((self.max_aircraft, 15), dtype=np.float32)

        for i, aircraft in enumerate(active_aircraft[:self.max_aircraft]):
            # Convert aircraft to observation format
            aircraft_dict = {
                'position': [
                    aircraft.current_position['lat'] * 111000,  # Convert to meters (approx)
                    aircraft.current_position['lon'] * 111000,
                    aircraft.current_position['alt'] * 0.3048  # Convert feet to meters
                ],
                'velocity': [
                    aircraft.current_velocity * 0.514,  # Convert knots to m/s
                    0,  # Simplified: no lateral velocity
                    0   # Simplified: no vertical velocity (could calculate from climb rate)
                ],
                'heading': aircraft.current_heading,
                'status': aircraft.status,
                'route_info': [
                    aircraft.time_elapsed,
                    aircraft.time_elapsed + 1800,  # Estimated arrival (simplified)
                    10000  # Remaining distance (simplified)
                ]
            }

            aircraft_states[i] = build_aircraft_state_vector(
                aircraft_dict,
                bounds,
                normalize=True
            )

        # Build conflict info (100 conflicts * 4)
        conflicts = predict_conflicts(
            [self._aircraft_to_dict(ac) for ac in active_aircraft],
            lookahead_seconds=self.config['observation_space']['prediction_horizon_seconds']
        )

        conflict_info = np.zeros((100, 4), dtype=np.float32)
        for i, conflict in enumerate(conflicts[:100]):
            conflict_info[i] = [
                conflict['aircraft1_idx'] / self.max_aircraft,  # Normalize
                conflict['aircraft2_idx'] / self.max_aircraft,
                normalize_value(conflict['min_distance'], 0, 20000),
                1.0 if conflict['severity'] == 'critical' else 0.5
            ]

        # Build density grid (10x10x10)
        aircraft_dicts = [self._aircraft_to_dict(ac) for ac in active_aircraft]
        density_grid = build_density_grid(
            aircraft_dicts,
            grid_size=tuple(self.config['observation_space']['density_grid_size']),
            bounds=bounds
        )

        # Build global state (3)
        global_state = np.array([
            normalize_value(self.simulation_manager.simulation_time, 0, self.episode_length),
            len(active_aircraft) / self.max_aircraft,
            0.5  # Controller workload (simplified)
        ], dtype=np.float32)

        # Flatten all components
        obs_dict = {
            'aircraft_states': aircraft_states,
            'conflict_info': conflict_info,
            'airspace_density': density_grid,
            'global_state': global_state
        }

        flattened = flatten_observation(obs_dict)

        return flattened

    def _aircraft_to_dict(self, aircraft) -> Dict:
        """
        Convert Aircraft object to dictionary format for observation builder.

        Args:
            aircraft: Aircraft object

        Returns:
            Dictionary with position, velocity, etc.
        """
        return {
            'position': [
                aircraft.current_position['lat'] * 111000,
                aircraft.current_position['lon'] * 111000,
                aircraft.current_position['alt'] * 0.3048
            ],
            'velocity': [
                aircraft.current_velocity * 0.514 * np.cos(np.radians(aircraft.current_heading)),
                aircraft.current_velocity * 0.514 * np.sin(np.radians(aircraft.current_heading)),
                0
            ]
        }

    def _get_env_state(self) -> Dict:
        """
        Get current environment state for reward calculation.

        Returns:
            Dictionary with environment state information
        """
        return {
            'simulation_time': self.simulation_manager.simulation_time,
            'active_aircraft': len([
                ac for ac in self.simulation_manager.aircraft_list
                if ac.status != 'LANDED'
            ]),
            'total_aircraft': len(self.simulation_manager.aircraft_list)
        }

    def _get_info(self) -> Dict[str, Any]:
        """
        Get auxiliary information about the environment.

        Returns:
            Dictionary with diagnostic information
        """
        active_aircraft = [
            ac for ac in self.simulation_manager.aircraft_list
            if ac.status != 'LANDED'
        ]

        return {
            'simulation_time': self.simulation_manager.simulation_time,
            'active_aircraft': len(active_aircraft),
            'completed_flights': self.simulation_manager.completed_flights,
            'total_conflicts': len(self.episode_conflicts),
            'current_step': self.current_step
        }

    def _is_terminated(self) -> bool:
        """
        Check if episode should terminate (all aircraft landed).

        Returns:
            True if all aircraft have landed
        """
        active_aircraft = [
            ac for ac in self.simulation_manager.aircraft_list
            if ac.status != 'LANDED'
        ]

        # Episode terminates when all aircraft have landed
        return len(active_aircraft) == 0

    def _is_truncated(self) -> bool:
        """
        Check if episode should be truncated (max time reached).

        Returns:
            True if maximum episode length reached
        """
        return self.simulation_manager.simulation_time >= self.episode_length

    def render(self, mode: str = 'human'):
        """
        Render current environment state.

        Args:
            mode: Rendering mode ('human' for text output)
        """
        if mode == 'human':
            active_aircraft = [
                ac for ac in self.simulation_manager.aircraft_list
                if ac.status != 'LANDED'
            ]

            print(f"\n=== ATC Environment State ===")
            print(f"Simulation Time: {self.simulation_manager.simulation_time:.0f}s")
            print(f"Active Aircraft: {len(active_aircraft)}")
            print(f"Completed Flights: {self.simulation_manager.completed_flights}")
            print(f"Episode Conflicts: {len(self.episode_conflicts)}")
            print(f"Current Step: {self.current_step}")

            if len(active_aircraft) > 0:
                print(f"\nSample Aircraft:")
                for ac in active_aircraft[:3]:
                    print(f"  {ac.id}: {ac.status} @ {ac.current_position['alt']:.0f}ft, "
                          f"{ac.current_heading:.0f}°, {ac.current_velocity:.0f}kts")

    def close(self):
        """
        Clean up environment resources.
        """
        pass
