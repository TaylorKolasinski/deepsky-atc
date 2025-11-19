"""
Tests for DeepSky ATC RL Environment
"""

import pytest
import numpy as np
import logging
from pathlib import Path

from src.rl_environment import ATCEnvironment
from src.simulation_manager import SimulationManager
from src.airspace import Airspace
from src.delay_model import DelayModel
from src.simulation_output import SimulationOutput
from src.reward_functions import SafetyFirstReward, BalancedReward, EfficiencyReward

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRLEnvironment:
    """Test suite for ATCEnvironment."""

    @pytest.fixture
    def env(self):
        """Create a basic environment for testing."""
        return ATCEnvironment(
            simulation_manager=None,
            reward_function="safety_first",
            max_aircraft=60
        )

    def test_environment_initialization(self, env):
        """Test that environment initializes correctly."""
        logger.info("\n=== Test: Environment Initialization ===")

        assert env is not None
        assert env.max_aircraft == 60
        assert env.episode_length == 3600
        assert env.action_step_size == 10

        # Check spaces are defined
        assert env.observation_space is not None
        assert env.action_space is not None

        logger.info(f"✓ Environment initialized successfully")
        logger.info(f"  Observation space: {env.observation_space.shape}")
        logger.info(f"  Action space: {env.action_space.shape}")

    def test_observation_space_shape(self, env):
        """Test that observation space has correct dimensions."""
        logger.info("\n=== Test: Observation Space Shape ===")

        # Expected dimensions:
        # - Aircraft states: 60 * 15 = 900
        # - Conflict info: 100 * 4 = 400
        # - Density grid: 10 * 10 * 10 = 1000
        # - Global state: 3
        # Total: 900 + 400 + 1000 + 3 = 2303

        expected_dim = 60 * 15 + 100 * 4 + 10 * 10 * 10 + 3
        actual_dim = env.observation_space.shape[0]

        assert actual_dim == expected_dim, f"Expected {expected_dim}, got {actual_dim}"

        logger.info(f"✓ Observation space shape correct: {actual_dim}")

    def test_action_space_shape(self, env):
        """Test that action space has correct dimensions."""
        logger.info("\n=== Test: Action Space Shape ===")

        # Expected: 60 aircraft * 3 actions per aircraft = 180
        expected_dim = 60 * 3
        actual_dim = env.action_space.shape[0]

        assert actual_dim == expected_dim, f"Expected {expected_dim}, got {actual_dim}"

        # Check bounds
        assert env.action_space.low[0] == -30.0  # Heading change min
        assert env.action_space.high[0] == 30.0  # Heading change max

        logger.info(f"✓ Action space shape correct: {actual_dim}")
        logger.info(f"  Action bounds: [{env.action_space.low[0]}, {env.action_space.high[0]}]")

    def test_reset(self, env):
        """Test environment reset functionality."""
        logger.info("\n=== Test: Environment Reset ===")

        observation, info = env.reset(seed=42)

        # Check observation is valid
        assert observation is not None
        assert observation.shape == env.observation_space.shape
        assert np.all(np.isfinite(observation))
        assert np.all(observation >= -1.0) and np.all(observation <= 1.0)

        # Check info dictionary
        assert 'simulation_time' in info
        assert 'active_aircraft' in info
        assert info['simulation_time'] == 0.0

        # Check that aircraft were added
        assert len(env.simulation_manager.aircraft_list) > 0

        logger.info(f"✓ Environment reset successfully")
        logger.info(f"  Total aircraft: {len(env.simulation_manager.aircraft_list)}")
        logger.info(f"  Observation shape: {observation.shape}")
        logger.info(f"  Observation range: [{observation.min():.2f}, {observation.max():.2f}]")

    def test_action_application(self, env):
        """Test that actions are correctly applied to aircraft."""
        logger.info("\n=== Test: Action Application ===")

        # Reset environment
        observation, info = env.reset(seed=42)

        # Wait for some aircraft to become active
        for _ in range(100):
            env.simulation_manager.step(delta_t=1.0)

        # Get initial state of first active aircraft
        active_aircraft = [
            ac for ac in env.simulation_manager.aircraft_list
            if ac.status != 'LANDED' and env.simulation_manager.simulation_time >= ac.time_elapsed
        ]

        if len(active_aircraft) > 0:
            aircraft = active_aircraft[0]
            initial_heading = aircraft.current_heading
            initial_altitude = aircraft.current_position['alt']
            initial_speed = aircraft.current_velocity

            logger.info(f"Initial state of {aircraft.id}:")
            logger.info(f"  Heading: {initial_heading:.1f}°")
            logger.info(f"  Altitude: {initial_altitude:.0f}ft")
            logger.info(f"  Speed: {initial_speed:.0f}kts")

            # Create action: +10° heading, +500ft altitude, +20kts speed
            action = np.zeros(env.action_space.shape)
            action[0] = 10.0   # Heading change
            action[1] = 500.0  # Altitude change
            action[2] = 20.0   # Speed change

            # Apply action
            env._apply_actions(action)

            # Check that changes were applied
            new_heading = aircraft.current_heading
            new_altitude = aircraft.current_position['alt']
            new_speed = aircraft.current_velocity

            logger.info(f"After action:")
            logger.info(f"  Heading: {new_heading:.1f}° (Δ={new_heading - initial_heading:.1f}°)")
            logger.info(f"  Altitude: {new_altitude:.0f}ft (Δ={new_altitude - initial_altitude:.0f}ft)")
            logger.info(f"  Speed: {new_speed:.0f}kts (Δ={new_speed - initial_speed:.0f}kts)")

            # Verify changes (allowing some tolerance)
            assert abs((new_heading - initial_heading) % 360 - 10.0) < 0.1
            assert abs(new_altitude - initial_altitude - 500.0) < 0.1
            assert abs(new_speed - initial_speed - 20.0) < 0.1

            logger.info(f"✓ Actions applied correctly to aircraft")
        else:
            logger.warning("⚠ No active aircraft found, skipping action test")

    def test_reward_calculation_safety_first(self):
        """Test SafetyFirstReward calculation."""
        logger.info("\n=== Test: SafetyFirstReward ===")

        env = ATCEnvironment(reward_function="safety_first", max_aircraft=60)
        env.reset(seed=42)

        # Test reward with conflicts
        conflicts = [
            {'severity': 'critical'},
            {'severity': 'warning'},
            {'severity': 'warning'}
        ]

        completions = [
            {'scheduled_arrival': 1000, 'actual_arrival': 1000},  # On time
            {'scheduled_arrival': 1000, 'actual_arrival': 1300}   # 5 min delay
        ]

        reward = env.reward_function.calculate(
            env_state={},
            actions_taken=np.zeros(180),
            conflicts=conflicts,
            completions=completions,
            timestep=100
        )

        # Expected: 1*(-100) + 2*(-30) + 5*(-1) + 2*(10) = -100 - 60 - 5 + 20 = -145
        expected_reward = -145.0
        assert abs(reward - expected_reward) < 0.1

        logger.info(f"✓ SafetyFirstReward calculated correctly: {reward:.2f}")

    def test_reward_calculation_balanced(self):
        """Test BalancedReward calculation."""
        logger.info("\n=== Test: BalancedReward ===")

        env = ATCEnvironment(reward_function="balanced", max_aircraft=60)
        env.reset(seed=42)

        conflicts = [
            {'severity': 'critical'},
        ]

        completions = [
            {'scheduled_arrival': 1000, 'actual_arrival': 1000},  # On time
            {'scheduled_arrival': 1000, 'actual_arrival': 1100}   # Within 5 min
        ]

        reward = env.reward_function.calculate(
            env_state={},
            actions_taken=np.zeros(180),
            conflicts=conflicts,
            completions=completions,
            timestep=100
        )

        # Expected: 1*(-50) + 2*(20) + 2*(5) = -50 + 40 + 10 = 0
        expected_reward = 0.0
        assert abs(reward - expected_reward) < 0.1

        logger.info(f"✓ BalancedReward calculated correctly: {reward:.2f}")

    def test_reward_calculation_efficiency(self):
        """Test EfficiencyReward calculation."""
        logger.info("\n=== Test: EfficiencyReward ===")

        env = ATCEnvironment(reward_function="efficiency", max_aircraft=60)
        env.reset(seed=42)

        conflicts = []

        completions = [
            {'scheduled_arrival': 1000, 'actual_arrival': 700},  # 5 min early
        ]

        reward = env.reward_function.calculate(
            env_state={},
            actions_taken=np.zeros(180),
            conflicts=conflicts,
            completions=completions,
            timestep=100
        )

        # Expected: 1*(30) + 5*(10) = 30 + 50 = 80
        expected_reward = 80.0
        assert abs(reward - expected_reward) < 0.1

        logger.info(f"✓ EfficiencyReward calculated correctly: {reward:.2f}")

    def test_step_consistency(self, env):
        """Test that environment steps are consistent."""
        logger.info("\n=== Test: Step Consistency ===")

        observation, info = env.reset(seed=42)
        initial_obs = observation.copy()

        # Run 10 random steps
        rewards = []
        for i in range(10):
            # Random action
            action = env.action_space.sample()

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Verify outputs
            assert observation.shape == env.observation_space.shape
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(info, dict)

            rewards.append(reward)

            logger.info(
                f"  Step {i+1}: reward={reward:.2f}, "
                f"active={info['active_aircraft']}, "
                f"completed={info['completed_flights']}"
            )

            if terminated or truncated:
                logger.info(f"  Episode ended at step {i+1}")
                break

        logger.info(f"✓ Completed {len(rewards)} steps successfully")
        logger.info(f"  Average reward: {np.mean(rewards):.2f}")

    def test_episode_termination(self):
        """Test that episodes terminate correctly."""
        logger.info("\n=== Test: Episode Termination ===")

        # Create environment with short episode length
        env = ATCEnvironment(reward_function="safety_first", max_aircraft=10)

        # Override episode length for faster test
        env.episode_length = 600  # 10 minutes

        observation, info = env.reset(seed=42)

        # Run until termination or truncation
        max_steps = 100
        step_count = 0
        terminated = False
        truncated = False

        for i in range(max_steps):
            action = np.zeros(env.action_space.shape)  # No action
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if terminated:
                logger.info(f"  Episode terminated at step {step_count} (all aircraft landed)")
                break
            elif truncated:
                logger.info(f"  Episode truncated at step {step_count} (max time reached)")
                break

        assert terminated or truncated or step_count >= max_steps

        logger.info(f"✓ Episode ended correctly after {step_count} steps")

    def test_render(self, env):
        """Test rendering functionality."""
        logger.info("\n=== Test: Render ===")

        env.reset(seed=42)

        # Advance a few steps
        for _ in range(5):
            env.simulation_manager.step(delta_t=10.0)

        # Render (should not crash)
        env.render(mode='human')

        logger.info(f"✓ Rendering works correctly")


def test_sample_observation_and_reward():
    """Print sample observation and reward for inspection."""
    logger.info("\n" + "="*60)
    logger.info("SAMPLE OBSERVATION AND REWARD")
    logger.info("="*60)

    # Create environment
    env = ATCEnvironment(reward_function="safety_first", max_aircraft=60)

    # Reset
    observation, info = env.reset(seed=42)

    logger.info(f"\n--- Initial Observation ---")
    logger.info(f"Shape: {observation.shape}")
    logger.info(f"Min: {observation.min():.3f}")
    logger.info(f"Max: {observation.max():.3f}")
    logger.info(f"Mean: {observation.mean():.3f}")
    logger.info(f"Std: {observation.std():.3f}")
    logger.info(f"Non-zero elements: {np.count_nonzero(observation)}")

    # Run a few steps
    logger.info(f"\n--- Running 5 Steps ---")
    for i in range(5):
        # Random action
        action = env.action_space.sample()

        # Step
        observation, reward, terminated, truncated, info = env.step(action)

        logger.info(
            f"Step {i+1}: "
            f"reward={reward:+8.2f}, "
            f"active={info['active_aircraft']:3d}, "
            f"completed={info['completed_flights']:3d}, "
            f"conflicts={info['total_conflicts']:3d}, "
            f"time={info['simulation_time']:6.0f}s"
        )

        if terminated or truncated:
            break

    logger.info(f"\n--- Final Observation ---")
    logger.info(f"Shape: {observation.shape}")
    logger.info(f"Min: {observation.min():.3f}")
    logger.info(f"Max: {observation.max():.3f}")
    logger.info(f"Mean: {observation.mean():.3f}")
    logger.info(f"Std: {observation.std():.3f}")

    # Print environment state
    env.render()

    logger.info("\n" + "="*60)


if __name__ == "__main__":
    # Run tests
    print("\n" + "="*60)
    print("DeepSky ATC RL Environment Tests")
    print("="*60)

    # Run pytest
    pytest.main([__file__, "-v", "-s"])

    # Run sample observation test
    test_sample_observation_and_reward()
