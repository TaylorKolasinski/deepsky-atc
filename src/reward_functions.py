"""
DeepSky ATC - Reward Functions
Modular reward function implementations for RL training.
"""

import numpy as np
import logging
from typing import Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RewardFunction(ABC):
    """
    Base class for reward functions.
    """

    def __init__(self, config: Dict):
        """
        Initialize reward function with configuration.

        Args:
            config: Dictionary containing reward parameters
        """
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")

    @abstractmethod
    def calculate(
        self,
        env_state: Dict,
        actions_taken: np.ndarray,
        conflicts: List[Dict],
        completions: List[Dict],
        timestep: int
    ) -> float:
        """
        Calculate reward for the current step.

        Args:
            env_state: Current environment state
            actions_taken: Actions applied this step
            conflicts: List of conflicts that occurred
            completions: List of aircraft that completed their flight
            timestep: Current simulation timestep

        Returns:
            Scalar reward value
        """
        pass

    def _count_conflicts_by_severity(self, conflicts: List[Dict]) -> Dict[str, int]:
        """
        Count conflicts by severity level.

        Args:
            conflicts: List of conflict dictionaries

        Returns:
            Dictionary with counts: {'critical': int, 'warning': int}
        """
        counts = {'critical': 0, 'warning': 0}

        for conflict in conflicts:
            severity = conflict.get('severity', 'warning')
            if severity in counts:
                counts[severity] += 1

        return counts

    def _calculate_delay_minutes(self, completions: List[Dict]) -> float:
        """
        Calculate total delay in minutes for completed flights.

        Args:
            completions: List of completion dictionaries with scheduled vs actual times

        Returns:
            Total delay in minutes
        """
        total_delay = 0.0

        for completion in completions:
            scheduled = completion.get('scheduled_arrival', 0)
            actual = completion.get('actual_arrival', 0)
            delay = max(0, actual - scheduled)  # Only count positive delays
            total_delay += delay / 60.0  # Convert to minutes

        return total_delay

    def _calculate_throughput(self, completions: List[Dict], elapsed_hours: float) -> float:
        """
        Calculate throughput in aircraft per hour.

        Args:
            completions: List of completed aircraft
            elapsed_hours: Elapsed simulation time in hours

        Returns:
            Aircraft per hour
        """
        if elapsed_hours <= 0:
            return 0.0

        return len(completions) / elapsed_hours


class SafetyFirstReward(RewardFunction):
    """
    Safety-focused reward function.
    Heavily penalizes conflicts, lightly penalizes delays.
    """

    def calculate(
        self,
        env_state: Dict,
        actions_taken: np.ndarray,
        conflicts: List[Dict],
        completions: List[Dict],
        timestep: int
    ) -> float:
        """
        Calculate safety-first reward.

        Components:
        - Heavy conflict penalty: -100 per critical conflict
        - Medium penalty: -30 per warning conflict
        - Small delay penalty: -1 per minute delayed
        - Completion bonus: +10 per safe landing
        """
        reward = 0.0

        # Conflict penalties
        conflict_counts = self._count_conflicts_by_severity(conflicts)
        reward += conflict_counts['critical'] * self.config['critical_conflict_penalty']
        reward += conflict_counts['warning'] * self.config['warning_conflict_penalty']

        # Delay penalty
        total_delay = self._calculate_delay_minutes(completions)
        reward += total_delay * self.config['delay_penalty_per_minute']

        # Safe landing bonus
        safe_landings = len(completions)
        reward += safe_landings * self.config['safe_landing_bonus']

        logger.debug(
            f"SafetyFirst reward: {reward:.2f} "
            f"(critical: {conflict_counts['critical']}, "
            f"warning: {conflict_counts['warning']}, "
            f"delay: {total_delay:.1f}min, "
            f"landings: {safe_landings})"
        )

        return reward


class BalancedReward(RewardFunction):
    """
    Balanced reward function.
    Moderately penalizes conflicts while rewarding efficiency and throughput.
    """

    def calculate(
        self,
        env_state: Dict,
        actions_taken: np.ndarray,
        conflicts: List[Dict],
        completions: List[Dict],
        timestep: int
    ) -> float:
        """
        Calculate balanced reward.

        Components:
        - Moderate conflict penalty: -50 per critical, -20 per warning
        - Efficiency bonus: +20 per on-time completion
        - Throughput bonus: +5 per aircraft/hour
        """
        reward = 0.0

        # Conflict penalties
        conflict_counts = self._count_conflicts_by_severity(conflicts)
        reward += conflict_counts['critical'] * self.config['critical_conflict_penalty']
        reward += conflict_counts['warning'] * self.config['warning_conflict_penalty']

        # On-time completion bonus
        on_time_count = 0
        for completion in completions:
            scheduled = completion.get('scheduled_arrival', 0)
            actual = completion.get('actual_arrival', 0)
            # On-time = within 5 minutes
            if abs(actual - scheduled) <= 300:
                on_time_count += 1

        reward += on_time_count * self.config['on_time_completion_bonus']

        # Throughput bonus
        elapsed_hours = timestep / 3600.0 if timestep > 0 else 0.01
        throughput = self._calculate_throughput(completions, elapsed_hours)
        # Give bonus based on cumulative completions
        reward += len(completions) * self.config['throughput_bonus_per_aircraft_hour']

        logger.debug(
            f"Balanced reward: {reward:.2f} "
            f"(critical: {conflict_counts['critical']}, "
            f"warning: {conflict_counts['warning']}, "
            f"on_time: {on_time_count}, "
            f"throughput: {throughput:.2f}/h)"
        )

        return reward


class EfficiencyReward(RewardFunction):
    """
    Efficiency-focused reward function.
    Lightly penalizes conflicts, heavily rewards throughput and speed.
    """

    def calculate(
        self,
        env_state: Dict,
        actions_taken: np.ndarray,
        conflicts: List[Dict],
        completions: List[Dict],
        timestep: int
    ) -> float:
        """
        Calculate efficiency-focused reward.

        Components:
        - Light conflict penalty: -30 per critical, -10 per warning
        - Heavy throughput bonus: +30 per completion
        - Speed bonus: +10 per minute saved (early arrival)
        """
        reward = 0.0

        # Light conflict penalties
        conflict_counts = self._count_conflicts_by_severity(conflicts)
        reward += conflict_counts['critical'] * self.config['critical_conflict_penalty']
        reward += conflict_counts['warning'] * self.config['warning_conflict_penalty']

        # Completion bonus (high value to encourage throughput)
        reward += len(completions) * self.config['completion_bonus']

        # Speed bonus for early arrivals
        total_time_saved = 0.0
        for completion in completions:
            scheduled = completion.get('scheduled_arrival', 0)
            actual = completion.get('actual_arrival', 0)
            time_saved = scheduled - actual  # Positive if early
            if time_saved > 0:
                total_time_saved += time_saved / 60.0  # Convert to minutes

        reward += total_time_saved * self.config['speed_bonus_per_minute_saved']

        logger.debug(
            f"Efficiency reward: {reward:.2f} "
            f"(critical: {conflict_counts['critical']}, "
            f"warning: {conflict_counts['warning']}, "
            f"completions: {len(completions)}, "
            f"time_saved: {total_time_saved:.1f}min)"
        )

        return reward


def create_reward_function(name: str, config: Dict) -> RewardFunction:
    """
    Factory function to create reward function instances.

    Args:
        name: Name of the reward function ('safety_first', 'balanced', 'efficiency')
        config: Configuration dictionary for the reward function

    Returns:
        RewardFunction instance

    Raises:
        ValueError: If reward function name is not recognized
    """
    reward_functions = {
        'safety_first': SafetyFirstReward,
        'balanced': BalancedReward,
        'efficiency': EfficiencyReward
    }

    if name not in reward_functions:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(reward_functions.keys())}"
        )

    reward_class = reward_functions[name]
    return reward_class(config)
