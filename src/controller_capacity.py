"""
Controller capacity and staffing model for DeepSky ATC.

Models real-world air traffic controller workload limits, performance
degradation under overload, and staffing scenarios including government
shutdown conditions.
"""

from typing import Dict, Any
import math


class ControllerStaffing:
    """
    Models air traffic controller staffing and workload constraints.

    Simulates realistic controller capacity limits and performance degradation
    when workload exceeds capacity. Models scenarios from full staffing to
    government shutdown conditions.

    Staffing Configurations:
        - full: 4 controllers, 15 aircraft each = 60 total capacity
        - normal: 3 controllers, 12 aircraft each = 36 total capacity
        - reduced: 2 controllers, 12 aircraft each = 24 total capacity
        - shutdown: 1 controller, 15 aircraft each = 15 total capacity (overworked)
        - none: 0 controllers = 0 capacity (AI-only baseline)

    Attributes:
        staffing_level: Current staffing configuration
        num_controllers: Number of active controllers
        aircraft_per_controller: Aircraft capacity per controller
        total_capacity: Total aircraft capacity
    """

    # Staffing configuration presets
    STAFFING_CONFIGS = {
        'full': {
            'controllers': 4,
            'aircraft_per_controller': 15,
            'description': 'Full Staffing - Optimal Operations',
            'scenario': 'Peak traffic period with full staffing complement'
        },
        'normal': {
            'controllers': 3,
            'aircraft_per_controller': 12,
            'description': 'Normal Staffing - Typical Day',
            'scenario': 'Standard operations with typical staffing levels'
        },
        'reduced': {
            'controllers': 2,
            'aircraft_per_controller': 12,
            'description': 'Reduced Staffing - Budget Cuts',
            'scenario': 'Cost-cutting measures with reduced staffing'
        },
        'shutdown': {
            'controllers': 1,
            'aircraft_per_controller': 15,
            'description': 'Government Shutdown - Minimal Staff',
            'scenario': 'Emergency staffing with only essential personnel'
        },
        'none': {
            'controllers': 0,
            'aircraft_per_controller': 0,
            'description': 'AI-Only - Zero Human Controllers',
            'scenario': 'Fully automated AI-based air traffic control'
        }
    }

    def __init__(self, staffing_level: str = "normal"):
        """
        Initialize controller staffing model.

        Args:
            staffing_level: Staffing configuration - "full", "normal", "reduced",
                          "shutdown", or "none"

        Raises:
            ValueError: If staffing_level is invalid
        """
        if staffing_level not in self.STAFFING_CONFIGS:
            valid_levels = ', '.join(self.STAFFING_CONFIGS.keys())
            raise ValueError(
                f"Invalid staffing_level: {staffing_level}. "
                f"Must be one of: {valid_levels}"
            )

        self.staffing_level = staffing_level
        config = self.STAFFING_CONFIGS[staffing_level]

        self.num_controllers = config['controllers']
        self.aircraft_per_controller = config['aircraft_per_controller']
        self.total_capacity = self.num_controllers * self.aircraft_per_controller

        self._description = config['description']
        self._scenario = config['scenario']

    def get_capacity(self) -> int:
        """
        Get total aircraft capacity for current staffing.

        Returns:
            Total number of aircraft that can be safely managed

        Example:
            >>> staffing = ControllerStaffing("normal")
            >>> staffing.get_capacity()
            36
        """
        return self.total_capacity

    def calculate_workload_factor(self, active_aircraft_count: int) -> float:
        """
        Calculate current workload factor.

        Returns ratio of active aircraft to capacity. Values:
        - < 1.0: Under capacity (safe operations)
        - = 1.0: At capacity (nominal workload)
        - > 1.0: Over capacity (performance degradation)

        Args:
            active_aircraft_count: Current number of active aircraft

        Returns:
            Workload factor (1.0 = nominal, >1.0 = overloaded)

        Example:
            >>> staffing = ControllerStaffing("normal")  # capacity = 36
            >>> staffing.calculate_workload_factor(30)
            0.833  # Under capacity
            >>> staffing.calculate_workload_factor(45)
            1.25   # Overloaded by 25%
        """
        if self.total_capacity == 0:
            # AI-only mode - no capacity limits
            return 0.0

        return active_aircraft_count / self.total_capacity

    def get_performance_penalty(self, active_aircraft_count: int) -> Dict[str, float]:
        """
        Calculate performance degradation due to controller overload.

        Returns multipliers and probabilities for performance penalties:
        - delay_multiplier: Scales departure delays (exponential growth when overloaded)
        - conflict_risk_multiplier: Increases conflict likelihood (power growth)
        - human_error_probability: Chance of controller mistake (0.0 to 0.10)

        Formulas:
        - delay_multiplier: 1.0 if under capacity, else 1.0 + (workload - 1.0)^2
        - conflict_risk_multiplier: 1.0 if under capacity, else 1.0 + (workload - 1.0)^1.5
        - human_error_probability: 0.01 at normal load, exponential increase when overloaded

        Args:
            active_aircraft_count: Current number of active aircraft

        Returns:
            Dictionary with performance penalty values

        Example:
            >>> staffing = ControllerStaffing("shutdown")  # capacity = 15
            >>> penalties = staffing.get_performance_penalty(30)  # 2x overload
            >>> penalties['delay_multiplier']
            2.0  # Delays doubled
            >>> penalties['conflict_risk_multiplier']
            1.41  # ~40% more conflicts
        """
        workload = self.calculate_workload_factor(active_aircraft_count)

        # AI-only mode has no penalties (unlimited capacity)
        if self.total_capacity == 0:
            return {
                'delay_multiplier': 1.0,
                'conflict_risk_multiplier': 1.0,
                'human_error_probability': 0.0
            }

        # Under capacity - normal performance
        if workload <= 1.0:
            return {
                'delay_multiplier': 1.0,
                'conflict_risk_multiplier': 1.0,
                'human_error_probability': 0.01  # Baseline 1% error rate
            }

        # Overloaded - performance degradation
        overload = workload - 1.0

        # Delay multiplier: quadratic growth (delays compound quickly)
        delay_multiplier = 1.0 + (overload ** 2)

        # Conflict risk: power growth (harder to maintain separation)
        conflict_risk_multiplier = 1.0 + (overload ** 1.5)

        # Human error: exponential growth (fatigue, stress, mistakes)
        # Cap at 10% error probability
        human_error_probability = min(0.10, 0.01 * math.exp(overload))

        return {
            'delay_multiplier': delay_multiplier,
            'conflict_risk_multiplier': conflict_risk_multiplier,
            'human_error_probability': human_error_probability
        }

    def get_staffing_description(self) -> Dict[str, Any]:
        """
        Get human-readable staffing description.

        Returns:
            Dictionary with staffing information:
            - staffing_level: Configuration name
            - num_controllers: Number of controllers
            - capacity: Total aircraft capacity
            - aircraft_per_controller: Capacity per controller
            - description: Short description
            - scenario: Scenario explanation

        Example:
            >>> staffing = ControllerStaffing("shutdown")
            >>> desc = staffing.get_staffing_description()
            >>> print(desc['description'])
            'Government Shutdown - Minimal Staff'
        """
        return {
            'staffing_level': self.staffing_level,
            'num_controllers': self.num_controllers,
            'capacity': self.total_capacity,
            'aircraft_per_controller': self.aircraft_per_controller,
            'description': self._description,
            'scenario': self._scenario
        }

    def __repr__(self) -> str:
        """String representation of staffing configuration."""
        return (
            f"ControllerStaffing(level='{self.staffing_level}', "
            f"controllers={self.num_controllers}, "
            f"capacity={self.total_capacity})"
        )


if __name__ == "__main__":
    """Test controller capacity model when module is executed directly."""
    print("Testing ControllerStaffing...")
    print()

    # Test all staffing levels
    for level in ControllerStaffing.STAFFING_CONFIGS.keys():
        print(f"=== {level.upper()} STAFFING ===")
        staffing = ControllerStaffing(level)
        desc = staffing.get_staffing_description()

        print(f"Description: {desc['description']}")
        print(f"Controllers: {desc['num_controllers']}")
        print(f"Capacity: {desc['capacity']} aircraft")
        print(f"Scenario: {desc['scenario']}")
        print()

        # Test workload at different aircraft counts
        if staffing.total_capacity > 0:
            test_counts = [
                int(staffing.total_capacity * 0.5),  # 50% capacity
                staffing.total_capacity,              # 100% capacity
                int(staffing.total_capacity * 1.5)    # 150% capacity
            ]
        else:
            test_counts = [10, 50, 100]  # AI-only

        print("Workload Analysis:")
        for count in test_counts:
            workload = staffing.calculate_workload_factor(count)
            penalties = staffing.get_performance_penalty(count)

            print(f"  {count} aircraft:")
            print(f"    Workload: {workload:.2f}x")
            print(f"    Delay multiplier: {penalties['delay_multiplier']:.2f}x")
            print(f"    Conflict risk: {penalties['conflict_risk_multiplier']:.2f}x")
            print(f"    Error probability: {penalties['human_error_probability']:.1%}")

        print()

    print("ControllerStaffing test complete!")
