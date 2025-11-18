"""
Departure delay modeling for DeepSky ATC.

Uses log-normal distribution to model flight delays. This matches empirical
FAA data showing right-skewed delays: most flights have small delays (0-30min),
while a small percentage have large delays (60+ min). The log-normal distribution
naturally produces this pattern and ensures all delays are non-negative.

The log-normal distribution is characterized by:
- Right skewness: Long tail toward higher delays
- Non-negativity: All sampled values are >= 0
- Realistic shape: Matches real-world airport delay patterns
- Two parameters: median (scale) and sigma (shape/spread)

Reference: FAA ASPM (Aviation System Performance Metrics) shows departure
delays follow approximately log-normal distributions.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from scipy import stats


class DelayModel:
    """
    Probabilistic model for flight departure delays.

    Uses log-normal distributions to generate realistic departure delays
    based on time of day and operational conditions. Supports multiple
    delay profiles (normal, weather, peak hours, overnight) and time-based
    profile selection.

    Attributes:
        config: Loaded delay configuration dictionary
        rng: NumPy RandomState for reproducible random sampling
        probability_no_delay: Probability of zero delay (on-time departure)
        max_delay_minutes: Maximum allowed delay in minutes
        delay_distributions: Dictionary of delay profiles
        delay_by_hour: Mapping of hour (0-23) to profile name
    """

    def __init__(self, config_path: Optional[str] = None, seed: Optional[int] = None):
        """
        Initialize delay model from configuration file.

        Args:
            config_path: Path to delay configuration JSON file.
                        If None, uses default: data/delay_config.json
            seed: Random seed for reproducibility. If None, uses random seed.

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or missing required fields
        """
        # Default config path
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "data" / "delay_config.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Delay configuration not found: {config_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Extract configuration parameters
        try:
            self.delay_distributions = self.config["delay_distributions"]
            self.probability_no_delay = self.config["probability_no_delay"]
            self.max_delay_minutes = self.config["max_delay_minutes"]
            self.delay_by_hour = self.config["delay_by_hour"]
        except KeyError as e:
            raise ValueError(f"Invalid delay config: missing field {e}")

        # Initialize random number generator
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible delay generation.

        Args:
            seed: Integer seed value
        """
        self.rng = np.random.RandomState(seed)

    def _get_profile_for_hour(self, hour_of_day: int) -> Dict[str, Any]:
        """
        Get delay profile based on hour of day.

        Args:
            hour_of_day: Hour (0-23)

        Returns:
            Dictionary with delay profile parameters

        Raises:
            ValueError: If hour is invalid or profile not found
        """
        if not (0 <= hour_of_day <= 23):
            raise ValueError(f"Hour must be 0-23, got {hour_of_day}")

        # Get profile name for this hour
        profile_name = self.delay_by_hour.get(str(hour_of_day))

        if profile_name is None:
            raise ValueError(f"No profile defined for hour {hour_of_day}")

        # Get profile configuration
        profile = self.delay_distributions.get(profile_name)

        if profile is None:
            raise ValueError(f"Profile '{profile_name}' not found in configuration")

        return profile

    def calculate_departure_delay(
        self,
        scheduled_time: Optional[float] = None,
        hour_of_day: Optional[int] = None,
        profile_name: Optional[str] = None
    ) -> float:
        """
        Calculate departure delay in minutes.

        Determines appropriate delay profile based on hour of day (or uses
        specified profile), then samples from log-normal distribution with
        some probability of zero delay.

        Args:
            scheduled_time: Scheduled departure time in seconds since epoch
                           (used to extract hour if hour_of_day not provided)
            hour_of_day: Hour of day (0-23). If None, extracted from scheduled_time
            profile_name: Explicit profile name to use. Overrides time-based selection.

        Returns:
            Delay in minutes (float, >= 0)

        Example:
            >>> delay_model = DelayModel(seed=42)
            >>> # Morning rush hour (7 AM)
            >>> delay = delay_model.calculate_departure_delay(hour_of_day=7)
            >>> print(f"Delay: {delay:.1f} minutes")
            Delay: 15.3 minutes
        """
        # Determine which profile to use
        if profile_name is not None:
            # Explicit profile specified
            profile = self.delay_distributions.get(profile_name)
            if profile is None:
                raise ValueError(f"Unknown profile: {profile_name}")
        elif hour_of_day is not None:
            # Use hour-based profile
            profile = self._get_profile_for_hour(hour_of_day)
        elif scheduled_time is not None:
            # Extract hour from scheduled time
            # Assuming scheduled_time is in seconds since epoch or simulation start
            hour_of_day = int((scheduled_time / 3600) % 24)
            profile = self._get_profile_for_hour(hour_of_day)
        else:
            # Default to "normal" profile
            profile = self.delay_distributions["normal"]

        # Check for no delay (on-time departure)
        if self.rng.random() < self.probability_no_delay:
            return 0.0

        # Sample from log-normal distribution
        # Log-normal parameters:
        #   s (sigma): shape parameter - controls spread/variability
        #   scale: scale parameter - equals the median for log-normal
        median_minutes = profile["median_minutes"]
        sigma = profile["sigma"]

        # Sample from log-normal distribution
        # scipy.stats.lognorm uses: scale = median, s = sigma
        delay_minutes = stats.lognorm.rvs(
            s=sigma,
            scale=median_minutes,
            random_state=self.rng
        )

        # Clip to maximum delay
        delay_minutes = min(delay_minutes, self.max_delay_minutes)

        return delay_minutes

    def get_actual_departure_time(
        self,
        scheduled_time: float,
        hour_of_day: Optional[int] = None,
        profile_name: Optional[str] = None
    ) -> float:
        """
        Calculate actual departure time given scheduled time.

        Args:
            scheduled_time: Scheduled departure time in seconds
            hour_of_day: Hour of day (0-23). If None, extracted from scheduled_time
            profile_name: Explicit profile name. Overrides time-based selection.

        Returns:
            Actual departure time in seconds

        Example:
            >>> delay_model = DelayModel(seed=42)
            >>> scheduled = 3600 * 8  # 8:00 AM
            >>> actual = delay_model.get_actual_departure_time(scheduled, hour_of_day=8)
            >>> delay_min = (actual - scheduled) / 60
            >>> print(f"Delay: {delay_min:.1f} minutes")
        """
        # Calculate delay in minutes
        delay_minutes = self.calculate_departure_delay(
            scheduled_time=scheduled_time,
            hour_of_day=hour_of_day,
            profile_name=profile_name
        )

        # Convert to seconds and add to scheduled time
        delay_seconds = delay_minutes * 60.0
        actual_time = scheduled_time + delay_seconds

        return actual_time

    def generate_delay_statistics(
        self,
        profile_name: str = "normal",
        num_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Generate statistics for a delay profile.

        Samples many delays from the specified profile and computes
        descriptive statistics. Useful for validating that distributions
        match expected real-world patterns.

        Args:
            profile_name: Name of delay profile to analyze
            num_samples: Number of samples to generate (default 10,000)

        Returns:
            Dictionary with statistics:
            - mean: Mean delay in minutes
            - median: Median delay in minutes
            - std: Standard deviation in minutes
            - percentiles: Dict with 50th, 75th, 90th, 95th, 99th percentiles
            - percent_zero_delay: Percentage of on-time departures
            - min: Minimum delay
            - max: Maximum delay
            - profile_name: Name of profile analyzed

        Example:
            >>> delay_model = DelayModel(seed=42)
            >>> stats = delay_model.generate_delay_statistics("normal", 10000)
            >>> print(f"Mean delay: {stats['mean']:.1f} minutes")
            >>> print(f"90th percentile: {stats['percentiles']['p90']:.1f} minutes")
        """
        if profile_name not in self.delay_distributions:
            raise ValueError(f"Unknown profile: {profile_name}")

        # Generate samples
        samples = np.array([
            self.calculate_departure_delay(profile_name=profile_name)
            for _ in range(num_samples)
        ])

        # Calculate statistics
        percentiles = {
            'p50': np.percentile(samples, 50),
            'p75': np.percentile(samples, 75),
            'p90': np.percentile(samples, 90),
            'p95': np.percentile(samples, 95),
            'p99': np.percentile(samples, 99)
        }

        # Count zero delays
        zero_count = np.sum(samples == 0)
        percent_zero = (zero_count / num_samples) * 100

        return {
            'profile_name': profile_name,
            'num_samples': num_samples,
            'mean': float(np.mean(samples)),
            'median': float(np.median(samples)),
            'std': float(np.std(samples)),
            'percentiles': {k: float(v) for k, v in percentiles.items()},
            'percent_zero_delay': float(percent_zero),
            'min': float(np.min(samples)),
            'max': float(np.max(samples))
        }

    def get_profile_info(self, profile_name: str) -> Dict[str, Any]:
        """
        Get configuration information for a specific profile.

        Args:
            profile_name: Name of profile

        Returns:
            Profile configuration dictionary
        """
        if profile_name not in self.delay_distributions:
            raise ValueError(f"Unknown profile: {profile_name}")

        return self.delay_distributions[profile_name].copy()

    def __repr__(self) -> str:
        """String representation of the delay model."""
        profiles = list(self.delay_distributions.keys())
        return (
            f"DelayModel(profiles={profiles}, "
            f"p_no_delay={self.probability_no_delay}, "
            f"max_delay={self.max_delay_minutes}min)"
        )


if __name__ == "__main__":
    """Test delay model when module is executed directly."""
    print("Testing DelayModel...")
    print()

    # Create delay model with fixed seed
    delay_model = DelayModel(seed=42)

    print(f"Delay model: {delay_model}")
    print()

    # Test each profile
    profiles = ["overnight", "normal", "peak_hours", "weather"]

    for profile in profiles:
        print(f"Profile: {profile}")
        profile_info = delay_model.get_profile_info(profile)
        print(f"  Median: {profile_info['median_minutes']} min")
        print(f"  Sigma: {profile_info['sigma']}")

        # Generate a few samples
        samples = [delay_model.calculate_departure_delay(profile_name=profile) for _ in range(5)]
        print(f"  Sample delays: {[f'{s:.1f}' for s in samples]} minutes")
        print()

    # Generate statistics for normal profile
    print("Generating statistics for 'normal' profile (10,000 samples)...")
    stats = delay_model.generate_delay_statistics("normal", 10000)

    print(f"  Mean: {stats['mean']:.1f} min")
    print(f"  Median: {stats['median']:.1f} min")
    print(f"  Std dev: {stats['std']:.1f} min")
    print(f"  90th percentile: {stats['percentiles']['p90']:.1f} min")
    print(f"  On-time rate: {stats['percent_zero_delay']:.1f}%")
    print()

    print("DelayModel test complete!")
