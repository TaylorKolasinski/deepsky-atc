I'm building DeepSky ATC, an air traffic control simulator. This is Phase 1, Deliverable 1.1.

Create the airspace boundary system for a 150nm x 150nm region centered on KJFK airport.

Requirements:

1. Create `data/airspace_config.json` with:
   - KJFK center coordinates (lat: 40.6413, lon: -73.7781)
   - Boundaries: 150nm radius (convert to degrees for lat/lon bounds)
   - Altitude range: 0 to 45,000 feet (FL000 to FL450)

2. Create `src/coordinates.py` with:
   - Function to convert lat/lon/alt to local Cartesian x/y/z (meters)
   - Function to convert local x/y/z back to lat/lon/alt
   - Use flat-earth approximation (local tangent plane) centered on KJFK
   - Earth radius = 6371000 meters
   - 1 nautical mile = 1852 meters

3. Create `src/airspace.py` with:
   - Class `Airspace` that loads from config
   - Method `is_in_bounds(lat, lon, alt)` - returns True/False
   - Method `get_bounds_info()` - returns dict with boundary details

4. Create `tests/test_airspace.py` that:
   - Loads the airspace
   - Tests KJFK center is in bounds
   - Tests a point 160nm away is out of bounds
   - Tests altitude limits (below 0ft and above 45,000ft)
   - Prints coordinate conversions for KJFK center

Use Python 3.11, numpy for calculations. Keep it clean and well-documented.





I'm building DeepSky ATC Phase 1, Deliverable 1.3: Delay Profile Model.

Create a system to inject realistic, probabilistic departure delays using log-normal distributions.

Requirements:

1. Create `data/delay_config.json` with:
   - delay_distributions: Different profiles for different scenarios
     * "normal": median=12min, sigma=0.8 (typical operations)
     * "weather": median=40min, sigma=1.0 (bad weather, high variability)
     * "peak_hours": median=20min, sigma=0.9 (rush hour congestion)
     * "overnight": median=3min, sigma=0.5 (late night, minimal delays)
   - probability_no_delay: 0.20 (20% of flights depart exactly on time)
   - max_delay_minutes: 180 (cap at 3 hours for extreme cases)
   - delay_by_hour: Dictionary mapping hour (0-23) to profile name
     * 0-5: "overnight"
     * 6-9: "peak_hours" (morning rush)
     * 10-16: "normal"
     * 17-20: "peak_hours" (evening rush)
     * 21-23: "normal"

2. Create `src/delay_model.py` with:

   Class `DelayModel`:
   - __init__(config_path="data/delay_config.json", seed=None):
     * Load delay configuration
     * Set up random number generator with optional seed
     * Store numpy RandomState for reproducibility
   
   - calculate_departure_delay(scheduled_time=None, hour_of_day=None):
     * Determine which profile to use based on hour_of_day
     * If no hour provided and scheduled_time is given, extract hour from it
     * Return 0 with probability_no_delay chance
     * Otherwise sample from log-normal distribution:
       - Use scipy.stats.lognorm
       - Parameters: s=sigma, scale=median from profile
       - Log-normal is right-skewed: many small delays, few large ones
     * Clip result to [0, max_delay_minutes]
     * Return delay in minutes (float)
   
   - get_actual_departure_time(scheduled_time, hour_of_day=None):
     * Calculate delay in minutes
     * Convert to seconds and add to scheduled_time
     * Return actual departure time
   
   - generate_delay_statistics(profile_name="normal", num_samples=10000):
     * Generate large sample from specified profile
     * Return dict with:
       - mean, median, std
       - percentiles: [50th, 75th, 90th, 95th, 99th]
       - percentage with zero delay
     * This validates our distributions match real-world patterns
   
   - set_seed(seed):
     * Set random seed for reproducibility

   Include docstring explaining log-normal choice:
   "Uses log-normal distribution to model flight delays. This matches empirical 
   FAA data showing right-skewed delays: most flights have small delays (0-30min),
   while a small percentage have large delays (60+ min). The log-normal distribution
   naturally produces this pattern and ensures all delays are non-negative."

3. Update `src/route_generator.py`:
   - Add `scheduled_departure_time` attribute to FlightRoute class (default 0.0)
   - Add `actual_departure_time` attribute (initially None)
   - Add method: `apply_delay(delay_model, hour_of_day=None)`
     * Uses delay_model.get_actual_departure_time()
     * Stores result in actual_departure_time
     * Returns the delay amount in minutes

4. Create `tests/test_delays.py` that:
   - Loads delay model with fixed seed for reproducibility
   - Tests each profile (normal, weather, peak_hours, overnight) with 10,000 samples
   - Prints comprehensive statistics for each:
     * Mean, median, std
     * Percentile breakdown (50th, 75th, 90th, 95th, 99th)
     * Percentage of on-time departures
   - Validates delays are within [0, max_delay_minutes]
   - Tests time-based profile selection (morning vs overnight)
   - Shows ASCII histogram for "normal" profile
   - Tests integration: creates FlightRoute, applies delay, verifies results
   - Verifies log-normal produces realistic right-skewed distribution

Use scipy.stats.lognorm for sampling. The sigma parameter controls spread - 
higher sigma = more variability and longer tail of extreme delays.

Key validation: "normal" profile should show ~70% of flights under 20min delay,
~90% under 40min, with rare delays up to the 180min cap.