"""
Generate documentation visuals for DeepSky ATC.

Creates publication-quality plots for blog posts and documentation.
Run once to generate artifacts for LinkedIn, Medium, etc.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

from src.route_generator import load_routes_from_data, generate_waypoints
from src.delay_model import DelayModel
from src.aircraft import Aircraft
from src.airspace import Airspace


# Output directory
OUTPUT_DIR = project_root / "docs" / "images"


def setup_plotting_style():
    """Set up matplotlib style for professional-looking plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9


def plot_jfk_route_map():
    """
    Plot 1: JFK Route Map
    Shows all real-world routes from JFK database.
    """
    print("Generating Plot 1: JFK Route Map...")

    # Load routes
    routes = load_routes_from_data()

    if len(routes) == 0:
        print("  ✗ No routes found. Run data acquisition first.")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # KJFK location
    jfk_lat, jfk_lon = 40.6413, -73.7781

    # Categorize routes by distance
    short_routes = []  # < 1000nm
    medium_routes = []  # 1000-3000nm
    long_routes = []  # > 3000nm

    for route in routes:
        if route.total_distance_nm < 1000:
            short_routes.append(route)
        elif route.total_distance_nm < 3000:
            medium_routes.append(route)
        else:
            long_routes.append(route)

    # Plot routes by category
    def plot_route_category(route_list, color, label, alpha=0.3, linewidth=0.5):
        for route in route_list:
            if len(route.waypoints) >= 2:
                # Get first and last waypoint
                start = route.waypoints[0]
                end = route.waypoints[-1]

                # Plot great circle as straight line (simple approximation)
                lats = [start[0], end[0]]
                lons = [start[1], end[1]]
                ax.plot(lons, lats, color=color, alpha=alpha, linewidth=linewidth)

    # Plot long routes first (background)
    plot_route_category(long_routes, 'red', f'Long-haul (>{3000}nm): {len(long_routes)}', alpha=0.2, linewidth=0.6)
    plot_route_category(medium_routes, 'green', f'Medium (1000-3000nm): {len(medium_routes)}', alpha=0.4, linewidth=0.7)
    plot_route_category(short_routes, 'blue', f'Short (<1000nm): {len(short_routes)}', alpha=0.6, linewidth=0.8)

    # Plot JFK as a star
    ax.plot(jfk_lon, jfk_lat, 'k*', markersize=20, label='KJFK', zorder=10)

    # Labels and title
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(f'DeepSky ATC: {len(routes)} Real-World Routes from JFK\n'
                 f'Data from OpenFlights Database', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label=f'Short-haul (<1000nm): {len(short_routes)}'),
        plt.Line2D([0], [0], color='green', linewidth=2, label=f'Medium-haul (1000-3000nm): {len(medium_routes)}'),
        plt.Line2D([0], [0], color='red', linewidth=2, label=f'Long-haul (>3000nm): {len(long_routes)}'),
        plt.Line2D([0], [0], marker='*', color='k', linewidth=0, markersize=10, label='KJFK')
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    # Save
    output_path = OUTPUT_DIR / "jfk_route_map.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved to {output_path}")
    return output_path


def plot_delay_distributions():
    """
    Plot 2: Delay Distributions
    Shows histograms of all 4 delay profiles.
    """
    print("Generating Plot 2: Delay Distributions...")

    # Load delay model
    delay_model = DelayModel(seed=42)

    # Generate samples for each profile
    profiles = ['overnight', 'normal', 'peak_hours', 'weather']
    num_samples = 10000

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, profile_name in enumerate(profiles):
        ax = axes[i]

        # Generate samples
        samples = [delay_model.calculate_departure_delay(profile_name=profile_name)
                  for _ in range(num_samples)]

        # Calculate statistics
        mean = np.mean(samples)
        median = np.median(samples)
        p95 = np.percentile(samples, 95)

        # Plot histogram
        ax.hist(samples, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

        # Add vertical lines for statistics
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f} min')
        ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.1f} min')
        ax.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95th: {p95:.1f} min')

        # Labels
        ax.set_xlabel('Delay (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{profile_name.replace("_", " ").title()} Profile', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Log-Normal Delay Distributions by Time Period\n'
                 f'{num_samples:,} samples per profile',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = OUTPUT_DIR / "delay_distributions.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved to {output_path}")
    return output_path


def plot_sample_flight_path():
    """
    Plot 3: Sample Flight Path
    Shows JFK→LAX flight with altitude/ground track/speed profiles.
    """
    print("Generating Plot 3: Sample Flight Path (JFK→LAX)...")

    # Generate JFK→LAX route
    jfk_lat, jfk_lon = 40.6413, -73.7781
    lax_lat, lax_lon = 33.9416, -118.4085
    waypoints, distance = generate_waypoints(jfk_lat, jfk_lon, lax_lat, lax_lon)

    from src.route_generator import FlightRoute
    route = FlightRoute(
        route_id="JFK_LAX_DEMO",
        aircraft_type="B738",
        departure_icao="KJFK",
        arrival_icao="KLAX",
        waypoints=waypoints,
        total_distance_nm=distance
    )

    # Create aircraft
    aircraft = Aircraft(id="DEMO001", route=route, departure_time=0.0)

    # Simulate flight and collect data
    positions = []
    altitudes = []
    speeds = []
    times = []
    statuses = []
    waypoint_indices = []

    # Run for 2 hours
    for t in range(0, 7201, 10):  # Every 10 seconds for 2 hours
        aircraft.update_position(delta_t=10.0)
        aircraft.check_waypoint_reached(threshold_nm=2.0)

        state = aircraft.get_state()
        positions.append((state['position']['lat'], state['position']['lon']))
        altitudes.append(state['position']['alt'])
        speeds.append(state['velocity'])
        times.append(t / 60)  # Convert to minutes
        statuses.append(state['status'])
        waypoint_indices.append(state['waypoint_info']['current_index'])

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Altitude Profile
    ax1.plot(times, altitudes, 'b-', linewidth=2)
    ax1.fill_between(times, altitudes, alpha=0.3)

    # Mark waypoints
    waypoint_times = []
    waypoint_alts = []
    for i, (t, wp_idx) in enumerate(zip(times, waypoint_indices)):
        if i > 0 and wp_idx != waypoint_indices[i-1]:
            waypoint_times.append(t)
            waypoint_alts.append(altitudes[i])

    ax1.plot(waypoint_times, waypoint_alts, 'ro', markersize=8, label='Waypoints reached')

    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Altitude (feet)')
    ax1.set_title('Altitude Profile', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Ground Track
    lats, lons = zip(*positions)
    ax2.plot(lons, lats, 'g-', linewidth=2)

    # Mark departure and arrival
    ax2.plot(jfk_lon, jfk_lat, 'b*', markersize=15, label='KJFK')
    ax2.plot(lax_lon, lax_lat, 'r*', markersize=15, label='KLAX')

    # Mark waypoints
    for wp in waypoints:
        ax2.plot(wp[1], wp[0], 'ko', markersize=4, alpha=0.5)

    ax2.set_xlabel('Longitude (degrees)')
    ax2.set_ylabel('Latitude (degrees)')
    ax2.set_title('Ground Track', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Speed Profile
    ax3.plot(times, speeds, 'r-', linewidth=2)
    ax3.fill_between(times, speeds, alpha=0.3, color='red')

    # Annotate status transitions
    prev_status = statuses[0]
    for i, status in enumerate(statuses):
        if status != prev_status:
            ax3.axvline(times[i], color='gray', linestyle='--', alpha=0.5)
            ax3.text(times[i], speeds[i] + 20, status, rotation=90,
                    verticalalignment='bottom', fontsize=8)
            prev_status = status

    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Speed (knots)')
    ax3.set_title('Speed Profile', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Sample Flight: JFK to LAX ({distance:.0f} nm)\n'
                 f'Boeing 737-800, {len(waypoints)} waypoints',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = OUTPUT_DIR / "sample_flight_path.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved to {output_path}")
    return output_path


def plot_airspace_coverage():
    """
    Plot 4: Airspace Coverage
    Shows KJFK 150nm x 150nm airspace with route density heatmap.
    """
    print("Generating Plot 4: Airspace Coverage...")

    # Load airspace
    airspace = Airspace()

    # Load routes
    routes = load_routes_from_data()

    if len(routes) == 0:
        print("  ✗ No routes found. Run data acquisition first.")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Extract first waypoints from all routes
    waypoint_lats = []
    waypoint_lons = []

    for route in routes:
        if len(route.waypoints) > 1:
            # Get second waypoint (first is departure airport)
            wp = route.waypoints[1]
            waypoint_lats.append(wp[0])
            waypoint_lons.append(wp[1])

    # Create 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(
        waypoint_lons, waypoint_lats,
        bins=30,
        range=[[-110, -40], [25, 55]]
    )

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower',
                   cmap='YlOrRd', alpha=0.7, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Route Density', rotation=270, labelpad=20)

    # Plot KJFK
    ax.plot(airspace.center_lon, airspace.center_lat, 'b*',
            markersize=20, label='KJFK', zorder=10)

    # Draw airspace boundary (approximate circle as box)
    # Convert 150nm radius to degrees (rough approximation)
    radius_deg_lat = 150 / 60  # 1 degree ≈ 60 nm
    radius_deg_lon = 150 / 60 / np.cos(np.radians(airspace.center_lat))

    boundary = mpatches.Rectangle(
        (airspace.center_lon - radius_deg_lon, airspace.center_lat - radius_deg_lat),
        2 * radius_deg_lon,
        2 * radius_deg_lat,
        linewidth=2,
        edgecolor='blue',
        facecolor='none',
        linestyle='--',
        label='150nm x 150nm Airspace'
    )
    ax.add_patch(boundary)

    # Labels
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(f'KJFK Airspace Coverage (150nm x 150nm)\n'
                 f'First waypoint density from {len(routes)} routes',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # Set axis limits to focus on relevant area
    ax.set_xlim(-100, -55)
    ax.set_ylim(30, 50)

    # Save
    output_path = OUTPUT_DIR / "airspace_coverage.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved to {output_path}")
    return output_path


def print_summary(output_files):
    """Print summary of generated files."""
    print("\n" + "=" * 70)
    print("Documentation Visuals Generated Successfully!")
    print("=" * 70)
    print()

    print("Files created:")
    total_size = 0

    for file_path in output_files:
        if file_path and file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            total_size += size_kb

            # Get image dimensions
            import PIL.Image
            with PIL.Image.open(file_path) as img:
                width, height = img.size

            print(f"  ✓ {file_path.name}")
            print(f"    - Path: {file_path}")
            print(f"    - Size: {size_kb:.1f} KB")
            print(f"    - Dimensions: {width} x {height} pixels")
            print()

    print(f"Total size: {total_size:.1f} KB ({total_size/1024:.2f} MB)")
    print()
    print("These images are ready for:")
    print("  - LinkedIn posts")
    print("  - Medium articles")
    print("  - GitHub README")
    print("  - Technical documentation")
    print()


def main():
    """Generate all documentation visuals."""
    print("\n" + "=" * 70)
    print("DeepSky ATC - Documentation Visual Generator")
    print("=" * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Set up plotting style
    setup_plotting_style()

    # Generate plots
    output_files = []

    try:
        output_files.append(plot_jfk_route_map())
        output_files.append(plot_delay_distributions())
        output_files.append(plot_sample_flight_path())
        output_files.append(plot_airspace_coverage())

        # Print summary
        print_summary(output_files)

    except Exception as e:
        print(f"\n✗ Error generating visuals: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
