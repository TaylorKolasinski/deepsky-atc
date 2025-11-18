# DeepSky ATC Documentation Images

This directory contains publication-quality visuals for blog posts, documentation, and social media.

## Generated Images

All images are 300 DPI, suitable for print and web publishing.

### 1. `jfk_route_map.png` (958 KB)
- **Description**: Global route map showing all 322 unique routes from JFK
- **Data Source**: OpenFlights database (911 total routes, deduplicated to 322 unique airport pairs)
- **Color Coding**:
  - Blue: Short-haul routes (<1000nm)
  - Green: Medium-haul routes (1000-3000nm)
  - Red: Long-haul routes (>3000nm)
- **Use Cases**: Overview of simulation scope, route diversity demonstration

### 2. `delay_distributions.png` (319 KB)
- **Description**: Log-normal delay distributions for 4 time-based profiles
- **Profiles**:
  - Overnight: 3min median, low variability
  - Normal: 12min median, typical operations
  - Peak Hours: 20min median, rush hour congestion
  - Weather: 40min median, high variability
- **Sample Size**: 10,000 samples per profile
- **Statistics Shown**: Mean, median, 95th percentile
- **Use Cases**: Explaining delay modeling methodology, showing realistic right-skewed distributions

### 3. `sample_flight_path.png` (307 KB)
- **Description**: Complete flight profile for JFK→LAX (2,146 nm)
- **Aircraft**: Boeing 737-800
- **Subplots**:
  1. Altitude profile over time (climb/cruise/descent phases)
  2. Ground track (lat/lon path)
  3. Speed profile with status transitions
- **Features**: Waypoint markers, status annotations (TAXI→CLIMBING→CRUISING)
- **Use Cases**: Demonstrating kinematic model, showing realistic flight phases

### 4. `airspace_coverage.png` (168 KB)
- **Description**: KJFK 150nm x 150nm airspace with route density heatmap
- **Visualization**: 2D histogram of first waypoints from all routes
- **Boundary**: Blue dashed box showing 150nm radius airspace
- **Color Scale**: Yellow-Orange-Red indicating route density
- **Use Cases**: Airspace boundary visualization, traffic flow patterns

## Generation

To regenerate these images:
```bash
python3 scripts/generate_docs_visuals.py
```

**Requirements**:
- matplotlib
- pillow
- numpy
- All DeepSky ATC data files (routes, airports)

**Total Size**: 1.71 MB (4 images)

## Usage Rights

These images are generated from:
- Real-world data: OpenFlights database (Open Database License)
- Simulation code: DeepSky ATC (your project license)

Suitable for:
- Technical blog posts (Medium, LinkedIn)
- GitHub repository README
- Conference presentations
- Academic papers (with attribution)
- Social media promotion

## Metadata

- **Generated**: November 18, 2024
- **Software**: matplotlib 3.x, DeepSky ATC Phase 1
- **Resolution**: 300 DPI
- **Format**: PNG with transparency support
- **Dimensions**: ~4000 x 3000 pixels (varies by plot)
