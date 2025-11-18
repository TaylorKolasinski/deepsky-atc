"""
Data acquisition module for DeepSky ATC.

Downloads and processes real-world flight route data from OpenFlights database.
Filters routes relevant to KJFK airspace.
"""

import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


# OpenFlights database URLs
OPENFLIGHTS_ROUTES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
OPENFLIGHTS_AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def download_file(url: str, max_retries: int = 3) -> Optional[bytes]:
    """
    Download a file from URL with retry logic.

    Args:
        url: URL to download from
        max_retries: Maximum number of retry attempts

    Returns:
        File content as bytes, or None if download failed

    Raises:
        None - errors are caught and logged
    """
    for attempt in range(max_retries):
        try:
            print(f"Downloading {url} (attempt {attempt + 1}/{max_retries})...")

            # Set a reasonable timeout
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
                print(f"✓ Downloaded {len(data)} bytes")
                return data

        except urllib.error.URLError as e:
            print(f"✗ Network error: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to download after {max_retries} attempts")
                return None

        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

    return None


def download_airports_database(output_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Download and parse OpenFlights airports database.

    The airports.dat format:
    0: Airport ID
    1: Name
    2: City
    3: Country
    4: IATA code (3-letter, e.g., "JFK")
    5: ICAO code (4-letter, e.g., "KJFK")
    6: Latitude
    7: Longitude
    8: Altitude (feet)
    9: Timezone
    10: DST
    11: Tz database timezone
    12: Type
    13: Source

    Args:
        output_path: Path to save CSV file. If None, uses data/airports_raw.csv

    Returns:
        DataFrame with airport data, or None if download failed
    """
    if output_path is None:
        output_path = DATA_DIR / "airports_raw.csv"

    # Ensure data directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download data
    data = download_file(OPENFLIGHTS_AIRPORTS_URL)
    if data is None:
        return None

    # Parse CSV
    try:
        # Define column names based on OpenFlights schema
        columns = [
            'airport_id', 'name', 'city', 'country',
            'iata', 'icao', 'latitude', 'longitude',
            'altitude_ft', 'timezone', 'dst', 'tz_database',
            'type', 'source'
        ]

        # Read CSV from bytes
        import io
        df = pd.read_csv(
            io.BytesIO(data),
            header=None,
            names=columns,
            na_values=['\\N', 'null', ''],
            encoding='utf-8'
        )

        print(f"✓ Parsed {len(df)} airports")

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")

        return df

    except Exception as e:
        print(f"✗ Failed to parse airports data: {e}")
        return None


def download_routes_database(output_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Download and parse OpenFlights routes database.

    The routes.dat format:
    0: Airline (2-letter IATA or 3-letter ICAO)
    1: Airline ID
    2: Source airport (3-letter IATA)
    3: Source airport ID
    4: Destination airport (3-letter IATA)
    5: Destination airport ID
    6: Codeshare
    7: Stops
    8: Equipment (aircraft types)

    Args:
        output_path: Path to save CSV file. If None, uses data/routes_raw.csv

    Returns:
        DataFrame with route data, or None if download failed
    """
    if output_path is None:
        output_path = DATA_DIR / "routes_raw.csv"

    # Ensure data directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download data
    data = download_file(OPENFLIGHTS_ROUTES_URL)
    if data is None:
        return None

    # Parse CSV
    try:
        # Define column names based on OpenFlights schema
        columns = [
            'airline', 'airline_id',
            'source_iata', 'source_id',
            'dest_iata', 'dest_id',
            'codeshare', 'stops', 'equipment'
        ]

        # Read CSV from bytes
        import io
        df = pd.read_csv(
            io.BytesIO(data),
            header=None,
            names=columns,
            na_values=['\\N', 'null', ''],
            encoding='utf-8'
        )

        print(f"✓ Parsed {len(df)} routes")

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")

        return df

    except Exception as e:
        print(f"✗ Failed to parse routes data: {e}")
        return None


def filter_jfk_routes(
    routes_df: pd.DataFrame,
    airports_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter routes where source or destination is JFK.

    Args:
        routes_df: DataFrame with all routes
        airports_df: DataFrame with all airports

    Returns:
        Tuple of (filtered_routes_df, jfk_airports_df)
    """
    print("\nFiltering JFK routes...")

    # Find JFK in airports database
    jfk_airports = airports_df[
        (airports_df['iata'] == 'JFK') |
        (airports_df['icao'] == 'KJFK')
    ].copy()

    if len(jfk_airports) == 0:
        print("✗ JFK airport not found in database!")
        return pd.DataFrame(), pd.DataFrame()

    print(f"✓ Found JFK: {jfk_airports.iloc[0]['name']}")
    print(f"  Location: ({jfk_airports.iloc[0]['latitude']}, {jfk_airports.iloc[0]['longitude']})")
    print(f"  Altitude: {jfk_airports.iloc[0]['altitude_ft']} ft")

    # Filter routes with JFK as source or destination
    jfk_routes = routes_df[
        (routes_df['source_iata'] == 'JFK') |
        (routes_df['dest_iata'] == 'JFK')
    ].copy()

    print(f"✓ Found {len(jfk_routes)} routes involving JFK")

    # Count unique routes (by source-destination pairs)
    unique_pairs = jfk_routes[['source_iata', 'dest_iata']].drop_duplicates()
    print(f"  {len(unique_pairs)} unique airport pairs")

    # Count departures vs arrivals
    departures = len(jfk_routes[jfk_routes['source_iata'] == 'JFK'])
    arrivals = len(jfk_routes[jfk_routes['dest_iata'] == 'JFK'])
    print(f"  {departures} departures from JFK")
    print(f"  {arrivals} arrivals to JFK")

    return jfk_routes, jfk_airports


def acquire_jfk_route_data(force_download: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Main function to acquire and filter JFK route data.

    Downloads OpenFlights data if needed, filters for JFK routes,
    and saves to CSV files.

    Args:
        force_download: If True, re-download even if files exist

    Returns:
        Tuple of (jfk_routes_df, airports_df) or (None, None) if failed
    """
    print("=" * 70)
    print("DeepSky ATC - Route Data Acquisition")
    print("=" * 70)
    print()

    routes_file = DATA_DIR / "routes_raw.csv"
    airports_file = DATA_DIR / "airports_raw.csv"
    jfk_routes_file = DATA_DIR / "jfk_routes.csv"

    # Download or load airports
    if force_download or not airports_file.exists():
        print("Downloading airports database...")
        airports_df = download_airports_database(airports_file)
        if airports_df is None:
            return None, None
    else:
        print(f"Loading airports from {airports_file}...")
        airports_df = pd.read_csv(airports_file)
        print(f"✓ Loaded {len(airports_df)} airports")

    # Download or load routes
    if force_download or not routes_file.exists():
        print("\nDownloading routes database...")
        routes_df = download_routes_database(routes_file)
        if routes_df is None:
            return None, None
    else:
        print(f"\nLoading routes from {routes_file}...")
        routes_df = pd.read_csv(routes_file)
        print(f"✓ Loaded {len(routes_df)} routes")

    # Filter for JFK routes
    jfk_routes, jfk_airports = filter_jfk_routes(routes_df, airports_df)

    if len(jfk_routes) == 0:
        print("✗ No JFK routes found")
        return None, None

    # Save filtered JFK routes
    jfk_routes.to_csv(jfk_routes_file, index=False)
    print(f"\n✓ Saved JFK routes to {jfk_routes_file}")

    print("\n" + "=" * 70)
    print("Data acquisition complete!")
    print("=" * 70)

    return jfk_routes, airports_df


if __name__ == "__main__":
    """Run data acquisition when module is executed directly."""
    jfk_routes, airports = acquire_jfk_route_data(force_download=True)

    if jfk_routes is not None:
        print(f"\nSuccess! Acquired {len(jfk_routes)} JFK routes")
        print("\nSample routes:")
        print(jfk_routes.head(10))
    else:
        print("\nFailed to acquire route data")
