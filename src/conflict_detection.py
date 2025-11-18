"""
Conflict detection and tracking for DeepSky ATC.

Detects separation violations between aircraft based on FAA standards
and tracks conflicts over time for analysis and AI training data.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

from src.physics import haversine_distance


# FAA separation standards
DEFAULT_HORIZONTAL_SEP_NM = 5.0  # Nautical miles
DEFAULT_VERTICAL_SEP_FT = 1000.0  # Feet


class ConflictDetector:
    """
    Detects separation violations between aircraft.

    Uses FAA separation minimums to identify conflicts. A conflict occurs
    when BOTH horizontal AND vertical separation are violated simultaneously.

    Attributes:
        horizontal_sep_nm: Minimum horizontal separation in nautical miles
        vertical_sep_ft: Minimum vertical separation in feet
    """

    def __init__(
        self,
        horizontal_sep_nm: float = DEFAULT_HORIZONTAL_SEP_NM,
        vertical_sep_ft: float = DEFAULT_VERTICAL_SEP_FT
    ):
        """
        Initialize conflict detector with separation minimums.

        Args:
            horizontal_sep_nm: Minimum horizontal separation (default 5.0 nm)
            vertical_sep_ft: Minimum vertical separation (default 1000 ft)
        """
        self.horizontal_sep_nm = horizontal_sep_nm
        self.vertical_sep_ft = vertical_sep_ft

    def check_separation(self, aircraft1, aircraft2) -> Dict[str, Any]:
        """
        Check separation between two aircraft.

        Calculates both horizontal (great circle distance) and vertical
        (altitude difference) separation. A conflict exists when BOTH
        are below minimums.

        Args:
            aircraft1: First Aircraft object
            aircraft2: Second Aircraft object

        Returns:
            Dictionary with separation details:
            - is_conflict: True if both horizontal and vertical violated
            - horizontal_distance_nm: Horizontal separation in nm
            - vertical_distance_ft: Vertical separation in feet
            - separation_margin_nm: Margin below minimum (negative if violation)

        Example:
            >>> detector = ConflictDetector()
            >>> result = detector.check_separation(aircraft1, aircraft2)
            >>> if result['is_conflict']:
            ...     print(f"CONFLICT! Separation: {result['horizontal_distance_nm']:.1f} nm")
        """
        # Get aircraft states
        state1 = aircraft1.get_state()
        state2 = aircraft2.get_state()

        # Get positions
        pos1 = state1['position']
        pos2 = state2['position']

        # Calculate horizontal distance (great circle)
        horizontal_distance_nm = haversine_distance(
            pos1['lat'], pos1['lon'],
            pos2['lat'], pos2['lon']
        )

        # Calculate vertical distance
        vertical_distance_ft = abs(pos1['alt'] - pos2['alt'])

        # Check if conflict exists (BOTH horizontal AND vertical violated)
        horizontal_violated = horizontal_distance_nm < self.horizontal_sep_nm
        vertical_violated = vertical_distance_ft < self.vertical_sep_ft
        is_conflict = horizontal_violated and vertical_violated

        # Calculate separation margin (negative means violation)
        separation_margin_nm = horizontal_distance_nm - self.horizontal_sep_nm

        return {
            'is_conflict': is_conflict,
            'horizontal_distance_nm': horizontal_distance_nm,
            'vertical_distance_ft': vertical_distance_ft,
            'separation_margin_nm': separation_margin_nm,
            'horizontal_violated': horizontal_violated,
            'vertical_violated': vertical_violated
        }

    def _classify_severity(self, horizontal_distance_nm: float) -> str:
        """
        Classify conflict severity based on horizontal distance.

        Args:
            horizontal_distance_nm: Horizontal separation in nautical miles

        Returns:
            Severity level: "CRITICAL", "WARNING", or "NEAR"
        """
        if horizontal_distance_nm < 3.0:
            return "CRITICAL"
        elif horizontal_distance_nm < 5.0:
            return "WARNING"
        elif horizontal_distance_nm < 7.0:
            return "NEAR"
        else:
            return "SAFE"

    def detect_all_conflicts(
        self,
        aircraft_list: List,
        simulation_time: float
    ) -> List[Dict[str, Any]]:
        """
        Detect all conflicts in a list of aircraft.

        Performs pairwise comparison of all aircraft (O(n²)). Skips pairs
        where either aircraft is TAXI or LANDED (not airborne).

        Args:
            aircraft_list: List of Aircraft objects
            simulation_time: Current simulation time in seconds

        Returns:
            List of conflict dictionaries, each containing:
            - aircraft1_id: First aircraft ID
            - aircraft2_id: Second aircraft ID
            - horizontal_distance_nm: Horizontal separation
            - vertical_distance_ft: Vertical separation
            - severity: Conflict severity level
            - timestamp: When conflict was detected

        Example:
            >>> detector = ConflictDetector()
            >>> conflicts = detector.detect_all_conflicts(aircraft_list, 100.0)
            >>> print(f"Detected {len(conflicts)} conflicts")
        """
        conflicts = []

        # Check all pairs of aircraft
        for i in range(len(aircraft_list)):
            for j in range(i + 1, len(aircraft_list)):
                aircraft1 = aircraft_list[i]
                aircraft2 = aircraft_list[j]

                # Skip if either aircraft is not airborne
                state1 = aircraft1.get_state()
                state2 = aircraft2.get_state()

                if state1['status'] in ['TAXI', 'LANDED']:
                    continue
                if state2['status'] in ['TAXI', 'LANDED']:
                    continue

                # Check separation
                sep_result = self.check_separation(aircraft1, aircraft2)

                if sep_result['is_conflict']:
                    # Classify severity
                    severity = self._classify_severity(
                        sep_result['horizontal_distance_nm']
                    )

                    # Create conflict record
                    conflict = {
                        'aircraft1_id': state1['id'],
                        'aircraft2_id': state2['id'],
                        'horizontal_distance_nm': sep_result['horizontal_distance_nm'],
                        'vertical_distance_ft': sep_result['vertical_distance_ft'],
                        'severity': severity,
                        'timestamp': simulation_time,
                        'separation_margin_nm': sep_result['separation_margin_nm']
                    }

                    conflicts.append(conflict)

        return conflicts

    def __repr__(self) -> str:
        """String representation of conflict detector."""
        return (
            f"ConflictDetector(horizontal_sep={self.horizontal_sep_nm}nm, "
            f"vertical_sep={self.vertical_sep_ft}ft)"
        )


class ConflictTracker:
    """
    Tracks conflicts over time and maintains history.

    Records when conflicts start, how long they last, and when they resolve.
    Provides statistics for analysis and AI training.

    Attributes:
        active_conflicts: Dictionary of currently active conflicts
        conflict_history: List of all conflicts (including resolved)
        conflict_id_counter: Counter for assigning unique IDs
    """

    def __init__(self):
        """Initialize conflict tracker."""
        # Active conflicts: key = (aircraft1_id, aircraft2_id), value = conflict data
        self.active_conflicts: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Complete conflict history
        self.conflict_history: List[Dict[str, Any]] = []

        # Counter for unique conflict IDs
        self.conflict_id_counter = 0

    def _get_conflict_key(self, aircraft1_id: str, aircraft2_id: str) -> Tuple[str, str]:
        """
        Get normalized conflict key (sorted order).

        Args:
            aircraft1_id: First aircraft ID
            aircraft2_id: Second aircraft ID

        Returns:
            Tuple with IDs in sorted order
        """
        return tuple(sorted([aircraft1_id, aircraft2_id]))

    def update(
        self,
        simulation_time: float,
        conflicts: List[Dict[str, Any]]
    ) -> None:
        """
        Update conflict tracking with current detections.

        Tracks new conflicts, updates ongoing conflicts, and detects
        when conflicts resolve.

        Args:
            simulation_time: Current simulation time in seconds
            conflicts: List of conflict dictionaries from detector

        Example:
            >>> tracker = ConflictTracker()
            >>> conflicts = detector.detect_all_conflicts(aircraft_list, 100.0)
            >>> tracker.update(100.0, conflicts)
        """
        # Create set of currently detected conflict pairs
        current_conflicts = set()

        for conflict in conflicts:
            key = self._get_conflict_key(
                conflict['aircraft1_id'],
                conflict['aircraft2_id']
            )
            current_conflicts.add(key)

            # Check if this is a new conflict
            if key not in self.active_conflicts:
                # New conflict detected
                conflict_id = f"CONFLICT_{self.conflict_id_counter:04d}"
                self.conflict_id_counter += 1

                conflict_record = {
                    'conflict_id': conflict_id,
                    'aircraft1_id': conflict['aircraft1_id'],
                    'aircraft2_id': conflict['aircraft2_id'],
                    'start_time': simulation_time,
                    'end_time': None,
                    'duration': None,
                    'min_horizontal_distance_nm': conflict['horizontal_distance_nm'],
                    'min_vertical_distance_ft': conflict['vertical_distance_ft'],
                    'max_severity': conflict['severity'],
                    'updates': [conflict]
                }

                self.active_conflicts[key] = conflict_record

            else:
                # Update existing conflict
                conflict_record = self.active_conflicts[key]
                conflict_record['updates'].append(conflict)

                # Update minimum distances
                if conflict['horizontal_distance_nm'] < conflict_record['min_horizontal_distance_nm']:
                    conflict_record['min_horizontal_distance_nm'] = conflict['horizontal_distance_nm']

                if conflict['vertical_distance_ft'] < conflict_record['min_vertical_distance_ft']:
                    conflict_record['min_vertical_distance_ft'] = conflict['vertical_distance_ft']

                # Update max severity
                severity_order = {'CRITICAL': 3, 'WARNING': 2, 'NEAR': 1, 'SAFE': 0}
                if severity_order[conflict['severity']] > severity_order[conflict_record['max_severity']]:
                    conflict_record['max_severity'] = conflict['severity']

        # Check for resolved conflicts
        resolved_keys = []
        for key, conflict_record in self.active_conflicts.items():
            if key not in current_conflicts:
                # Conflict has resolved
                conflict_record['end_time'] = simulation_time
                conflict_record['duration'] = simulation_time - conflict_record['start_time']

                # Move to history
                self.conflict_history.append(conflict_record)
                resolved_keys.append(key)

        # Remove resolved conflicts from active list
        for key in resolved_keys:
            del self.active_conflicts[key]

    def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active conflicts.

        Returns:
            List of active conflict dictionaries
        """
        return list(self.active_conflicts.values())

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get conflict statistics.

        Returns:
            Dictionary with comprehensive statistics:
            - total_conflicts: Total unique conflicts detected
            - active_conflicts: Currently ongoing conflicts
            - resolved_conflicts: Conflicts that have resolved
            - average_duration_seconds: Mean conflict duration
            - max_duration_seconds: Longest conflict duration
            - min_duration_seconds: Shortest conflict duration
            - severity_breakdown: Count by severity level
            - closest_approach_nm: Minimum separation observed
            - total_conflict_time_seconds: Sum of all conflict durations

        Example:
            >>> tracker = ConflictTracker()
            >>> stats = tracker.get_statistics()
            >>> print(f"Total conflicts: {stats['total_conflicts']}")
        """
        total_conflicts = len(self.conflict_history) + len(self.active_conflicts)
        active_count = len(self.active_conflicts)
        resolved_count = len(self.conflict_history)

        # Calculate duration statistics (only for resolved conflicts)
        durations = [c['duration'] for c in self.conflict_history if c['duration'] is not None]

        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            total_conflict_time = sum(durations)
        else:
            avg_duration = 0.0
            max_duration = 0.0
            min_duration = 0.0
            total_conflict_time = 0.0

        # Severity breakdown
        severity_counts = {'CRITICAL': 0, 'WARNING': 0, 'NEAR': 0}
        for conflict in self.conflict_history:
            severity_counts[conflict['max_severity']] = severity_counts.get(conflict['max_severity'], 0) + 1
        for conflict in self.active_conflicts.values():
            severity_counts[conflict['max_severity']] = severity_counts.get(conflict['max_severity'], 0) + 1

        # Find closest approach
        all_conflicts = list(self.conflict_history) + list(self.active_conflicts.values())
        if all_conflicts:
            closest_approach = min(c['min_horizontal_distance_nm'] for c in all_conflicts)
        else:
            closest_approach = None

        return {
            'total_conflicts': total_conflicts,
            'active_conflicts': active_count,
            'resolved_conflicts': resolved_count,
            'average_duration_seconds': avg_duration,
            'max_duration_seconds': max_duration,
            'min_duration_seconds': min_duration,
            'severity_breakdown': severity_counts,
            'closest_approach_nm': closest_approach,
            'total_conflict_time_seconds': total_conflict_time
        }

    def __repr__(self) -> str:
        """String representation of conflict tracker."""
        stats = self.get_statistics()
        return (
            f"ConflictTracker(active={stats['active_conflicts']}, "
            f"resolved={stats['resolved_conflicts']}, "
            f"total={stats['total_conflicts']})"
        )


if __name__ == "__main__":
    """Test conflict detection when module is executed directly."""
    print("Testing ConflictDetector...")
    print()

    detector = ConflictDetector()
    print(f"Detector: {detector}")
    print()

    tracker = ConflictTracker()
    print(f"Tracker: {tracker}")
    print()

    print("ConflictDetector test complete!")
