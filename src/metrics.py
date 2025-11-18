"""
Performance metrics system for DeepSky ATC.

Tracks and analyzes ATC system performance including safety (conflicts),
efficiency (delays, throughput), and provides comprehensive statistics
for comparing baseline vs AI controller performance.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime


class PerformanceMetrics:
    """
    Comprehensive performance metrics tracker for ATC simulation.

    Tracks conflicts, delays, throughput, and calculates safety scores.
    Provides baseline for AI controller comparison in Phase 3.

    Attributes:
        aircraft_count_history: List of (time, count) tuples
        conflict_history: List of conflicts over time
        completed_flights: List of completed flight records
        conflict_timesteps: Number of timesteps with conflicts
        total_timesteps: Total simulation timesteps
    """

    def __init__(self):
        """Initialize performance metrics tracker."""
        # Time series data
        self.aircraft_count_history: List[tuple] = []
        self.conflict_history: List[Dict] = []

        # Flight completion data
        self.completed_flights: List[Dict] = []

        # Conflict tracking
        self.conflict_timesteps = 0
        self.total_timesteps = 0
        self.peak_concurrent_aircraft = 0

        # Start time for rate calculations
        self.simulation_start_time = 0.0
        self.simulation_end_time = 0.0

    def update(
        self,
        simulation_time: float,
        aircraft_list: List,
        conflicts: List[Dict[str, Any]]
    ) -> None:
        """
        Update metrics with current simulation state.

        Called every timestep to track aircraft counts and conflicts.

        Args:
            simulation_time: Current simulation time in seconds
            aircraft_list: List of active Aircraft objects
            conflicts: List of current conflicts from detector

        Example:
            >>> metrics = PerformanceMetrics()
            >>> metrics.update(100.0, aircraft_list, conflicts)
        """
        # Track timesteps
        self.total_timesteps += 1
        self.simulation_end_time = simulation_time

        # Track aircraft count
        active_count = len(aircraft_list)
        self.aircraft_count_history.append((simulation_time, active_count))

        # Update peak
        if active_count > self.peak_concurrent_aircraft:
            self.peak_concurrent_aircraft = active_count

        # Track conflicts
        if len(conflicts) > 0:
            self.conflict_timesteps += 1

            # Record conflicts
            for conflict in conflicts:
                self.conflict_history.append({
                    'time': simulation_time,
                    'conflict': conflict
                })

    def record_flight_completion(self, aircraft) -> None:
        """
        Record completion metrics for a landed aircraft.

        Calculates delays, flight duration, and other performance metrics.

        Args:
            aircraft: Landed Aircraft object

        Example:
            >>> metrics = PerformanceMetrics()
            >>> metrics.record_flight_completion(aircraft)
        """
        state = aircraft.get_state()

        # Get route information
        route = aircraft.route

        # Calculate scheduled arrival time
        # Scheduled = departure + estimated flight time based on distance
        # Rough estimate: distance (nm) / average speed (kts) * 3600 (seconds)
        estimated_flight_time_hours = route.total_distance_nm / 450.0  # 450kts average
        estimated_flight_time_seconds = estimated_flight_time_hours * 3600
        scheduled_arrival_time = route.scheduled_departure_time + estimated_flight_time_seconds

        # Actual arrival time
        actual_arrival_time = state['time_elapsed']

        # Calculate delay (actual - scheduled)
        delay_seconds = actual_arrival_time - scheduled_arrival_time
        delay_minutes = delay_seconds / 60.0

        # Flight duration
        flight_duration_seconds = actual_arrival_time - route.actual_departure_time
        flight_duration_minutes = flight_duration_seconds / 60.0

        # Record completion
        flight_record = {
            'aircraft_id': state['id'],
            'route_id': route.route_id,
            'departure_icao': route.departure_icao,
            'arrival_icao': route.arrival_icao,
            'distance_nm': route.total_distance_nm,
            'scheduled_departure': route.scheduled_departure_time,
            'actual_departure': route.actual_departure_time,
            'departure_delay_minutes': (route.actual_departure_time - route.scheduled_departure_time) / 60.0,
            'scheduled_arrival': scheduled_arrival_time,
            'actual_arrival': actual_arrival_time,
            'arrival_delay_minutes': delay_minutes,
            'flight_duration_minutes': flight_duration_minutes
        }

        self.completed_flights.append(flight_record)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.

        Returns:
            Dictionary with complete performance metrics:
            - conflict_metrics: Conflict statistics
            - on_time_performance: Delay and punctuality metrics
            - throughput: Capacity and flow metrics
            - efficiency: System efficiency metrics
            - safety_score: Overall safety rating (0-100)

        Example:
            >>> metrics = PerformanceMetrics()
            >>> stats = metrics.get_summary_statistics()
            >>> print(f"Safety Score: {stats['safety_score']}")
        """
        # Calculate simulation hours
        simulation_duration_seconds = self.simulation_end_time - self.simulation_start_time
        simulation_hours = simulation_duration_seconds / 3600.0 if simulation_duration_seconds > 0 else 1.0

        # === CONFLICT METRICS ===
        unique_conflicts = self._count_unique_conflicts()
        conflicts_per_hour = unique_conflicts / simulation_hours if simulation_hours > 0 else 0

        # Calculate total time in conflict
        conflict_minutes = self.conflict_timesteps / 60.0  # Assuming 1-second timesteps

        # Average conflict duration
        conflict_durations = self._calculate_conflict_durations()
        avg_conflict_duration = np.mean(conflict_durations) if conflict_durations else 0.0

        # Severity breakdown
        severity_counts = self._count_conflicts_by_severity()

        conflict_metrics = {
            'total_conflicts': unique_conflicts,
            'conflicts_per_hour': conflicts_per_hour,
            'conflict_minutes': conflict_minutes,
            'avg_conflict_duration_seconds': avg_conflict_duration,
            'critical_conflicts': severity_counts.get('CRITICAL', 0),
            'warning_conflicts': severity_counts.get('WARNING', 0),
            'near_conflicts': severity_counts.get('NEAR', 0),
            'severity_breakdown': severity_counts
        }

        # === ON-TIME PERFORMANCE ===
        total_completed = len(self.completed_flights)

        if total_completed > 0:
            delays = [f['arrival_delay_minutes'] for f in self.completed_flights]

            # On-time: within 15 minutes of schedule
            on_time_count = sum(1 for d in delays if abs(d) <= 15)
            on_time_percentage = (on_time_count / total_completed) * 100

            # Early arrivals (negative delay)
            early_count = sum(1 for d in delays if d < -1)

            # Statistics
            avg_delay = np.mean(delays)
            median_delay = np.median(delays)

            # Percentiles
            percentiles = {
                'p50': np.percentile(delays, 50),
                'p75': np.percentile(delays, 75),
                'p90': np.percentile(delays, 90),
                'p95': np.percentile(delays, 95)
            }
        else:
            on_time_percentage = 0.0
            early_count = 0
            avg_delay = 0.0
            median_delay = 0.0
            percentiles = {'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0}

        on_time_performance = {
            'total_flights_completed': total_completed,
            'on_time_percentage': on_time_percentage,
            'average_delay_minutes': avg_delay,
            'median_delay_minutes': median_delay,
            'delay_percentiles': percentiles,
            'flights_early': early_count
        }

        # === THROUGHPUT ===
        flights_per_hour = total_completed / simulation_hours if simulation_hours > 0 else 0

        # Average concurrent aircraft
        if self.aircraft_count_history:
            counts = [count for _, count in self.aircraft_count_history]
            avg_concurrent = np.mean(counts)
        else:
            avg_concurrent = 0.0

        throughput = {
            'flights_per_hour': flights_per_hour,
            'peak_concurrent_aircraft': self.peak_concurrent_aircraft,
            'average_concurrent_aircraft': avg_concurrent
        }

        # === EFFICIENCY ===
        # Conflict-free percentage
        if self.total_timesteps > 0:
            conflict_free_percentage = ((self.total_timesteps - self.conflict_timesteps) / self.total_timesteps) * 100
        else:
            conflict_free_percentage = 100.0

        # Separation violations per 100 flights
        if total_completed > 0:
            violations_per_100 = (unique_conflicts / total_completed) * 100
        else:
            violations_per_100 = 0.0

        efficiency = {
            'conflict_free_percentage': conflict_free_percentage,
            'separation_violations_per_100_flights': violations_per_100
        }

        # === SAFETY SCORE (0-100) ===
        safety_score = self._calculate_safety_score(
            conflict_free_percentage,
            severity_counts.get('CRITICAL', 0),
            unique_conflicts
        )

        return {
            'simulation_duration_hours': simulation_hours,
            'conflict_metrics': conflict_metrics,
            'on_time_performance': on_time_performance,
            'throughput': throughput,
            'efficiency': efficiency,
            'safety_score': safety_score
        }

    def _count_unique_conflicts(self) -> int:
        """
        Count unique conflict pairs.

        Returns:
            Number of unique aircraft pairs that conflicted
        """
        unique_pairs = set()

        for record in self.conflict_history:
            conflict = record['conflict']
            pair = tuple(sorted([conflict['aircraft1_id'], conflict['aircraft2_id']]))
            unique_pairs.add(pair)

        return len(unique_pairs)

    def _calculate_conflict_durations(self) -> List[float]:
        """
        Calculate duration of each unique conflict.

        Returns:
            List of conflict durations in seconds
        """
        # Group conflicts by aircraft pair
        conflict_groups = {}

        for record in self.conflict_history:
            conflict = record['conflict']
            pair = tuple(sorted([conflict['aircraft1_id'], conflict['aircraft2_id']]))
            time = record['time']

            if pair not in conflict_groups:
                conflict_groups[pair] = {'start': time, 'end': time}
            else:
                conflict_groups[pair]['end'] = time

        # Calculate durations
        durations = []
        for pair, times in conflict_groups.items():
            duration = times['end'] - times['start'] + 1  # +1 for inclusive
            durations.append(duration)

        return durations

    def _count_conflicts_by_severity(self) -> Dict[str, int]:
        """
        Count conflicts by severity level.

        Returns:
            Dictionary with severity counts
        """
        severity_counts = {}

        # Track unique pairs with their max severity
        pair_severities = {}

        for record in self.conflict_history:
            conflict = record['conflict']
            pair = tuple(sorted([conflict['aircraft1_id'], conflict['aircraft2_id']]))
            severity = conflict['severity']

            if pair not in pair_severities:
                pair_severities[pair] = severity
            else:
                # Keep the most severe
                severity_order = {'CRITICAL': 3, 'WARNING': 2, 'NEAR': 1}
                if severity_order.get(severity, 0) > severity_order.get(pair_severities[pair], 0):
                    pair_severities[pair] = severity

        # Count by severity
        for severity in pair_severities.values():
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return severity_counts

    def _calculate_safety_score(
        self,
        conflict_free_pct: float,
        critical_conflicts: int,
        total_conflicts: int
    ) -> float:
        """
        Calculate overall safety score (0-100).

        Weighted combination:
        - 60% conflict-free time
        - 30% no critical conflicts
        - 10% minimal total conflicts

        Args:
            conflict_free_pct: Percentage of time with no conflicts
            critical_conflicts: Number of critical severity conflicts
            total_conflicts: Total number of conflicts

        Returns:
            Safety score from 0 to 100
        """
        # Component 1: Conflict-free time (60%)
        component1 = (conflict_free_pct / 100.0) * 60

        # Component 2: No critical conflicts (30%)
        if critical_conflicts == 0:
            component2 = 30.0
        elif critical_conflicts <= 2:
            component2 = 15.0
        else:
            component2 = 0.0

        # Component 3: Minimal conflicts (10%)
        if total_conflicts == 0:
            component3 = 10.0
        elif total_conflicts <= 5:
            component3 = 5.0
        else:
            component3 = max(0, 10 - total_conflicts)

        safety_score = component1 + component2 + component3
        return min(100.0, max(0.0, safety_score))

    def export_to_json(self, filepath: str) -> None:
        """
        Export all metrics to JSON file.

        Args:
            filepath: Path to output JSON file

        Example:
            >>> metrics = PerformanceMetrics()
            >>> metrics.export_to_json('data/metrics/baseline.json')
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get summary statistics
        summary = self.get_summary_statistics()

        # Add detailed flight data
        export_data = {
            'summary': summary,
            'completed_flights': self.completed_flights,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_timesteps': self.total_timesteps,
                'simulation_duration_seconds': self.simulation_end_time - self.simulation_start_time
            }
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Metrics exported to {output_path}")

    def compare_to_baseline(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current metrics to baseline.

        Args:
            baseline_metrics: Dictionary with baseline summary statistics

        Returns:
            Dictionary with improvement percentages (positive = better)

        Example:
            >>> current_metrics = metrics.get_summary_statistics()
            >>> improvements = metrics.compare_to_baseline(baseline_stats)
            >>> print(f"Conflict reduction: {improvements['conflict_reduction_percent']:.1f}%")
        """
        current = self.get_summary_statistics()

        # Calculate improvements
        baseline_conflicts = baseline_metrics['conflict_metrics']['total_conflicts']
        current_conflicts = current['conflict_metrics']['total_conflicts']

        if baseline_conflicts > 0:
            conflict_reduction_pct = ((baseline_conflicts - current_conflicts) / baseline_conflicts) * 100
        else:
            conflict_reduction_pct = 0.0

        # Delay improvement
        baseline_delay = baseline_metrics['on_time_performance']['average_delay_minutes']
        current_delay = current['on_time_performance']['average_delay_minutes']

        if baseline_delay != 0:
            delay_improvement_pct = ((baseline_delay - current_delay) / abs(baseline_delay)) * 100
        else:
            delay_improvement_pct = 0.0

        # Safety score improvement
        baseline_safety = baseline_metrics['safety_score']
        current_safety = current['safety_score']
        safety_improvement = current_safety - baseline_safety

        return {
            'conflict_reduction_percent': conflict_reduction_pct,
            'delay_improvement_percent': delay_improvement_pct,
            'safety_score_improvement': safety_improvement,
            'baseline_safety_score': baseline_safety,
            'current_safety_score': current_safety
        }

    def generate_metrics_report(self, output_path: str) -> None:
        """
        Generate markdown report with metrics.

        Creates a formatted markdown file ready for blog post inclusion.

        Args:
            output_path: Path to output markdown file

        Example:
            >>> metrics = PerformanceMetrics()
            >>> metrics.generate_metrics_report('docs/metrics_report.md')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_summary_statistics()

        # Build markdown report
        report = []
        report.append("# DeepSky ATC Performance Metrics Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Simulation Duration: {stats['simulation_duration_hours']:.2f} hours\n")
        report.append("\n## Safety Score\n")
        report.append(f"**Overall Safety Score: {stats['safety_score']:.1f}/100**\n")
        report.append("\n## Conflict Metrics\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        for key, value in stats['conflict_metrics'].items():
            if key != 'severity_breakdown':
                report.append(f"| {key.replace('_', ' ').title()} | {value:.2f} |\n")

        report.append("\n## On-Time Performance\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        otp = stats['on_time_performance']
        report.append(f"| Total Flights Completed | {otp['total_flights_completed']} |\n")
        report.append(f"| On-Time Percentage | {otp['on_time_percentage']:.1f}% |\n")
        report.append(f"| Average Delay | {otp['average_delay_minutes']:.1f} min |\n")
        report.append(f"| Median Delay | {otp['median_delay_minutes']:.1f} min |\n")

        report.append("\n## Throughput\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        for key, value in stats['throughput'].items():
            report.append(f"| {key.replace('_', ' ').title()} | {value:.2f} |\n")

        report.append("\n## Efficiency\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        for key, value in stats['efficiency'].items():
            report.append(f"| {key.replace('_', ' ').title()} | {value:.2f} |\n")

        # Write report
        with open(output_path, 'w') as f:
            f.writelines(report)

        print(f"✓ Metrics report generated: {output_path}")

    def __repr__(self) -> str:
        """String representation of metrics tracker."""
        return (
            f"PerformanceMetrics(flights_completed={len(self.completed_flights)}, "
            f"conflicts={self._count_unique_conflicts()}, "
            f"timesteps={self.total_timesteps})"
        )


if __name__ == "__main__":
    """Test metrics when module is executed directly."""
    print("Testing PerformanceMetrics...")
    print()

    metrics = PerformanceMetrics()
    print(f"Metrics: {metrics}")
    print()

    print("PerformanceMetrics test complete!")
