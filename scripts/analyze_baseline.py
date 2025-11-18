"""
Baseline analysis and visualization for DeepSky ATC.

Loads baseline simulation results and generates comparison visualizations
for blog posts, LinkedIn, and documentation.

Usage:
    python scripts/analyze_baseline.py [--input data/baseline/]
    python scripts/analyze_baseline.py --input data/baseline_short_haul/
"""

import sys
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze DeepSky ATC baseline simulation results'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/baseline',
        help='Input directory containing baseline results (default: data/baseline)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='docs/images',
        help='Output directory for visualizations (default: docs/images)'
    )
    return parser.parse_args()


def validate_input_directory(input_dir):
    """
    Validate that input directory exists and contains required files.

    Args:
        input_dir: Path to input directory

    Returns:
        True if valid, False otherwise
    """
    input_path = project_root / input_dir

    if not input_path.exists():
        print(f"✗ ERROR: Input directory does not exist: {input_path}")
        return False

    if not input_path.is_dir():
        print(f"✗ ERROR: Input path is not a directory: {input_path}")
        return False

    # Check for scenario files
    scenario_files = list(input_path.glob("scenario_*.json"))
    if len(scenario_files) == 0:
        print(f"✗ ERROR: No scenario files found in: {input_path}")
        print(f"  Expected files like: scenario_A_full_staffing.json")
        return False

    print(f"✓ Found {len(scenario_files)} scenario files in {input_path}")
    return True


def load_baseline_results(input_dir):
    """
    Load all baseline scenario results from input directory.

    Args:
        input_dir: Path to directory containing scenario JSON files

    Returns:
        List of result dictionaries
    """
    baseline_dir = project_root / input_dir

    print()
    print(f"Loading baseline results from: {baseline_dir}")
    print()

    # Find all scenario files
    scenario_files = sorted(baseline_dir.glob("scenario_*.json"))

    results = []
    for filepath in scenario_files:
        print(f"  Loading: {filepath.name}")

        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"    ⚠ WARNING: Failed to load {filepath.name}: {e}")
            continue

    print()
    print(f"✓ Loaded {len(results)} scenario results")
    print()

    return results


def create_conflicts_bar_chart(results, output_dir):
    """Create bar chart comparing conflicts per scenario."""
    print("Generating conflicts bar chart...")

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    # Extract data
    scenarios = [r['scenario_name'] for r in results]
    conflicts_per_hour = [r['results']['performance_metrics']['conflict_metrics']['conflicts_per_hour']
                          for r in results]

    # Color code: green for AI-only, red for shutdown, others gray
    colors = []
    for r in results:
        if r['scenario_id'] == 'E':  # AI-only
            colors.append('#2ecc71')  # Green
        elif r['scenario_id'] == 'D':  # Shutdown
            colors.append('#e74c3c')  # Red
        else:
            colors.append('#95a5a6')  # Gray

    # Create bar chart
    bars = plt.bar(scenarios, conflicts_per_hour, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xlabel('Staffing Scenario', fontsize=13, fontweight='bold')
    plt.ylabel('Conflicts Per Hour', fontsize=13, fontweight='bold')
    plt.title('Air Traffic Conflicts by Staffing Scenario\nDeepSky ATC Baseline Comparison',
             fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    # Save
    output_path = output_dir / "baseline_comparison_conflicts.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def create_safety_line_chart(results, output_dir):
    """Create line chart showing safety score by scenario."""
    print("Generating safety score line chart...")

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    # Extract data
    scenarios = [r['scenario_name'] for r in results]
    safety_scores = [r['results']['performance_metrics']['safety_score'] for r in results]

    # Create line chart with markers
    plt.plot(scenarios, safety_scores, marker='o', linewidth=3, markersize=12,
            color='#3498db', markerfacecolor='#e74c3c', markeredgecolor='black',
            markeredgewidth=2)

    # Add value labels
    for i, (scenario, score) in enumerate(zip(scenarios, safety_scores)):
        plt.text(i, score + 1.5, f'{score:.1f}', ha='center', fontsize=11, fontweight='bold')

    # Add target zones
    plt.axhspan(90, 100, alpha=0.1, color='green', label='Excellent (90-100)')
    plt.axhspan(75, 90, alpha=0.1, color='yellow', label='Good (75-90)')
    plt.axhspan(60, 75, alpha=0.1, color='orange', label='Marginal (60-75)')
    plt.axhspan(0, 60, alpha=0.1, color='red', label='Poor (<60)')

    plt.xlabel('Staffing Scenario', fontsize=13, fontweight='bold')
    plt.ylabel('Safety Score (0-100)', fontsize=13, fontweight='bold')
    plt.title('Safety Score by Staffing Scenario\nDeepSky ATC Baseline Comparison',
             fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 105)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()

    # Save
    output_path = output_dir / "baseline_comparison_safety.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def create_delay_box_plot(results, output_dir):
    """Create box plot for delay distributions by scenario."""
    print("Generating delay box plot...")

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    # Extract data - we'll use summary statistics to simulate distribution
    data_for_plot = []
    labels = []

    for result in results:
        otp = result['results']['performance_metrics']['on_time_performance']

        # Get delay statistics
        avg_delay = otp['average_delay_minutes']
        median_delay = otp['median_delay_minutes']

        # Simulate distribution from percentiles (rough approximation)
        # In a real scenario, we'd have the full delay data
        delays = [
            median_delay,  # p50
            otp['delay_percentiles'].get('p75', avg_delay * 1.2),
            otp['delay_percentiles'].get('p90', avg_delay * 1.5),
            otp['delay_percentiles'].get('p95', avg_delay * 1.8),
            avg_delay
        ]

        data_for_plot.append(delays)
        labels.append(result['scenario_name'])

    # Create box plot
    bp = plt.boxplot(data_for_plot, labels=labels, patch_artist=True,
                    notch=True, showmeans=True)

    # Color boxes
    colors = []
    for r in results:
        if r['scenario_id'] == 'E':  # AI-only
            colors.append('#2ecc71')
        elif r['scenario_id'] == 'D':  # Shutdown
            colors.append('#e74c3c')
        else:
            colors.append('#95a5a6')

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xlabel('Staffing Scenario', fontsize=13, fontweight='bold')
    plt.ylabel('Delay (minutes)', fontsize=13, fontweight='bold')
    plt.title('Flight Delay Distribution by Staffing Scenario\nDeepSky ATC Baseline Comparison',
             fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save
    output_path = output_dir / "baseline_comparison_delays.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def create_metrics_table(results, output_dir):
    """Create visual metrics comparison table."""
    print("Generating metrics comparison table...")

    # Set style
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Prepare table data
    columns = ['Scenario', 'Safety\nScore', 'Conflicts\n/Hour', 'Avg Delay\n(min)',
              'On-Time\n%', 'Flights\nCompleted']
    data = []

    for result in results:
        pm = result['results']['performance_metrics']
        row = [
            result['scenario_name'],
            f"{pm['safety_score']:.1f}",
            f"{pm['conflict_metrics']['conflicts_per_hour']:.2f}",
            f"{pm['on_time_performance']['average_delay_minutes']:.1f}",
            f"{pm['on_time_performance']['on_time_percentage']:.1f}%",
            f"{pm['on_time_performance']['total_flights_completed']}"
        ]
        data.append(row)

    # Create table
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(len(data)):
        # Determine color based on scenario
        if results[i]['scenario_id'] == 'E':  # AI-only
            color = '#d5f4e6'  # Light green
        elif results[i]['scenario_id'] == 'D':  # Shutdown
            color = '#fadbd8'  # Light red
        else:
            color = '#ecf0f1'  # Light gray

        for j in range(len(columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(color)

            # Bold scenario name
            if j == 0:
                cell.set_text_props(weight='bold')

    plt.title('DeepSky ATC Baseline Comparison - Key Metrics\n',
             fontsize=16, fontweight='bold', pad=20)

    # Save
    output_path = output_dir / "baseline_comparison_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def print_key_findings(results):
    """Print key findings from baseline analysis."""
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Find specific scenarios
    full_result = next((r for r in results if r['scenario_id'] == 'A'), None)
    shutdown_result = next((r for r in results if r['scenario_id'] == 'D'), None)
    ai_result = next((r for r in results if r['scenario_id'] == 'E'), None)

    if not full_result or not shutdown_result:
        print("⚠ Missing required scenarios for comparison")
        return

    full_pm = full_result['results']['performance_metrics']
    shutdown_pm = shutdown_result['results']['performance_metrics']

    print("1. WORST SCENARIO:")
    # Find worst by safety score
    worst = min(results, key=lambda r: r['results']['performance_metrics']['safety_score'])
    print(f"   {worst['scenario_name']} (Scenario {worst['scenario_id']})")
    print(f"   Safety Score: {worst['results']['performance_metrics']['safety_score']:.1f}")
    print(f"   Conflicts/Hour: {worst['results']['performance_metrics']['conflict_metrics']['conflicts_per_hour']:.2f}")
    print()

    print("2. GOVERNMENT SHUTDOWN DEGRADATION vs FULL STAFFING:")

    # Safety
    safety_drop = full_pm['safety_score'] - shutdown_pm['safety_score']
    safety_pct = (safety_drop / full_pm['safety_score']) * 100
    print(f"   Safety Score: {full_pm['safety_score']:.1f} → {shutdown_pm['safety_score']:.1f} "
          f"({safety_drop:+.1f} points, {safety_pct:.0f}% degradation)")

    # Conflicts
    full_conflicts = full_pm['conflict_metrics']['conflicts_per_hour']
    shutdown_conflicts = shutdown_pm['conflict_metrics']['conflicts_per_hour']

    if full_conflicts > 0:
        conflict_increase = ((shutdown_conflicts - full_conflicts) / full_conflicts) * 100
        print(f"   Conflicts/Hour: {full_conflicts:.2f} → {shutdown_conflicts:.2f} "
              f"({conflict_increase:+.0f}% increase)")
    else:
        print(f"   Conflicts/Hour: {full_conflicts:.2f} → {shutdown_conflicts:.2f}")

    # Delays
    full_delay = full_pm['on_time_performance']['average_delay_minutes']
    shutdown_delay = shutdown_pm['on_time_performance']['average_delay_minutes']

    if full_delay != 0:
        delay_increase = ((shutdown_delay - full_delay) / abs(full_delay)) * 100
        print(f"   Avg Delay: {full_delay:.1f} → {shutdown_delay:.1f} min "
              f"({delay_increase:+.0f}% increase)")
    else:
        print(f"   Avg Delay: {full_delay:.1f} → {shutdown_delay:.1f} min")

    print()

    print("3. TARGET METRICS FOR PHASE 3 AI CONTROLLER:")
    print()
    print(f"   MUST BEAT (Shutdown Scenario):")
    print(f"     • Safety Score: >{shutdown_pm['safety_score']:.1f}")
    print(f"     • Conflicts/Hour: <{shutdown_conflicts:.2f}")
    print(f"     • Avg Delay: <{shutdown_delay:.1f} min")
    print()
    print(f"   SHOULD MATCH (Full Staffing):")
    print(f"     • Safety Score: ~{full_pm['safety_score']:.1f}")
    print(f"     • Conflicts/Hour: ~{full_conflicts:.2f}")
    print(f"     • Avg Delay: ~{full_delay:.1f} min")
    print()
    print(f"   STRETCH GOAL:")
    print(f"     • Safety Score: >90.0")
    print(f"     • Conflicts/Hour: <{full_conflicts:.2f}")
    print(f"     • Better on-time performance than full staffing")
    print()

    if ai_result:
        ai_pm = ai_result['results']['performance_metrics']
        print("4. AI-ONLY BASELINE (No Human Controllers):")
        print(f"   Safety Score: {ai_pm['safety_score']:.1f}")
        print(f"   Conflicts/Hour: {ai_pm['conflict_metrics']['conflicts_per_hour']:.2f}")
        print(f"   Avg Delay: {ai_pm['on_time_performance']['average_delay_minutes']:.1f} min")
        print(f"   (This represents uncontrolled baseline - Phase 3 will add AI control)")
        print()


def main():
    """Analyze baseline results and generate visualizations."""
    print("\n")
    print("*" * 80)
    print("* DeepSky ATC - Baseline Analysis & Visualization")
    print("* Phase 2, Deliverable 2.3")
    print("*" * 80)
    print()

    # Parse command line arguments
    args = parse_arguments()

    # Show which directory is being analyzed
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print()

    # Validate input directory
    if not validate_input_directory(args.input):
        print()
        print("✗ ERROR: Invalid input directory")
        print()
        print("Usage:")
        print("  python scripts/analyze_baseline.py [--input data/baseline/]")
        print("  python scripts/analyze_baseline.py --input data/baseline_short_haul/")
        print()
        return 1

    print()

    # Load baseline results
    results = load_baseline_results(args.input)

    if len(results) == 0:
        print("✗ ERROR: No baseline results found!")
        print("   Run: python scripts/run_baseline.py")
        return 1

    # Create output directory
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("Generating visualizations...")
    print()

    try:
        create_conflicts_bar_chart(results, output_dir)
        create_safety_line_chart(results, output_dir)
        create_delay_box_plot(results, output_dir)
        create_metrics_table(results, output_dir)

        print()
        print("✓ All visualizations generated")

    except Exception as e:
        print(f"\n✗ ERROR generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print key findings
    print_key_findings(results)

    print("=" * 80)
    print("BASELINE ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Analyzed dataset:")
    print(f"  • Input: {args.input}")
    print()
    print("Outputs:")
    print(f"  • Visualizations: {output_dir}")
    print(f"  • Individual scenarios: {args.input}/scenario_*.json")
    print(f"  • Summary CSV: {args.input}/baseline_summary.csv")
    print()
    print("Next steps:")
    print("  • Review visualizations for blog posts / LinkedIn")
    print("  • Use target metrics to design Phase 3 AI controller")
    print("  • Reference shutdown degradation in problem statement")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
