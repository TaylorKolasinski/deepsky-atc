# DeepSky ATC Staffing Scenario Comparison
Generated: 2025-11-18 13:45:27

## Overview

Comparison of air traffic control performance across different staffing scenarios, from full staffing to government shutdown conditions.

## Staffing Configurations

| Scenario | Controllers | Capacity | Description |
|----------|-------------|----------|-------------|
| **FULL** | 4 | 60 aircraft | Full Staffing - Optimal Operations |
| **NORMAL** | 3 | 36 aircraft | Normal Staffing - Typical Day |
| **REDUCED** | 2 | 24 aircraft | Reduced Staffing - Budget Cuts |
| **SHUTDOWN** | 1 | 15 aircraft | Government Shutdown - Minimal Staff |
| **NONE** | 0 | 0 aircraft | AI-Only - Zero Human Controllers |

## Performance Metrics Comparison

### Safety Score (0-100)

| Scenario | Safety Score | Grade |
|----------|--------------|-------|
| **FULL** | 20.6 | F |
| **NORMAL** | 20.0 | F |
| **REDUCED** | 20.5 | F |
| **SHUTDOWN** | 20.5 | F |
| **NONE** | 20.1 | F |

### Conflict Metrics

| Scenario | Total Conflicts | Conflicts/Hour | Critical | Warning | Near |
|----------|-----------------|----------------|----------|---------|------|
| **FULL** | 274 | 137.00 | 204 | 70 | 0 |
| **NORMAL** | 231 | 115.50 | 178 | 53 | 0 |
| **REDUCED** | 253 | 126.50 | 198 | 55 | 0 |
| **SHUTDOWN** | 251 | 125.50 | 184 | 67 | 0 |
| **NONE** | 253 | 126.50 | 198 | 55 | 0 |

### On-Time Performance

| Scenario | Flights Completed | On-Time % | Avg Delay (min) | Median Delay (min) |
|----------|-------------------|-----------|-----------------|--------------------|
| **FULL** | 4 | 0.0% | 19.5 | 19.1 |
| **NORMAL** | 4 | 0.0% | 20.1 | 20.0 |
| **REDUCED** | 4 | 0.0% | 23.0 | 23.6 |
| **SHUTDOWN** | 4 | 0.0% | 20.5 | 19.7 |
| **NONE** | 4 | 0.0% | 21.3 | 21.6 |

### Throughput

| Scenario | Flights/Hour | Peak Concurrent | Avg Concurrent |
|----------|--------------|-----------------|----------------|
| **FULL** | 2.00 | 48 | 17.94 |
| **NORMAL** | 2.00 | 47 | 17.19 |
| **REDUCED** | 2.00 | 48 | 16.89 |
| **SHUTDOWN** | 2.00 | 47 | 17.86 |
| **NONE** | 2.00 | 48 | 17.03 |

### Controller Workload

| Scenario | Avg Workload | Peak Workload | Time Overloaded (%) |
|----------|--------------|---------------|---------------------|
| **FULL** | 0.30x | 0.80x | 0.0% |
| **NORMAL** | 0.48x | 1.31x | 27.6% |
| **REDUCED** | 0.70x | 2.00x | 36.8% |
| **SHUTDOWN** | 1.19x | 3.13x | 40.9% |
| **NONE** | 0.00x | 0.00x | 0.0% |

## Key Insights

- **Government Shutdown Impact**: Conflicts increased by -8% compared to full staffing
- **Delay Impact**: Average delays increased by 5% under shutdown conditions
- **Safety Degradation**: Safety score dropped 0.1 points (from 20.6 to 20.5)

- **AI vs Shutdown**: AI safety score (20.1) vs shutdown (20.5)
- **AI Conflict Performance**: 126.50 conflicts/hour vs 125.50 under shutdown
