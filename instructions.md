# Emperor Penguin Thermoregulation Simulation
## User Guide

This guide explains how to use the Emperor penguin thermoregulation simulation system to study how penguins maintain body temperature in harsh Antarctic conditions.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Running the Simulation](#running-the-simulation)
4. [Parameter Sweeps](#parameter-sweeps)
5. [Analyzing Results](#analyzing-results)
6. [Advanced Usage](#advanced-usage)
7. [Parameter Reference](#parameter-reference)

## Overview

This simulation system models Emperor penguin thermoregulation and group behavior in a dynamic environment. Each penguin is represented as a circle with:

- A body temperature that changes based on environmental factors
- Movement that responds to temperature state (cold penguins seek warmth, hot ones avoid it)
- Visualization showing temperature via color and heading via a yellow arrow

The system is composed of:

- **PenguinSimulation**: Core simulation logic
- **PenguinVisualizer**: Visualization and animation
- **Analysis Tools**: For analyzing parameter sweep results

## Setup

Save these Python files to your project directory:

1. `penguin_simulation_core.py` - The core simulation logic
2. `penguin_visualization.py` - The visualization system
3. `penguin_main.py` - The main script with examples
4. `penguin_analysis.py` - Analysis tools for parameter sweeps

Dependencies:
- NumPy
- Matplotlib
- SciPy (for analysis tools)
- Seaborn (for prettier plots in analysis)

## Running the Simulation

### Option 1: Using the main script

The simplest approach is to run the `penguin_main.py` script:

```bash
python penguin_main.py

## Parameter Reference

The simulation has many parameters you can adjust:

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `N` | Number of penguins | 10 | 1-100 |
| `R` | Penguin radius (m) | 0.15 | 0.1-0.2 |
| `hard_stop` | Minimum distance between penguins | 0.33 | 0.3-0.4 |
| `T_air` | Air temperature (°C) | -20.0 | -70.0 to -10.0 |
| `U` | Base wind speed (m/s) | 5.0 | 0.0-15.0 |
| `T_opt` | Optimal core temperature (°C) | 38.0 | 37.0-39.0 |
| `T_min` | Min temperature for display | 35.0 | - |
| `T_max` | Max temperature for display | 41.0 | - |
| `T_cold_death` | Fatal hypothermia temperature (°C) | 34.0 | 32.0-35.0 |
| `T_hot_death` | Fatal hyperthermia temperature (°C) | 42.0 | 41.0-43.0 |
| `k_air` | Heat loss to air coefficient | 0.005 | 0.001-0.01 |
| `k_wind` | Wind chill effect coefficient | 0.008 | 0.001-0.02 |
| `k_rad` | Heat radiation between penguins | 0.06 | 0.01-0.1 |
| `k_body` | Metabolic heat generation | 0.02 | 0.01-0.05 |
| `dt` | Time step | 0.01 | 0.001-0.05 |
| `m_neighbors` | Number of neighbors to consider | 7 | 3-10 |
| `heading_inertia` | Inertia factor for penguin heading (0-1) | 0.9 | 0.7-0.99 |
| `domain_radius` | Simulation domain radius | 5.0 | 2.0-10.0 |

### Realistic Values

For the most realistic Emperor penguin simulation, use these values:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `T_opt` | 38.0°C | Actual core temperature of Emperor penguins |
| `T_cold_death` | 34.0°C | Temperature at which severe hypothermia becomes fatal |
| `T_hot_death` | 42.0°C | Temperature at which severe hyperthermia becomes fatal |
| `T_air` | -40.0 to -60.0°C | Typical Antarctic winter temperature range |
| `U` | 5.0 to 15.0 m/s | Antarctic katabatic wind speeds |
| `k_air` | 0.005 | Heat loss coefficient accounting for thick feather insulation |
| `k_wind` | 0.008 | Wind chill effect reduced by feather and fat layers |
| `k_rad` | 0.06 | Radiative heat transfer between huddling penguins |
| `k_body` | 0.02 | Metabolic heat generation including shivering thermogenesis |
| `m_neighbors` | 7 | Typical number of neighboring penguins in a huddle |
| `heading_inertia` | 0.9 | Realistic value for smooth penguin movement |
| `R` | 0.15 | Approximates half the diameter of an adult Emperor penguin |