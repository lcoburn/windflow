import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
from scipy.interpolate import griddata
import seaborn as sns

def load_simulation_results(result_dir):
    """
    Load results from a simulation directory
    
    Args:
        result_dir: Path to the simulation results directory
        
    Returns:
        Dictionary containing metadata, parameters, and history
    """
    # Load metadata and parameters
    with open(f"{result_dir}/metadata.json", 'r') as f:
        data = json.load(f)
    
    # Load position and temperature data if needed
    positions = np.load(f"{result_dir}/positions.npy")
    temperatures = np.load(f"{result_dir}/temperatures.npy")
    headings = np.load(f"{result_dir}/headings.npy")
    
    # Add raw data to the results
    results = {
        "metadata": data,
        "params": data["params"],
        "history": data["history"],
        "positions": positions,
        "temperatures": temperatures,
        "headings": headings
    }
    
    return results

def find_simulation_results(base_dir='.', pattern='penguin_sim_*'):
    """
    Find all simulation result directories in the given base directory
    
    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for simulation directories
        
    Returns:
        List of paths to result directories
    """
    return glob.glob(os.path.join(base_dir, pattern))

def extract_parameter_value(result_dir, param_name):
    """
    Extract a parameter value from the result directory name
    
    Args:
        result_dir: Path to result directory
        param_name: Name of the parameter to extract
        
    Returns:
        Parameter value or None if not found
    """
    dir_name = os.path.basename(result_dir)
    
    # Look for param_name=value pattern
    param_pattern = f"{param_name}="
    if param_pattern in dir_name:
        parts = dir_name.split('_')
        for part in parts:
            if part.startswith(param_pattern):
                value_str = part[len(param_pattern):]
                try:
                    # Try to convert to number
                    if '.' in value_str:
                        return float(value_str)
                    else:
                        return int(value_str)
                except ValueError:
                    return value_str
    
    return None

def create_survival_time_heatmap(result_dirs, param_x, param_y):
    """
    Create a heatmap showing survival time as a function of two parameters
    
    Args:
        result_dirs: List of simulation result directories
        param_x: Name of parameter for x-axis
        param_y: Name of parameter for y-axis
        
    Returns:
        Figure and axes objects
    """
    # Extract parameter values and survival times
    x_values = []
    y_values = []
    survival_times = []
    
    for result_dir in result_dirs:
        # Load results
        results = load_simulation_results(result_dir)
        
        # Get parameter values
        x_value = results["params"][param_x]
        y_value = results["params"][param_y]
        
        # Calculate survival time (time until the first penguin dies)
        active_counts = results["history"]["active_counts"]
        initial_count = active_counts[0]
        survival_time = results["history"]["times"][-1]  # Default to max time
        
        for i, count in enumerate(active_counts):
            if count < initial_count:
                survival_time = results["history"]["times"][i]
                break
        
        x_values.append(x_value)
        y_values.append(y_value)
        survival_times.append(survival_time)
    
    # Create a grid for heatmap
    x_unique = sorted(list(set(x_values)))
    y_unique = sorted(list(set(y_values)))
    X, Y = np.meshgrid(x_unique, y_unique)
    
    # Interpolate survival times onto grid
    if len(x_unique) > 1 and len(y_unique) > 1:
        Z = griddata((x_values, y_values), survival_times, (X, Y), method='linear')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Survival Time (s)')
        
        # Label axes
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f'Penguin Survival Time as Function of {param_x} and {param_y}')
        
        # Add text annotations with values
        for i, x in enumerate(x_unique):
            for j, y in enumerate(y_unique):
                survival = Z[j, i]
                if not np.isnan(survival):
                    ax.text(x, y, f'{survival:.1f}', 
                            ha='center', va='center', 
                            color='white' if survival < np.max(survival_times)/2 else 'black')
        
        return fig, ax
    else:
        # Not enough unique values for a heatmap
        # Create a line or bar chart instead
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(x_unique) > 1:
            # X varies, Y is constant
            ax.plot(x_values, survival_times, 'o-')
            ax.set_xlabel(param_x)
            ax.set_ylabel('Survival Time (s)')
            ax.set_title(f'Penguin Survival Time vs {param_x} (Fixed {param_y}={y_unique[0]})')
        else:
            # Y varies, X is constant
            ax.plot(y_values, survival_times, 'o-')
            ax.set_xlabel(param_y)
            ax.set_ylabel('Survival Time (s)')
            ax.set_title(f'Penguin Survival Time vs {param_y} (Fixed {param_x}={x_unique[0]})')
        
        ax.grid(True)
        return fig, ax

def create_temperature_profile_plot(result_dirs, param_to_compare):
    """
    Create a plot showing temperature profiles over time for different parameter values
    
    Args:
        result_dirs: List of simulation result directories
        param_to_compare: Parameter to compare across simulations
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort result_dirs by parameter value
    sorted_dirs = sorted(
        result_dirs, 
        key=lambda d: extract_parameter_value(d, param_to_compare) or 0
    )
    
    # Create color palette
    num_lines = len(sorted_dirs)
    palette = sns.color_palette("viridis", num_lines)
    
    # Plot temperature profiles
    for i, result_dir in enumerate(sorted_dirs):
        # Load results
        results = load_simulation_results(result_dir)
        
        # Get parameter value for label
        param_value = results["params"][param_to_compare]
        
        # Plot temperature profile
        times = results["history"]["times"]
        temps = results["history"]["mean_temperatures"]
        
        # Filter out None values
        valid_indices = [i for i, t in enumerate(temps) if t is not None]
        valid_times = [times[i] for i in valid_indices]
        valid_temps = [temps[i] for i in valid_indices]
        
        ax.plot(valid_times, valid_temps, '-', 
                color=palette[i], 
                linewidth=2, 
                label=f'{param_to_compare}={param_value}')
    
    # Add reference lines
    first_result = load_simulation_results(sorted_dirs[0])
    ax.axhline(y=first_result["params"]["T_opt"], color='k', linestyle='--', alpha=0.5, label='Optimal Temp')
    ax.axhline(y=first_result["params"]["T_cold_death"], color='b', linestyle='--', alpha=0.5, label='Cold Death')
    ax.axhline(y=first_result["params"]["T_hot_death"], color='r', linestyle='--', alpha=0.5, label='Heat Death')
    
    # Add labels and legend
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average Body Temperature (°C)')
    ax.set_title(f'Penguin Temperature Profiles for Different {param_to_compare} Values')
    ax.grid(True)
    ax.legend(loc='best')
    
    return fig, ax

def create_wind_chill_visualization(result_dir):
    """
    Create a visualization of wind chill effect on penguins
    
    Args:
        result_dir: Path to a simulation result directory
        
    Returns:
        Figure and axes objects
    """
    # Load results
    results = load_simulation_results(result_dir)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Time values
    times = results["history"]["times"]
    
    # Plot temperature and wind speed
    ax1 = axs[0]
    ax1.plot(times, results["history"]["mean_temperatures"], 'r-', linewidth=2, label='Body Temp')
    ax1.set_ylabel('Temperature (°C)')
    
    # Calculate wind chill
    air_temps = results["history"]["air_temperatures"]
    wind_speeds = results["history"]["wind_speeds"]
    wind_chill = [t - (1.59 * 0.112 * (3.6 * w)) for t, w in zip(air_temps, wind_speeds)]
    
    ax1.plot(times, air_temps, 'b-', linewidth=2, label='Air Temp')
    ax1.plot(times, wind_chill, 'g--', linewidth=2, label='Wind Chill')
    
    # Add reference lines
    ax1.axhline(y=results["params"]["T_opt"], color='k', linestyle='--', alpha=0.5, label='Optimal Temp')
    ax1.axhline(y=results["params"]["T_cold_death"], color='b', linestyle='--', alpha=0.5, label='Cold Death')
    
    ax1.set_title(f'Wind Chill Effect on Penguin Temperature')
    ax1.grid(True)
    ax1.legend(loc='best')
    
    # Plot wind speed and active penguin count
    ax2 = axs[1]
    ax2.plot(times, wind_speeds, 'b-', linewidth=2, label='Wind Speed')
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.set_xlabel('Time (s)')
    
    ax3 = ax2.twinx()
    ax3.plot(times, results["history"]["active_counts"], 'g-', linewidth=2, label='Active Penguins')
    ax3.set_ylabel('Number of Active Penguins')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    ax2.grid(True)
    
    plt.tight_layout()
    
    return fig, axs

def create_dashboard(result_dirs):
    """
    Create a comprehensive dashboard of analysis plots
    
    Args:
        result_dirs: List of simulation result directories
        
    Returns:
        None (saves plots to files)
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"penguin_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameters to compare
    first_results = load_simulation_results(result_dirs[0])
    params = list(first_results["params"].keys())
    
    # Find parameters that vary across simulations
    varying_params = []
    for param in params:
        values = [extract_parameter_value(d, param) for d in result_dirs]
        unique_values = set(v for v in values if v is not None)
        if len(unique_values) > 1:
            varying_params.append(param)
    
    # Create heatmaps for each pair of varying parameters
    for i, param_x in enumerate(varying_params):
        for param_y in varying_params[i+1:]:
            fig, _ = create_survival_time_heatmap(result_dirs, param_x, param_y)
            fig.savefig(f"{output_dir}/heatmap_{param_x}_vs_{param_y}.png", dpi=150)
            plt.close(fig)
    
    # Create temperature profile plots for each varying parameter
    for param in varying_params:
        fig, _ = create_temperature_profile_plot(result_dirs, param)
        fig.savefig(f"{output_dir}/temperature_profile_{param}.png", dpi=150)
        plt.close(fig)
    
    # Create wind chill visualization for a representative result
    fig, _ = create_wind_chill_visualization(result_dirs[0])
    fig.savefig(f"{output_dir}/wind_chill_analysis.png", dpi=150)
    plt.close(fig)
    
    print(f"Analysis dashboard saved to {output_dir}")

if __name__ == "__main__":
    # Find all simulation results
    result_dirs = find_simulation_results()
    
    if not result_dirs:
        print("No simulation results found. Run some simulations first!")
    else:
        print(f"Found {len(result_dirs)} simulation result directories.")
        create_dashboard(result_dirs)