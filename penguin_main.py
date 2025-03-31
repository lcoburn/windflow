import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Import simulation and visualization classes
from penguin_simulation_core import PenguinSimulation
from penguin_visualization import PenguinVisualizer
from matplotlib.animation import FuncAnimation

def run_single_simulation(params=None, with_visualization=True, num_steps=1000, save_results=True):
    """
    Run a single simulation with optional visualization
    
    Args:
        params: Custom simulation parameters (dict)
        with_visualization: Whether to show animation
        num_steps: Number of simulation steps
        save_results: Whether to save results to disk
        
    Returns:
        Path to saved results if save_results=True
    """
    # Create simulation
    sim = PenguinSimulation(params)
    
    if with_visualization:
        # Create visualizer and animation
        viz = PenguinVisualizer(sim)
        anim = viz.create_animation(frames=num_steps)
        
        # Show the animation
        plt.tight_layout()
        plt.show()
    else:
        # Run without visualization
        sim.run_simulation(num_steps)
    
    # Save results if requested
    if save_results:
        return sim.save_results()
    
    return None

def run_parameter_sweep(param_grid, steps_per_config=500, create_plots=True):
    """
    Run a parameter sweep and generate analysis plots
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        steps_per_config: Number of simulation steps for each parameter combination
        create_plots: Whether to generate summary plots
        
    Returns:
        List of result directories
    """
    # Create simulation object
    sim = PenguinSimulation()
    
    # Run parameter sweep
    result_dirs = sim.run_parameter_sweep(param_grid, steps_per_config)
    
    if create_plots:
        # Generate plots for each result directory
        for result_dir in result_dirs:
            plot_simulation_results(result_dir)
    
    return result_dirs

def plot_simulation_results(result_dir):
    """
    Create plots from saved simulation results
    
    Args:
        result_dir: Path to directory with saved results
    """
    # Load the metadata and parameters
    with open(f"{result_dir}/metadata.json", 'r') as f:
        data = json.load(f)
    
    params = data["params"]
    history = data["history"]
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Time axis (in seconds)
    times = history["times"]
    
    # Plot 1: Temperature over time
    ax = axs[0]
    ax.plot(times, history["mean_temperatures"], 'r-', linewidth=2)
    ax.set_ylabel('Average Body Temp (°C)')
    ax.set_title(f'Penguin Simulation Results - {os.path.basename(result_dir)}')
    ax.grid(True)
    
    # Add reference lines for optimal temperature and death thresholds
    ax.axhline(y=params["T_opt"], color='g', linestyle='--', alpha=0.7, label='Optimal Temp')
    ax.axhline(y=params["T_cold_death"], color='b', linestyle='--', alpha=0.7, label='Cold Death')
    ax.axhline(y=params["T_hot_death"], color='r', linestyle='--', alpha=0.7, label='Heat Death')
    ax.legend()
    
    # Plot 2: Active penguins over time
    ax = axs[1]
    ax.plot(times, history["active_counts"], 'b-', linewidth=2)
    ax.set_ylabel('Active Penguins')
    ax.set_ylim(0, params["N"] + 1)
    ax.grid(True)
    
    # Plot 3: Environmental conditions
    ax = axs[2]
    ax.plot(times, history["air_temperatures"], 'c-', label='Air Temp')
    
    # Calculate wind chill
    wind_chill = [t - (1.59 * 0.112 * (3.6 * w)) for t, w in 
                 zip(history["air_temperatures"], history["wind_speeds"])]
    
    ax.plot(times, wind_chill, 'm--', label='Wind Chill')
    ax2 = ax.twinx()
    ax2.plot(times, history["wind_speeds"], 'y-', label='Wind Speed')
    ax2.set_ylabel('Wind Speed (m/s)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.grid(True)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{result_dir}/summary_plot.png", dpi=150)
    plt.close()

def generate_comparison_plot(result_dirs, key_param):
    """
    Generate a comparison plot across multiple simulation runs
    
    Args:
        result_dirs: List of directories with simulation results
        key_param: Parameter name to compare across runs
    """
    plt.figure(figsize=(12, 8))
    
    for result_dir in result_dirs:
        # Load the metadata
        with open(f"{result_dir}/metadata.json", 'r') as f:
            data = json.load(f)
        
        params = data["params"]
        history = data["history"]
        
        # Get parameter value for label
        param_value = params[key_param]
        label = f"{key_param}={param_value}"
        
        # Plot mean temperature over time
        plt.plot(history["times"], history["mean_temperatures"], label=label, linewidth=2)
    
    plt.title(f'Temperature Comparison Across Different {key_param} Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Body Temperature (°C)')
    plt.grid(True)
    plt.legend()
    
    # Save the comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"comparison_{key_param}_{timestamp}.png", dpi=150)
    plt.close()

def load_and_visualize_results(result_dir):
    """
    Load saved simulation results and create a visualization
    
    Args:
        result_dir: Path to directory with saved results
    """
    # Load metadata and parameters
    with open(f"{result_dir}/metadata.json", 'r') as f:
        data = json.load(f)
    
    params = data["params"]
    
    # Load position and temperature data
    positions = np.load(f"{result_dir}/positions.npy")
    temperatures = np.load(f"{result_dir}/temperatures.npy")
    headings = np.load(f"{result_dir}/headings.npy")
    
    # Create a new simulation with the same parameters
    sim = PenguinSimulation(params)
    
    # Create visualizer
    viz = PenguinVisualizer(sim)
    
    # Create a playback function that uses saved data
    def playback_frame(frame):
        # Update simulation state from saved data
        sim.positions = positions[frame]
        sim.temperatures = temperatures[frame]
        sim.headings = headings[frame]
        sim.time = data["history"]["times"][frame]
        
        # Update visualization
        return viz.update_visualization(frame)
    
    # Create animation with the playback function
    anim = FuncAnimation(
        viz.fig, 
        playback_frame, 
        frames=min(len(positions), 500), 
        interval=50, 
        blit=True
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example 1: Run a single penguin simulation
    single_penguin_params = {
        "N": 100,
        "T_air": 38.0,
        "U": 5.0,
        "domain_radius": 3.0
    }
    
    print("Running single penguin simulation...")
    result_dir = run_single_simulation(single_penguin_params)
    print(f"Results saved to: {result_dir}")
    
    # Example 2: Run a small flock simulation
    # flock_params = {
    #     "N": 20,
    #     "T_air": -30.0,
    #     "U": 8.0,
    #     "domain_radius": 5.0
    # }
    
    # print("\nRunning flock simulation...")
    # result_dir = run_single_simulation(flock_params)
    # print(f"Results saved to: {result_dir}")
    
    # Example 3: Run a parameter sweep
    # print("\nRunning parameter sweep...")
    # params_to_sweep = {
    #     "T_air": [-10.0, -20.0, -30.0],
    #     "U": [3.0, 6.0, 9.0]
    # }
    
    # result_dirs = run_parameter_sweep(params_to_sweep, steps_per_config=300)
    # print(f"Parameter sweep results saved to: {result_dirs}")
    
    # Generate comparison plot
    # generate_comparison_plot(result_dirs, "T_air")
    
    # print("\nAnalysis complete! Check the output directories for results.")