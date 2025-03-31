import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class PenguinSimulation:
    """
    Emperor Penguin Thermoregulation Simulation
    
    This class handles the core simulation logic independent of visualization,
    allowing parameter sweeps and headless simulations.
    """
    
    def __init__(self, params=None):
        """Initialize the simulation with default or custom parameters"""
        # Set default parameters
        self.params = {
            # Penguin population parameters
            "N": 10,                   # Number of penguins
            "R": 0.15,                 # Penguin radius (m)
            "hard_stop": 0.33,         # Minimum distance between penguins
            
            # Environment parameters
            "T_air": -20.0,            # Air temperature (°C)
            "U": 5.0,                  # Base wind speed (m/s)
            
            # Penguin body temperature parameters
            "T_opt": 38.0,             # Optimal core temperature (°C)
            "T_min": 34.0,             # Min temperature for display
            "T_max": 42.0,             # Max temperature for display
            "T_cold_death": 34.0,      # Fatal hypothermia temperature (°C)
            "T_hot_death": 42.0,       # Fatal hyperthermia temperature (°C)
            
            # Heat transfer coefficients
            "k_air": 0.005,            # Heat loss to air
            "k_wind": 0.008,           # Wind chill effect
            "k_rad": 0.06,             # Heat radiation between penguins
            "k_body": 0.02,            # Metabolic heat generation
            
            # Movement parameters
            "dt": 0.01,                # Time step
            "m_neighbors": 7,          # Number of neighbors to consider
            "heading_inertia": 0.9,    # Inertia factor for penguin heading (0-1)
            
            # Domain parameters
            "domain_radius": 5.0,      # Simulation domain radius
        }
        
        # Update with custom parameters if provided
        if params:
            for key, value in params.items():
                if key in self.params:
                    self.params[key] = value
        
        # Initialize the simulation state
        self.initialize_simulation()
        
        # History tracking
        self.history = {
            "positions": [],
            "temperatures": [],
            "headings": [],
            "times": [],
            "active_counts": [],
            "mean_temperatures": [],
            "wind_speeds": [],
            "air_temperatures": []
        }
        
        # Current simulation time
        self.time = 0
    
    def initialize_simulation(self):
        """Initialize penguin positions, temperatures, and headings"""
        N = self.params["N"]
        R = self.params["R"]
        
        # Calculate cluster radius based on number of penguins
        cluster_radius = R * np.sqrt(N) * 1.5
        
        # Initialize arrays
        self.positions = np.zeros((N, 2), dtype=float)
        self.temperatures = np.full(N, self.params["T_opt"])
        self.headings = np.random.uniform(0, 2*np.pi, N)  # Random initial headings
        
        # Place penguins randomly in a circle with no overlap
        for i in range(N):
            max_attempts = 1000
            placed = False
            
            for attempt in range(max_attempts):
                # Generate random position in circle
                angle = np.random.uniform(0, 2 * np.pi)
                radius = cluster_radius * np.sqrt(np.random.uniform(0, 0.8))
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                # Check for overlap with existing penguins
                overlap = False
                for j in range(i):
                    if not self.is_active(self.positions[j]):
                        continue
                    dist = np.linalg.norm(np.array([x, y]) - self.positions[j])
                    if dist < 2.2 * R:
                        overlap = True
                        break
                
                if not overlap:
                    self.positions[i] = [x, y]
                    placed = True
                    break
            
            # Fallback if no valid position found
            if not placed:
                if i > 0:
                    # Place near the previous penguin
                    self.positions[i] = self.positions[i-1] + np.array([2.2 * R, 0])
                else:
                    # Place at origin
                    self.positions[i] = np.array([0, 0])
        
        # Set up grid for wind calculation
        domain_radius = self.params["domain_radius"]
        grid_size = 50
        x = np.linspace(-domain_radius, domain_radius, grid_size)
        y = np.linspace(-domain_radius, domain_radius, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
    
    def is_active(self, pos):
        """Check if a penguin is active in the simulation"""
        return abs(pos[0]) < 9999
    
    def compute_velocity_field(self, time_factor=0):
        """Compute wind velocity field across the domain"""
        U = self.params["U"] * (1 + 0.2 * np.sin(time_factor / 10))
        Vx = U * np.ones_like(self.X)
        Vy = np.zeros_like(self.Y)
        
        for i, (x0, y0) in enumerate(self.positions):
            # Skip inactive penguins
            if not self.is_active((x0, y0)):
                continue
                
            dx, dy = self.X - x0, self.Y - y0
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            r = np.where(r < self.params["R"], self.params["R"], r)
            
            Vr = U * (1 - (self.params["R"]**2 / r**2)) * np.cos(theta)
            Vt = -U * (1 + (self.params["R"]**2 / r**2)) * np.sin(theta)
            Vx_c = Vr * np.cos(theta) - Vt * np.sin(theta)
            Vy_c = Vr * np.sin(theta) + Vt * np.cos(theta)
            
            outside = r > self.params["R"]
            Vx[outside] += (Vx_c[outside] - U)
            Vy[outside] += Vy_c[outside]
        
        speed = np.sqrt(Vx**2 + Vy**2)
        return Vx, Vy, speed, U
    
    def update_temperature(self, T, T_air, wind_speed, neighbor_temps, distances):
        """Update penguin body temperature based on environmental factors"""
        k_air = self.params["k_air"]
        k_wind = self.params["k_wind"]
        k_rad = self.params["k_rad"]
        k_body = self.params["k_body"]
        T_opt = self.params["T_opt"]
        
        # Calculate metabolic heat production (increases when cold)
        delta = T_opt - T
        
        # Sigmoid function for metabolic response
        if delta > 0:  # Cold condition - increased metabolism
            Q_body = k_body * (1.5 + 1.0 / (1 + np.exp(-3 * delta)))
        else:  # Warm condition - decreased metabolism
            Q_body = k_body * (1.0 / (1 + np.exp(5 * delta)))
        
        # Heat loss to air (conduction/convection)
        T_air_loss = k_air * (T - T_air)
        
        # Wind chill effect
        T_wind_loss = k_wind * (wind_speed**0.5) * (T - T_air)
        
        # Heat gain from neighboring penguins
        T_rad_gain = k_rad * np.sum([
            max(0, Tn - T) / (d**2 + 1e-6)
            for Tn, d in zip(neighbor_temps, distances)
        ]) if neighbor_temps else 0
        
        # Temperature change
        return T - T_air_loss - T_wind_loss + Q_body + T_rad_gain
    
    def get_local_wind(self, pos, speed):
        """Get wind speed at a specific position"""
        # Find closest grid point
        domain_radius = self.params["domain_radius"]
        grid_size = len(self.X)
        
        # Convert position to grid index
        i = int((pos[0] + domain_radius) / (2 * domain_radius) * (grid_size - 1))
        j = int((pos[1] + domain_radius) / (2 * domain_radius) * (grid_size - 1))
        
        # Ensure indices are in bounds
        i = max(0, min(i, grid_size - 1))
        j = max(0, min(j, grid_size - 1))
        
        return speed[j, i]
    
    def update_heading(self, current_heading, target_vector):
        """Update penguin heading with inertia"""
        inertia = self.params["heading_inertia"]
        
        if np.linalg.norm(target_vector) < 1e-6:
            # No significant direction, maintain current heading
            return current_heading
        
        # Calculate target angle
        target_heading = np.arctan2(target_vector[1], target_vector[0])
        
        # Find the smallest angle difference (accounting for -π to π discontinuity)
        diff = target_heading - current_heading
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        
        # Apply inertia to heading change
        new_heading = current_heading + (1 - inertia) * diff
        
        # Normalize to [0, 2π)
        return new_heading % (2 * np.pi)
    
    def step(self):
        """Advance the simulation by one time step"""
        dt = self.params["dt"]
        R = self.params["R"]
        m_neighbors = self.params["m_neighbors"]
        hard_stop = self.params["hard_stop"]
        T_cold_death = self.params["T_cold_death"]
        T_hot_death = self.params["T_hot_death"]
        T_min = self.params["T_min"]
        T_max = self.params["T_max"]
        
        # Current time factor
        time_factor = self.time / 5.0
        
        # Compute wind field
        Vx, Vy, speed, current_wind = self.compute_velocity_field(time_factor)
        
        # Copy current state
        new_positions = self.positions.copy()
        new_temperatures = self.temperatures.copy()
        new_headings = self.headings.copy()
        
        # Get active penguins
        active_indices = [i for i, pos in enumerate(self.positions) if self.is_active(pos)]
        
        if active_indices:
            # Calculate center of active penguins
            center = np.mean(self.positions[active_indices], axis=0)
        else:
            center = np.array([0.0, 0.0])
        
        # Update each penguin
        for i, pos in enumerate(self.positions):
            # Skip inactive penguins
            if not self.is_active(pos):
                continue
            
            # Find active neighbors
            neighbor_data = []
            for j in range(len(self.positions)):
                if j != i and self.is_active(self.positions[j]):
                    dist = np.linalg.norm(self.positions[j] - pos)
                    neighbor_data.append((j, dist))
            
            # Sort neighbors by distance
            neighbor_data.sort(key=lambda x: x[1])
            
            # Get closest neighbors
            num_neighbors = min(m_neighbors, len(neighbor_data))
            
            if num_neighbors > 0:
                # Extract neighbor information
                neighbor_indices = [data[0] for data in neighbor_data[:num_neighbors]]
                neighbor_positions = [self.positions[j] for j in neighbor_indices]
                neighbor_temps = [self.temperatures[j] for j in neighbor_indices]
                neighbor_distances = [data[1] for data in neighbor_data[:num_neighbors]]
                
                # Movement: follow the heat gradient
                avg_pos = np.mean(neighbor_positions, axis=0)
                direction = avg_pos - pos
                temp = self.temperatures[i]
                
                if temp < self.params["T_opt"]:
                    direction = direction  # move toward warmth
                elif temp > self.params["T_opt"]:
                    direction = -direction  # move away from heat
                
                # Update heading with inertia
                new_headings[i] = self.update_heading(self.headings[i], direction)
                
                # Move in the direction of the new heading
                heading_vector = np.array([np.cos(new_headings[i]), np.sin(new_headings[i])])
                norm = np.linalg.norm(heading_vector)
                if norm > 1e-6:
                    heading_vector /= norm
                new_pos = pos + heading_vector * dt
                
                # Handle collisions
                for j, other in enumerate(self.positions):
                    if i != j and self.is_active(other):
                        d = np.linalg.norm(new_pos - other)
                        if d < hard_stop:
                            repulse = (new_pos - other) / (d + 1e-6) * (hard_stop - d)
                            new_pos += repulse
                
                # Re-center active penguins
                new_pos -= center / 100.0
                
                # Ensure penguins stay in domain
                domain_radius = self.params["domain_radius"]
                dist_from_center = np.linalg.norm(new_pos)
                if dist_from_center > domain_radius - R:
                    new_pos = new_pos * (domain_radius - R) / dist_from_center
                
                new_positions[i] = new_pos
                
                # Get local wind speed
                wind_local = self.get_local_wind(pos, speed)
                
                # Update temperature
                T_new = self.update_temperature(
                    temp, self.params["T_air"], wind_local, 
                    neighbor_temps, neighbor_distances
                )
                
                # Check for death conditions
                if T_new < T_cold_death or T_new > T_hot_death:
                    new_positions[i] = np.array([-9999.0, -9999.0])
                    new_temperatures[i] = np.nan
                else:
                    new_temperatures[i] = np.clip(T_new, T_min, T_max)
            
            else:
                # No neighbors, penguin is isolated
                # Get local wind speed
                wind_local = self.get_local_wind(pos, speed)
                
                # Update temperature with no neighbors
                T_new = self.update_temperature(
                    self.temperatures[i], self.params["T_air"], 
                    wind_local, [], []
                )
                
                # Check for death conditions
                if T_new < T_cold_death or T_new > T_hot_death:
                    new_positions[i] = np.array([-9999.0, -9999.0])
                    new_temperatures[i] = np.nan
                else:
                    new_temperatures[i] = np.clip(T_new, T_min, T_max)
        
        # Update simulation state
        self.positions = new_positions
        self.temperatures = new_temperatures
        self.headings = new_headings
        
        # Record history
        active_temps = [t for pos, t in zip(self.positions, self.temperatures) 
                       if self.is_active(pos) and not np.isnan(t)]
        
        self.history["positions"].append(self.positions.copy())
        self.history["temperatures"].append(self.temperatures.copy())
        self.history["headings"].append(self.headings.copy())
        self.history["times"].append(self.time)
        self.history["active_counts"].append(len(active_temps))
        self.history["mean_temperatures"].append(np.mean(active_temps) if active_temps else np.nan)
        self.history["wind_speeds"].append(current_wind)
        self.history["air_temperatures"].append(self.params["T_air"])
        
        # Increment time
        self.time += dt
        
        return {
            "positions": self.positions.copy(),
            "temperatures": self.temperatures.copy(),
            "headings": self.headings.copy(),
            "time": self.time,
            "active_count": len(active_temps),
            "mean_temperature": np.mean(active_temps) if active_temps else np.nan,
            "wind_speed": current_wind,
            "air_temperature": self.params["T_air"]
        }
    
    def run_simulation(self, num_steps, callback=None):
        """Run simulation for specified number of steps"""
        results = []
        for i in range(num_steps):
            state = self.step()
            results.append(state)
            if callback:
                callback(state, i)
        return results
    
    def save_results(self, params_description="default"):
        """Save simulation results to a JSON file"""
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirname = f"penguin_sim_{params_description}_{timestamp}"
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        # Prepare data for JSON
        data = {
            "params": self.params,
            "history": {
                "times": self.history["times"],
                "active_counts": self.history["active_counts"],
                "mean_temperatures": [float(t) if not np.isnan(t) else None for t in self.history["mean_temperatures"]],
                "wind_speeds": self.history["wind_speeds"],
                "air_temperatures": self.history["air_temperatures"]
            }
        }
        
        # Add positions, temps, and headings as separate files to avoid huge JSON
        positions_array = np.array(self.history["positions"])
        temperatures_array = np.array(self.history["temperatures"])
        headings_array = np.array(self.history["headings"])
        
        # Save metadata and aggregated metrics
        with open(f"{dirname}/metadata.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save raw data as numpy arrays
        np.save(f"{dirname}/positions.npy", positions_array)
        np.save(f"{dirname}/temperatures.npy", temperatures_array)
        np.save(f"{dirname}/headings.npy", headings_array)
        
        return dirname
    
    def run_parameter_sweep(self, parameter_grid, steps_per_config=1000):
        """
        Run simulations with different parameter combinations
        
        Args:
            parameter_grid: Dictionary mapping parameter names to lists of values
            steps_per_config: Number of simulation steps for each configuration
        
        Returns:
            List of result directories
        """
        import itertools
        
        # Generate all combinations of parameters
        param_names = list(parameter_grid.keys())
        param_values = list(itertools.product(*[parameter_grid[name] for name in param_names]))
        
        results = []
        
        for values in param_values:
            # Set parameters for this run
            params = self.params.copy()
            param_description = []
            
            for name, value in zip(param_names, values):
                params[name] = value
                param_description.append(f"{name}={value}")
            
            # Create a new simulation with these parameters
            sim = PenguinSimulation(params)
            
            # Run the simulation
            sim.run_simulation(steps_per_config)
            
            # Save results
            result_dir = sim.save_results("_".join(param_description))
            results.append(result_dir)
        
        return results


# Example usage:
if __name__ == "__main__":
    # Example 1: Run a single simulation
    sim = PenguinSimulation()
    sim.run_simulation(500)
    sim.save_results()
    
    # Example 2: Parameter sweep
    params_to_sweep = {
        "T_air": [-10.0, -20.0, -30.0],
        "U": [3.0, 6.0, 9.0]
    }
    
    sim = PenguinSimulation()
    result_dirs = sim.run_parameter_sweep(params_to_sweep, steps_per_config=500)
    print(f"Results saved to: {result_dirs}")