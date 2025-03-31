import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Emperor Penguin Thermoregulation Simulation
# Simulates how penguins move and regulate their body temperature in a group
# Penguins that get too hot or cold (outside survival range) are removed from simulation

# === Simulation Constants ===
N = 1
R = 0.15
hard_stop = 1.01 * 2 * R
U = 1.0
T_air = 30.0
T_min, T_max = 28.0, 44.0
T_opt = 37.5
T_cold_death = 28.0
T_hot_death = 44.0
m_neighbors = 7
k_air = 0.01
k_wind = 0.015
k_rad = 0.06
k_body = 0.05
dt = 0.005

# === Random Initialization within a Circle ===
pad = 0.5
# Calculate domain size based on number of penguins
cluster_radius = R * np.sqrt(N) * 1.5  # Radius to contain all penguins with some spacing
xlim = cluster_radius + pad
ylim = cluster_radius + pad

# Initialize positions randomly within the circle
cylinder_positions = np.zeros((N, 2), dtype=float)
cylinder_temps = np.full(N, T_opt)

# Place penguins one by one, ensuring no overlap
for i in range(N):
    max_attempts = 1000
    placed = False
    
    for attempt in range(max_attempts):
        # Generate random angle and radius (with square root for uniform distribution)
        angle = np.random.uniform(0, 2 * np.pi)
        # Use square root to ensure uniform distribution within circle
        radius = cluster_radius * np.sqrt(np.random.uniform(0, 0.8))  # 0.8 factor to keep inside boundary
        
        # Convert to Cartesian coordinates
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Check overlap with existing penguins
        overlap = False
        for j in range(i):
            dist = np.sqrt((x - cylinder_positions[j, 0])**2 + (y - cylinder_positions[j, 1])**2)
            if dist < 2.2 * R:  # Allow small spacing between penguins
                overlap = True
                break
        
        if not overlap:
            cylinder_positions[i] = [x, y]
            placed = True
            break
    
    # If we couldn't place after max attempts, place it closely packed
    if not placed:
        print(f"Warning: Could not place penguin {i} without overlap after {max_attempts} attempts")
        # Place near the origin with some offset from previous penguin
        if i > 0:
            cylinder_positions[i] = cylinder_positions[i-1] + np.array([2.2 * R, 0])
        else:
            cylinder_positions[i] = np.array([0, 0])

# Domain
Nx, Ny = 200, 200  # Make grid square for better visualization
x_vals = np.linspace(-xlim, xlim, Nx)
y_vals = np.linspace(-ylim, ylim, Ny)
X, Y = np.meshgrid(x_vals, y_vals)

# === ScalarMap for Color Coding ===
norm = Normalize(vmin=T_min, vmax=T_max)
scalar_map = ScalarMappable(norm=norm, cmap='RdYlBu_r')

# === Plot Setup ===
fig_width = 12
fig_height = 10
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

wind_colormap = ax.imshow(np.zeros_like(X), extent=[-xlim, xlim, -ylim, ylim],
                          origin='lower', cmap='cool', vmin=0, vmax=2.5)
cax_wind = inset_axes(ax, width="3%", height="100%", loc='center right',
                      bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
cbar_wind = plt.colorbar(wind_colormap, cax=cax_wind, orientation='vertical')
cbar_wind.set_label('Wind Speed (m/s)', fontsize=8)
cbar_wind.ax.tick_params(labelsize=7)

cax_temp = inset_axes(ax, width="3%", height="100%", loc='center left',
                      bbox_to_anchor=(-0.09, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
cbar_temp = plt.colorbar(scalar_map, cax=cax_temp, orientation='vertical')
cbar_temp.set_label('Body Temp (\u00b0C)', fontsize=8)
cbar_temp.ax.tick_params(labelsize=7)
cbar_temp.ax.yaxis.set_ticks_position('left')
cbar_temp.ax.yaxis.set_label_position('left')

ax.set_xticks([])
ax.set_yticks([])
frame_text = ax.text(0.5, 1.03, '', transform=ax.transAxes, color='black',
                     fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
quiver = ax.quiver(X[::4, ::4], Y[::4, ::4],
                   np.zeros_like(X[::4, ::4]), np.zeros_like(Y[::4, ::4]),
                   color='white', scale=75, width=0.0015, alpha=0.5)

cylinder_patches = [plt.Circle((x, y), R, color=scalar_map.to_rgba(T_opt), zorder=5)
                    for x, y in cylinder_positions]
for patch in cylinder_patches:
    ax.add_patch(patch)

# === Check if penguin is active ===
def is_active(pos):
    return abs(pos[0]) < 9999

# === Compute Wind Field ===
def compute_velocity_field(time_factor, cylinders):
    Vx = U * np.ones_like(X)
    Vy = np.zeros_like(Y)
    for (x0, y0) in cylinders:
        # Skip inactive penguins
        if not is_active((x0, y0)):
            continue
        dx, dy = X - x0, Y - y0
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        r = np.where(r < R, R, r)
        U_mod = U * (1 + 0.2 * np.sin(time_factor))
        Vr = U_mod * (1 - (R**2 / r**2)) * np.cos(theta)
        Vt = -U_mod * (1 + (R**2 / r**2)) * np.sin(theta)
        Vx_c = Vr * np.cos(theta) - Vt * np.sin(theta)
        Vy_c = Vr * np.sin(theta) + Vt * np.cos(theta)
        outside = r > R
        Vx[outside] += (Vx_c[outside] - U_mod)
        Vy[outside] += Vy_c[outside]
    return Vx, Vy

# === Temperature Update ===
def update_temperature(T, T_air, wind_speed, neighbor_temps, distances,
                       k_air=0.01, k_wind=0.015, k_rad=0.06, k_body=0.05):
    delta = T_opt - T
    Q_body = k_body * (1 / (1 + np.exp(-5 * delta)))
    T_air_loss = k_air * (T - T_air)
    T_wind_loss = k_wind * wind_speed**0.6
    T_rad_gain = k_rad * np.sum([
        max(0, Tn - T) / (d**2 + 1e-6)
        for Tn, d in zip(neighbor_temps, distances)
    ]) if neighbor_temps else 0
    return T - T_air_loss - T_wind_loss + Q_body + T_rad_gain

# === Animation Frame ===
def animate(frame):
    global cylinder_positions, cylinder_temps
    time_factor = frame / 5.0
    new_positions = cylinder_positions.copy()
    new_temps = cylinder_temps.copy()
    Vx, Vy = compute_velocity_field(time_factor, cylinder_positions)
    speed = np.sqrt(Vx**2 + Vy**2)

    # Get only active penguins for calculations
    active_indices = [i for i, pos in enumerate(cylinder_positions) if is_active(pos)]
    active_positions = np.array([cylinder_positions[i] for i in active_indices])
    
    if len(active_positions) > 0:
        # Calculate center of active penguins only
        center = np.mean(active_positions, axis=0)
    else:
        center = np.array([0.0, 0.0])

    for i, pos in enumerate(cylinder_positions):
        # Skip inactive penguins
        if not is_active(pos):
            continue

        # Find active neighbors only
        neighbor_data = []
        for j in range(len(cylinder_positions)):
            if j != i and is_active(cylinder_positions[j]):
                dist = np.linalg.norm(cylinder_positions[j] - pos)
                neighbor_data.append((j, dist))
        
        # Sort neighbors by distance
        neighbor_data.sort(key=lambda x: x[1])
        
        # Get the m closest neighbors
        num_neighbors = min(m_neighbors, len(neighbor_data))
        if num_neighbors > 0:
            # Extract neighbor information
            neighbor_indices = [data[0] for data in neighbor_data[:num_neighbors]]
            neighbor_positions = [cylinder_positions[j] for j in neighbor_indices]
            neighbor_temps = [cylinder_temps[j] for j in neighbor_indices]
            neighbor_distances = [data[1] for data in neighbor_data[:num_neighbors]]
            
            # Movement: follow the heat gradient
            avg_pos = np.mean(neighbor_positions, axis=0)
            direction = avg_pos - pos
            temp = cylinder_temps[i]
            
            if temp < T_opt:
                direction = direction  # move toward warmth
            elif temp > T_opt:
                direction = -direction  # move away from heat
            
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            new_pos = pos + direction * dt

            # Handle collisions with other active penguins
            for j, other in enumerate(cylinder_positions):
                if i != j and is_active(other):
                    d = np.linalg.norm(new_pos - other)
                    if d < hard_stop:
                        repulse = (new_pos - other) / (d + 1e-6) * (hard_stop - d)
                        new_pos += repulse

            # Re-center active penguins
            new_pos -= center / 100.0
            new_pos[0] = np.clip(new_pos[0], -xlim + R, xlim - R)
            new_pos[1] = np.clip(new_pos[1], -ylim + R, ylim - R)
            new_positions[i] = new_pos
        
            # Get local wind speed at penguin position
            ix = np.argmin(np.abs(x_vals - pos[0]))
            iy = np.argmin(np.abs(y_vals - pos[1]))
            wind_local = speed[iy, ix]

            # Update temperature
            T_new = update_temperature(temp, T_air, wind_local, neighbor_temps, neighbor_distances,
                                    k_air, k_wind, k_rad, k_body)
            
            # Check for death conditions
            if T_new < T_cold_death or T_new > T_hot_death:
                new_positions[i] = np.array([-9999.0, -9999.0])  # Move to (-9999, -9999)
                new_temps[i] = np.nan  # Mark temperature as NaN for dead penguins
            else:
                new_temps[i] = np.clip(T_new, T_min, T_max)
        else:
            # No neighbors, penguin is isolated
            # Get local wind speed at penguin position
            ix = np.argmin(np.abs(x_vals - pos[0]))
            iy = np.argmin(np.abs(y_vals - pos[1]))
            wind_local = speed[iy, ix]
            
            # Update temperature with no neighbors
            T_new = update_temperature(cylinder_temps[i], T_air, wind_local, [], [],
                                   k_air, k_wind, k_rad, k_body)
            
            # Check for death conditions
            if T_new < T_cold_death or T_new > T_hot_death:
                new_positions[i] = np.array([-9999.0, -9999.0])
                new_temps[i] = np.nan
            else:
                new_temps[i] = np.clip(T_new, T_min, T_max)

    cylinder_positions = new_positions
    cylinder_temps = new_temps
    
    # Update visualization
    wind_colormap.set_data(speed)
    quiver.set_UVC(Vx[::4, ::4], Vy[::4, ::4])

    for patch, (x, y), temp in zip(cylinder_patches, cylinder_positions, cylinder_temps):
        patch.center = (x, y)
        if np.isnan(temp):
            patch.set_color(scalar_map.to_rgba(T_min))
        else:
            patch.set_color(scalar_map.to_rgba(temp))
        patch.set_visible(is_active((x, y)))

    # Calculate average temperature of active penguins
    active_temps = [t for pos, t in zip(cylinder_positions, cylinder_temps) if is_active(pos) and not np.isnan(t)]
    avg_temp = np.mean(active_temps) if active_temps else np.nan
    active_count = sum(1 for pos in cylinder_positions if is_active(pos))
    frame_text.set_text(f"Frame {frame:,} | Active: {active_count}/{N} | Avg Body Temp: {avg_temp:.2f}\u00b0C")
    
    return [wind_colormap, quiver, frame_text] + cylinder_patches

# Fix the warning by adding frames and save_count
anim = FuncAnimation(fig, animate, interval=100, blit=True, cache_frame_data=False)
plt.tight_layout()
plt.show()