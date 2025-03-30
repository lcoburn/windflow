import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
k_air=0.01
k_wind=0.015
k_rad=0.06
k_body=0.05
dt = 0.001

# === Auto Grid & Domain Sizing ===
cols = int(np.ceil(np.sqrt(N)))
rows = int(np.ceil(N / cols))
pad = 0.5
spacing = 2 * R * 1.2
x_grid = np.arange(-(cols - 1) / 2, (cols + 1) / 2) * spacing
y_grid = np.arange(-(rows - 1) / 2, (rows + 1) / 2) * spacing
coords = [(x, y) for y in y_grid for x in x_grid][:N]
cylinder_positions = np.array(coords, dtype=float)
cylinder_temps = np.full(N, T_opt)

# Domain
xlim = spacing * cols / 2 + pad
ylim = spacing * rows / 2 + pad
Nx, Ny = 200, 100
x_vals = np.linspace(-xlim, xlim, Nx)
y_vals = np.linspace(-ylim, ylim, Ny)
X, Y = np.meshgrid(x_vals, y_vals)

# === ScalarMap for Color Coding ===
norm = Normalize(vmin=T_min, vmax=T_max)
scalar_map = ScalarMappable(norm=norm, cmap='RdYlBu_r')

# === Plot Setup ===
fig_width = 1200#(12 * xlim) / 2
fig_height = 1000#(10 * ylim) / 2
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Main wind colormap
wind_colormap = ax.imshow(np.zeros_like(X), extent=[-xlim, xlim, -ylim, ylim],
                          origin='lower', cmap='cool', vmin=0, vmax=2.5)

# Right-side wind speed colorbar
cax_wind = inset_axes(ax, width="3%", height="100%", loc='center right',
                      bbox_to_anchor=(0.05, 0, 1, 1),  # Was 0.02
                      bbox_transform=ax.transAxes, borderpad=0)
cbar_wind = plt.colorbar(wind_colormap, cax=cax_wind, orientation='vertical')
cbar_wind.set_label('Wind Speed (m/s)', fontsize=8)
cbar_wind.ax.tick_params(labelsize=7)

# Left-side body temp colorbar (legend style)
cax_temp = inset_axes(ax, width="3%", height="100%", loc='center left',
                      bbox_to_anchor=(-0.09, 0, 1, 1),  # Was -0.06
                      bbox_transform=ax.transAxes, borderpad=0)
cbar_temp = plt.colorbar(scalar_map, cax=cax_temp, orientation='vertical')
cbar_temp.set_label('Body Temp (°C)', fontsize=8)
cbar_temp.ax.tick_params(labelsize=7)
cbar_temp.ax.yaxis.set_ticks_position('left')
cbar_temp.ax.yaxis.set_label_position('left')

# Axes formatting
ax.set_xticks([])
ax.set_yticks([])
frame_text = ax.text(0.5, 1.03, '', transform=ax.transAxes, color='black',
                     fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Quiver plot for airflow
quiver = ax.quiver(X[::4, ::4], Y[::4, ::4],
                   np.zeros_like(X[::4, ::4]), np.zeros_like(Y[::4, ::4]),
                   color='white', scale=75, width=0.0015, alpha=0.5)

# Penguin patches
cylinder_patches = [plt.Circle((x, y), R, color=scalar_map.to_rgba(T_opt), zorder=5)
                    for x, y in cylinder_positions]
for patch in cylinder_patches:
    ax.add_patch(patch)

# === Compute Wind Field ===
def compute_velocity_field(time_factor, cylinders):
    Vx = U * np.ones_like(X)
    Vy = np.zeros_like(Y)
    for (x0, y0) in cylinders:
        if abs(x0) > 1e4:
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
def update_temperature(T, T_air, wind_speed, neighbors,
                       k_air=0.01, k_wind=0.015, k_rad=0.06, k_body=0.05):
    delta = T_opt - T
    Q_body = k_body * (1 / (1 + np.exp(-5 * delta)))  # Sigmoid: tapers to 0 or k_body
    T_air_loss = k_air * (T - T_air)
    T_wind_loss = k_wind * wind_speed**0.6
    T_rad_gain = k_rad * np.mean([max(0, Tn - T) for Tn in neighbors]) if neighbors else 0

    return T - T_air_loss - T_wind_loss + Q_body + T_rad_gain

# === Animation Frame ===
def animate(frame):
    global cylinder_positions, cylinder_temps
    time_factor = frame / 5.0
    new_positions, new_temps = [], []

    Vx, Vy = compute_velocity_field(time_factor, cylinder_positions)
    speed = np.sqrt(Vx**2 + Vy**2)
    print(cylinder_temps)
    if N > 1:
        if N > 1:
            avg_neighbor_dist = np.mean(distances[nearest_indices])
            avg_neighbor_pos = np.mean(neighbor_positions, axis=0)
        else:
            avg_neighbor_dist = 0
            avg_neighbor_pos = pos
    else:
        neighbor_positions = []
        neighbor_temps = []
        if abs(pos[0]) > 1e4:
            new_positions.append(pos)
            new_temps.append(cylinder_temps[i])
            continue

        distances = np.linalg.norm(cylinder_positions - pos, axis=1)
        nearest_indices = np.argsort(distances)[1:m_neighbors+1]
        neighbor_positions = cylinder_positions[nearest_indices]
        neighbor_temps = [cylinder_temps[j] for j in nearest_indices]
        temp = cylinder_temps[i]

        avg_neighbor_dist = np.mean(distances[nearest_indices])
        avg_neighbor_pos = np.mean(neighbor_positions, axis=0)

        # Determine direction based on temperature optimization
        direction = np.zeros(2)
        if temp > T_opt + 0.8:
            # Repel from neighbors when too hot
            direction = pos - avg_neighbor_pos
        elif temp < T_opt - 0.8:
            # Attract to neighbors if cold, but taper if already close
            if avg_neighbor_dist > 2 * R:
                direction = avg_neighbor_pos - pos
            else:
                # Huddle force decreases as already close
                direction = (avg_neighbor_pos - pos) * (avg_neighbor_dist / (2 * R))
        else:
            # No strong huddle/repel needed
            direction = np.zeros(2)


        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        new_pos = pos + direction * dt

        for j, other in enumerate(cylinder_positions):
            if i != j and abs(other[0]) < 1e4:
                d = np.linalg.norm(new_pos - other)
                if d < hard_stop:
                    repulse = (new_pos - other) / (d + 1e-6) * (hard_stop - d)
                    new_pos += repulse

        center = np.mean(cylinder_positions[cylinder_positions[:, 0] < 9999], axis=0)
        new_pos -= center / 100.0
        new_pos[0] = np.clip(new_pos[0], -xlim + R, xlim - R)
        new_pos[1] = np.clip(new_pos[1], -ylim + R, ylim - R)
        new_positions.append(new_pos)

        ix = np.argmin(np.abs(x_vals - new_pos[0]))
        iy = np.argmin(np.abs(y_vals - new_pos[1]))
        wind_local = speed[iy, ix]
        neighbors = [cylinder_temps[j] for j in nearest_indices]
        T_new = update_temperature(temp, T_air, wind_local, neighbors, k_air, k_wind, k_rad, k_body)

        if T_new < T_cold_death or T_new > T_hot_death:
            new_positions[-1] = np.array([9999.0, 9999.0])
            T_new = np.nan

        new_temps.append(np.clip(T_new, T_min, T_max))

    cylinder_positions[:] = np.array(new_positions)
    cylinder_temps[:] = np.array(new_temps)

    wind_colormap.set_data(speed)
    quiver.set_UVC(Vx[::4, ::4], Vy[::4, ::4])
    for patch, (x, y), temp in zip(cylinder_patches, cylinder_positions, cylinder_temps):
        patch.center = (x, y)
        patch.set_color(scalar_map.to_rgba(temp if not np.isnan(temp) else T_min))
        patch.set_visible(abs(x) < 999)

    avg_temp = np.nanmean(cylinder_temps)
    frame_text.set_text(f"Frame {frame:,} | Avg Body Temp: {avg_temp:.2f}°C")
    return [wind_colormap, quiver, frame_text] + cylinder_patches

# Run animation
anim = FuncAnimation(fig, animate, interval=100, blit=True)
plt.tight_layout()
plt.show()
