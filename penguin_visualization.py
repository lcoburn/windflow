import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from datetime import datetime

class PenguinVisualizer:
    """
    Visualization for Emperor Penguin Thermoregulation Simulation
    
    This class handles the animation and visualization of penguin simulations.
    """
    
    def __init__(self, simulation):
        """
        Initialize the visualizer with a simulation object
        
        Args:
            simulation: A PenguinSimulation instance
        """
        self.sim = simulation
        self.params = simulation.params
        
        # Set up the visualization
        self.setup_visualization()
    
    def setup_visualization(self):
        """Set up the matplotlib figure and visualization elements"""
        # Get domain size
        self.domain_radius = self.params["domain_radius"]
        
        # Create figure and axes
        fig_width = 12
        fig_height = 10
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Set up normalization for temperature color map
        self.norm = Normalize(vmin=self.params["T_min"], vmax=self.params["T_max"])
        self.scalar_map = ScalarMappable(norm=self.norm, cmap='RdYlBu_r')
        
        # Create wind speed visualization
        grid_size = 50
        x = np.linspace(-self.domain_radius, self.domain_radius, grid_size)
        y = np.linspace(-self.domain_radius, self.domain_radius, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initial velocity field
        Vx, Vy, speed, current_wind = self.sim.compute_velocity_field()
        
        # Wind speed colormap
        self.wind_colormap = self.ax.imshow(
            speed, 
            extent=[-self.domain_radius, self.domain_radius, -self.domain_radius, self.domain_radius],
            origin='lower', 
            cmap='cool', 
            vmin=0, 
            vmax=self.params["U"] * 2.5
        )
        
        # Add wind speed colorbar
        cax_wind = inset_axes(
            self.ax, 
            width="3%", 
            height="100%", 
            loc='center right',
            bbox_to_anchor=(0.05, 0, 1, 1), 
            bbox_transform=self.ax.transAxes, 
            borderpad=0
        )
        cbar_wind = plt.colorbar(self.wind_colormap, cax=cax_wind, orientation='vertical')
        cbar_wind.set_label('Wind Speed (m/s)', fontsize=8)
        cbar_wind.ax.tick_params(labelsize=7)
        
        # Add temperature colorbar
        cax_temp = inset_axes(
            self.ax, 
            width="3%", 
            height="100%", 
            loc='center left',
            bbox_to_anchor=(-0.09, 0, 1, 1), 
            bbox_transform=self.ax.transAxes, 
            borderpad=0
        )
        cbar_temp = plt.colorbar(self.scalar_map, cax=cax_temp, orientation='vertical')
        cbar_temp.set_label('Body Temp (°C)', fontsize=8)
        cbar_temp.ax.tick_params(labelsize=7)
        cbar_temp.ax.yaxis.set_ticks_position('left')
        cbar_temp.ax.yaxis.set_label_position('left')
        
        # Set up axis
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(-self.domain_radius, self.domain_radius)
        self.ax.set_ylim(-self.domain_radius, self.domain_radius)
        self.ax.set_aspect('equal')
        
        # Add text displays
        self.frame_text = self.ax.text(
            0.5, 1.03, '', 
            transform=self.ax.transAxes, 
            color='black',
            fontsize=12, 
            ha='center', 
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        self.env_text = self.ax.text(
            0.5, -0.05, '', 
            transform=self.ax.transAxes, 
            color='black',
            fontsize=10, 
            ha='center', 
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Wind velocity quiver plot
        self.quiver = self.ax.quiver(
            self.X[::4, ::4], 
            self.Y[::4, ::4],
            Vx[::4, ::4], 
            Vy[::4, ::4],
            color='white', 
            scale=75, 
            width=0.0015, 
            alpha=0.5
        )
        
        # Create penguin visualization components
        self.penguin_circles = []
        self.outline_arrows = []  # Store outline arrows separately
        self.inner_arrows = []    # Store yellow arrows separately
        
        for i, (pos, heading) in enumerate(zip(self.sim.positions, self.sim.headings)):
            # Penguin body (circle with black outline)
            circle = Circle(
                pos, 
                self.params["R"], 
                facecolor=self.scalar_map.to_rgba(self.sim.temperatures[i]),
                edgecolor='black', 
                linewidth=3.0,
                zorder=5
            )
            self.ax.add_patch(circle)
            self.penguin_circles.append(circle)
            
            # Penguin heading (yellow arrow)
            dx = np.cos(heading) * self.params["R"] * 1.5
            dy = np.sin(heading) * self.params["R"] * 1.5
            
            # Create a black outline arrow first (slightly larger)
            outline_arrow = FancyArrowPatch(
                pos, 
                pos + np.array([dx, dy]),
                color='black',
                linewidth=4,  # Thicker than the yellow arrow
                arrowstyle='-|>',
                mutation_scale=18,  # Slightly larger than the yellow arrow
                zorder=6
            )
            self.ax.add_patch(outline_arrow)
            self.outline_arrows.append(outline_arrow)
            
            # Create the yellow arrow on top
            inner_arrow = FancyArrowPatch(
                pos, 
                pos + np.array([dx, dy]),
                color='yellow',
                linewidth=2,
                arrowstyle='-|>',
                mutation_scale=15,
                zorder=7  # Higher zorder to appear on top of outline
            )
            self.ax.add_patch(inner_arrow)
            self.inner_arrows.append(inner_arrow)
            
        # Initialize text displays with starting values
        wind_chill = self.params["T_air"] - (1.59 * 0.112 * (3.6 * current_wind))
        self.frame_text.set_text(f"Time: 0.0s | Active: {self.params['N']}/{self.params['N']} | Avg Body Temp: {self.params['T_opt']:.1f}°C")
        self.env_text.set_text(f"Air Temp: {self.params['T_air']:.1f}°C | Wind: {current_wind:.1f} m/s | Wind Chill: {wind_chill:.1f}°C")
    
    def animate(self, frame):
        """Animation function for FuncAnimation"""
        # Step the simulation
        self.sim.step()
        
        # Update the visualization
        return self.update_visualization(frame)
    
    def create_animation(self, frames=500, interval=50, save_path=None):
        """
        Create and optionally save an animation of the simulation
        
        Args:
            frames: Number of frames in the animation
            interval: Delay between frames in milliseconds
            save_path: If provided, save the animation to this file path
            
        Returns:
            The animation object
        """
        # Create animation
        anim = FuncAnimation(
            self.fig, 
            self.animate, 
            frames=frames, 
            interval=interval, 
            blit=True, 
            cache_frame_data=False
        )
        
        # Save if requested
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow')
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg')
            else:
                raise ValueError("Save path must end with .gif or .mp4")
        
        return anim
    
    def show(self):
        """Display the current state without animation"""
        plt.tight_layout()
        plt.show()
    
    def save_frame(self, filename):
        """Save the current visualization frame as an image"""
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        return filename

    def update_visualization(self, frame=None):
        """Update the visualization with the current simulation state"""
        # Get wind field
        Vx, Vy, speed, current_wind = self.sim.compute_velocity_field(self.sim.time / 5.0)
        
        # Update wind colormap and quiver
        self.wind_colormap.set_data(speed)
        self.quiver.set_UVC(Vx[::4, ::4], Vy[::4, ::4])
        
        # Update each penguin's visualization
        for i, (circle, outline_arrow, inner_arrow, pos, temp, heading) in enumerate(zip(
            self.penguin_circles, 
            self.outline_arrows,
            self.inner_arrows,
            self.sim.positions, 
            self.sim.temperatures, 
            self.sim.headings
        )):
            # Update circle position and color
            circle.center = (pos[0], pos[1])
            if np.isnan(temp):
                circle.set_facecolor(self.scalar_map.to_rgba(self.params["T_min"]))
            else:
                circle.set_facecolor(self.scalar_map.to_rgba(temp))
            
            # Make inactive penguins invisible
            active = self.sim.is_active(pos)
            circle.set_visible(active)
            outline_arrow.set_visible(active)
            inner_arrow.set_visible(active)
            
            if active:
                # Update arrows to show heading
                dx = np.cos(heading) * self.params["R"] * 1.5
                dy = np.sin(heading) * self.params["R"] * 1.5
                
                # Update both arrows
                outline_arrow.set_positions(pos, pos + np.array([dx, dy]))
                inner_arrow.set_positions(pos, pos + np.array([dx, dy]))
        
        # Update text displays
        active_count = sum(1 for pos in self.sim.positions if self.sim.is_active(pos))
        active_temps = [t for pos, t in zip(self.sim.positions, self.sim.temperatures) 
                       if self.sim.is_active(pos) and not np.isnan(t)]
        
        # Calculate wind chill temperature
        wind_chill = self.params["T_air"] - (1.59 * 0.112 * (3.6 * current_wind))
        
        if active_count > 0:
            avg_temp = np.mean(active_temps)
            self.frame_text.set_text(
                f"Time: {self.sim.time:.1f}s | Active: {active_count}/{self.params['N']} | "
                f"Avg Body Temp: {avg_temp:.1f}°C"
            )
        else:
            self.frame_text.set_text(
                f"Time: {self.sim.time:.1f}s | All penguins inactive"
            )
            
        self.env_text.set_text(
            f"Air Temp: {self.params['T_air']:.1f}°C | Wind: {current_wind:.1f} m/s | "
            f"Wind Chill: {wind_chill:.1f}°C"
        )
        
        # Return all artists that need to be redrawn
        return [
            self.wind_colormap, 
            self.quiver,
            self.frame_text, 
            self.env_text
        ] + self.penguin_circles + self.outline_arrows + self.inner_arrows