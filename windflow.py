#
# 2D flow around a cylinder
#
#%%
import os
import cv2
import numpy as np
from numpy import array, zeros, fromfunction, sin, roll, sqrt, pi
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

#from matplotlib import cm
from tqdm import tqdm

mpl.rcParams['figure.max_open_warning'] = 0

#%%
###### Flow definition #########################################################
maxIter = 50000 # Total number of time iterations.
Re = 220.0         # Reynolds number.
nx, ny = 840, 360 # Numer of lattice nodes.
ly = ny-1         # Height of the domain in lattice units.
cx, cy, r = nx//4, ny//2, ny//9 # Coordinates of the cylinder/square.
uLB     = 0.04                  # Velocity in lattice units.
nulb    = uLB * r/Re;             # Viscoscity in lattice units.
omega = 1 / (3 * nulb + 0.5);    # Relaxation parameter.

###### Lattice Constants #######################################################
v = array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
            [ 0, -1], [-1,  1], [-1,  0], [-1, -1] ])
t = array([ 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])

col1 = array([0, 1, 2])
col2 = array([3, 4, 5])
col3 = array([6, 7, 8])

now = datetime.now().strftime("%d%m%Y_%H%M%S")
frames_dir = 'square_sim_frames' + now

if not(os.path.isdir(frames_dir)):
    os.mkdir(frames_dir)

#%%
###### Function Definitions ####################################################
def macroscopic(fin):
    rho = np.sum(fin, axis=0)
    u = zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u

def equilibrium(rho, u):              # Equilibrium distribution function.
    usqr = 3/2 * (u[0]**2 + u[1]**2)
    feq = zeros((9,nx,ny))
    for i in range(9):
        cu = 3 * (v[i, 0]*u[0, :, :] + v[i, 1]*u[1, :, :])
        feq[i, :, :] = rho*t[i] * (1 + cu + 0.5*cu**2 - usqr)
    return feq

###### Setup: cylindrical obstacle and velocity inlet with perturbation ########
# Creation of a mask with 1/0 values, defining the shape of the obstacle.
#def obstacle_fun(x, y):
    #return (x-cx)**2+(y-cy)**2<r**2

def obstacle_fun(x, y):
    return (x - cx)**2 + (y - cy)**2 <= r**2                                                                               


# Initial velocity profile: almost zero, with a slight perturbation to trigger
# the instability.
def inivel(d, x, y):
    return (1 - d) * uLB * (1 + 1e-4*sin(y/ly*2*pi))
#%%

obstacle = fromfunction(obstacle_fun, (nx, ny))
vel = fromfunction(inivel, (2,nx, ny))

# Initialization of the populations at equilibrium with the given velocity.
fin = equilibrium(1, vel)

#%%
###### Main time loop ##########################################################
for time in tqdm(range(maxIter)):
    # Right wall: outflow condition.
    fin[col3, -1, :] = fin[col3, -2, :] 

    # Compute macroscopic variables, density and velocity.
    rho, u = macroscopic(fin)

    # Left wall: inflow condition.
    u[:, 0, :] = vel[:, 0, :]
    rho[0, :] = 1/(1-u[0, 0, :]) * ( np.sum(fin[col2, 0, :], axis = 0) +
                                  2*np.sum(fin[col3, 0, :], axis = 0) )
    # Compute equilibrium.
    feq = equilibrium(rho, u)
    fin[[0, 1, 2], 0, :] = feq[[0, 1, 2], 0, :] + fin[[8, 7, 6], 0, :] - feq[[8, 7, 6], 0, :]

    # Collision step.
    fout = fin - omega * (fin - feq)

    # Bounce-back condition for obstacle.
    for i in range(9):
        fout[i, obstacle] = fin[8-i, obstacle]

    # Streaming step.
    for i in range(9):
        fin[i, :, :] = roll(roll(fout[i, :, :], v[i, 0], axis = 0), v[i, 1], axis = 1)
    
    # Visualization of the velocity.
    if (time%100 == 0):
        fig = plt.figure(figsize = (20, 10))
        ax = plt.gca()
        im = ax.imshow(sqrt(u[0]**2+u[1]**2).transpose(), cmap = 'magma')
        
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size = "5%", pad  = 0.5)

        

        clb = fig.colorbar(im, cax=cax)
        clb.set_label('Velocity Magnitude', rotation = 270, labelpad = 20)
        
        plt.xticks([])
        plt.yticks([])
        plt.savefig(frames_dir + "/vel.{0:05d}.png".format(time//100))
        plt.close(fig)
        
#%%
imshape = cv2.imread(os.path.join(frames_dir, os.listdir(frames_dir)[0])).shape
imshape = (imshape[1], imshape[0])


if not os.path.isdir(frames_dir):
    os.mkdir(frames_dir)

result = cv2.VideoWriter(f'square_simulation_re{Re}t{maxIter}_{now}.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         20, imshape)

for file in tqdm(sorted(os.listdir(frames_dir))):
    image = cv2.imread(os.path.join(frames_dir, file))
    result.write(image)
result.release()