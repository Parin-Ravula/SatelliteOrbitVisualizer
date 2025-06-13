import numpy as np
import matplotlib.pyplot as plt
import math

### VERSION 1
### ////////////////////////////////////////////////////////////////
# ## Storing satellite information provided by user
# altitude = float(input("How high above the Earth will the satellite's orbit? "))
# eccentricity = float(input("What is the eccentricity of the satellite's orbit (between 0 and 1)? "))
# inclination = float(input("What is the inclination of the satellite's orbit (in degrees)? "))

# ## Hard-coding for testing
# # altitude = 1100
# # eccentricity = 0.02
# # inclination = 13

# ## Orbit parameters
# earth_radius = 6378
# total_radius = earth_radius + altitude
# theta = np.linspace(0, 2 * np.pi, 10000)

# ## Parametric equations
# x = total_radius * np.cos(theta)
# y = total_radius * np.sin(theta)

# fig, ax = plt.subplots(figsize=(8, 8))
# earth = plt.Circle((0, 0), radius=6371, color='green', alpha=0.5, label='Earth')

# # Add to plot
# plt.plot(x, y)
# ax.add_patch(earth)
# plt.gca().set_aspect('equal', adjustable='box')  
# plt.title("Satellite Orbit Trajectory")
# plt.grid(True)
# plt.legend()
# plt.show()
### ////////////////////////////////////////////////////////////////



### VERSION 2
### ////////////////////////////////////////////////////////////////
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

## Global variables defined
earth_radius = 6378
theta = np.linspace(0, 2 * np.pi, 10000)

## Generate Earth
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(earth_x, earth_y, earth_z, color="skyblue", alpha=0.5)

## Apply rotations to orbit's plane
def apply_rotation(angle, axis_of_rotation, x, y, z):
    rad = np.deg2rad(angle)
    ## Assign correct axis of rotation
    if axis_of_rotation == "x":
        rotation_matrix = np.matrix([
            [1, 0, 0], 
            [0, np.cos(rad), -1 * np.sin(rad)], 
            [0, np.sin(rad), np.cos(rad)]
        ])
    elif axis_of_rotation == "z":
        rotation_matrix = np.matrix([
            [np.cos(rad), -1 * np.sin(rad), 0], 
            [np.sin(rad), np.cos(rad), 0], 
            [0,0,1]
        ])
        
    ## Initialize rotation matrix
    x_trajectory = []
    y_trajectory = []
    z_trajectory = []
    
    for i in range(0,10000):
        orbit_matrix = np.matrix([x[i], y[i], z[i]])
        rotated_matrix = np.round(np.dot(orbit_matrix, rotation_matrix), decimals=10)
        x_trajectory.append(rotated_matrix[0, 0])
        y_trajectory.append(rotated_matrix[0, 1])
        z_trajectory.append(rotated_matrix[0, 2])
        
    trajectory_plots = {
        "x_trajectory": x_trajectory,
        "y_trajectory": y_trajectory,
        "z_trajectory": z_trajectory
    }
    
    return trajectory_plots

def graph_circular_orbit(satellite_name, altitude, inclination, raan):
    total_radius = earth_radius + altitude
    
    ## Orbit parameters    
    x = total_radius * np.cos(theta)
    y = total_radius * np.sin(theta) 
    z = np.zeros(len(x))
    
    ## Apply rotation about x axis (inclination) and z axis (RAAN) + plot   
    x_rotation = apply_rotation(inclination, "x", x, y, z) 
    z_rotation = apply_rotation(raan, "z", x_rotation["x_trajectory"], x_rotation["y_trajectory"], x_rotation["z_trajectory"])
    ax.plot(z_rotation["x_trajectory"], z_rotation["y_trajectory"], z_rotation["z_trajectory"], zdir='z', linestyle='--', 
            label=f"{satellite_name}'s orbit trajectory")

    
    
while(True):
    ## Store satellite information provided by user
    satellite_name = input("Enter satellite name ")
    if satellite_name == "stop":
        break
    eccentricity = float(input("Enter satellite orbit's eccentricity (between 0 and 1): "))
    altitude = float(input("Enter satellite altitude above the Earth: "))
    inclination = float(input("Enter satellite orbit's inclination (in degrees): "))
    raan = float(input("Enter the RAAN (rotation of the orbit around the axis of the Earth (in degrees): "))
    
    ## Plot circular orbit trajectory
    graph_circular_orbit(satellite_name, altitude, inclination, raan)

ax.set_aspect('equal')
ax.legend()
plt.show()
### ////////////////////////////////////////////////////////////////



