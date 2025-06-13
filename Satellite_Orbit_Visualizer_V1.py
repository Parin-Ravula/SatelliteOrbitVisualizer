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
## Storing satellite information provided by user
satellite_name = input("What is the name of the satellite? ")
altitude = float(input("How high above the Earth will the satellite's orbit? "))
eccentricity = float(input("What is the eccentricity of the satellite's orbit (between 0 and 1)? "))
inclination = np.deg2rad(float(input("What is the inclination of the satellite's orbit (in degrees)? ")))

## Hard-coding for testing
# satellite_name = "International Space Station"
# altitude = 11000
# eccentricity = 0.02
# inclination = np.deg2rad(0)

## Orbit parameters
earth_radius = 6378
total_radius = earth_radius + altitude
theta = np.linspace(0, 2 * np.pi, 10000)

## Parametric equations of orbit (circular)
x = total_radius * np.cos(theta)
y = total_radius * np.sin(theta) * np.cos(inclination)
z = total_radius * np.sin(theta) * np.sin(inclination)

# Generate Earth
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surfaces
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, zdir='z', linestyle='--', label=f"{satellite_name}'s orbit trajectory")
ax.plot_surface(earth_x, earth_y, earth_z, color="skyblue", alpha=0.5)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# Set an equal aspect ratio
ax.set_aspect('equal')
ax.legend()
plt.show()
### ////////////////////////////////////////////////////////////////



