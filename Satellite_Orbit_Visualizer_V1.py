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
altitude = float(input("How high above the Earth will the satellite's orbit? "))
eccentricity = float(input("What is the eccentricity of the satellite's orbit (between 0 and 1)? "))
inclination = np.deg2rad(float(input("What is the inclination of the satellite's orbit (in degrees)? ")))