import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import math

##### VERSION 5

fig = plt.figure(figsize=(18, 6))  

# 3D subplot - orbit simulations
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_position([-0.02, 0.1, 0.32, 0.8])  

## Set position labels & simulation title
ax.set_xlabel('X (km)', labelpad=10)
ax.set_ylabel('Y (km)', labelpad=10)
ax.set_zlabel('Z (km)', labelpad=10)
ax.set_title('Satellite Orbit Visualization')

## Change color of 3D graph to black, mimicking space 
fig.patch.set_facecolor('black')
ax.set_facecolor('black') 
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))  
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

## Change color of axis labels and title to white
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.title.set_color('white')
ax.tick_params(colors='white')

# 2D subplot - ground track visualization
gt = fig.add_subplot(1, 2, 2)
gt.set_position([0.38, 0.15, 0.60, 0.7])  

## Set longitude and latitude limits for full Earth map
gt.set_xlim(-180, 180)
gt.set_ylim(-90, 90)

## Add axis labels and map title
gt.set_xlabel('Longitude (°)')
gt.set_ylabel('Latitude (°)')
gt.set_title('Satellite Ground Track')

## Change color of axis labels and title to white
gt.xaxis.label.set_color('white')
gt.yaxis.label.set_color('white')
gt.title.set_color('white')
gt.tick_params(colors='white')

# Add Earth map background to 2d graph 
img = plt.imread(r'C:\Users\Parin Ravula\OneDrive\Desktop\Aerospace Projects\Python\Satellite_Orbit_Visualizer\world_map.jpg')
gt.imshow(img, extent=[-180, 180, -90, 90], aspect='auto', zorder=0)

## Adjust 2D graph settings
gt.set_aspect('auto') 
gt.grid(True, color='white', linewidth=0.5, alpha=0.5)

## Global variables defined
gravitational_constant = 6.6743 * math.pow(10, -11)
mass_of_earth = 5.97219 * math.pow(10, 24)
earth_radius = 6378
theta = np.linspace(0, 2 * np.pi, 5000)

## Store trajectory data for all orbits to enable simultaneous animation after user input
orbits_data = []
##
ground_track_data = []
## Keep references to all animation objects
animations = []

## Generate Earth
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(earth_x, earth_y, earth_z, color="skyblue", alpha=0.4)

## Determine orbit trajectories & corresponding animation's colors
color_index = 0



## Determine & set color of specific orbit 
def apply_orbit_color(color_index):
    colors = ["lime", "crimson", "gold", "mediumorchid", "orange", "hotpink", "chartreuse", "magenta", "white", "springgreen"]
    chosen_color = colors[color_index % 10]
    return chosen_color



## Determine orbital period of orbit       
def get_orbital_period(semi_major_axis):
    a = semi_major_axis * 1000  # convert to meters
    return 2 * np.pi * np.sqrt(a**3 / (gravitational_constant * mass_of_earth))    



# Reference: LEO at 400 km altitude
reference_altitude = 400
reference_semi_major = earth_radius + reference_altitude
reference_period = get_orbital_period(reference_semi_major)  
reference_anim_duration = 5 
time_scale = reference_anim_duration / reference_period



## Calculate animation duration
def get_anim_duration(semi_major_axis_km):
    real_period = get_orbital_period(semi_major_axis_km)
    return real_period * time_scale 



## Compute velocity (magnitude only) at each orbital position for circular orbits
def determine_velocity_circular(r):
    all_velocities = []

    for i in range (0, 5000):
        vel = math.sqrt((gravitational_constant * mass_of_earth) / r)
        all_velocities.append(vel)
    
    return all_velocities
 
 
        
## Compute velocity (magnitude only) at each orbital position for elliptical orbits        
def determine_velocity_elliptical(semi_major_axis, x, y, z):
    all_velocities = []

    for i in range (0, 5000):
        r = math.sqrt(x[i]**2 +  y[i]**2 + z[i]**2)
        vel = math.sqrt((gravitational_constant * mass_of_earth) * ((2 / r) - (1 / semi_major_axis)))
        all_velocities.append(vel)
        
    return all_velocities



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
        
    ## Initialize modified points arrays
    x_trajectory = []
    y_trajectory = []
    z_trajectory = []
    
    ## Apply rotation matrix to each point in the orbit to get the transformed (rotated) trajectory
    for i in range(len(x)):
        orbit_matrix = np.matrix([x[i], y[i], z[i]])
        rotated_matrix = np.round(np.dot(orbit_matrix, rotation_matrix), decimals=10)
        x_trajectory.append(rotated_matrix[0, 0])
        y_trajectory.append(rotated_matrix[0, 1])
        z_trajectory.append(rotated_matrix[0, 2])
    
    ## Save rotated points    
    trajectory_plots = {
        "x_trajectory": x_trajectory,
        "y_trajectory": y_trajectory,
        "z_trajectory": z_trajectory
    }
    
    return trajectory_plots



## Apply argument of periapsis rotation
def apply_argument_of_periapsis(angle, x, y, z): 
    rad = np.deg2rad(angle)  
    
    ## Find axis of rotation
    vec1 = [x[600] - x[250], y[600] - y[250], z[600] - z[250]]
    vec2 = [x[900] - x[250], y[900] - y[250], z[900] - z[250]]
    
    ## Take cross product to find normal vector, then normalize
    normal_vector = np.cross(vec1, vec2)
    normal_vector /= np.linalg.norm(normal_vector)
    
    ## Initialize new trajectory
    x_trajectory = []
    y_trajectory = []
    z_trajectory = []
    
    ## Apply rotation matrix to each point in the orbit to get the transformed (rotated) trajectory
    for i in range(0,5000):
        v = np.array([x[i], y[i], z[i]])
        k = normal_vector
        
        ## Rodrigues' rotation formula
        rotated = (v * np.cos(rad) +
                   np.cross(k, v) * np.sin(rad) +
                   k * (np.dot(k, v)) * (1 - np.cos(rad)))

        x_trajectory.append(rotated[0])
        y_trajectory.append(rotated[1])
        z_trajectory.append(rotated[2])
       
    ## Save rotated points    
    trajectory_plots = {
        "x_trajectory": x_trajectory,
        "y_trajectory": y_trajectory,
        "z_trajectory": z_trajectory
    }
    
    return trajectory_plots  



## Plot satellite ground track by projecting 3D orbit onto Earth's 2D map
def graph_satellite_ground_visualization(orbit_color, x, y, z):     
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))

    gt.plot(lon, lat, linewidth=2.5, color=orbit_color)



## Determine orbit animation details 
def animate_orbit(satellite_name, orbit_color, x_traj, y_traj, z_traj, desired_anim_duration=10, fps=60):
    N = len(x_traj)
    total_frames = int(desired_anim_duration * fps)
    indices = np.linspace(0, N-1, total_frames).astype(int)
    x_anim = np.array(x_traj)[indices]
    y_anim = np.array(y_traj)[indices]
    z_anim = np.array(z_traj)[indices]
    trail, = ax.plot([], [], [], color=orbit_color, lw=2.5)
    satellite, = ax.plot([], [], [], 'o', markersize=8, color='red')
    label = ax.text2D(0, 0, satellite_name, fontsize=9, color='black',
                  bbox=dict(boxstyle='round,pad=0.6', fc='white', ec='black', lw=1, alpha=0.8),
                  ha='left', va='center')

    orbits_data.append({
        "x": x_anim,
        "y": y_anim,
        "z": z_anim,
        "trail": trail,
        "satellite": satellite,
        "label": label,  
        "frames": total_frames
    })



## Prepare orbit visualization by applying transformations and setting up trajectory
def process_and_visualize_orbit(satellite_name, orbit_color, altitude_or_semi_major, inclination, raan, periapsis, x, y, z):
    total_radius = earth_radius + altitude_or_semi_major
    
    ## Apply rotation about x axis (inclination) and z axis (RAAN & Argument of Periapsis)     
    z_rotation = apply_rotation(raan, "z", x, y, z) 
    x_rotation = apply_rotation(inclination, "x", z_rotation["x_trajectory"], z_rotation["y_trajectory"], z_rotation["z_trajectory"])
    periapsis_rotation = apply_argument_of_periapsis(periapsis, x_rotation["x_trajectory"], x_rotation["y_trajectory"], x_rotation["z_trajectory"])  
    
    ## Plot satellite's ground track on the Earth 
    graph_satellite_ground_visualization(orbit_color, periapsis_rotation["x_trajectory"], periapsis_rotation["y_trajectory"], periapsis_rotation["z_trajectory"])
    
    ## Create animation of orbit
    anim_duration = get_anim_duration(total_radius)
    animate_orbit(
        satellite_name,
        apply_orbit_color(color_index),
        periapsis_rotation["x_trajectory"], 
        periapsis_rotation["y_trajectory"], 
        periapsis_rotation["z_trajectory"],
        desired_anim_duration=anim_duration,
        fps=60
    )

    ## Plot orbit trajectory
    ax.plot(periapsis_rotation["x_trajectory"], periapsis_rotation["y_trajectory"], periapsis_rotation["z_trajectory"], 
            zdir='z', color=orbit_color, alpha=0.4)
    


## Graph circular orbit
def graph_circular_orbit(satellite_name, orbit_color, altitude, inclination, raan, periapsis):
    total_radius = earth_radius + altitude
    
    ## Define orbit parameters    
    x = total_radius * np.cos(theta)
    y = total_radius * np.sin(theta) 
    z = np.zeros(len(x))
    
    ## Plot orbit trajectory and set up components for animation
    process_and_visualize_orbit(satellite_name, orbit_color, altitude, inclination, raan, periapsis, x, y, z)
    

 
## Graph elliptical orbit 
def graph_elliptical_orbit(satellite_name, orbit_color, eccentricity, altitude, inclination, raan, periapsis):
    ## Define & calculate both semi major axis and semi minor axis
    semi_major_axis = earth_radius + altitude
    semi_minor_axis = semi_major_axis * math.sqrt(1 - math.pow(eccentricity, 2))
    
    ## Define orbit parameters    
    x = (semi_major_axis) * np.cos(theta)
    y = (semi_minor_axis) * np.sin(theta)
    z = np.zeros(len(x))
    
    ## Find foci points, center ellipse about negative foci (-c, 0)
    c = math.sqrt(math.pow((semi_major_axis), 2) - math.pow((semi_minor_axis), 2))
    x += c
    
    ## Plot orbit trajectory and set up components for animation
    process_and_visualize_orbit(satellite_name, orbit_color, semi_major_axis, inclination, raan, periapsis, x, y, z)
    


## User input loop for orbit visualization 
while(True):
    ## Store satellite parameters provided by user
    satellite_name = input("Enter satellite name: ")
    if satellite_name == "stop":
        break
    eccentricity = float(input("Enter satellite orbit's eccentricity (between 0 and 1): "))
    altitude = float(input("Enter satellite's maximum altitude above the Earth: "))
    inclination = float(input("Enter satellite orbit's inclination (in degrees): "))
    raan = float(input("Enter the RAAN (rotation of the orbit around the axis of the Earth (in degrees): "))
    periapsis = float(input("Enter the argument of periapsis (rotation of the orbit around the axis of the Earth (in degrees): "))
    
    ## Determine orbit type/shape and generate corresponding trajectory
    if eccentricity == 0:    
        ## Plot circular orbit trajectory
        graph_circular_orbit(satellite_name, apply_orbit_color(color_index), altitude, inclination, raan, periapsis)
    else:
       ## Plot elliptical orbit trajectory 
       graph_elliptical_orbit(satellite_name, apply_orbit_color(color_index), eccentricity, altitude, inclination, raan, periapsis)
    
    color_index += 1



### Animation functionality START -------------------
## Find the maximum number of frames needed for any orbit
max_frames = max(orbit["frames"] for orbit in orbits_data)

def update(frame):
    for orbit in orbits_data:
        idx = frame % orbit["frames"]
        # Update satellite and trail
        orbit["satellite"].set_data_3d([orbit["x"][idx]], [orbit["y"][idx]], [orbit["z"][idx]])
        orbit["trail"].set_data_3d(orbit["x"][:idx+1], orbit["y"][:idx+1], orbit["z"][:idx+1])
        
        # Project 3D position to 2D screen coordinates
        x2, y2, _ = proj3d.proj_transform(
            orbit["x"][idx], orbit["y"][idx], orbit["z"][idx], ax.get_proj()
        )
        orbit["label"].set_position((x2, y2))  # Update 2D label position
    return (
        [o["satellite"] for o in orbits_data] +
        [o["trail"] for o in orbits_data] +
        [o["label"] for o in orbits_data]
    )

ani = FuncAnimation(
    fig,
    update,
    frames=None,
    interval=1000/60,
    blit=True
)
### Animation functionality END -------------------



## Plot orbits
ax.set_aspect('equal')
ax.legend()
plt.show()