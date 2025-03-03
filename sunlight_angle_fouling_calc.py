import numpy as np
import random
import matplotlib.pyplot as plt
import mesa
import math
import seaborn as sns
from mesa.experimental.cell_space.property_layer import PropertyLayer
# import plotly.graph_objects as go


class HullModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, width, length, max_depth, seed=None):
        super().__init__(seed=seed)
        self.depths = np.zeros((width, length))
        self.grid = mesa.space.MultiGrid(width, length, True)
        self.width = width
        self.length = length
        self.max_depth = max_depth

        # Create agents
        # Create x and y coordinates for agents
        for x in range(width):
            for y in range(length):

                if x == 0 or x == (self.width - 1) or y == 0 or y == (self.length - 1):
                    self.depths[x][y] = 0
                else:

                    self.depths[x][y] = self.max_depth * ((self.cross_profile(x) + self.length_profile(y)) / 2)
                # self.depths[x][y] = self.max_depth * (self.length_profile(y))

    def cross_profile(self, x):
        '''returns a percentage (out of 1) of the maximum depth of the hull'''
        ys = np.linspace(-1.25, 1.25, self.width)
        local_max_height = (0.4 * (abs(math.tan(1.25))))
        self.scaled_height = 0.4 * abs(math.tan(ys[x])) / local_max_height
        self.scaled_depth = 1 - self.scaled_height  # opposite of height
        return self.scaled_depth

    def length_profile(self, y):
        '''returns a percentage of the maximum depth of the hull'''
        xs = np.linspace(-(0.2 ** (1 / 8)), (0.2 ** (1 / 8)), self.length)
        self.local_depth = (xs[y]) ** 8
        '''if xs[y] < 0.0375:
            self.local_depth = 4*(xs[y]-0.0375) + 0.15
        elif xs[y] > 1.0375:
            self.local_depth = -3*(xs[y]-1.0375) + 0.15
        else:
            self.local_depth = 0.15'''

        return 1 - (self.local_depth / 0.2)


# model = HullModel(10,10,10)
# depths = model.depths
# width = model.width
# length = model.length
# X, Y = np.meshgrid(np.arange(width), np.arange(length), indexing='ij')
# Z = depths
#
# X_flat = X.flatten()
# Y_flat = Y.flatten()
# Z_flat = Z.flatten()
#
# # print(X_flat, 'X_flat', Y_flat, 'Y_flat', Z_flat, 'Z_flat')
# flattened_data = np.column_stack((X_flat, Y_flat, Z_flat))
# print(flattened_data)
#



# model = HullModel(10, 10, 10)
# depths = model.depths
# x_column = 5
# angles = calculate_angles(depths, x_column)
#
# print("Angles between vertical normal and depth vectors:", angles)

# this does not make much sense as the coordinates do not show the shape i would expect.


""" THIS BIT (ABOVE) OUGHT TO BE LOOKED AT —— THIS WHOLE BIT"""
# sgasdjdjfdjdfjdfjdfsj
# asjgsdjkfdjkdsfjksdfkjdsfkjj9g5c2


def calculate_surface_angles(depths, x_columns):
    """
    calc surface angles along selected x-columns of hull.
    rets:
    dict with x-cols as keys and lists of angles as values.
    """
    length = depths.shape[1] // 2 # no. of y-coords
    angles_dict = {}

    for x_column in x_columns:
        angles = []
        for y in range(length - 1):
            # extract (x, y, z) positions
            z1 = depths[x_column, y]
            z2 = depths[x_column, y + 1]

            # calc surface vector (change in y and z)
            vector = np.array([0, 1, z2 - z1])  # movement in y

            # normal vector (points straight up)
            normal = np.array([0, 0, 1])

            # calc dot product and magnitudes
            dot_product = np.dot(vector, normal)
            magnitude_vector = np.linalg.norm(vector)
            magnitude_normal = np.linalg.norm(normal)

            # calc angle in rad.s
            angle_rad = np.arccos(dot_product / (magnitude_vector * magnitude_normal))

            # convert to deg.s
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)

        angles_dict[x_column] = angles  # store angles for this column

    return angles_dict

def check_sunlight_hit(surface_angles, sunlight_angle):
    """
    whether sunlight will hit each hull surface based on angles.
    rets:
    dictionary of if light will hit (True) or miss (False).
    """
    hit_results = {}
    for x_column, angles in surface_angles.items():
        hit_results[x_column] = [sunlight_angle > angle for angle in angles]

    return hit_results


def plot_surface_angles(surface_angles):
    plt.figure(figsize=(10, 6))

    for x_column, angles in surface_angles.items():
        y_values = np.arange(len(angles))  # corresponding y-coords
        plt.plot(y_values, angles, marker='o', label=f'X = {x_column}')
    # for x_column, angles in surface_angles.items():
    #     y_values = np.arange(len(angles) // 2)  # Ignore latter half of y-coordinates
    #     plt.plot(y_values, angles[:len(angles) // 2], marker='o', label=f'X = {x_column}')

    plt.xlabel('Y-Coordinate')
    plt.ylabel('Surface Angle (degrees)')
    plt.title('Surface Angles Along Selected X-Columns')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_hull_cross_section(depths, x_column, sunlight_angle):
    y_values = np.arange(depths.shape[1] // 2)  # still only first half?
    z_values = depths[x_column, :len(y_values)]

    plt.figure(figsize=(10, 6))
    plt.plot(y_values, z_values, marker='o', label=f'Hull Cross-Section at X = {x_column}')

    # # get sunlight line
    # max_y = max(y_values)
    # max_z = max(z_values)
    # sunlight_x = np.linspace(0, max_y, 100)
    # sunlight_y = max_z - np.tan(np.radians(sunlight_angle)) * sunlight_x
    # plt.plot(sunlight_x, sunlight_y, 'r--', label=f'Sunlight Angle = {sunlight_angle}°')

    # get sunlight line to act as tangent to hull
    tangent_index = len(y_values) // 2  # pick midpoint for tangent
    tangent_y = y_values[tangent_index]
    tangent_z = z_values[tangent_index]

    sunlight_x = np.linspace(tangent_y - 10, tangent_y + 10, 100)  # Extend both directions
    sunlight_y = tangent_z + np.tan(np.radians(sunlight_angle)) * (sunlight_x - tangent_y)
    plt.plot(sunlight_x, sunlight_y, 'r--', label=f'Sunlight Angle = {sunlight_angle}°')

    plt.xlabel('Y-Coordinate')
    plt.ylabel('Depth (Z)')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title('Hull Cross-Section with Sunlight Angle')
    plt.legend()
    plt.grid(True)
    plt.show()

# create hull thingamigig
model = HullModel(40, 40, 40)
depths = model.depths

x_columns = [5, 15, 25, 35]  # select multiple x-coordinate columns
sunlight_angle = 50  # sunlight angle in degrees

# surface angles
surface_angles = calculate_surface_angles(depths, x_columns)
print("Surface angles:", surface_angles)

# if sunlight will hit
light_hits = check_sunlight_hit(surface_angles, sunlight_angle)
print("Light hitting the hull:", light_hits)


plot_surface_angles(surface_angles)

plot_hull_cross_section(depths, x_columns[0], sunlight_angle)