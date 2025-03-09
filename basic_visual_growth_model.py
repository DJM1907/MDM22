from scipy.spatial import Delaunay
import numpy as np
import random
import matplotlib.pyplot as plt
import mesa
import math
import pandas as pd
import seaborn as sns
from mesa.experimental.cell_space.property_layer import PropertyLayer
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
from mesa.datacollection import DataCollector

def read_kcs_grid(file):
    
    with open(file,'r') as file:
        lines = file.readlines()
        
    X = np.zeros(len(lines))
    Y = np.zeros(len(lines))
    Z = np.zeros(len(lines))
        
    lines = [line.split() for line in lines]
    
    for line in range(len(lines)):
        X[line] = lines[line][0]
        Y[line] = lines[line][1]
        Z[line] = lines[line][2]
    
    return X,Y,Z

def new_KCS_hull():
    X_bow,Y_bow,Z_bow = read_kcs_grid("kcs_bow2.dat")
    X_stn,Y_stn,Z_stn = read_kcs_grid("kcs_stn2.dat")

    X = np.concatenate((X_bow,X_stn))
    Y = np.concatenate((Y_bow,Y_stn))
    Z = np.concatenate((Z_bow,Z_stn))

    mesh1 = np.stack((X,Y,Z))
    mesh2 = np.stack((X,-Y,Z))
    mesh = np.vstack((mesh1,mesh2))

    X = np.concatenate((X,X))
    Y = np.concatenate((Y,-Y))
    Z = np.concatenate((Z,Z))
    return mesh[:,0],mesh[:,1],mesh[:,2]

def KCS_hull():
    X_bow,Y_bow,Z_bow = read_kcs_grid("kcs_bow2.dat")
    X_stn,Y_stn,Z_stn = read_kcs_grid("kcs_stn2.dat")

    X = np.concatenate((X_bow,X_stn))
    Y = np.concatenate((Y_bow,Y_stn))
    Z = np.concatenate((Z_bow,Z_stn))

    mesh1 = np.array([X,Y,Z]).transpose()
    mesh2 = np.array([X,-Y,Z]).transpose()
    mesh = np.concatenate((mesh1,mesh2))

    X = np.concatenate((X,X))
    Y = np.concatenate((Y,-Y))
    Z = np.concatenate((Z,Z))
    return X,Y,Z


class HullAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        super().__init__(model)
        self.fouling_level = 0
        self.temp = self.model.temp
        self.depth = model.depths[self.pos]
        self.max_fouling = 8



    def grow(self):
        '''function deciding if the level of biofouling should increase'''
        # sigmoid distribution used to determine the probability of biofouling growth
        rand_var = np.random.uniform(0,1) 
        x,y = self.pos
        self.depth = self.model.depths[x][y]
        
        # create distributions for the optimal temperatures of each stage of biofouling
        optimal_temps = [30, 20, 15] 
        x,y = self.pos
        self.sunlight = self.model.sunlight_map[x][y]


        # sigmoid distribution for biofouling growth probability from depths
        growth_prob = self.model.sigmoid(self.depth)
        self.growth = 0 
        self.growth_prob = 0
        if rand_var > growth_prob:
            if self.fouling_level == 0:  # no fouling currently 
                self.growth += np.random.uniform(0.8,1.0)
                     
            elif self.fouling_level < self.max_fouling:  # keeps level below max fouling
                # condition statements control how much growing occurs due to light intensity
                if self.sunlight > 0.8: # high light intensity
                    self.growth += np.random.uniform(0.8,1.0)
                elif self.sunlight > 0.6:  # medium-high light intensity
                    self.growth += np.random.uniform(0.65,0.8)
                elif self.sunlight > 0.4:  # medium light intensity
                    self.growth += np.random.uniform(0.35,0.55)
                elif self.sunlight > 0.2:  # low light intensity
                    self.growth += np.random.uniform(0.15,0.35)

                '''
                # condition statements control the growing due to proximity to optimal temperature
                if self.fouling_level > 5:  # large organisms
                    if optimal_temps[2] - 1 < self.temp < optimal_temps[2] + 1: # if temperature is in optimal range then do more fouling
                        self.growth += np.random.uniform(0.35,0.55)
                    elif optimal_temps[2] - 5 < self.temp < optimal_temps[2] + 5: # if temperature is in optimal range then do more fouling 
                        self.growth += np.random.uniform(0.15,0.30)
                    else:
                        self.growth += np.random.uniform(0.05,0.15) 

                elif 2 < self.fouling_level <= 5:  # mid level organisms
                    if optimal_temps[1] - 1 < self.temp < optimal_temps[1] + 1: # if temperature is in optimal range then do more fouling
                        self.growth += np.random.uniform(0.65,0.80)
                    elif optimal_temps[1] - 5 < self.temp < optimal_temps[1] + 5: # if temperature is in optimal range then do more fouling 
                        self.growth += np.random.uniform(0.35,0.55)
                    else:
                        self.growth += np.random.uniform(0.15,0.35)
                else:  # slime  
                    if optimal_temps[0] - 1 < self.temp < optimal_temps[0] + 1: # if temperature is in optimal range then do more fouling
                        self.growth += np.random.uniform(0.80,1.00)
                    elif optimal_temps[0] - 5 < self.temp < optimal_temps[0] + 5: # if temperature is in optimal range then do more fouling 
                        self.growth += np.random.uniform(0.35,0.55)
                    else:
                        self.growth +=np.random.uniform(0.15,0.35)
                '''
                
                if self.fouling_level > 5:  # large organisms
                    index = 2
                elif 2 < self.fouling_level <= 5:  # mid level organisms
                    index = 1
                else:  # slime  
                    index = 0
                # determine difference from optimal temperature
                temp_diff = abs(self.temp - optimal_temps[index])
                # linear decay for growth 
                temp_growth = 1-(temp_diff)/10
                # ensure growing occurs
                if temp_growth < 0.25 :
                    self.growth += 0.25
                else:
                    self.growth += temp_growth    
                  

            self.fouling_level += self.growth/2             
                      

class HullModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, width, length, max_depth,temp=20,sunlight_angle=50, seed=None):
        super().__init__(seed=seed)
        self.depths = np.zeros((width, length))
        self.areas = np.zeros((width - 1, length - 1))
        self.point_areas = np.zeros((width, length))

        self.width = width
        self.length = length
        self.max_depth = max_depth
        self.temp = temp
        self.large_growth_area, self.mid_growth_area, self.slime_area = 0,0,0
        
        # create  grid
        self.grid = mesa.space.MultiGrid(self.width, self.length, False) 
        # setup agents and calculate depths   
        self.setup()    
        # get sunlight mapping
        self.sunlight_map = self.get_sunlight_map()
        # get total surface area
        self.total_area, _, self.point_areas = self.getsurface_areas()

        # Create data collector for fouling rating 
        self.datacollector = DataCollector(
            model_reporters={"Fouling Rating": self.getFR,
                             "Large Organism Area": self.get_large,
                             "Small Organism Area" : self.get_mid,
                             "Slime Area": self.get_slime
                             },
        )

    def setup(self):
        # Create agents
        # Create x and y coordinates for agents
        for x in range(self.width):
            for y in range(self.length):
                agent = HullAgent(self)
                self.grid.place_agent(agent, (x, y))
                if x==0 or x == (self.width-1) or y == 0 or y == (self.length-1):
                    self.depths[x][y] = 0
                else:
                    self.depths[x][y] = self.max_depth * ((self.cross_profile(x)+self.length_profile(y))/2)


    
    def cross_profile(self, x):
        '''returns a percentage (out of 1) of the maximum depth of the hull'''
        ys = np.linspace(-1.25, 1.25, self.width)
        local_max_height = (0.4*(abs(math.tan(1.25))))
        self.scaled_height = 0.4*abs(math.tan(ys[x]))/local_max_height
        self.scaled_depth = 1-self.scaled_height  # opposite of height
        return self.scaled_depth
    
    def length_profile(self, y):

        '''returns a percentage of the maximum depth of the hull''' 
        xs = np.linspace(-(0.2**(1/8)), (0.2**(1/8)), self.length)
        self.local_depth = (xs[y])**8
        return 1-(self.local_depth/0.2)
    
    def getsurface_areas(self):
        total_area = 0
        point_areas = np.zeros((self.width, self.length))
        point_counts = np.zeros((self.width, self.length))
        for x in range(1, self.width - 1):
            for y in range(1, self.length - 1):
                p1 = np.array([x, y, self.depths[x, y]])
                p2 = np.array([x + 1, y, self.depths[x + 1, y]])
                p3 = np.array([x, y + 1,self.depths[x, y + 1]])
                p4 = np.array([x + 1, y + 1, self.depths[x + 1, y + 1]])
                
                # Two triangular facets per square cell
                area1 = np.linalg.norm(np.cross(p2 - p1, p3 - p1)) / 2
                area2 = np.linalg.norm(np.cross(p4 - p2, p3 - p2)) / 2
                
                cell_area = area1 + area2
                self.areas[x - 1, y - 1] = cell_area
                total_area += cell_area

                # Assign area to each point
                for px, py in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]:
                    point_areas[px, py] += cell_area / 4  # Each point gets 1/4 of surrounding triangles
                    point_counts[px, py] += 1
        
        # Normalize the area per point
        self.point_areas = np.divide(point_areas, point_counts, out=np.zeros_like(point_areas), where=point_counts > 0)
        
        return total_area, self.areas, self.point_areas



    def get_sunlight_map(self):

        #x_col = np.random.choice(np.linspace(self.width/4, 3*self.width/4))
        x_columns = np.arange(0, self.width,1)  # select multiple x-coordinate columns
        sunlight_angle = 50  # sunlight angle in degrees

        # surface angles
        surface_angles = self.calculate_surface_angles(self.depths, x_columns)
        # if sunlight will hit for each column
        light_hits = self.check_sunlight_hit(surface_angles, sunlight_angle)
        # convert to array and fill in last row
        light_hits_arr = np.array(list(light_hits.values()))
        light_hits_arr = np.hstack((light_hits_arr, light_hits_arr[:, [0]]))  # Add last row

        return light_hits_arr

    def light_intensities(self, surface_angles):
        angles = np.array(list(surface_angles.values()))
        intensities = np.ones((self.width,self.length))
        for i in range(self.width):
            for j in range(self.length-1):
                distance_underwater = self.depths[i][j]/np.sin(np.radians(angles[i][j])) 
                intensities[i][j] = self.get_intensity()





    def calculate_surface_angles(self,depths, x_columns):
        """
        calc surface angles along selected x-columns of hull.
        rets:
        dict with x-cols as keys and lists of angles as values.
        """
        length = depths.shape[1] 
        angles_dict = {}

        for x_column in x_columns:
            angles = []
            for y in range(length-1):
                # extract (x, y, z) positions
                z1 = depths[x_column][y]
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

    def check_sunlight_hit(self,surface_angles, sunlight_angle):
        """
        whether sunlight will hit each hull surface based on angles.
        rets:
        dictionary of if light will hit (True) or miss (False).
        """
        hit_results = {}
        for x_column, angles in surface_angles.items():
            hit_results[x_column] = [sunlight_angle > angle for angle in angles]

        return hit_results

    def get_slime(self):
        return self.slime_area
    
    def get_large(self):
        return self.large_growth_area
    
    def get_mid(self):
        return self.mid_growth_area
                
    def getFR(self, return_types=False):
        '''returns the fouling rate of the hull'''
        total_area = self.total_area
        # Create lists to store fouling levels
        fouling_levels = np.zeros_like(self.point_areas)
        # Iterate through all agents on the grid
        for cell_content, pos in self.grid.coord_iter():
            x, y = pos
            if cell_content:  # Ensure there's an agent in the cell
                fouling_levels[x][y] = cell_content[0].fouling_level
        fouling_levels = np.array(fouling_levels)
        self.large_growth_area = np.sum(fouling_levels*self.point_areas > 5)/total_area
        self.mid_growth_area = np.sum((fouling_levels*self.point_areas > 2) & (fouling_levels*self.point_areas <= 5))/total_area
        self.slime_area = np.sum((fouling_levels*self.point_areas > 0) & (fouling_levels*self.point_areas <= 2))/total_area

        if return_types==True:
            return self.large_growth_area, self.mid_growth_area, self.slime_area
        else:
            return 15*self.large_growth_area + 0.5*self.mid_growth_area + 0.2*self.slime_area

            
    def getFR_old(self, return_types=False):
        '''returns the fouling rate of the hull'''
        # Create lists to store fouling levels
        fouling_levels = []
        total_area = self.length*self.width
        # Iterate through all agents on the grid
        for cell_content, pos in self.grid.coord_iter():
            if cell_content:  # Ensure there's an agent in the cell
                fouling_levels.append(cell_content[0].fouling_level) 
        fouling_levels = np.array(fouling_levels)
        self.large_growth_area = np.sum(fouling_levels > 5)/total_area
        self.mid_growth_area = np.sum((fouling_levels > 2) & (fouling_levels <= 5))/total_area
        self.slime_area = np.sum((fouling_levels > 0) & (fouling_levels <= 2))/total_area

        if return_types==True:
            return self.large_growth_area, self.mid_growth_area, self.slime_area
        else:
            return 15*self.large_growth_area + 0.5*self.mid_growth_area + 0.2*self.slime_area


    def sigmoid(self, x):
        'sigmoid distribution for biofouling growth probability'
        x0 = 7
        k = 0.5
        return 1 / (1 + np.exp(-k * (x - x0)))
    

    def step(self):

        self.agents.shuffle_do("grow")    
        self.large_growth_area, self.mid_growth_area, self.slime_area = self.getFR(True)
        #self.agents.shuffle_do("detach")

        self.datacollector.collect(self)


def plot_fouling_classes(large, mid, slime):
    plt.figure()
    time_steps = np.linspace(0,n_steps,n_steps)
    
    plt.plot(time_steps, 100*large,'r-', label='Large Organisms (Eg. Barnacles)')
    plt.plot(time_steps, 100*mid,'g-', label='Small Organisms (Eg. Weed)')
    plt.plot(time_steps, 100*slime,'b-', label='Slime Layer')

    plt.xlabel('Time Steps')
    plt.ylabel('Fouling Coverage (percentage cover over the hull)')
    plt.title('Fouling Coverage Over time')
    plt.legend()
    plt.show()   

def plotly_plot():
    '''Visualise the hull in 3D using Plotly'''
    # Create mesh grid
    depths = model.depths
    width = model.width
    length = model.length
    X, Y = np.meshgrid(np.arange(width), np.arange(length), indexing='ij')
    Z = depths 
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues')])
    fig.update_layout(title='3D Hull Depth Profile', scene=dict(
        xaxis_title='X (Width)',
        yaxis_title='Y (Length)',
        zaxis_title='Depth',
        zaxis=dict(autorange='reversed'),  # Ensures depth is plotted downward
        aspectmode='auto',  # Ensures all axes have the same scale
    ))
    fig.show()
    #return X,Y,Z


def plot_prob_dist():
    '''plot sigmoid distribution for biofouling growth probability'''
    plt.figure()
    xs = np.linspace(0, model.max_depth, 100)
    ys = np.array([model.sigmoid(x) for x in xs])
    plt.plot(xs, ys)    
    plt.show()


# instantiate model
model = HullModel(width=12, length=120, max_depth=9)
total, arr, point_arr = model.getsurface_areas()
print('Area:',total)
print('Area arr:',arr.shape)
print('point_arr', point_arr)

# set parameters for batch_run
params = {"length": 120,
          "width": 12,
          "max_depth": 9,
          "temp": [15,20,25,30,35],
          "sunlight_angle": [25,30,35,40,45,50],
           "seed": 42}

n_steps = 80
n_iter = 3

# run model
results = mesa.batch_run(
    HullModel,
    iterations=n_iter,
    parameters=params,
    max_steps=n_steps,
    data_collection_period=1,
)
# convert results to dataframe
results_df = pd.DataFrame(results)
# plot overall fouling
plt.figure()
g5 = sns.lineplot(data=results_df, x="Step", y="Fouling Rating", hue='sunlight_angle')  
plt.figure()
g1 = sns.lineplot(data=results_df, x="Step", y="Fouling Rating", hue='temp')
plt.xlabel('Time')
plt.ylabel('FR')
plt.title('Fouling rating over time')
plt.legend()
plt.tight_layout()
# plot other fouling types
plt.figure()
g2 = sns.lineplot(data=results_df, x="Step", y="Large Organism Area")
g3 = sns.lineplot(data=results_df, x="Step", y="Small Organism Area")
g4 = sns.lineplot(data=results_df, x="Step", y="Slime Area")
plt.show()
'''
# Run the model and update the heatmap dynamically
n_steps = 2
FRs = np.zeros(n_steps)  # fouling ratings 
large_fouling = np.zeros(n_steps)
medium_fouling = np.zeros(n_steps)
slime_fouling = np.zeros(n_steps)

for step in range(n_steps):
    model.step()
    #FRs[step] = model.getFR(False)
    #large_fouling[step], medium_fouling[step], slime_fouling[step] = model.getFR(True)
    # Initialize agent count grid
    agent_levels = np.zeros((model.length, model.width))
    # Populate the grid with agent counts
    for cell_content, (x, y) in model.grid.coord_iter():
            agent_levels[y][x] = cell_content[0].fouling_level
    # Clear previous plot
    plt.clf()    
    # Draw updated heatmap
    sns.heatmap(agent_levels, cmap="Greens", annot=False, cbar=True,square=False, vmin=0,vmax=8)
    #plt.set_axis_off()
    plt.xlabel('y')
    plt.ylabel('x')
    plt.suptitle('Visual Representation of Biofouling Over Time (Plan View)') 
    plt.title(f"Step {step + 1}")

    # Pause to update plot
    plt.pause(0.1)  # Adjust for desired speed

# plot total biofouling growth over time
#plot_fouling_classes(large_fouling, medium_fouling, slime_fouling)
# plot sigmoid distribution for biofouling growth probability
#plot_prob_dist()

# show hull
#plotly_plot()
'''