from scipy.spatial import Delaunay
import numpy as np
import random
import matplotlib.pyplot as plt
import mesa
import math
import seaborn as sns
from mesa.experimental.cell_space.property_layer import PropertyLayer
import plotly.graph_objects as go
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


def doublet_velocity(strength, x_doublet, y_doublet, x, y):
    """Calculate velocity due to a doublet at a specific point (x_doublet, y_doublet)"""
    r_squared = (x - x_doublet)**2 + (y - y_doublet)**2
    u = -strength / (2 * np.pi) * ((x - x_doublet)**2 - (y - y_doublet)**2) / r_squared**2
    v = -strength / (2 * np.pi) * (2 * (x - x_doublet) * (y - y_doublet)) / r_squared**2
    return u, v

class HullAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        super().__init__(model)
        self.fouling_level = 0
        self.temp = 20
        self.depth = model.depths[self.pos]
        # initialise the velocity of the fluid flow 
        self.velocity = model.ship_speed


    def grow(self):
        '''function deciding if the level of biofouling should increase'''
        # sigmoid distribution used to determine the probability of biofouling growth
        rand_var = np.random.uniform(0,1)
        x,y = self.pos
        self.depth = self.model.depths[x][y]

        '''# old probability distribution:
        if self.depth < 5:
            threshold = 49
        elif self.depth < 20:
            threshold = 60
        elif self.depth < 30:
            threshold = 65    
        else:
            threshold = 85       
        '''
        # sigmoid distribution for biofouling growth probability
        growth_prob = self.model.sigmoid(self.depth)

        if rand_var > growth_prob:
            self.fouling_level+=1      

    

    def detach(self):
        '''function capturing the case when the organisms detach due to hydrodynamic forces'''
        rand_var = random.randint(0,100)
        # calculations for the speed of the fluid flow in the area could be used to determine if the organism detatches
        if rand_var > 95 and self.fouling_level>2:  # only mid level biofouling and above can detach
            self.fouling_level = 0         



class HullModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, width, length, max_depth,ship_speed=2.0, seed=None):
        super().__init__(seed=seed)
        self.depths = np.zeros((width, length))
    
        self.width = width
        self.length = length
        self.max_depth = max_depth
        self.ship_speed = ship_speed
        self.grid = mesa.space.MultiGrid(self.width, self.length, False)
        # create water grid
        #self.grid = mesa.space.ContinuousSpace(self.width, self.length, False)

        self.setup()
        #self.KCS_setup()
        

    def KCS_setup(self):
        '''---------NEEDS CHANGING---------'''
        self.depths = Z
        self.max_depth = max(Z)  
        self.width = max(Y)   
        self.length = max(X)   

        plt.scatter(Y,Z, s=1)
        plt.xlabel('y')
        plt.show()
        # Create agents
        for x in range(self.width):
            print('x')
            for y in range(self.length):
                 agent = HullAgent(self)
                 self.grid.place_agent(agent, (y, x))


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
        '''if xs[y] < 0.0375:
            self.local_depth = 4*(xs[y]-0.0375) + 0.15
        elif xs[y] > 1.0375:
            self.local_depth = -3*(xs[y]-1.0375) + 0.15
        else:
            self.local_depth = 0.15'''


        return 1-(self.local_depth/0.2)

    def getFR(self):
        '''returns the fouling rate of the hull'''
        # Create lists to store fouling levels
        fouling_levels = []

        # Iterate through all agents on the grid
        for cell_content, pos in self.grid.coord_iter():
            if cell_content:  # Ensure there's an agent in the cell
                fouling_levels.append(cell_content[0].fouling_level) 
        fouling_levels = np.array(fouling_levels)
        large_growth_area = np.sum(fouling_levels > 4)
        mid_growth_area = np.sum((fouling_levels > 2) & (fouling_levels < 5))
        slime_area = np.sum((fouling_levels > 0) & (fouling_levels < 3))


        return 15*large_growth_area + 0.5*mid_growth_area + 0.2*slime_area

    def sigmoid(self, x):
        'sigmoid distribution for biofouling growth probability'
        x0 = 7
        k = 0.5
        return 1 / (1 + np.exp(-k * (x - x0)))
    

    def step(self):
        self.agents.shuffle_do("grow")    
        self.agents.shuffle_do("detach")




def plotly_plot():
    '''Visualise the hull in 3D using Plotly'''
    # Create mesh grid
    depths = model.depths
    width = model.width
    length = model.length
    X, Y = np.meshgrid(np.arange(width), np.arange(length), indexing='ij')
    Z = depths  # Negative for proper visualization
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues')])
    fig.update_layout(title='3D Hull Depth Profile', scene=dict(
        xaxis_title='X (Width)',
        yaxis_title='Y (Length)',
        zaxis_title='Depth',
        zaxis=dict(autorange='reversed'),  # Ensures depth is plotted downward
        aspectmode='auto',  # Ensures all axes have the same scale
    ))
    return X,Y,Z
    fig.show()



def plot_prob_dist():
    '''plot sigmoid distribution for biofouling growth probability'''
    plt.figure()
    xs = np.linspace(0, model.max_depth, 100)
    ys = np.array([model.sigmoid(x) for x in xs])
    plt.plot(xs, ys)    
    plt.show()

def plot_FR():
    '''plot the fouling rating of the hull over time'''
    plt.figure()
    time_steps = np.linspace(0,n_steps,n_steps)
    plt.plot(time_steps, FRs)
    plt.xlabel('Time Steps')
    plt.ylabel('Fouling Rating')
    plt.title('Fouling Rating over Time')
    plt.show()    

def KCS_surface():
    points = np.zeros((24604, 3))  # Adjust shape (Width, Length, Depth)
    #points[:, 2] = -np.abs(points[:, 2])  # Keep hull below waterline
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = Z

    tri = Delaunay(points[:, :2])  # Triangulate hull base

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri.simplices, cmap="copper")

    ax.set_title("3D Triangulated Hull")
    plt.show()


X, Y, Z = KCS_hull()


model = HullModel(60, 120, 15)
# Run the model and update the heatmap dynamically
n_steps = 20
FRs = np.zeros(n_steps)  # fouling ratings 
for step in range(n_steps):
    model.step()
    FRs[step] = model.getFR()
    # Initialize agent count grid
    agent_levels = np.zeros((model.length, model.width))
    # Populate the grid with agent counts
    for cell_content, (x, y) in model.grid.coord_iter():
        agent_levels[y][x] = cell_content[0].fouling_level
    # Clear previous plot
    plt.clf()    
    # Draw updated heatmap
    sns.heatmap(agent_levels, cmap="Greens", annot=False, cbar=True,square=True, vmin=0,vmax=10)
    plt.title(f"Step {step + 1}")

    # Pause to update plot
    plt.pause(0.1)  # Adjust for desired speed

# plot velocities over the hull for each agent
'''velocities = np.zeros((model.length, model.width))
for cell_content, (x, y) in model.grid.coord_iter():
    velocities[y][x] = cell_content[0].velocity
'''

#plt.figure()
#sns.heatmap(velocities, cmap="coolwarm", annot=False, cbar=True, square=True)
#plt.show()


# plot total biofouling growth over time
#plot_FR()


# plot sigmoid distribution for biofouling growth probability
#plot_prob_dist()


# visualise hull with Plotly library
# comment this stuff out below
x_dat, y_dat, z_dat = plotly_plot()
print(x_dat)
hull_dat = np.zeros((len(x_dat), len(y_dat), len(z_dat)))
hull_dat[:,0], hull_dat[:,1], hull_dat[:,2] = x_dat, y_dat, z_dat

print(hull_dat)

# visualise KCS mesh
#KCS_surface()
#plt.show()

