import numpy as np
import random
import matplotlib.pyplot as plt
import mesa
import math
import seaborn as sns
from mesa.experimental.cell_space.property_layer import PropertyLayer
import plotly.graph_objects as go



class HullAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        super().__init__(model)
        self.fouling_level = 0
        self.temp = 20
        self.depth = model.depths[self.pos]

    def grow(self):
        '''function deciding if the level of biofouling should increase'''
        # Currently growth occurs when a random number is >7 (this will be changed to incorperate actual parameters like SST and sunlight)
        rand_var = random.randint(0,100)
        x,y = self.pos
        self.depth = model.depths[x][y]
        if self.depth < 5:
            threshold = 49
        elif self.depth < 20:
            threshold = 60
        elif self.depth < 30:
            threshold = 65    
        else:
            threshold = 85       

        if rand_var > threshold:
            self.fouling_level+=1      
        

    def detach(self):
        '''function capturing the case when the organisms detach due to hydrodynamic forces'''
        rand_var = random.randint(0,100)
        # calculations for the speed of the fluid flow in the area could be used to determine if the organism detatches
        if rand_var > 90:
            self.fouling_level = 0         



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
                agent = HullAgent(self)
                self.grid.place_agent(agent, (x, y))
                if x==0 or x == (self.width-1) or y == 0 or y == (self.length-1):
                    self.depths[x][y] = 0
                else:

                    self.depths[x][y] = self.max_depth * ((self.cross_profile(x)+self.length_profile(y))/2)
                #self.depths[x][y] = self.max_depth * (self.length_profile(y))
    
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

    def step(self):
        self.agents.shuffle_do("grow")    
        self.agents.shuffle_do("detach")

model = HullModel(32, 230, 19)
# Run the model and update the heatmap dynamically
for step in range(1):
    model.step()

    # Initialize agent count grid
    #agent_counts = np.zeros((model.length, model.width))
    agent_levels = np.zeros((model.length, model.width))
    # Populate the grid with agent counts
    for cell_content, (x, y) in model.grid.coord_iter():
        #agent_counts[y][x] = len(cell_content)  # Ensure proper indexing
        agent_levels[y][x] = cell_content[0].fouling_level
    # Clear previous plot
    plt.clf()    
    # Draw updated heatmap
    sns.heatmap(agent_levels, cmap="Greens", annot=False, cbar=True,square=True, cbar_kws={"orientation": "horizontal"}, vmin=0,vmax=10)
    plt.title(f"Step {step + 1}")

    # Pause to update plot
    plt.pause(0.2)  # Adjust for desired speed

#plt.figure()
#sns.heatmap(model.depths, cmap="Greens", annot=False, cbar=True, square=False, cbar_kws={"orientation": "horizontal"})


# Create mesh grid
depths = model.depths
width = model.width
length = model.length
X, Y = np.meshgrid(np.arange(width), np.arange(length), indexing='ij')
Z = depths  # Negative for proper visualization

# Plot with Plotly
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues')])
fig.update_layout(title='3D Hull Depth Profile', scene=dict(
    xaxis_title='X (Width)',
    yaxis_title='Y (Length)',
    zaxis_title='Depth',
    zaxis=dict(autorange='reversed'),  # Ensures depth is plotted downward
    aspectmode='auto',  # Ensures all axes have the same scale
))

fig.show()