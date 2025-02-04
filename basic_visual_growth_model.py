import numpy as np
import random
import matplotlib.pyplot as plt
import mesa
import seaborn as sns

class HullAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        super().__init__(model)
        self.fouling_level = 0
        self.temp = 20

    def grow(self):
        '''function deciding if the level of biofouling should increase'''
        # Currently growth occurs when a random number is >7 (this will be changed to incorperate actual parameters like SST)
        rand_var = random.randint(0,10)
        if rand_var > 7:
            self.fouling_level+=1      

    def detach(self):
        '''function capturing the case when the organisms detach due to hydrodynamic forces'''
        rand_var = random.randint(0,100)
        # calculations for the speed of the fluid flow in the area could be used to determine if the organism detatches
        if rand_var > 85:
            self.fouling_level-+1           



class HullModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, width, height, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.grid = mesa.space.MultiGrid(width, height, True)

        # Create agents
        # Create x and y coordinates for agents
        for x in range(width):
            for y in range(height):
                agent = HullAgent(self)
                self.grid.place_agent(agent, (x, y))

    def step(self):
        self.agents.shuffle_do("grow")    

model = HullModel(100, 50, 30)
# Run the model and update the heatmap dynamically
for step in range(10):
    model.step()

    # Initialize agent count grid
    agent_counts = np.zeros((model.grid.height, model.grid.width))
    agent_levels = np.zeros((model.grid.height, model.grid.width))
    # Populate the grid with agent counts
    for cell_content, (x, y) in model.grid.coord_iter():
        #agent_counts[y][x] = len(cell_content)  # Ensure proper indexing
        agent_levels[y][x] = cell_content[0].fouling_level
    # Clear previous plot
    plt.clf()
    
    # Draw updated heatmap
    sns.heatmap(agent_levels, cmap="Greens", annot=False, cbar=True, square=True, cbar_kws={"orientation": "horizontal"})
    plt.title(f"Step {step + 1}")

    # Pause to update plot
    plt.pause(0.5)  # Adjust for desired speed

# Ensure the final frame remains visible
#plt.show()



'''        
ALIVE = 1
n = 15
#create grid
def create_grid(n):
    grid = np.zeros((n,n), dtype=np.int8)
    return grid
    
   
#update grid
def update_grid(grid, n):

    for i in range(n):
        for j in range(n):
            neighbours_list = []

            neighbours_list.append(grid[i][(j+1)%n])
            neighbours_list.append(grid[i][(j-1)%n])
            neighbours_list.append(grid[(i+1)%n][(j)%n])
            neighbours_list.append(grid[(i-1)%n][(j)%n])
            neighbours_list.append(grid[(i+1)%n][(j+1)%n])
            neighbours_list.append(grid[(i+1)%n][(j-1)%n])
            neighbours_list.append(grid[(i-1)%n][(j-1)%n])
            neighbours_list.append(grid[(i-1)%n][(j-1)%n])
            
            grid[i][j] = update_value(neighbours_list,grid[i][j])
    

def update_value(neighbours, current_element):
    #new value
    new_value = current_element
    #count 1s and 0s
    ones = neighbours.count(1)
    zeros = neighbours.count(0)

    if current_element == 1:   
        if ones <= 1: # loneliness
            new_value = 0
        elif ones >= 4: # starvation
            new_value = 0
        elif ones == 2 or ones == 3 : # survival rule
            new_value = 1        
    elif current_element == 0:
        if ones == 3: # birth rule
            new_value = 1
      
    return new_value        
            
        
def run_tests():
    
    assert update_value([1,1,1,1,1,1,1,1,1],0)==0, "test 1"
    assert update_value([1,1,1,1,1,1,1,1,0],0)==0, "test 2"
    assert update_value([1,1,1,1,1,1,1,0,0],0)==0, "test 3"
    assert update_value([1,1,1,1,1,1,0,0,0],0)==0, "test 4"
    assert update_value([1,1,1,1,1,0,0,0,0],0)==0, "test 5"
    assert update_value([1,1,1,1,0,0,0,0,0],0)==0, "test 6"
    assert update_value([1,1,1,0,0,0,0,0,0],0)==1, "test 7"
    assert update_value([1,1,0,0,0,0,0,0,0],0)==0, "test 8"
    assert update_value([1,0,0,0,0,0,0,0,0],0)==0, "test 9"
    assert update_value([0,0,0,0,0,0,0,0,0],0)==0, "test 10"
        
    assert update_value([1,1,1,1,1,1,1,1,1],1)==0, "test 11"
    assert update_value([1,1,1,1,1,1,1,1,0],1)==0, "test 12"
    assert update_value([1,1,1,1,1,1,1,0,0],1)==0, "test 13"
    assert update_value([1,1,1,1,1,1,0,0,0],1)==0, "test 14"
    assert update_value([1,1,1,1,1,0,0,0,0],1)==0, "test 15"
    assert update_value([1,1,1,1,0,0,0,0,0],1)==0, "test 16"
    assert update_value([1,1,1,0,0,0,0,0,0],1)==1, "test 17"
    assert update_value([1,1,0,0,0,0,0,0,0],1)==1, "test 18"
    assert update_value([1,0,0,0,0,0,0,0,0],1)==0, "test 19"
    assert update_value([0,0,0,0,0,0,0,0,0],1)==0, "test 20"





grid = create_grid(n)
print(grid) 

update_grid(grid, n)
print("new grid")
print(grid) 


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(211)
ax1.set_axis_off()
image1 = np.zeros((n, n), dtype = np.int8)

#Copy CA_grid into the image array
image1[:, :] = grid

#Display the image
ax1.imshow(image1, interpolation='none', cmap='RdPu')
plt.pause(10)
'''