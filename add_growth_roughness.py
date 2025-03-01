import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
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

X_bow,Y_bow,Z_bow = read_kcs_grid("kcs_bow2.dat")
X_stn,Y_stn,Z_stn = read_kcs_grid("kcs_stn2.dat")

X = np.concatenate((X_bow,X_stn))
Y = np.concatenate((Y_bow,Y_stn))
Z = np.concatenate((Z_bow,Z_stn))

mesh1 = np.array([X,Y,Z]).transpose()
mesh2 = np.array([X,-Y,Z]).transpose()
mesh = np.concatenate((mesh1,mesh2))

# Plotting as a surface using delaunay triangulation
# cloud = pv.PolyData(mesh)
# surf = cloud.delaunay_2d()
# surf.plot()

# Plotting as a surface using pyvista stuctured surfaces
#X,Y,Z = np.meshgrid(X,Y,Z)
X = np.concatenate((X,X))
Y = np.concatenate((Y,-Y))
Z = np.concatenate((Z,Z))
grid = pv.StructuredGrid(X,Y,Z)
#grid.plot()

n_points = len(X)
# fouling ratings denote how rough the surface should be 
fouling_ratings = np.random.randint(0,8, n_points)
# Parameters for the logistic distribution
mu = 3      # Mean (center of the sigmoid)
scale = 4   # Spread of the distribution

# Generate 1000 random numbers
random_numbers = np.random.logistic(loc=mu, scale=scale, size=n_points)

# Scale and shift to the desired range [0, 8]
min_val = 0
max_val = 8
scaled_numbers = min_val + (random_numbers - np.min(random_numbers)) * (max_val - min_val) / (np.max(random_numbers) - np.min(random_numbers))

# Plot the histogram of scaled numbers
plt.hist(scaled_numbers, bins=50, density=True, alpha=0.6, color='b')

plt.title("Fouling Ratings in Range [0, 8] Sampled from Sigmoid-Like Distribution")
#plt.show()

fouling_noise  = 0.0001* scaled_numbers
coords = [X, Y, Z]
for i in range(n_points):
    ind = np.random.randint(0,3)
    if coords[ind][i] > 0:
        coords[ind][i]+= fouling_noise[i]
    else:
        coords[ind][i]-= fouling_noise[i]


X2, Y2, Z2 = coords[0], coords[1], coords[2]
grid = pv.StructuredGrid(X2,Y2,Z2)
grid.plot()