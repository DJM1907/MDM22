import numpy as np
import pyvista as pv

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
grid.plot()

# Improve model by first mirroring so it's a full hull
# then work out how to make it into a full mesh instead of a point cloud
