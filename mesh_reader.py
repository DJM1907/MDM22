import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
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

mesh = np.array([X,Y,Z]).transpose()

pl = pv.Plotter()
pl.add_mesh(mesh, color='black')
pl.show()