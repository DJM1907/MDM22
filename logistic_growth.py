import numpy as np
import matplotlib.pyplot as plt

def logistic(time):
    max_val = 10
    steepness = 0.35
    t_0 = 50
    return (max_val/(2+(np.e**(-steepness*(time-t_0)))))



ts = np.linspace(0,100, 1000)
ys = logistic(ts)
plt.plot(ts,ys)
plt.show()