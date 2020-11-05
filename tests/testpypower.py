import numpy as np
import pypower
import geometry
A = np.array([[ 0, 0, 1],
              [ 0, 0,-1],
              [ 0, 1, 0],
              [ 0,-1, 0],
              [ 1, 0, 0],
              [-1, 0, 0]])
b = np.array([100,100,1,1,1,1])
Y = np.random.rand(10,3)
psi = np.ones(10)
cells = pypower.power_diagram(Y,psi,A,b)
vols = [geometry.volume_of_cell(c) for c in cells]
print(vols)
print("s={}".format(sum(vols)))
