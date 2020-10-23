from geometry import *


def areas_of_power_diagram_quadric_intersection(X,w,quad):
    N = len(w)
    cells = pypower.power_diagram(X,w)
    areas = np.zeros(N)
    for i in range(N):
        curves = intersect_cell_with_quadric(X,cells[i],quad,20)
        if curves is None:
            print("empty cell {}".format(i))
            continue
        # compute area of cell i by integrating over the boundary
        # j is the number of the adjacent cell (negative number if adjacency 
        # is with the boundary)
        for j,curvesj in curves.items():
            for crv in curvesj:
                crv = np.array(crv)
                for k in range(crv.shape[0]-1):
                    a = crv[k,0:2]
                    b = crv[k+1,0:2]
                    areas[i] += np.linalg.det(np.array([a,b]))/2
    return areas
 
N = 10
quad = Quadric(np.diag([-1,-1,0]), np.array([0,0,1]), 1)
X = np.random.rand(N,3)
w = np.ones(N)
w[0] = 3
print(areas_of_power_diagram_quadric_intersection(X,w,quad))
