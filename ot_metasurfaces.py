from geometry import *
import numpy as np
import pypower
import scipy.optimize as opt
import matplotlib.pyplot as plt

def make_power(X,w):
    A = np.array([[ 0, 0, 1],
                  [ 0, 0,-1],
                  [ 0, 1, 0],
                  [ 0,-1, 0],
                  [ 1, 0, 0],
                  [-1, 0, 0]])
    b = np.array([100,100,1,1,1,1])
    return pypower.power_diagram(X,w,A,b)

# X=(x,y,z) is above the graph of (x,y) -> sqrt(|(x,y)-p|^2 + beta**2) + b_i
# <=> z >= sqrt((x-px)^2 + (y-py)^2 + beta**2) + b_i
# <=> (z-b_i)^2 >= (x-px)^2 + (y-py)^2 + beta**2 (AND z>=b_i...)
# <=> <Q(X-R)|(X-R)> - beta**2 >= 0 and z-b_i>=0
# with Q = diag(-1,-1,1), R = (px,py,bi)
# <=> <QX|X> - 2<QR|X> + <QR|R> - beta**2 >= 0

def make_hyperboloid(p, bi, beta):
    Q = np.array([-1,-1,1])
    R = np.array([p[0],p[1],bi])
    return Quadric(Q,b=-2*Q*R,c=np.dot(Q*R,R) - beta**2,
                   truncating_plane=True,p=np.array([0,0,1]),d=-bi)

def make_laguerre(Y, psi, beta, k=20):
    N = len(Y)
    X = np.zeros((N,3))
    X[:,0:2] = Y
    X[:,2] = -psi
    w = 2*psi**2
    
    cells = make_power(X,w)
    curves = []
    for i in range(N):
        quad = make_hyperboloid(Y[i], psi[i], beta)
        c = intersect_cell_with_quadric(X,cells[i],quad,k)
        curves.append(c)
    return curves

def plot_laguerre_cell(curves):
    area = 0
    for j,curvesj in curves.items():
        for crv in curvesj:
            crv = np.array(crv)
            for i in range(crv.shape[0]-1):
                a = crv[i,0:2]
                b = crv[i+1,0:2]
                area += np.linalg.det(np.array([a,b]))/2
            plt.plot(crv[:,0], crv[:,1], 'k')
    return area

def plot_laguerre(allcurves):
    areas = []
    for curves in allcurves:
        area = plot_laguerre_cell(curves)
        areas.append(area)
    return areas

from numba import jit

def nabla_y_cost(x,y,beta):
    return (y-x)/ np.sqrt(np.linalg.norm(x-y)**2 + beta**2)

def compute_H_DH_from_laguerre(Y, beta, allcurves):
    N = len(allcurves)
    areas = np.zeros(N)
    DH = np.zeros((N,N))
    for i in range(N):
        curves = allcurves[i]
        for j,curvesj in curves.items():
            for crv in curvesj:
                crv = np.array(crv)
                for k in range(crv.shape[0]-1):
                    a = crv[k,0:2]
                    b = crv[k+1,0:2]
                    areas[i] += np.linalg.det(np.array([a,b]))/2
                    if j<0:
                        continue
                    length  = np.linalg.norm(b-a)
                    x = (a+b)/2
                    hij = length/np.linalg.norm(nabla_y_cost(x,Y[i],beta) - nabla_y_cost(x,Y[j],beta))
                    DH[i,j] += hij
                    DH[i,i] -= hij
    return np.array(areas), DH

def solve_ot(H_DH, psi, stoperr=1e-8):
    N = len(psi)
    h,dh = H_DH(psi)
    it = 0
    while True:
        dh = dh[0:N-1,0:N-1]
        h = h[0:N-1]
        d = np.zeros(N)
        d[0:N-1] = -np.linalg.solve(dh, h - 4/N)
        err = np.linalg.norm(h - 4/N)
        print("it={}, error={}".format(it,err))
        if err <= stoperr:
            return psi
        t = 1
        d = d - min(d)
        psi0 = psi.copy()
        while True:
            psi = psi0 + t*d
            h,dh = H_DH(psi)
            if min(h) > 0:
                print("t=%g"%t)
                break
            t = t/2
        it += 1
    return psi
