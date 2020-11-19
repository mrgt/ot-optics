#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from geometry import *
from ot_metasurfaces import *
from gje import *
import numpy as np
import pypower
import scipy.optimize as opt
import matplotlib.pyplot as plt


def check_gradient(f,gradf,x0):
    N = len(x0)
    gg = np.zeros((N,N))
    for i in range(N):
        eps = 1e-6
        e = np.zeros(N)
        e[i] = eps
        gg[i,:] = (f(x0+e) - f(x0-e))/(2*eps)
    print('erreur numérique dans le calcul du gradient: %g (doit être petit)' % np.linalg.norm(gradf(x0)-gg))
    print(gradf(x0))
    print(gg)



#---------------- bloc de test calcul de gradient-----------------------------

N = 5
Y = np.array([[ 0.62074898, -0.00207409],
[ 0.9568784 , -0.06380766],
[ 0.69871951,  0.87968212],
[-0.87548466, -0.91384685],
[-0.41198645, -0.76459244]])

H = lambda psi: compute_H_DH_from_mobius(Y, psi, make_mobius(Y, psi, k=100))[0]
DH = lambda psi: compute_H_DH_from_mobius(Y, psi, make_mobius(Y, psi, k=100))[1]

psi = 0.1 *(np.ones(N) + 0.01 * np.array([0.1, 0.6, -0.1, 1.7, -0.6]))
print("twist = ",test_twist(Y, psi))
np.set_printoptions(2)

psi = 0.1*np.ones(N)
check_gradient(H,DH,psi)

M = make_mobius(Y, psi, k=100)

h,dh = compute_H_DH_from_mobius(Y, psi, M)
dh2 = dh.copy()
dh2[0,0] += 1
print("H = ",h)
print("DH = ", dh)
print("det(DH) =", np.linalg.det(dh), "det(DH + e_1,1) =", np.linalg.det(dh2))
#--------------------------------------------------------------------------------




# N = 5
# Y = np.random.rand(N,2) 
# test_twist(Y, np.ones(N))
# psi = solve_gje(Y, np.ones(N))
# plot_laguerre(M)

# plt.show()
