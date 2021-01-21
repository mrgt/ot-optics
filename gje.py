#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from geometry import *
from ot_metasurfaces import *
import numpy as np
import pypower
import scipy.optimize as opt
import matplotlib.pyplot as plt

"""
This code is dedicated to solve the Near Field parallel reflector problem using a damped newton algorithm,
The source is fixed, it is the square [-1,1]^2 with a uniform measure.
"""


def mob_to_pow(y, psi):
    """Return the 3d point c and distance r of the 3d 
    power diagram associated to the 2d mobius diagram of center y and distance k"""
    y1 = y[0]
    y2 = y[1]
    c = np.array([psi * y1 / 2, psi * y2 / 2, -psi / 4])
    r = psi**2 / 16 + 1 / (2 * psi) + (psi**2 / 4 - psi / 2)*(y1**2 + y2**2)
    return [c,r]

def G(x,y,v):
    """Generating function for NF-par"""
    return 1 / (2*v) - (v/2)*np.linalg.norm(x - y)**2

def make_mobius(Y, psi, k = 20):
    """return the Mobius diagram associated to the cost G defined above"""
    quad = Quadric(np.array([-1,-1,0]), np.array([0,0,1]), 0)
    N = len(Y)
    X = np.zeros((N,3))
    w = np.zeros(N)
    for i in range(N):
        v = mob_to_pow(Y[i], psi[i])
        X[i] = v[0]
        w[i] = v[1]
    cells = make_power(X,w)
    curves = []
    for i in range(N):
        c = intersect_cell_with_quadric(X,cells[i],quad,k)
        curves.append(c)
    return curves



def g(x,y_i,y_j,psi_i, psi_j):
    """Function to integrate for computation of DH_i/Dpsi_j of NF-par problem"""
    return (1 / (2 * psi_j**2) + np.linalg.norm((x - y_j)**2/2) )/np.linalg.norm(psi_i * y_i - psi_j * y_j + (psi_j - psi_i)*x)


def test_twist(Y, psi):
    """ Test the twist assumption on psi for NF-parallel reflector, which is psi_i < gamma = inf 1 / ||x - y|| """
    x = np.array([[1,1], [-1,1],[-1,-1], [1, -1]])
    N = len(Y)
    Y1 = np.copy(Y)
    Y2 = np.copy(Y)
    Y3 = np.copy(Y)
    Y4 = np.copy(Y)
    for i in range(N):
        Y1[i] = Y1[i] - x[0]
        Y2[i] = Y2[i] - x[1]
        Y3[i] = Y3[i] - x[2]
        Y4[i] = Y4[i] - x[3]
    m = 1e-15
    for i in range(N):
        ma = max(np.linalg.norm(Y1[i]), np.linalg.norm(Y2[i]), np.linalg.norm(Y3[i]), np.linalg.norm(Y4[i]))
        if ma > m:
            m = ma
    m = 1/m
    for i in range(N):
        if psi[i] > m:
            return False
    return True



def compute_H_DH_from_mobius(Y, psi, allcurves):
    """Computes H and DH from a mobius diagram stored in allcurves"""
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
                    #x = (a + b) / 2
                    #hij = length * g(x, Y[i], Y[j], psi[i], psi[j])
                    hij = length * (g(a, Y[i], Y[j], psi[i], psi[j]) + g(b, Y[i], Y[j], psi[i], psi[j]))/2
                    DH[i,j] += hij
                    DH[i,i] -= hij
    return areas, DH

def compute_H_DH(Y, psi):
    return compute_H_DH_from_mobius(Y, psi, make_mobius(Y, psi))

def solve_gje(Y, psi, nu = [], stoperr=1e-6, verb = True, threshold = 1e-18, maxit = 50):
    """Resolution of Generated Jacobian equation using damped Newton algorithm"""
    N = len(psi)
    if nu == []:
        nu = (4/N) * np.ones(N) #if no measure where given we chose it to be uniform
    if len(nu) != N or len(Y) != N:
        print("ERROR : Y, psi and nu must be arrays of the same size")
        return [[]]
    h,dh = compute_H_DH(Y, psi)
    delta = min(min(h), min(nu))/2
    err = np.linalg.norm(h - nu, 1)
    it = 0
    iterations = [0]
    errors = [err]
    if verb:
        print("it={}, error={}, delta={} \n".format(it,err,delta))
    while err > stoperr:
        if it > maxit:
            print("Newton failed to converge \n")
            print("min height = ", min_height(Y, psi))
            return [[]]
        it += 1
        t = 1
        h = h[0:N-1] # les 4 lignes suivantes servent à supprimer la derniere ligne et colonne du systeme en fixant u[N-1] = 0
        dh = dh[0:N-1, 0:N-1]
        u = np.zeros(N)
        u[0:N-1] = np.linalg.solve(dh, h - nu[0:N-1])
        psi -=  u
        h,dh = compute_H_DH(Y, psi)
        newerr = np.linalg.norm(h - nu, 1)
        while min(h) < delta or newerr > err:
            t /= 2
            psi += t * u
            h,dh = compute_H_DH(Y, psi)
            newerr = np.linalg.norm(h - nu, 1)
            if t < threshold:
                print("Newton failed to converge \n")
                print("min height = ", min_height(Y, psi))
                return [[]]
        err = newerr
        if verb:
            print("it={}, error={}, tau={} \n".format(it,err,t))
        errors.append(err)
        iterations.append(it)
    return [psi, iterations, errors]

