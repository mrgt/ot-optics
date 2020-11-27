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


def plot_random_mobius(N = 10, k = 20):
	"""Trace un diagramme de Mobius de N cellules de centres aléatoires dans le carré [-1,1]^2"""
	Y = 2 * (np.random.rand(N,2) -0.5) 
	psi = 5 * np.ones(N) + 5 * np.random.rand(N) #Twist not satisfied but not important for a single diagram
	M = make_mobius(Y, psi, k)
	plot_laguerre(M)
	return

def compare_mobius_naive(N = 10,  gridpoints = 50, k = 20):
	"""Trace le diagram de mobius et verifie que tous les points d'une cellule calculés naivement sont bien dedans
	N est le nombre de cellules, gridpoints et le nombre de points dans la discretisation de chaque axe de du carré"""
	Y = 2 * (np.random.rand(N,2) -0.5) 
	psi =  np.ones(N) + 5 * np.random.rand(N) #Twist not satisfied but not import for a single diagram
	X = [[-1 + 2 * i / gridpoints , -1 + 2 * j / gridpoints] for j in range(gridpoints) for i in range(gridpoints)]
	Lag_i = []
	M = make_mobius(Y, psi, k)
	plot_laguerre(M)
	i = np.random.randint(0, N)
	for a in range(gridpoints**2):
			x = X[a]
			tab = [G(x, Y[k], psi[k]) for k in range(N)]
			if i == np.argmax(tab):
				Lag_i.append(x)
	for x in Lag_i:
		plt.plot(x[0], x[1], 'o', color = 'r')
	return

def uniform_circle(N, ray = 3):
	"""return an array of points uniformly distributed on the circle of center (0,0) and ray 5"""
	pi = np.pi
	return np.array([[ray * np.cos(2 * k * pi / N) , ray * np.sin(2 * k * pi / N )] for k in range(N)])

def random_around_circle(N, ray = 1,  diff = 0.3):
	"""N random points between circles of rays ray - diff and ray + diff """
	pi = np.pi
	angles = 2 * pi * np.random.rand(N)
	rays = ray + 2 * diff * (np.random.rand(N) - 0.5)
	return np.transpose([rays * np.cos(angles), rays * np.sin(angles)])

def koch_iter(L, n):
	if n == 0:
		return L
	N = len(L)
	R = np.zeros((4*N,2))
	for i in range(len(L)):
		t = np.sqrt(3)/2
		a = L[i]
		e = L[(i + 1) % N]
		b = (1/3) * a + (2/3) * e
		d = (2/3) * a + (1/3) * e
		c = (b + d)/2
		c[0] += t * (d[1] - b[1])
		c[1] += t * (b[0] - d[0])
		R[4 * i] = a
		R[4 * i + 1] = d
		R[4 * i + 2] = c
		R[4 * i + 3] = b
	return koch_iter(R, n-1)

def koch(n):
	t = np.sqrt(3)/2
	a = np.array([1.2, - 0.8])
	c = np.array([3, - 0.8])
	b = (a + c) /2
	b[0] += t * (a[1] - c[1])
	b[1] -= t * (a[0] - c[0])
	L = np.array([a,b,c])
	return koch_iter(L,n)


def test_newton_koch(n = 2, k = 20):
	"""test de newton avec mesure cible nu uniforme sur le flocon de koch après n itérations"""
	Y = koch(n)
	N = len(Y)
	psi = 0.1 * np.ones(N) 
	print("twist = ", test_twist(Y, psi))
	M0 = make_mobius(Y, psi, k)
	psi = solve_gje(Y, psi)
	M = make_mobius(Y, psi, k)
	compare_mobius(M0,M,Y)
	return psi

def test_newton_uniform(N = 10, k = 20):
	"""test de newton avec mesure cible nu uniforme sur N points"""
	Y = (np.random.rand(N,2))
	psi = 0.1 * np.ones(N) 
	#nu = np.random.rand(N) #Random target measure
	#nu =  4 * nu / np.linalg.norm(nu, 1)
	#print("nu =", nu, "\n")
	print("twist = ", test_twist(Y, psi))
	M0 = make_mobius(Y, psi, k)
	psi = solve_gje(Y, psi)
	M = make_mobius(Y, psi, k)
	compare_mobius(M0,M,Y)
	return psi

def test_newton_random(N = 10, k = 20):
	"""test de newton avec mesure cible nu aleatoire sur N points"""
	Y =  (np.random.rand(N,2) )
	psi =  0.3 * np.ones(N)
	nu = np.random.rand(N) #Random target measure
	nu =  4 * nu / np.linalg.norm(nu, 1)
	print("nu =", nu, "\n")
	print("twist = ", test_twist(Y, psi))
	M0 = make_mobius(Y, psi, k)
	psi = solve_gje(Y, psi, nu)
	M = make_mobius(Y, psi, k)
	compare_mobius(M0,M,Y)
	return psi

def compare_mobius(M,N,Y=[]):
	"""Plot the two mobius diagram M and N"""
	fig, axs = plt.subplots(2,1)
	axs[0].set_title("Initial diagram")
	axs[1].set_title("Computed diagram")
	plt.axes(axs[0])
	plt.axis('equal')
	plot_laguerre(M)
	if len(Y) > 0:
		plt.plot(Y[:,0], Y[:,1], 'o', color = "r")
	plt.axes(axs[1])
	plt.axis('equal')
	plot_laguerre(N)
	if len(Y) > 0:
		plt.plot(Y[:,0], Y[:,1], 'o', color = "r")
	return
	

#---------------- bloc de test calcul de gradient-----------------------------

# N = 5
# Y = np.array([[ 0.62074898, -0.00207409],
# [ 0.9568784 , -0.06380766],
# [ 0.69871951,  0.87968212],
# [-0.87548466, -0.91384685],
# [-0.41198645, -0.76459244]])

# H = lambda psi: compute_H_DH_from_mobius(Y, psi, make_mobius(Y, psi, k=100))[0]
# DH = lambda psi: compute_H_DH_from_mobius(Y, psi, make_mobius(Y, psi, k=100))[1]

# psi = 0.3 *(np.ones(N) + 0.01 * np.array([0.1, 0.6, -0.1, 0.7, -0.6]))
# print("twist = ",test_twist(Y, psi))
# np.set_printoptions(2)
# check_gradient(H,DH,psi)

# M = make_mobius(Y, psi, k=100)

# h,dh = compute_H_DH_from_mobius(Y, psi, M)
# dh2 = dh.copy()
# dh2[0,0] += 1
# print("H = ",h)
# print("DH = ", dh)
# print("det(DH) =", np.linalg.det(dh), "det(DH + e_1,1) =", np.linalg.det(dh2))


# plot_random_mobius(50)

# compare_mobius_naive()

# psi = test_newton_random(50, 50)

psi = test_newton_uniform(50, 50)

plt.show()
