#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from geometry import *
from ot_metasurfaces import *
from gje import *
import numpy as np
import pypower
import scipy.optimize as opt
import matplotlib.pyplot as plt

def compare_mobius(M,N,Y=[]):
	"""Plot the two mobius diagram M and N"""
	fig, axs = plt.subplots(1,2)
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

def uniform_setup(N = 10):
	"""Generate a discrete uniform measure associated with a set of N random points Y in the square [0,1]^2 """
	nu = (4/N) * np.ones(N)
	Y = (np.random.rand(N,2))
	return Y, nu

def random_setup(N = 10):
	"""Generate a discrete random measure associated with a set of N random points Y in the square [0,1]^2 """
	Y =  (np.random.rand(N,2) )
	nu = np.random.rand(N) #Random target measure
	nu =  4 * nu / np.linalg.norm(nu, 1)
	return Y, nu

def test_newton(Y, nu, plot = False, verb = True, stoperr = 1e-6, eps = 0.01):
	"""
	Tries solving NF parallel reflector problem  with source [-1,1]^2 associated with Lebesgue measure and target space Y and measure nu
	Variable plot set to True will plot the initial and final mobius diagram if the algorithm converges , verb = True will print the error
	value and damping parameter at each iteration. The algorithm will stop when ||H(psi) - nu||_1 < stoperr. k is the discretisation parameter
	of the interface between two cells of the diagram.
	"""
	N = len(Y)
	psi = 0.004 * np.ones(N) #psi is chosen of the form lambda * np.ones(N) to ensure that none of the cells is empty. The value of lambda is arbitrary.
	print("twist = ", test_twist(Y, psi))
	M0 = make_mobius(Y, psi, eps=eps)
	result = solve_gje(Y, psi, nu, stoperr, verb)
	psi = result[0]
	if plot:
		M = make_mobius(Y, psi, eps=eps)
		if N <= 100:
			compare_mobius(M0,M,Y)
		else:
			compare_mobius(M0,M)
	fig2 = plt.figure()
	plt.semilogy(result[1], result[2])
	plt.xlabel("Iteration number")
	plt.ylabel("Error in log scale")
	return 



N = 50

Y, nu = uniform_setup(N)
#Y, nu = random_setup(N)


test_newton(Y, nu, plot = True, verb = True)

Y, nu = random_setup(N)

test_newton(Y, nu, plot = True, verb = True)

plt.show()
