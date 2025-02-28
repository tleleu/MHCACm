#   Timothee Leleu, Sam Reifenstein
#   Non-Equilibrium Dynamics of Hybrid Continuous-Discrete Ground-State Sampling
#   ICLR2025

import numpy as np
import scipy as sp

#SEED = random seed for instance generation
#N = number of spins
#M = second dim. of wishart matrix (rank of J matrix), alpha = M/N
#D (1 <= D < N) = number of ground states to plant
#R (-1 <= R < N) = distance between ground states (number of spins to flip). If R = -1 then choose GS randomly and indep.
def gen_dWPE(SEED, N, M, D = 1, R = -1):
	
	np.random.seed(SEED)
	
	#choose GS
	gs = np.zeros((N,D))
	for i in range(D):
		if(R > 0):
			gs[:,i] = np.random.permutation(np.sign(np.array(range(N)) - 0.5 - R))
			
		else:
			gs[:,i] = np.sign(np.random.randn(N))
	
	
	gs_orth = sp.linalg.orth(gs)
	
	W = np.random.randn(N,M)
	
	W = np.dot(np.eye(N) - np.dot(gs_orth, gs_orth.T), W)
	
	J = np.dot(W,W.T)
	E0 = -np.sum(np.diag(J))/2
	J = -(J - np.diag(np.diag(J)))
	
	return J, E0, gs


def gen_dWPE_cluster(SEED, N, M, D1 = 1, R1 = -1, D2 = 1, R2 = -1, bias = 0):
	
	D = D1 + D2
	np.random.seed(SEED)
	
	#choose GS
	gs = np.zeros((N,D))
	
	for i in range(D1):
		if(R1 > 0):
			gs[:,i] = np.random.permutation(np.sign(np.array(range(N)) - 0.5 - R1))
			
		else:
			gs[:,i] = np.sign(np.random.randn(N))
			
	for i in range(D2):
		if(R2 > 0):
			gs[:,D1 + i] = np.random.permutation(np.sign(np.array(range(N)) - 0.5 - R2))
			
		else:
			gs[:,D1 + i] = np.sign(np.random.randn(N))
	
	
	gs_orth = sp.linalg.orth(gs)
	
	W = np.random.randn(N,M)
	
	bv = np.ones((N,1))
	bv = bv/np.sum(bv**2)**0.5
	
	W = np.dot(np.eye(N) + bias*np.dot(bv, bv.T),  W)
	
	W = np.dot(np.eye(N) - np.dot(gs_orth, gs_orth.T), W)
	
	
	J = np.dot(W,W.T)
	E0 = -np.sum(np.diag(J))/2
	J = -(J - np.diag(np.diag(J)))
	
	return J, E0, gs




#test
if(0):	
	J, E0, gs = gen_dWPE(0, 50, 20, 5)
	
	print(E0, -0.5*np.sum(gs*np.dot(J, gs), axis = 0))
	
	
	J , E0, gs = gen_dWPE_cluster(0, 50, 20, 5, 5, 5, -1, 0.5)
	
	print(E0, -0.5*np.sum(gs*np.dot(J, gs), axis = 0))
	
