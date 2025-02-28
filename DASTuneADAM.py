#   Timothee Leleu, Sam Reifenstein
#   Non-Equilibrium Dynamics of Hybrid Continuous-Discrete Ground-State Sampling
#   ICLR2025


import numpy as np

class Sampler:
	
	eps = 10**(-10)
	rms_beta = 0.99
	mom_beta = 0.0
	v = 1.0
	
	prev_xw = []
	prev_L = []
	prev_xsamp = []
	prev_ysamp = []
	
	CRN = True
	
	def __init__(self, sample, D, B):
		self.D = D
		self.sample = lambda  x,seed,fitness_beta : sample(x, seed, fitness_beta=fitness_beta)
		
		
		self.xw = np.zeros(D)
		self.L = np.diag(np.ones(D))
		
		self.tot_samp  = 0
		self.dt = 0.1
		self.g = 0.02
		self.g_exp = 0.0
		self.kappa = 1.0
		
		
		self.dt0 = 0.05
		
		self.fit_est = 0.00001
		self.fit_var = 0.0
		self.fit_est_beta = 0.05
		self.curv_est = 0.000
		self.curv_est_beta = 0.05
		self.grad_est = np.zeros(D)
		self.grad_est_beta = 0.05
		
		self.delta_fit = 0.0
		self.delta_fit2 = 0.0
		self.fit_prev = 0.0
		
		self.B = 300#50
		self.Bmax = 300#50
		
		self.best_fit = 0.01**2
		
		
		self.growth = 0.05
	
	
	def init_window(self, xw, L):
		self.xw = xw
		self.L = L
	
	def step(self, dt, B):
		self.tot_samp += B
		D = self.D
		#random samples (z = normalized coord, x = real coord)
		z_samp = np.random.randn(D, B)
		
		seed = range(B)
		if(self.CRN):
			z_samp[:, :B//2] = -z_samp[:, B//2 :]
			seed = [2*s % B for s in seed]
			
		
		x_samp = self.xw.reshape(D, 1) + np.dot(self.L, z_samp)
		fitness_beta = 1-np.max([1.0-2*self.tot_samp/self.tot_samp_max,0])
		fitness_beta = 0.8 + fitness_beta*0.2
		y_samp = self.sample(x_samp, seed,fitness_beta)
		
		
		best_fit_ = self.best_fit + 0.0
		self.best_fit = np.maximum(np.average(y_samp**2), self.best_fit*self.rms_beta)
		#RMS normalization
		self.v = self.v*( (self.best_fit + 0.001) /(best_fit_ + 0.001))**2
		self.v = self.best_fit + self.eps
		
		self.dt = self.dt0/np.sqrt(self.v)
		
		
		#differentials in normalized coordinates
		dz = np.average(z_samp*y_samp.reshape(1,B), axis = 1)
		dA = -np.diag(np.ones(D))*np.average(y_samp) + np.average(z_samp.reshape(D, 1, B)*z_samp.reshape(1, D, B)*y_samp.reshape(1, 1,B), axis = 2)
		
		dz = dz*(1 - self.mom_beta)
		dA = dA*(1 - self.mom_beta)
		
		#save info for momentum
		self.prev_xw.append(self.xw)
		self.prev_L.append(self.L)
		self.prev_xsamp.append(x_samp)
		self.prev_ysamp.append(y_samp)
		
		
		Lamb = np.dot(self.L.T, self.L)
		
		scale = np.trace(self.L)
		dxw =  np.dot(self.L, dz)*1
		
		dL =  np.dot(self.L, dA)*1/D
		
		L_ = self.L + dt*dL
		r = np.sum(L_**2)**0.5/np.sum(self.L**2)**0.5
		self.L = self.L + r*dt*dL
		
		
		#ensure step is not too big
		xw_step = r*dt*dxw
		zw_step = np.linalg.solve(self.L, xw_step)
		self.L = self.L 
		
		if(np.average(self.L**2)**0.5 < self.g_current):
			self.L =  self.L*self.g_current/np.average(self.L**2)**0.5
		
		#max size of window
		if(np.sqrt(np.average(self.L**2)) > 2):
			self.L =  self.L*2/np.sqrt(np.average(self.L**2 ))
		
		fest = np.average(y_samp)
		cest = np.average(y_samp*np.average(z_samp**2, axis = 0))
		gest = np.average(y_samp.reshape(1,B)*z_samp.reshape(D,B), axis = 1)
		
		self.fit_est = (1 - self.fit_est_beta)*self.fit_est + self.fit_est_beta*fest
		self.curv_est = (1 - self.curv_est_beta)*self.curv_est + self.curv_est_beta*cest
		self.grad_est = (1 - self.grad_est_beta)*self.grad_est + self.grad_est_beta*gest
		
		fvest = np.std(y_samp)**2/B
		self.fit_var = (1 - self.fit_est_beta)*self.fit_var + self.fit_est_beta*np.sqrt(fvest)
		
		dfit = self.fit_est - self.fit_prev
		self.delta_fit = (1 - self.fit_est_beta)*self.delta_fit + self.fit_est_beta*dfit
		self.delta_fit2 = (1 - self.fit_est_beta)*self.delta_fit2 + self.fit_est_beta*dfit**2
		
		self.fit_prev = self.fit_est
		r2 = np.minimum(1.0, 1.0/np.linalg.norm(zw_step))
		self.xw = self.xw + r2*xw_step
	
	
	
	def optimize(self, tot_samp_max = 50000, tr_min = 0, R_end = 10):
		self.tot_samp  =0
		self.tot_samp_max = tot_samp_max
		tot_samp_rec = []
		xw_rec = []
		L_rec = []
		
		count = 0
		while(self.tot_samp < tot_samp_max and np.abs(np.sum(self.L**2)) > tr_min and (self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est < R_end):
			R = (self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est
			
			if(count > 20):
				if((self.fit_var)/self.fit_est > 4*(1 - R)):
					self.B += 2
				elif((self.fit_var)/self.fit_est < 1*(1 - R)):
					self.B -= 2
				if(self.B < self.Bmax):
					self.B = self.Bmax
			
			self.g_current = self.g/(count + 1)**self.g_exp
			self.step(self.dt0/np.sqrt(self.v), self.B)
	
			tot_samp_rec.append(self.tot_samp)
			xw_rec.append(self.xw)
			L_rec.append(self.L)
			count += 1
		
		return 	tot_samp_rec, xw_rec, L_rec
	
	
	