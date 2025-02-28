#   Timothee Leleu, Sam Reifenstein
#   Non-Equilibrium Dynamics of Hybrid Continuous-Discrete Ground-State Sampling
#   ICLR2025

import numpy as np
import torch
import sys

class CAC:
    
    J = None
    h = None
    norm = 1
    
    dtype = torch.float64
   
    beta = np.log(1.0)
    kappa = np.log(1.0)
    lamb = np.log(1.0)
    xi = np.log(1.0)
    gamma = np.log(1.0)
    eps = np.log(1.0)
    a = np.log(1.0)
    
    PARAM_NAMES = ["beta","lamb","xi","gamma","a"]
    hyperparams = hyperparams = {'T': 10}
    
    solvertype='CACm'
  
    def __init__(self, pt_device, N, J = None, H0 = None, h = None, solvertype = 'CACm', precGS = 1):
        
        self.pt_device = pt_device
        self.N = N
        self.load_problem(J = J, h = h)
        self.H0 = torch.tensor(H0, dtype = self.dtype).to(self.pt_device)
        
        self.eps = torch.math.sqrt(self.N)
        
        self.solvertype = solvertype
        
        self.precGS = precGS #for calculation of energy within a certain precision
        
    #loads problem from numpy or pytorch tensors J and h (J assumed to be NxN symmetric matrix with zero diagonal)
    def load_problem(self, J = None, h = None):
        if(not J is None):
            self.J = torch.tensor(J, dtype = self.dtype).to(self.pt_device)
            self.N = self.J.shape[0]
        
        if(not h is None):
            self.h = torch.tensor(h, dtype = self.dtype).to(self.pt_device)
        
    
    #loads coupling matrix from numpy array with coordinate representation (i, j, Jij)
    def load_J_coo(self, N, ijJij, h = None):
        self.J = torch.zeros((N,N), dtype = self.dtype)
        
        for (i,j,Jij) in ijJij:
            
            self.J[int(i),int(j)] = Jij
            self.J[int(j),int(i)] = Jij
        
        self.J = self.J.to(self.pt_device)
        
        self.h = None
        
        if(not h is None):
            self.h = torch.tensor(h, dtype = self.dtype).to(self.pt_device)
    
    
    def cal_feedback(self, y):
        if(self.h is None):
            return torch.mm(self.J, y)*self.norm
        return  (torch.mm(self.J, y) + self.h)*self.norm
    
    def cal_E(self, s):
        if(self.h is None):
            E = -0.5*torch.sum(torch.mm(self.J, s)*s, dim=0)
            if self.precGS<1:
                E = torch.floor(E/self.precGS)
            return E
        E = -0.5*torch.sum(torch.mm(self.J, s)*s, dim=0) - torch.sum(self.h*s, dim=0)
        if self.precGS<1:
            E = torch.floor(E/self.precGS)
        return E
    
    def doflags(self, s):
        #flag for decrease of energy
        flag = self.E < self.E_opt
        self.E_opt = self.E*flag + self.E_opt*(~ flag)
        
        #save all GS found
        flagGS = self.E == self.H0
        if torch.any(flagGS*flag): #GS found for the first time
            idx = np.where((flagGS*flag).cpu().numpy() == 1)[0]
            for id0 in idx:
                s0 = s[:,id0].cpu().numpy()
                s0 = s0/s0[0]
                self.allGS.append(s0.tolist())

    #Init solver with problem and number of trajs
    def init(self, R, PARAM_NAMES, hyperparams):
        self.R = R
        
        self.PARAM_NAMES = PARAM_NAMES
        self.hyperparams = hyperparams
        self.T = self.hyperparams['T']
        
        if self.solvertype == 'MHCAC' or self.solvertype == 'MHCACm':
            
            self.nrep = int(self.T*self.hyperparams['fMH'])
            self.T = int(self.T/self.nrep)
    
            self.P2 = torch.rand(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.s1 = torch.rand(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.s1 = torch.sign(2 * self.s1 - 1)
            self.H1 = self.cal_E(self.s1)  
            self.E_opt = self.H1
            
            #for return
            self.e = torch.ones(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            
        
        if self.solvertype == 'CACm':
        
            self.P = 2*torch.rand(self.N, self.R, dtype = self.dtype).to(self.pt_device)-1
            self.s = torch.rand(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.s = torch.sign(2 * self.s - 1)
            self.H = self.cal_E(self.s)  
            self.u = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.up = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.upp = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.e = torch.ones(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.E_opt = self.H
            
        if self.solvertype == 'CAC':
        
            self.P = 2*torch.rand(self.N, self.R, dtype = self.dtype).to(self.pt_device)-1
            self.s = torch.rand(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.s = torch.sign(2 * self.s - 1)
            self.H = self.cal_E(self.s)  
            self.u = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.up = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.e = torch.ones(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.E_opt = self.H
        
        if self.solvertype == 'AIM':
        
            self.P = 2*torch.rand(self.N, self.R, dtype = self.dtype).to(self.pt_device)-1
            self.s = torch.rand(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.s = torch.sign(2 * self.s - 1)
            self.H = self.cal_E(self.s)  
            self.u = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.up = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.upp = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
            self.E_opt = self.H
        
        self.reachedGS = torch.zeros(self.R, dtype = self.dtype).to(self.pt_device)
        
        
        for param_name in self.PARAM_NAMES:
            setattr(self, param_name, torch.tensor(getattr(self, param_name), dtype = self.dtype).to(self.pt_device))
       
        if self.solvertype == 'MHCAC' or self.solvertype == 'MHCACm':
            self.beta_MH = self.beta / self.eps
            self.beta_de = self.beta_MH*self.kappa
            
            self.doa = self.hyperparams['doa']
            self.dosampling = self.hyperparams['dosampling']
            
        #list of all ground-states
        self.allGS = []
            
        
    def deterministic_pass(self, s, momentum):
        
        debug = 0
        
        u = torch.zeros(self.N, self.R, dtype = self.dtype).to(self.pt_device)
        up = u
        if momentum:
            upp = u
        
        e = torch.ones(self.N, self.R, dtype = self.dtype).to(self.pt_device)
        
        P = (s + 1) / 2
        
        if debug:
            Ptraj = []
            etraj = []
            Htraj = []
        
        for i in range(self.nrep):
            ### COUPLING ###
            mu = torch.matmul(self.J, (2 * P - 1))
    
            ### ITERATION ###
            if momentum:
                upp = up
            up = u
            
            u = up - self.lamb * up + e * mu
            if momentum:
                u = u + self.gamma * (up - upp)
                
            e = e - ((2 * P - 1) ** 2 - self.a) * e * self.xi
            P = 1 / (1 + torch.exp(- self.beta_de * u * 2))
            
            norm = torch.mean(e, dim=0)
            e = e / norm.unsqueeze(0)
            e = torch.abs(e)
            
            if debug:
                Ptraj.append(P.cpu().numpy()[0,:])
                etraj.append(e.cpu().numpy()[0,:])
                s = torch.sign(2*P-1)
                H = -0.5*torch.sum(s*torch.matmul(self.J,s),0)
                Htraj.append(H.cpu().numpy()[0])
                
            if self.dosampling==0:
                s = torch.sign(2 * P - 1).to(dtype = self.dtype)
                self.E = self.cal_E(s)
                self.doflags(s)
               
        P_new = 1/(1+torch.exp(-u*self.beta_de*2/self.lamb/e))
        
        
        if debug:
            #traj
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(Ptraj)
            plt.subplot(2,1,2)
            plt.plot(etraj)
            
            plt.figure()
            plt.plot(Htraj)
            
            sys.exit("Stop here")
        
        return P_new, e
        
    ##################################################################
    
    #step solver with tau = T_current/T_max
    def step_MHCAC(self, t, update_E = True, momentum = False):
                                
        ### GENERATE RANDOM BINARY OUTPUT: generate new attempt for s ###
        rdm = torch.rand(self.N, self.R,device=self.pt_device)
        s2 = (2 * (self.P2 > rdm).int() - 1).to(dtype = self.dtype)
        
        ### ENERGY CALCULATION FROM THIS BINARY OUTPUT ###
        H2 = self.cal_E(s2)
        
        # Get deterministic forward path from s
        P1p, e = self.deterministic_pass(s2,momentum)
        self.e = e #for return
        
        ### COMPUTE ACCEPTANCE CRITERION (of the previous pass) ###
        if self.doa:
            logQ1 = torch.nansum((1 + s2) / 2 * torch.log(self.P2) + (1 - s2) / 2 * torch.log(1 - self.P2), dim=0)  # transition proba s2/s1
            logQ2 = torch.nansum((1 + self.s1) / 2 * torch.log(P1p) + (1 - self.s1) / 2 * torch.log(1 - P1p), dim=0)  # transition proba s1/s2
    
            a = torch.exp(self.beta_MH * (-H2 + self.H1) - logQ1 + logQ2) 
            a = torch.clamp(a, 0.0, 1.0)
                
        else:
            a = torch.zeros(self.R,device=self.pt_device)
            
        ### ACCEPTANCE RULE ###
        if self.doa and t > 3:
            accept = (torch.rand(self.R,device=self.pt_device) < a).float()
            
            self.s1 = s2 * accept.unsqueeze(0) + self.s1 * (1 - accept.unsqueeze(0))
            self.H1 = H2 * accept + self.H1 * (1 - accept)
            
            self.P2 = P1p * accept.unsqueeze(0) + self.P2 * (1 - accept.unsqueeze(0))
            
        else:
            
            self.s1 = s2
            self.H1 = H2
            
            self.P2 = P1p
            
        ## Energy to save ##
        if(update_E):
            s = torch.sign(2 * self.P2 - 1).to(dtype = self.dtype)
            if self.dosampling == 0:
                self.E = self.cal_E(s)
            else:
                self.E = self.H1
                
            self.doflags(s)
        
      
    ##################################################################
              
    #step solver with tau = T_current/T_max
    def step_CACm(self, t, update_E = True):

        ### COUPLING ###
        mu = torch.matmul(self.J, self.P)

        ### ITERATION ###
        self.upp = self.up
        self.up = self.u
        
        self.u = self.up - self.lamb * self.up + (self.beta/self.eps) * self.e * mu + self.gamma * (self.up - self.upp)
        self.e = self.e - (self.P ** 2 - self.a) * self.e * self.xi

        self.P = torch.tanh(self.u)
        
        norm = torch.mean(self.e, dim=0)
        self.e = self.e / norm.unsqueeze(0)
        self.e = torch.abs(self.e)
        
        ## Energy to save ##
        if(update_E):
            s = torch.sign(self.P).to(dtype = self.dtype)
            self.E = self.cal_E(s)
                
            self.doflags(s)
        
    ##################################################################
  
    #step solver with tau = T_current/T_max
    def step_CAC(self, t, update_E = True):

        ### COUPLING ###
        mu = torch.matmul(self.J, self.P)

        ### ITERATION ###
        self.up = self.u
        
        self.u = self.up - self.lamb * self.up + (self.beta/self.eps) * self.e * mu
        self.e = self.e - (self.P ** 2 - self.a) * self.e * self.xi

        self.P = torch.tanh(self.u)
        self.e = torch.abs(self.e)
        
        ## Energy to save ##
        if(update_E):
            s = torch.sign(self.P).to(dtype = self.dtype)
            self.E = self.cal_E(s)
                
            self.doflags(s)
          
    ##################################################################
        
    #step solver with tau = T_current/T_max
    def step_AIM(self, t, update_E = True):

        ### COUPLING ###
        mu = torch.matmul(self.J, self.P)

        ### ITERATION ###
        self.upp = self.up
        self.up = self.u
        
        self.u = self.up - self.lamb * self.up + (self.beta/self.eps) * mu + self.gamma * (self.up - self.upp)
        self.P = torch.tanh(self.u)
        
        ## Energy to save ##
        if(update_E):
            s = torch.sign(self.P).to(dtype = self.dtype)
            self.E = self.cal_E(s)
                
            self.doflags(s)
            
            
    #function for annealing schedule (set to linear for now)
    def schedule(self, tau):
        return tau
    
    
    #compute trajectory with target energy. Returns best energy for each trajectory and succes probability. Should be called after init.
    def traj(self, target_E, R_rec = 0):
        
        x_rec = None
        e_rec = None
        E_rec = None
        T_rec = None
        tau_rec = None
        
        if(R_rec > 0):
            T_ = int(np.ceil(self.T))
            x_rec = np.zeros((T_, self.N, R_rec))
            e_rec = np.zeros((T_, self.N, R_rec))
            E_rec = np.zeros((T_, R_rec))
            T_rec = np.zeros(T_)
            tau_rec = np.zeros(T_)
        
        t = 0
        while(t < np.max(np.array(self.T))):
            tau = self.schedule(t/self.T)
            
            #if t==89:
            #    print('hello')
            
            if self.solvertype == 'MHCACm':
                self.step_MHCAC(t, update_E = True, momentum=True)
            if self.solvertype == 'MHCAC':
                self.step_MHCAC(t, update_E = True)
            if self.solvertype == 'CACm':
                self.step_CACm(t, update_E = True)
            if self.solvertype == 'CAC':
                self.step_CAC(t, update_E = True)
            if self.solvertype == 'AIM':
                self.step_AIM(t, update_E = True)
            
            if(R_rec > 0):
                
                if np.mod(t,int(self.T/10))==0:
                    print('%0.2f percent completed' % float(t/self.T))
                
                #x_rec[t ,:, :] = self.x[:, :R_rec]
                #e_rec[t ,:, :] = self.e[:, :R_rec]
                #x_rec[t ,:, :] = self.P2[:, :R_rec].cpu()
                #e_rec[t ,:, :] = self.e[:, :R_rec].cpu()
                E_rec[t, :] = self.E[:R_rec].cpu()
                T_rec[t] = t
                tau_rec[t] = tau
            
            t += 1
            
        E_opt_ = self.E_opt.cpu().numpy()
        
        if(target_E is None):
            target_E = np.min(E_opt_)
        
        Ps = np.sum(E_opt_ <= target_E)/self.R
        
        if(R_rec > 0):
            return  Ps, E_opt_, {"x": x_rec, "e": e_rec, "E": E_rec, "T": T_rec, "tau": tau_rec}
        
        
        return Ps, E_opt_
    
    
    
    
    
    
    
    
    