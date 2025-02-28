#   Timothee Leleu, Sam Reifenstein
#   Non-Equilibrium Dynamics of Hybrid Continuous-Discrete Ground-State Sampling
#   ICLR2025

import CAC_Ising
import DASTuneADAM as DASTuner
import lib
import numpy as np
import time
import os
import generate_dWPE

def tune_wishart(folder_name,instance,hyperparams,PARAM_NAMES,x,flags,tunerparams,data):

    global count, fit_est
    
    N = instance['N']
    alphatxt = instance['alphatxt']
    
    savetraj = flags['savetraj']
    pt_device = flags['pt_device']
    solvertype = flags['solvertype']
    
    nsamp_max = tunerparams['nsamp_max']
    R = tunerparams['R']
    
    x_init = np.array(x)
    L_init = np.diag(np.ones(len(x))*0.5)

    #generate problem and initialize solver

    def gen_problem():
        
        if data['datatype'] == 'load':
            #load instance
            i = int(10* np.random.rand())
            J, eps0, H0 = lib.load_wishart(N,alphatxt,i)
            prec = 1
            
        else:
            
            i = int(100000* np.random.rand())
            alpha = float(alphatxt)
            M = int(N*alpha)
            
            if data['datatype']=='unbias':
                J, H0, gs = generate_dWPE.gen_dWPE(i, N, M, data['D_WPE'], data['R_WPE'])
            elif data['datatype']=='bias':
                J, H0, gs = generate_dWPE.gen_dWPE_cluster(i, N, M, data['D1_WPE'], data['R1_WPE'], data['D2_WPE'], data['R2_WPE'], bias =  data['bias'])
                J_bias = np.ones((N,N))
                J_bias = J_bias - np.diag(np.diag(J_bias))
                J = J + (0)*J_bias*10**(-4)

            eps0 = np.mean(np.abs(J))
            
            prec = 10**(-6) #precision for GS energy
            H0 = np.floor(H0/prec)
            
            #not used here
            gs = (np.array(gs).T)
            gs = gs/np.expand_dims(gs[:,0],1)
            gs = gs.tolist()
            
        #setup solver

        solver = CAC_Ising.CAC(pt_device, N, J=J, H0=H0, solvertype=solvertype, precGS = prec)
        solver.eps = eps0

        return solver, H0

    #tuner parameters
    fitness_beta = -1
    
    def log(*args):
        if(debug):
            print(*args)

    T_index = -1

    for idx, param in enumerate(PARAM_NAMES):
        if(param == "T"):
            T_index = idx

    count = 0
    debug = True

    def sample(x, seed, fitness_beta=-1):
        global count, fit_est
        R = x.shape[1]
        D = x.shape[0]

        solver, E0 = gen_problem()

        #setup solver

        T_vec = None
        if(T_index >= 0):
            T_vec = x[T_index, :]


        for idx, param_name in enumerate(PARAM_NAMES):
            if(T_index == idx):
                setattr(solver, param_name, T_vec)
            else:
                setattr(solver, param_name, np.exp(x[idx, :]))

        solver.init(R,PARAM_NAMES,hyperparams)

        tstart = time.time()
        Ps, E_opt = solver.traj(E0)

        if(T_vec is None):
            T_vec = np.ones(R)

        out = None
        if(fitness_beta == -1):
            out = (E_opt <= E0)/T_vec
        elif(fitness_beta == 0):
            out = E_opt==E0
        else:
            out = E_opt<=fitness_beta*E0

        count += 1
        return out

    D = len(PARAM_NAMES)

    #####################################################


    #use DAS tuner

    fit_est_beta = 0.01

    tuner = DASTuner.Sampler(sample, D, R)
    tuner.fit_est_beta = fit_est_beta
    tuner.curv_est_beta = fit_est_beta
    tuner.grad_est_beta = fit_est_beta/D

    if(x_init is None):
        x_init = np.zeros(D)

    if(L_init is None):
        L_init = np.diag(np.ones(D))

    tuner.init_window(x_init, L_init)

    tuner.dt0 = 0.5

    tot_samp_rec, x_rec, L_rec = tuner.optimize(tot_samp_max = nsamp_max, R_end = 10.0)
    
    param_out = x_rec[len(x_rec)-1]


    #####################################################

    R_eval = 400
    count = 0
    f_eval = 0
    N_inst = 15
    evalist = []
    for i in range(N_inst):

        eva = np.average(sample(np.outer(param_out, np.ones(R_eval)),range(R_eval),fitness_beta=0))
        f_eval += eva
        evalist.append(eva)
    f_eval = f_eval/N_inst	

    info = {}
    info["L"] = tuner.L
    info["curv_est"] = tuner.curv_est


    #####################################################

    bias = data['bias']
    if 'fMH' in hyperparams:
        fMH = hyperparams['fMH']
    else:
        fMH = 0.0
    T = hyperparams['T']
    file_name = f"wishart_{N}_{alphatxt}_{bias}_{T}_{fMH}.txt"
    lib.save_to_file(folder_name, file_name, f_eval, evalist, param_out)
    
    
    #####################################################

    
    def save_to_file_and_plot(folder_name, plot_file_name, PARAM_NAMES, x_rec):
        # Construct the file name for data

        # Construct the file name for the plot
        plot_file_path = os.path.join(folder_name, plot_file_name)

        # Plot the figure
        plt.figure()
        for idx, PARAM in enumerate(PARAM_NAMES):
            plt.plot(np.exp(x_rec)[:,idx],label=PARAM)
        plt.xlabel('steps')
        plt.ylabel('parameters')
        
        plt.legend()
        
        plt.yscale('log')
        
        ax = plt.gca()
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(True)

        # Save the figure to a file
        plt.savefig(plot_file_path)

        # Close the figure
        plt.close()
        
    
    plot_file_name = f"wishart_{N}_{alphatxt}_{bias}_{T}_{fMH}.png"
    save_to_file_and_plot(folder_name, plot_file_name, PARAM_NAMES, x_rec)
    