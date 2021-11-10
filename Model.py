# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:45:35 2021

@author: Francesco Poli, Tommaso Ghilardi, Max Hinne
"""
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as T
import arviz as az
import matplotlib.pyplot as plt
from simulation_metalearning import SimulateData 


# Choose sampling method (ADVI vs MCMC)
useADVI=True

# Set if to run the model using the data collected or if to simulate the data
Simulation = True

# Set relative directory
os.chdir(os.path.abspath(os.path.dirname(__file__)))

data = pd.read_csv('Data' + os.sep + 'data_SL_LT_LA.csv', sep=',')

if Simulation:
    
    nsubj  = 70  # specify number of subjects
    ntrial = 15  # specify number of trials for seuence ( <= 15)
    nseq   = 10  # specify number of sequences (<= 16)
    
    #Pass all the parameter to the simulation script using a dictionary
    
    simulation_dictionary = {'nsubj':nsubj, 'ntrial':ntrial, 'nseq':nseq,                  # set up some stuff for later
        'b0_alpha':0.451, 'b1_alpha':0.162, 'b0_seq':0.092, 'b1_seq':0.034,   # set parameter values to change kl across sequences and trials (meta learning) 
        'sim_beta0_LT':-0.8, 'sim_beta1_LT':20, 'sim_noise_LT':0.5,           # specify the parameters that define the likelihoods for looking time
        'sim_beta0_SL':-0.5, 'sim_beta1_SL':10, 'sim_noise_SL':0.3,           # specify the parameters that define the likelihoods for saccadic latency
        'sim_lambda0' : 0.5, 'sim_beta_LA' : -50                              # set parameters for lambda 0 and beta_LA
        }
    
    [subj_idx, seq_idx, seq_vect, trial_vect,
    kl, ltime, slat,lookaway] = SimulateData(simulation_dictionary, missing = True, Plot =True) # simulate the data
    
    
elif not Simulation:
    # Import data
    data= pd.read_csv('Data' + os.sep + 'data_SL_LT_LA.csv', sep=',')
    
    ################ Convert data to a pymc-friendly version ######################
    # total number of subjects
    nsubj = len(data.subj_idx.unique())
    # max number of sequences in the task
    nseq = len(data.seq_idx.unique())
    # max number of trials in a sequence
    ntrial = len(data.trial_vect.unique())
    # index of each subject and each sequence for each trial
    subj_idx = data.subj_idx.values.astype(int)
    seq_idx = data.seq_idx.values.astype(int) # objective sequence number
    seq_vect = data.seq_vect.values # subjective sequence number
    trial_vect = data.seq_vect.values
    
    # use theano instead of numpy (better for pymc3)
    trial_vect = theano.shared(trial_vect.astype("float64"))
    seq_vect = theano.shared(seq_vect.astype("float64"))
    
    # counts and prev_counts specify the frequency of appearence of the target in each location for each trial of each sequence
    counts = np.array(data.iloc[:,-8:-4])
    prev_counts = np.array(data.iloc[:,-4:])
    
    ################ Compute KL Divergence ########################################
    # First, set flat prior for counts
    flat_count = np.ones(shape=(len(subj_idx),4))
    # Compute probabilities of seeing the given target in any given location
    probs = (counts+flat_count)/np.sum(counts+flat_count,axis=1).reshape((counts.shape[0], 1))
    prev_probs = (prev_counts+flat_count)/np.sum(prev_counts+flat_count,axis=1).reshape((counts.shape[0], 1))
    # Compute KL-Divergence for every trial
    kl =  np.sum(probs*np.log2(probs/prev_probs),axis=1)
    kl = theano.shared(kl.astype("float64"))
    
    ################ Dependent Variables ##########################################
    # If nans are present, variables must be masked to function in theano
    ltime = np.ma.masked_invalid(data.ltime.values) # looking time to the target (standardized across participants)
    slat = np.ma.masked_invalid(data.slat.values) # saccadic latency (standardized across participants)
    lookaway = data.lookaway.values # look-away from the screen (0 for no look-away, 1 for look-away)


################ Bayesian Model ###############################################
with pm.Model() as model: 
# =============================================================================
# Looking time and saccadic latency
# =============================================================================       
################################# Priors ###################################### 
    # Define beta0, beta1 and error for Looking Time  
    LT0 = pm.Normal('LT0', mu = 0, sigma=10, shape = nsubj)
    LT1 = pm.Normal('LT1', mu = 0, sigma=10, shape = nsubj)
    eps_LT = pm.HalfCauchy('eps_LT', beta = 1)
    
    # Define beta0, beta1 and error for Saccadic Latency 
    SL0 = pm.Normal('SL0', mu = 0, sigma=10, shape = nsubj)
    SL1 = pm.Normal('SL1', mu = 0, sigma=10, shape = nsubj)
    eps_SL = pm.HalfCauchy('eps_SL', beta = 1)
    
    # Define the parameters that regulate the up- and down-weighting of KL Divergence
    b0_seq = pm.Lognormal('b0_seq', mu=0, sigma=1)
    b1_seq = pm.Lognormal('b1_seq', mu=0, sigma=1) 
    b0_alpha = pm.Lognormal('b0_alpha', mu=0, sigma=1)
    b1_alpha = pm.Lognormal('b1_alpha', mu=0, sigma=1) 
    
    # Up- and down-weighting of KL-Divergence values
    kl_time= kl*(b0_alpha+b1_alpha*seq_vect)*np.exp(-(b0_seq+b1_seq*seq_vect)*trial_vect)

######################### Estimates and Likelihood ############################
    # Linear regression between Looking Time and KL-Divergence
    est_LT = LT0[subj_idx]+LT1[subj_idx]*kl_time            
    LT_like = pm.Normal("LT_like", mu=est_LT, sigma=eps_LT, observed=ltime)
    
    # Linear regression between Saccadic Latency and KL-Divergence
    est_SL = SL0[subj_idx]+SL1[subj_idx]*kl_time                
    SL_like = pm.Normal("SL_like", mu=est_SL, sigma=eps_SL, observed=slat)  
    
# =============================================================================
# Look Away
# =============================================================================       
################################# Priors ######################################
    # Define baseline hazard and beta1 for Look-Away
    lambda0 = pm.Gamma("lambda0", 0.01, 0.01, shape = nseq)
    beta_LA = pm.Normal("beta_LA", 0, sigma=10, shape = nsubj)

######################### Estimates and Likelihood ############################
    # proportional hazard model (https://docs.pymc.io/notebooks/survival_analysis.html)
    lambda1 = pm.Deterministic("lambda1", T.exp(beta_LA[subj_idx] * kl_time)*lambda0[seq_idx])
    
    # The piecewise-constant proportional hazard model is closely related to a Poisson regression model, hence:
    LA_like = pm.Poisson("LA_like", lambda1, observed=lookaway)

#Estimate posterior
with model:
    # Using Variational Inference
    if useADVI:
        inference = pm.ADVI()
        approx = pm.fit(n=30000, method=inference)
        trace = approx.sample(draws=50000)
    # Using MCMC NUTS sampling
    else:
        trace = pm.sample(10000,  chains=2, cores=64, tune=190000)

# check ELBO for ADVI
if useADVI:
    plt.plot(-inference.hist, label="new ADVI", alpha=0.3)
    plt.plot(approx.hist, label="old ADVI", alpha=0.3)
    plt.legend()
    plt.ylabel("ELBO")
    plt.xlabel("iteration");

# change trace format for further analysis
idata = az.from_pymc3(trace)

# Nan values are treated as zero likelihood by pymc, so masked (nan) values must be removed before computing model goodness of fit 
SL_like_nan=np.nan_to_num(idata.log_likelihood.SL_like, copy=True, nan=0)#mean_value)
idata.log_likelihood['SL_like']=(idata.log_likelihood['SL_like'].dims, SL_like_nan)

LT_like_nan=np.nan_to_num(idata.log_likelihood.LT_like, copy=True, nan=0)#mean_value)
idata.log_likelihood['LT_like']=(idata.log_likelihood['LT_like'].dims, LT_like_nan)

LA_like_nan=np.nan_to_num(idata.log_likelihood.LA_like, copy=True, nan=0)#mean_value)
idata.log_likelihood['LA_like']=(idata.log_likelihood['LA_like'].dims, LA_like_nan)

# Get LOO estimates
model_waic1 = az.loo(idata, var_name="LT_like")
print(model_waic1)
model_waic2 = az.loo(idata, var_name="SL_like")
print(model_waic2)
model_waic3 = az.loo(idata, var_name="LA_like")
print(model_waic3)

# Get WAIC estimates
model_waic1 = az.waic(idata, var_name="LT_like")
print(model_waic1)
model_waic2 = az.waic(idata, var_name="SL_like")
print(model_waic2)
model_waic3 = az.waic(idata, var_name="LA_like")
print(model_waic3)

############################### PLOTTING #####################################
# plot the parameters and save the plots
az.plot_forest(trace, kind='forestplot', var_names=["lambda0"])
plt.savefig('forest_lambda0.png')
 
az.plot_trace(trace, var_names=["b0_alpha"]);
plt.savefig('b0_alpha.png') 
az.plot_trace(trace, var_names=["b1_alpha"]);
plt.savefig('b1_alpha.png')  
az.plot_trace(trace, var_names=["b0_seq"]);
plt.savefig('b0_seq.png')  
az.plot_trace(trace, var_names=["b1_seq"]);
plt.savefig('b1_seq.png')
az.plot_trace(trace, var_names=["lambda0"]);
plt.savefig('lambda0.png')
az.plot_trace(trace, var_names=["beta_LA"]);
plt.savefig('beta_LA.png')
az.plot_trace(trace, var_names=["b0_seq","b1_seq","lambda0","beta_LA"]);
plt.savefig('trace.png')

az.plot_trace(trace, var_names=["LT0"]);
plt.savefig('LT0.png')
az.plot_trace(trace, var_names=["LT1"]);
plt.savefig('LT1.png')

az.plot_trace(trace, var_names=["SL0"]);
plt.savefig('SL0.png')
az.plot_trace(trace, var_names=["SL1"]);
plt.savefig('SL1.png')

axes = az.plot_forest(trace, kind='ridgeplot', var_names=['lambda0'],combined=True,ridgeplot_overlap=3,colors='white',figsize=(9, 7))
axes[0].set_title('lambda0')
plt.savefig('forest2_lambda0.png')

# Save summary statistics for the model parameters
with model:
    summary = az.summary(trace,var_names=["b0_alpha","b1_alpha","b0_seq","b1_seq","LT0","LT1","SL0","SL1","lambda0","beta_LA"], round_to=3)
    summary.to_csv('gen_summary.csv')
