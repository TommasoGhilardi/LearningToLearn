# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:45:35 2021

@author: Francesco Poli, Tommaso Ghilardi, Max Hinne
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as T
import arviz as az
from collections import Counter
import matplotlib.pyplot as plt
import os

#Choose sampling method (ADVI vs MCMC)
useADVI=True  # if True ADVI will be used, if False MCMC will be used

#Set directory
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# Import Saccadic Latency and Look Away Data
data= pd.read_csv('df.txt', sep=' ')
ltime=np.array(data.ltime)
slat=np.array(data.sl)
# Import what trials have been watched, and when infants looked away.
watched = pd.read_csv('watched.txt', sep=' ')
lookaway = pd.read_csv('lookaway.txt', sep=' ') # how to account for no LA in first three trials?
# Import sequences that were presented to infants
sequences= pd.read_csv('sequences_t.csv', header=None).transpose()
# Set up some stuff for later
nsubj = len(data.subj.unique())
subj_idx = data.subj.values
seq_idx = data.seq.values-1
ntrial, nseq = 15, 16

# Initialize variables for loop to structure data in the right format for the model
counts = np.zeros(shape=(len(subj_idx),4)) #counts at trial t
prev_counts = np.zeros(shape=(len(subj_idx),4)) #counts at trial t-1
pastseq_counts = np.zeros(shape=(len(subj_idx),4)) #counts at the end of the previous sequence
offtrials = np.zeros(shape=(len(subj_idx),))
trial_vect = np.zeros(shape=(len(subj_idx),))
seq_vect = np.zeros(shape=(len(subj_idx),))
# for every subject
for i in data.subj.unique():
    # reset which is the last sequence that has been watched to zero
    lastseq_watched=0 
    seq_count=-1
    # for every sequence
    for s in data.seq.unique():
        # if the sequence has been watched
        if any(watched.iloc[i*nseq+s-1,:]): 
            seq_count+=1
            n=-1
            # if this is the first sequence that has been observed by the participant
            if lastseq_watched==0:
                this_pastseq_counts=np.array([1,1,1,1]) 
            # if a sequence has been observed before, assign its final counts as pastseq_counts
            if lastseq_watched>0:
                this_pastseq_counts=list(this_count.values())
            lastseq_watched=s
            this_count=Counter({1:0, 2:0, 3:0, 4:0})
            # for every trial
            for t in data.trial.unique():
                # if the trial has been whatched, update count and assign
                if watched.iloc[i*nseq+s-1,t-1]==1:
                    prev_counts[i*nseq*ntrial+ntrial*(s-1)+t-1,:]=list(this_count.values())
                    this_count.update([sequences.iloc[t-1,s-1]])
                    counts[i*nseq*ntrial+ntrial*(s-1)+t-1,:]=list(this_count.values())
                    pastseq_counts[i*nseq*ntrial+ntrial*(s-1)+t-1,:] = this_pastseq_counts
                    n+=1
                    trial_vect[i*nseq*ntrial+ntrial*(s-1)+t-1]=n
                    seq_vect[i*nseq*ntrial+ntrial*(s-1)+t-1]=seq_count                  

# set up stuff in numpy or theano 
lookaway=np.array(lookaway)
watched=np.array(watched)

#reshape everything
watched=watched.reshape(subj_idx.shape).astype(theano.config.floatX)
lookaway = lookaway.reshape(subj_idx.shape).astype(theano.config.floatX)

# Keep only data in which participants looked at the screen 
lookaway=lookaway[watched==1]
ltime=ltime[watched==1]
slat=slat[watched==1]
subj_idx=subj_idx[watched==1]
seq_idx=seq_idx[watched==1]
counts=counts[watched==1]
prev_counts=prev_counts[watched==1]
pastseq_counts=pastseq_counts[watched==1]
trial_vect=trial_vect[watched==1]
trial_vect = theano.shared(trial_vect.astype("float64"))
seq_vect=seq_vect[watched==1]
seq_vect = theano.shared(seq_vect.astype("float64"))

# set flat prior for counts
flat_count = np.ones(shape=(len(subj_idx),4))
# Compute probabilities of seeing the given target in any given location
probs = (counts+flat_count)/np.sum(counts+flat_count,axis=1).reshape((counts.shape[0], 1))
prev_probs = (prev_counts+flat_count)/np.sum(prev_counts+flat_count,axis=1).reshape((counts.shape[0], 1))
# Compute KL-Divergence for every trial
kl =  np.sum(probs*np.log2(probs/prev_probs),axis=1)
kl = theano.shared(kl.astype("float64"))

# Mask nans (necessary for pymc to run)
ltime = np.ma.masked_invalid(ltime)
slat = np.ma.masked_invalid(slat)

# =============================================================================
# Looking Time (ltime)
# =============================================================================
with pm.Model() as model: #           
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
