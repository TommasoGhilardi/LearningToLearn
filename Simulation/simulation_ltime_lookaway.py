# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:45:35 2021

@author: krav
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as T
from collections import Counter
import matplotlib.pyplot as plt

Path= "C:\\Users\\krav\\Desktop\\BabyBrain\\Projects\\Weighting_evidence\\MODEL\\"

#import sequences
sequences= pd.read_csv(Path+'sequences_t.csv', header=None).transpose()

# ste up some stuff for later
nsubj = 10
ntrial=15
nseq=16
subj_idx = []

# Initialize variables for loop
counts=np.zeros(shape=(nsubj*nseq*ntrial,4)) #counts[:]=np.NaN # or zero?
prev_counts=np.zeros(shape=(nsubj*nseq*ntrial,4)) #prev_counts[:]=np.NaN # or zero?
# for every subject
for i in range(nsubj):
    # for every sequence
    for s in range(1,nseq+1):
        this_count=Counter({1:0, 2:0, 3:0, 4:0})
        # for every trial
        for t in range(1,ntrial+1):
            # if the trial has been whatched, update count and assign
            prev_counts[i*nseq*ntrial+ntrial*(s-1)+t-1,:]=list(this_count.values())
            this_count.update([sequences.iloc[t-1,s-1]])
            counts[i*nseq*ntrial+ntrial*(s-1)+t-1,:]=list(this_count.values())
            # update subject index
            subj_idx.append(i)
          
subj_idx=np.array(subj_idx)
subj_idx_reshaped = subj_idx.reshape((nsubj*nseq,ntrial))[:,1]

sim_alpha=2

prior_count=np.array([1,1,1,1]) 
sim_alpha_count = sim_alpha*counts+prior_count
sim_alpha_prev_count = sim_alpha*prev_counts+prior_count
sim_probs = sim_alpha_count/np.sum(sim_alpha_count,axis=1).reshape((sim_alpha_count.shape[0], 1))
sim_prev_probs = sim_alpha_prev_count/np.sum(sim_alpha_prev_count,axis=1).reshape((sim_alpha_prev_count.shape[0], 1))#alpha_prev_count/alpha_prev_count.norm(1, axis=1).reshape((alpha_prev_count.shape[0], 1))#alpha_prev_count/T.sum(1/alpha_prev_count,axis=1)

sim_kl = np.sum(sim_probs*np.log2(sim_probs/sim_prev_probs),axis=1)

sim_beta0=0
sim_beta1=10
sim_noise=0.5

sim_est_LT=sim_beta0+sim_beta1*sim_kl

sim_ltime=np.random.normal(sim_est_LT,sim_noise)

plt.scatter(sim_kl,sim_ltime)
plt.hist(sim_ltime)

sim_lambda0 = 0.15
sim_beta_LA = -30

sim_pre_lambda1 = np.reshape(np.exp(sim_beta_LA * sim_kl), (nsubj*nseq,ntrial))
sim_lambda1 = sim_pre_lambda1*sim_lambda0

pre_lookaway=np.random.poisson(sim_lambda1, (nsubj*nseq,ntrial))

watched=np.ones(pre_lookaway.shape)
sim_lookaway=np.zeros(pre_lookaway.shape)

pre_lookaway=np.where(pre_lookaway !=0, 1, pre_lookaway)


for i in range(len(watched)):
    idx=np.where(pre_lookaway[i,:]==1)[0]
    if idx.size>0:
        idx=idx[0]
        watched[i,idx+1:]=0
        sim_lookaway[i,idx]=1

counts=counts.astype(theano.config.floatX)
prev_counts=prev_counts.astype(theano.config.floatX) 
watched=watched.astype(theano.config.floatX) 
sim_lookaway=sim_lookaway.astype(theano.config.floatX) 

# =============================================================================
# Looking Time (ltime)
# =============================================================================
with pm.Model() as model: #           
############################## Hyperpriors ####################################
    mean_b0 = pm.Normal('mean_b0', mu = 0, tau = 0.01)
    prec_b0 = pm.HalfCauchy('prec_b0',  beta= 1 )
    
    mean_b1 = pm.Normal('mean_b1', mu = 0, tau = 0.01)
    prec_b1 = pm.HalfCauchy('prec_b1',  beta= 1 )

################################# Priors ######################################
    LT0 = pm.Normal('LT0', mu = mean_b0, tau=prec_b0, shape = nsubj)
    LT1 = pm.Normal('LT1', mu = mean_b1, tau=prec_b1, shape = nsubj)
    eps = pm.HalfCauchy('eps', beta = 5)
    
    alpha= pm.Gamma('alpha', 2, 1, shape = (nsubj,4)) #this will be a free parameter
    prior_count=np.array([1,1,1,1]) # this will depend on previous evidence with a free weight

########################### Information theory ################################
    # 1. we multiply the counts by alpha [subj_idx] or [seq_idx] and sum to the prior
    alpha_count = alpha[subj_idx]*counts+prior_count
    alpha_prev_count = alpha[subj_idx]*prev_counts+prior_count
    # 2. we trasform counts in probabilities
    probs = alpha_count/T.sum(alpha_count,axis=1).reshape((alpha_count.shape[0], 1))#alpha_count[subj_idx]/alpha_count.norm(1, axis=1).reshape((alpha_count.shape[0], 1))#alpha_count/T.sum(alpha_count,axis=1)
    prev_probs = alpha_prev_count/T.sum(alpha_prev_count,axis=1).reshape((alpha_prev_count.shape[0], 1))#alpha_prev_count/alpha_prev_count.norm(1, axis=1).reshape((alpha_prev_count.shape[0], 1))#alpha_prev_count/T.sum(1/alpha_prev_count,axis=1)
    # 3. we compute KL
    kl =  pm.Deterministic("kl", T.sum(probs*T.log2(probs/prev_probs),axis=1))
    
######################### Estimates and Likelihood ############################
    est_LT = LT0[subj_idx]+LT1[subj_idx]*kl                  
    LT_like = pm.Normal("LT_like", mu=est_LT, sigma=eps, observed=sim_ltime)
    
    
# =============================================================================
# Look Away
# =============================================================================
with model:
############################## Hyperpriors ####################################
    mean_LA = 0#pm.Normal('mean_LA', mu = 0, tau = 0.0001)
    prec_LA = 0.01#pm.HalfCauchy('prec_LA',  beta= 1 )
    
    hyper_alpha = 0.01 #these will be distributions
    hyper_beta = 0.01

################################# Priors ######################################
    lambda0 = pm.Gamma("lambda0", hyper_alpha, hyper_beta, shape = (nsubj, ntrial))
    beta_LA = pm.Normal("beta_LA", mean_LA, tau = prec_LA, shape = nsubj)

######################### Estimates and Likelihood ############################
    pre_lambda1 = T.reshape(T.exp(beta_LA[subj_idx] * kl), watched.shape)
    # multiply it by lambda0
    lambda1 = pm.Deterministic("lambda1", pre_lambda1[subj_idx_reshaped]*lambda0[subj_idx_reshaped])
    
    mu = pm.Deterministic("mu", watched * lambda1)
    LA_like = pm.Poisson("obs", mu, observed=sim_lookaway)


with model:
    start = pm.find_MAP() # Find starting value by optimization
    step = pm.NUTS(scaling=start) # Instantiate MCMC sampling algorithm
    trace = pm.sample(50000, step,  chains=4, cores=6, tune=50000, start=start) 
    
pm.traceplot(trace, var_names=["alpha","mean_b0","mean_b1","LT0","LT1"]);

np.exp(trace["kl"].mean())
pm.plot_posterior(trace, var_names=["beta_LA"], color="#87ceeb");
                  
np.exp(trace["lambda1"].mean())
pm.traceplot(trace, var_names=["lambda1"], coords={"lambda1_dim_0": range(73)});
    
    
    
    
    
    