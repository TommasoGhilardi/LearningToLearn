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
from collections import Counter
import matplotlib.pyplot as plt

Path= r"C:\Users\krav\Desktop\BabyBrain\Projects\Weighting_evidence\Model\Data\\"

#import sequences
sequences= pd.read_csv(Path+'sequences_t.csv', header=None).transpose()

# set up some stuff for later
nsubj = 70
ntrial=15
nseq=10
subj_idx = []
seq_idx=[]

# Initialize variables for loop
counts=np.zeros(shape=(nsubj*nseq*ntrial,4)) #counts[:]=np.NaN # or zero?
prev_counts=np.zeros(shape=(nsubj*nseq*ntrial,4)) #prev_counts[:]=np.NaN # or zero?
offtrials = np.zeros(shape=(nsubj*nseq*ntrial,))
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
            seq_idx.append(s-1)
            if sequences.iloc[t-1,s-1]!=np.bincount(sequences.iloc[:,s-1]).argmax():
                offtrials[i*nseq*ntrial+ntrial*(s-1)+t-1]=1

trial=np.tile(range(1,ntrial+1),nseq*nsubj) #indicates trial number
subj_idx=np.array(subj_idx) #indicates subject number
seq_idx=np.array(seq_idx) #indicates sequence number

# compute kl Divergence
evidence_weight=1 #always set - not a free parameter
prior_count=np.array([1,1,1,1]) #always set - not a free parameter
sim_alpha_count = evidence_weight*counts+prior_count # computes counts
sim_alpha_prev_count = evidence_weight*prev_counts+prior_count # compute previous counts
sim_probs = sim_alpha_count/np.sum(sim_alpha_count,axis=1).reshape((sim_alpha_count.shape[0], 1)) #from counts to probabilities
sim_prev_probs = sim_alpha_prev_count/np.sum(sim_alpha_prev_count,axis=1).reshape((sim_alpha_prev_count.shape[0], 1)) # same
sim_kl = np.sum(sim_probs*np.log2(sim_probs/sim_prev_probs),axis=1) # kl divergence


#plot example of kl in a given sequence
plt.plot(sim_kl[-15:])
plt.xticks(np.arange(1, 16, step = 2))
plt.ylabel('Information gain')
plt.xlabel('Trial number')

# set parameter values to change kl across sequences and trials (meta learning)
b0_alpha = 0.451	
b1_alpha = 0.162
b0_seq =0.092	
b1_seq = 0.034

sim_kl_time = sim_kl*(b0_alpha+b1_alpha*seq_idx)*np.exp(-(b0_seq+b1_seq*seq_idx)*trial)

# plot example
plt.figure(figsize=(5,4))
plt.plot(range(1,16),sim_kl_time[-15:])
plt.xticks(np.arange(1, 16, step = 2))
plt.ylabel('Information gain')
plt.xlabel('Trial number')

# generate the data, i.e., specify the parameters that define the likelihoods
# for looking time
sim_beta0=-0.8
sim_beta1=20
sim_noise=0.5

sim_est_LT=sim_beta0+sim_beta1*sim_kl_time

sim_ltime=np.random.normal(sim_est_LT,sim_noise)
# add random missing values to ltime
setasnan=np.random.randint(0,len(sim_ltime),int(len(sim_ltime)*.2))
#sim_ltime[setasnan]=np.nan

plt.scatter(sim_kl_time,sim_ltime)
plt.hist(sim_ltime)

# for saccadic latency
sim_beta0=-0.5
sim_beta1=10
sim_noise=0.3

sim_est_SL=sim_beta0+sim_beta1*sim_kl_time

sim_slat=np.random.normal(sim_est_SL,sim_noise)
# add random missing values to ltime
setasnan=np.random.randint(0,len(sim_slat),int(len(sim_slat)*.2))
#sim_slat[setasnan]=np.nan

plt.scatter(sim_kl_time,sim_slat)
plt.hist(sim_slat)

# for look-aways
# set parameters for lambda 0 and beta_LA
sim_lambda0 = 0.5
sim_beta_LA = -50

# get lambda 1
sim_pre_lambda1 = np.reshape(np.exp(sim_beta_LA * sim_kl), (nsubj*nseq,ntrial))
sim_lambda1 = sim_pre_lambda1*sim_lambda0

# sample from poisson distribution to get lookaways
pre_lookaway=np.random.poisson(sim_lambda1, (nsubj*nseq,ntrial))

# with poisson, multiple observations are possible. For us, only one. So we censor the data
#initialize some stuff
watched=np.ones(pre_lookaway.shape)
sim_lookaway=np.zeros(pre_lookaway.shape)

#set all observations to 1 (e.g., not 2 lookaways at the same time are possible)
pre_lookaway=np.where(pre_lookaway !=0, 1, pre_lookaway)

# now remove all lookaways observed after the first one
for i in range(len(watched)):
    idx=np.where(pre_lookaway[i,:]==1)[0]
    if idx.size>0:
        idx=idx[0]
        watched[i,idx+2:]=0
        if np.random.binomial(1,0.3,1)[0]==1:
            watched[i,0:3]=0
        sim_lookaway[i,idx]=1

counts=counts.astype(theano.config.floatX)
prev_counts=prev_counts.astype(theano.config.floatX) 
watched=watched.astype(theano.config.floatX) 
sim_lookaway=sim_lookaway.astype(theano.config.floatX) 

#reshape everything
watched=watched.reshape(subj_idx.shape)
sim_lookaway = sim_lookaway.reshape(subj_idx.shape)

la_trial = trial[sim_lookaway==1]
plt.hist(la_trial)

# keep only trials that have been watched
sim_lookaway=sim_lookaway[watched==1]
sim_ltime=sim_ltime[watched==1]
sim_slat=sim_slat[watched==1]
subj_idx=subj_idx[watched==1]
seq_idx=seq_idx[watched==1]
counts=counts[watched==1]
prev_counts=prev_counts[watched==1]

sim_ltime = np.ma.masked_invalid(sim_ltime) 
sim_slat = np.ma.masked_invalid(sim_slat) 

    
    
    
    