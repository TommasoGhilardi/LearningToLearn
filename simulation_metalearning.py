# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:45:35 2021

@author: Francesco Poli, Tommaso Ghilardi, Max Hinne
"""

import os
import theano
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from collections import Counter


def SimulateData( dict_sim = {'nsubj':70, 'ntrial':15, 'nseq':10,                               # set up some stuff for later
        'b0_alpha':0.451, 'b1_alpha':0.162, 'b0_seq':0.092, 'b1_seq':0.034,   # set parameter values to change kl across sequences and trials (meta learning) 
        'sim_beta0_LT':-0.8, 'sim_beta1_LT':20, 'sim_noise_LT':0.5,           # specify the parameters that define the likelihoods for looking time
        'sim_beta0_SL':-0.5, 'sim_beta1_SL':10, 'sim_noise_SL':0.3,           # specify the parameters that define the likelihoods for saccadic latency
        'sim_lambda0' : 0.5, 'sim_beta_LA' : -50                              # set parameters for lambda 0 and beta_LA
        }, missing = True,  Plot = False):


    #import sequences
    sequences= pd.read_csv('Data' + os.sep + 'sequences_t.csv',header=None).transpose()

    # set up some stuff for later
    sim_subj_idx = []
    sim_seq_idx=[]
    
    # Initialize variables for loop
    counts=np.zeros(shape=(dict_sim['nsubj']*dict_sim['nseq']*dict_sim['ntrial'],4)) #counts[:]=np.NaN # or zero?
    prev_counts=np.zeros(shape=(dict_sim['nsubj']*dict_sim['nseq']*dict_sim['ntrial'],4)) #prev_counts[:]=np.NaN # or zero?
    offtrials = np.zeros(shape=(dict_sim['nsubj']*dict_sim['nseq']*dict_sim['ntrial'],))
    # for every subject
    for i in range(dict_sim['nsubj']):
        # for every sequence
        for s in range(1,dict_sim['nseq']+1):
            this_count=Counter({1:0, 2:0, 3:0, 4:0})
            # for every trial
            for t in range(1,dict_sim['ntrial']+1):
                # if the trial has been whatched, update count and assign
                prev_counts[i*dict_sim['nseq']*dict_sim['ntrial']+dict_sim['ntrial']*(s-1)+t-1,:]=list(this_count.values())
                this_count.update([sequences.iloc[t-1,s-1]])
                counts[i*dict_sim['nseq']*dict_sim['ntrial']+dict_sim['ntrial']*(s-1)+t-1,:]=list(this_count.values())
                # update subject index
                sim_subj_idx.append(i)
                sim_seq_idx.append(s-1)
                if sequences.iloc[t-1,s-1]!=np.bincount(sequences.iloc[:,s-1]).argmax():
                    offtrials[i*dict_sim['nseq']*dict_sim['ntrial']+dict_sim['ntrial']*(s-1)+t-1]=1
    
    sim_trial_vect=np.tile(range(1,dict_sim['ntrial']+1),dict_sim['nseq']*dict_sim['nsubj']) #indicates trial number
    sim_subj_idx=np.array(sim_subj_idx) #indicates subject number
    sim_seq_idx=np.array(sim_seq_idx) #indicates sequence number
    
    # compute kl Divergence
    evidence_weight=1 #always set - not a free parameter
    prior_count=np.array([1,1,1,1]) #always set - not a free parameter
    sim_alpha_count = evidence_weight*counts+prior_count # computes counts
    sim_alpha_prev_count = evidence_weight*prev_counts+prior_count # compute previous counts
    sim_probs = sim_alpha_count/np.sum(sim_alpha_count,axis=1).reshape((sim_alpha_count.shape[0], 1)) #from counts to probabilities
    sim_prev_probs = sim_alpha_prev_count/np.sum(sim_alpha_prev_count,axis=1).reshape((sim_alpha_prev_count.shape[0], 1)) # same
    sim_kl = np.sum(sim_probs*np.log2(sim_probs/sim_prev_probs),axis=1) # kl divergence
    
    sim_kl_time = sim_kl*(dict_sim['b0_alpha']+dict_sim['b1_alpha']*sim_seq_idx)*np.exp(-(dict_sim['b0_seq']+dict_sim['b1_seq']*sim_seq_idx)*sim_trial_vect)
    

    ### Looking time 
    sim_est_LT=dict_sim['sim_beta0_LT']+dict_sim['sim_beta1_LT']*sim_kl_time
    
    sim_ltime=np.random.normal(sim_est_LT,dict_sim['sim_noise_LT'])
    
    if missing == True:
        # add random missing values to ltime
        setasnan=np.random.randint(0,len(sim_ltime),int(len(sim_ltime)*.2))
        sim_ltime[setasnan]=np.nan
    
    
    ### Saccadic latency
    sim_est_SL=dict_sim['sim_beta0_SL']+dict_sim['sim_beta1_SL']*sim_kl_time
    
    sim_slat=np.random.normal(sim_est_SL,dict_sim['sim_noise_SL'])
    
    if missing == True:
        # add random missing values to saccadic latency
        setasnan=np.random.randint(0,len(sim_slat),int(len(sim_slat)*.2))
        sim_slat[setasnan]=np.nan  
    
    
    ### Look away
    # get lambda 1
    sim_pre_lambda1 = np.reshape(np.exp(dict_sim['sim_beta_LA'] * sim_kl), (dict_sim['nsubj']*dict_sim['nseq'],dict_sim['ntrial']))
    sim_lambda1 = sim_pre_lambda1*dict_sim['sim_lambda0']
    
    # sample from poisson distribution to get lookaways
    pre_lookaway=np.random.poisson(sim_lambda1, (dict_sim['nsubj']*dict_sim['nseq'],dict_sim['ntrial']))
    
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
    watched=watched.reshape(sim_subj_idx.shape)
    sim_lookaway = sim_lookaway.reshape(sim_subj_idx.shape)
    
    la_trial = sim_trial_vect[sim_lookaway==1]

    
    
    #### Plot    
    if Plot == True :        
         
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(3,2,1)
        ax1.plot(range(1,16),sim_kl[-15:], label = 'actual kl')     #plot example of kl in a given sequence
        ax1.plot(range(1,16),sim_kl_time[-15:], label='simulated kl')     # plot example sim_kl
        ax1.set(xlabel= 'Trial number' , ylabel='Information gain', title = 'Kl in different sequences' )
        ax1.legend(title='KL at given sequence')

        ax2 = plt.subplot(3,2,2)
        ax2.hist(la_trial)
        ax2.set(title='Distribution Look Away')     

        ax3 = plt.subplot(3,2,3)
        ax3.scatter(sim_kl_time,sim_ltime,color='green')
        ax3.set(xlabel= 'Sim Kl' , ylabel='Sim Looking Time', title='Looking time') 
       
        ax4 = plt.subplot(3,2,4)
        ax4.hist(sim_ltime,color='green')
        ax4.set(title='Distribution Sim Looking time')
       
        ax5 = plt.subplot(3,2,5)
        ax5.scatter(sim_kl_time,sim_slat,color='blue')
        ax5.set(xlabel= 'Sim Kl' , ylabel='Sim Saccadic Latency',title='Saccadic latency')
       
        ax6 = plt.subplot(3,2,6)
        ax6.hist(sim_slat,color='blue')
        ax6.set(title='Distribution Sim Saccadic Latency')       
       
        plt.tight_layout()


    # keep only trials that have been watched
    sim_lookaway=sim_lookaway[watched==1]
    sim_ltime=sim_ltime[watched==1]
    sim_slat=sim_slat[watched==1]
    sim_subj_idx=sim_subj_idx[watched==1]
    sim_seq_idx=sim_seq_idx[watched==1]
    counts=counts[watched==1]
    prev_counts=prev_counts[watched==1]
    
    sim_ltime = sc.stats.zscore(sim_ltime, nan_policy='omit')
    sim_slat = sc.stats.zscore(sim_slat, nan_policy='omit')

    sim_ltime = np.ma.masked_invalid(sim_ltime) 
    sim_slat = np.ma.masked_invalid(sim_slat) 
    
    
        
    return(sim_subj_idx, sim_seq_idx, sim_seq_idx, sim_trial_vect, sim_kl, sim_ltime, sim_slat,sim_ltime)