 
<p align="center">
  <img src="https://img.shields.io/badge/python-3.6-green.svg"
              alt="Python 3.6">
  <a href="https://psyarxiv.com/dc9s6/">
          <img src="https://img.shields.io/badge/Doi-10.31234/osf.io/dc9s6-blue.svg"
              alt="follow on Twitter"></a>
  <a href="https://twitter.com/intent/follow?screen_name=francescpoli">
          <img src="https://img.shields.io/twitter/follow/francescpoli?style=social&logo=twitter"
              alt="follow on Twitter"></a>
  <a href="https://twitter.com/intent/follow?screen_name=tommasoghi">
          <img src="https://img.shields.io/twitter/follow/tommasoghi?style=social&logo=twitter"
              alt="follow on Twitter"></a>
</p>
   
      
<h1 align="center">Eight-Month-Old Infants Meta-Learn by Downweighting Irrelevant Evidence</h1>




- **[About the project](#about-the-project)**
  - **[The_task](#the-task)**
  - **[The_model](#the-model)**
- **[Try for yourself](#try-for-yourself )**
	- **[Install pymc3](#install-pymc3-and-its-dependencies)**
	- **[Run the model](#run-the-model)**



## About the project:
Meta-learning is the ability to make use of prior experience to learn how to learn better in the future. Whether infants possess meta-learning abilities is still unknown. We propose a Bayesian hierarchical model that keeps track of how the informational value of the stimuli changes across time. By fitting the model to infants’ eye-tracking data, we show that infants attention and information processing is in line with the presence of meta-learning abilities.


### The task
Infants were presented with sequences of cue-target trials. In each sequence, the cue consisted of a simple shape appearing in the middle of the screen. The target was the same shape reappearing in one of four screen quadrants around the cue location. The shape was the same across all trials of the sequence but changed across sequences. The target could appear in any location, but one location was more likely than the others. Infants could thus learn to predict the most likely target location of each sequence.
<br>

<p align="center">
  <img src="https://github.com/TommasoGhilardi/LearningToLearn/blob/main/task.jpeg" width="300" />
  <em><br>Fig1 Task representation</em>
</p>

<br>
Three different variables from the infants’ looking behavior were extracted:

1. **Look-away:** At each trial, we recorded whether infants kept looking at the screen or looked away;
2. **Saccadic latency:** How quickly infants moved their eyes from the cue to the target location, from the moment the target appeared. Negative times (i.e., anticipations to the target location) were also possible;
3. **Looking time:** How long infants looked at the target location, from the moment it appeared to 750ms after its disappearance.
<br>

For more information about the task and the eye-tracking processing check **[Infants tailor their attention to maximize learning](https://www.science.org/doi/10.1126/sciadv.abb5053)**


### The model

In every trial ***t*** of a sequence ***s***, a stimulus is shown in the target location ***x_(s,t)*** and the probability  ***P(X_(s,t))***  of seeing the stimulus in any given location is updated in light of the new evidence  ***x_(s,t)***,  starting from the initial uniform prior  ***γ_s*** , which assumes that the target is equally likely to appear in any of the four locations. In every trial,  ***P(X_(s,t))***  is used to compute the information gain carried by the new stimulus. Information gain, ***IG***,  is quantified using the Kullback-Leibler (KL) divergence: 

<p align="center">
 <img src="https://latex.codecogs.com/gif.image?\dpi{100}&space;\bg_white&space;D_{K&space;L}=\sum&space;P(X)_{s,&space;t}&space;\log&space;\left(\frac{P(X)_{s,&space;t}}{P(X)_{s,&space;t-1}}\right)" title="\bg_white D_{K L}=\sum P(X)_{s, t} \log \left(\frac{P(X)_{s, t}}{P(X)_{s, t-1}}\right)" />
</p>

Information gain is assumed to vary linearly with Saccadic Latency ( ***SL*** ), Looking Time ( ***LT*** ), and Look-Away ( ***LA*** ). When estimating the relationship between information gain and the dependent variables, the regression coefficients (see Figure 2, in yellow) are estimated for each participant, thus taking into account individual differences. To quantify infants’ meta-learning abilities, four additional parameters ***λ_α^0***,  ***λ_α^1***, ***β_α^0***,  and  ***β_α^1***  (see Figure 2, in green) are used to describe an exponential decay over trials. This allows us to track how the exponential decay of information gain varied across sequences, thus testing our hypothesis on up- and down-regulation of evidence   across the task. Specifically, ***λ_α^0***  and  ***λ_α^1***  regulate the up-weighting across sequences of the information acquired in trials early in the sequence, while ***β_α^0***  and  ***β_α^1***  regulate the down-weighting across sequences of the information acquired in trials late in the sequence. The parameter  ***λ_s***  controls for changes in baseline attention to the task across sequences

<p align="center">
  <img src="https://github.com/TommasoGhilardi/LearningToLearn/blob/main/model.jpg" width="400" />
  <em><br>Fig2 Model representation</em>
</p>



## Try for yourself 

### Install pymc3 and its dependencies
To facilitate the process and aim for the best reproducibility we provide the **pymc_enviroment.yml** file in the [General folder](https://github.com/TommasoGhilardi/LearningToLearn/tree/main/General). To create a conda enviroment called *pymc* run the command:  ```conda env create -f pymc_environment.yml``` in the conda terminal.
After setting the enviroment it can be accessed using ```conda activate pymc```, from here the ilde **Spyder** can be accessed running ```spyder```

### Run the model


You can choose between two sampling methods using the handle ```useADVI```. When set to ```True``` Pymc3 will use ADVI to sample the data, if set to ```False``` Pymc3 will use MCMC ([learn more abour ADVI](https://arxiv.org/pdf/1603.00788.pdf) ).

We advice to use MCMC to obtain reliable results and only use ADVI to check if the model is properly running after changes.

#### Simulation
In addition to the data we collected we provide the ```SimulateData``` function. This function allows to simulate similar data as the one collected. Differntly from the collected data the function allow to specify all the paramenters that will shape the data. Set the handle ```Simulation =  True``` to simulate and recover the data using the model.

Multiple parameters need to be passed to the ```SimulateData``` function as dictionary. Default parameters are provided.
```
{'nsubj':70, 'ntrial':15, 'nseq':10,                                  # experiment parameters
'b0_alpha':0.451, 'b1_alpha':0.162, 'b0_seq':0.092, 'b1_seq':0.034,   # set parameters to change kl across sequences and trials (meta learning) 
'sim_beta0_LT':-0.8, 'sim_beta1_LT':20, 'sim_noise_LT':0.5,           # specify the parameters that define the likelihoods for looking time
'sim_beta0_SL':-0.5, 'sim_beta1_SL':10, 'sim_noise_SL':0.3,           # specify the parameters that define the likelihoods for saccadic latency
'sim_lambda0' : 0.5, 'sim_beta_LA' : -50                              # set parameters for lambda 0 and beta_LA}
 ```

## Contributors:

|Name     |  Twitter  | 
|---------|-----------------|
|__[Francesco Poli](https://francescopoli.weebly.com/)__| [@FrancescPoli](https://twitter.com/francescpoli) |
|__[Tommaso Ghilardi](tommasoghilardi.github.io/])__ | [@TommasoGhi](https://twitter.com/tommasoghi) |

