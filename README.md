 
<p align="center">
  <img src="https://img.shields.io/badge/python-3.6-green.svg"
              alt="Python 3.6">
  <a href="the url to paper">
          <img src="https://img.shields.io/badge/Doi-the actual doi-blue.svg"
              alt="follow on Twitter"></a>
  <a href="https://twitter.com/intent/follow?screen_name=francescpoli">
          <img src="https://img.shields.io/twitter/follow/francescpoli?style=social&logo=twitter"
              alt="follow on Twitter"></a>
  <a href="https://twitter.com/intent/follow?screen_name=tommasoghi">
          <img src="https://img.shields.io/twitter/follow/tommasoghi?style=social&logo=twitter"
              alt="follow on Twitter"></a>
</p>
   
      
<h1 align="center">Learning to learn: evidence for meta-learning in 8-month-old infants</h1>




**[About the project](#about-the-project)**<br>
**[Try for yourself](#try-for-yourself )**<br>


## About the project:
Meta-learning is the ability to make use of prior experience to learn how to learn better in the future. Whether infants possess meta-learning abilities is still unknown. We propose a Bayesian hierarchical model that keeps track of how the informational value of the stimuli changes across time. By fitting the model to infants’ eye-tracking data, we show that infants attention and information processing is in line with the presence of meta-learning abilities.


### The task

Infants were presented with sequences of cue-target trials. In each sequence, the cue consisted of a simple shape appearing in the middle of the screen. The target was the same shape reappearing in one of four screen quadrants around the cue location. The shape was the same across all trials of the sequence but changed across sequences. The target could appear in any location, but one location was more likely than the others. Infants could thus learn to predict the most likely target location of each sequence.

<p align="center">
  <img src="https://www.science.org/cms/10.1126/sciadv.abb5053/asset/07688eb0-7eb8-4560-ae8c-b60b72e79a1e/assets/graphic/abb5053-f1.jpeg" width="300" height="300" />
</p>

For more information about the task and the experimental settings check __[Infants tailor their attention to maximize learning](https://www.science.org/doi/10.1126/sciadv.abb5053)__

### The model

In every trial *t* of a sequence *s*, a stimulus is shown in the target location *x_(s,t)* and the probability *P(X_(s,t))*  of seeing the stimulus in any given location is updated in light of the new evidence *x_(s,t)*, starting from the initial uniform prior *γ_s*, which assumes that the target is equally likely to appear in any of the four locations. In every trial, *P(X_(s,t))* is used to compute the information gain carried by the new stimulus. Information gain, *IG* ,is quantified using the Kullback-Leibler (KL) divergence: 
<p align="center">
<img src="http://www.sciweavers.org/tex2img.php?eq=D_%7BK%20L%7D%3D%5Csum%20P%28X%29_%7Bs%2C%20t%7D%20%5Clog%20%5Cleft%28%5Cfrac%7BP%28X%29_%7Bs%2C%20t%7D%7D%7BP%28X%29_%7Bs%2C%20t-1%7D%7D%5Cright%29%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="D_{K L}=\sum P(X)_{s, t} \log \left(\frac{P(X)_{s, t}}{P(X)_{s, t-1}}\right)" width="261" height="47" />
</p>

Information gain is assumed to vary linearly with Saccadic Latency (SL), Looking Time (LT), and Look-Away (LA). When estimating the relationship between information gain and the dependent variables, the regression coefficients (see Figure 2, in yellow) are estimated for each participant, thus taking into account individual differences. To quantify infants’ meta-learning abilities, four additional parameters *λ_α^0*, *λ_α^1*, *β_α^0*, and *β_α^1* are used to describe an exponential decay over trials. This allows us to track how the exponential decay of information gain varied across sequences, thus testing our hypothesis on up- and down-regulation of evidence   across the task. Specifically, *λ_α^0* and *λ_α^1* regulate the up-weighting across sequences of the information acquired in trials early in the sequence, while *β_α^0* and *β_α^1* regulate the down-weighting across sequences of the information acquired in trials late in the sequence. The parameter *λ_s* controls for changes in baseline attention to the task across sequences

<p align="center">
  <img src="https://github.com/TommasoGhilardi/LearningToLearn/blob/main/model.jpg" width="400" />
</p>



## Try for yourself 

### Install pymc3 and its dependencies
To facilitate the process and aim for the best reproducibility we provide the **pymc_enviroment.yml** file in the [General folder](https://github.com/TommasoGhilardi/LearningToLearn/tree/main/General). To create a conda enviroment called *pymc* run the command:  ```conda env create -f pymc_enviroment.yml``` in the conda terminal.
After setting the enviroment it can be accessed using ```conda activate pymc```, from here the ilde **Spyder** can be accessed running ```spyder```


## Contributors:

|Name     |  Twitter  | 
|---------|-----------------|
|__[Francesco Poli](https://francescopoli.weebly.com/)__| [@FrancescPoli](https://twitter.com/francescpoli) |
|__[Tommaso Ghilardi](tommasoghilardi.github.io/])__ | [@TommasoGhi](https://twitter.com/tommasoghi) |

