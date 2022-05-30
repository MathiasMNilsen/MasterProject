Below you will find an explanation of the different files

Data        -   Folder containing the data

mymain01.cc -   c++ code for generating the quark and gluon jets in Pythia

Sampler.py  -   Contains two calsses. JetSampler which performs the fits the 
                vacuum spectrum to the pp-data, does the Bayesian analysis, 
                samples and plots the posterior. The second class 
                QuarkGluonQuenching is makes the bayesian model for quark/gluon
                quenching. For details, look at the code.

Rederive.py -   Rederives the results of Xin Nian and others using Hamiltonian
                Monte Carlo. Results stored in folder named "Rederive"

Partonic.py -   Does the Bayesian inference for the final model Results 
                stored in folder named "Partonc"


Traces      -   Folder containing the MCMC samples (trace) of the models.

nPDF_effect.py  -   Shows the effect of the nuclear bound PDF. Produces the 
                    plot "nPDF"