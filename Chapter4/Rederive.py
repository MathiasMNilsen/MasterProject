'''
This script reproduces the result in doi.org/10.1103/PhysRevLett.122.252302
for  RAA measurments at 5.02TeV at PbPb colloisions using Hamiltonina Monte
Carlo (instead of Random Walk Metropolis)

This will produce the results stated in chapter 4.1 in the thesis  
'''

from Sampler import JetSampler, in_1sigma
from scipy.special import roots_laguerre

import matplotlib.pyplot as plt
import theano.tensor as tt
import numpy as np
import pymc3 as pm

import math
import os


class GammaQuenching(JetSampler):

    def __init__(self, e_cm=5020, rap=2.8):
        ''' Initializing the class instance '''
        super().__init__(e_cm, False, rap, 'Rederive', ['α', 'β', 'γ'], False)
    
    def gamma_dist(self, x, a):
        ''' 
        Gamma function for energyloss plotting

        Args
        ----------------------------------
        x :  Energy-loss values in D(x)
        a :  α-parameter in gamma dist

        Returns
        ----------------------------------
        Value of D(x|α) = α^α * x^(α-1) * e^(-α*x) / Γ(α)
        '''
        return a**a*np.exp(-a*x)*x**(a-1)/math.gamma(a)
    
    def raa_model(self, pT, theta: list): 
        '''
        Computes the convolution D(x)*dN(x+pT)/dpT 
        using Gauss–Laguerre quadrature.
        
        Args
        -------------------------------------------
        pT    : Transvers momenta of jet
        theta : list of parameters [α, β, γ]

        Returns
        -------------------------------------------
        values RAA(pT|theta)
        '''
        a = theta[0]
        b = theta[1]
        g = theta[2]
        integral = 0
        x_GL, w_GL = roots_laguerre(30)
        for x, w in zip(x_GL, w_GL):
            mean_eps = b*(pT)**g*np.log(pT)
            D_eps = x**(a-1)/tt.gamma(a)
            vac   = self.vacuum_func(pT + x*mean_eps/a, *self.vac_prms)
            integral  += w*D_eps*vac
        return integral/self.vacuum_func(pT, *self.vac_prms)
    
    def bayesian_model(self) -> pm.Model:
        '''
        Makes the bayesian model (pymc3 model)

        Returns
        --------------------------------------
        the built pymc3 model
        '''
        with pm.Model() as gamma_qunching:
            pT_data  = pm.Data('RAA_pT', self.raa_pT)
            RAA_data = pm.Data('RAA_data', self.raa)

            α = pm.Uniform('α', 0, 10)
            β = pm.Uniform('β', 0, 10)
            γ = pm.Uniform('γ', 0, 1)
            
            theta = [α, 0.8*β, γ]
            raa = pm.Deterministic('Raa', self.raa_model(pT_data, theta))
            pm.Normal('LH', mu=raa, sigma=self.raa_err, observed=RAA_data)
        return gamma_qunching
    
    def plot_energyloss_dist(self):
        '''
        Plots the energyloss distribution D(x), and plots the
        mean energyloss as function of pT. 
        NB! It only uses parameters with 1-sigma of mean (like the paper)
        '''
        n = 0
        x = np.linspace(0.01, 5, 300)
        dist = np.zeros_like(x)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        for a in self.trace['α']:
            if in_1sigma(a, self.trace['α']):
                dist += self.gamma_dist(x, a)
                ax1.plot(x, self.gamma_dist(x, a), color='lightsteelblue',
                         linewidth = 0.003)
                n += 1
        ax1.plot(x, dist/n, color='darkblue', label=f'Posterior expetation')
        ax1.set_xlabel(r'$x = \frac{\varepsilon}{<\varepsilon>}$')
        ax1.set_yscale('log')
        ax1.set_ylabel('$D(x)$')
        ax1.set_ylim(10e-4, 10e0)
        ax1.set_xlim(0.1, 5)
        ax1.legend()
        jet_pT  = np.linspace(50, 800, 100)
        mean_eps= np.zeros_like(jet_pT)
        n = 0
        for b, c in zip(self.trace['β'], self.trace['γ']):
            if (in_1sigma(b, self.trace['β']) and
                in_1sigma(c, self.trace['γ'])):
                current = b*(jet_pT)**c*np.log(jet_pT) 
                ax2.plot(jet_pT, current, color='lightsteelblue',
                         linewidth = 0.003)
                mean_eps += current
                n += 1

        ax2.plot(jet_pT, mean_eps/n, color='darkblue',
                 label=f'Posterior expetation')
        ax2.set_xlim(50, 800)
        ax2.legend()
        ax2.set_xlabel(r'$p_T$ [GeV]')
        ax2.set_ylabel(r'$<\varepsilon>$ [GeV]')
        plt.gcf().subplots_adjust(bottom=0.15)

        directory = f'{self.model_name}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_name = os.path.join(directory, 'Energyloss.png')
        plt.savefig(path_name, dpi=300)
        return

def main():
    gam = GammaQuenching()
    #gam.plot_vacuum_spectrum()
    gam_model = gam.bayesian_model()
    gam.mcmc_inference(gam_model, 50000, 5000, algo='HMC', load_trace=True)
    gam.plot_RAA_ppc(gam_model)
    gam.plot_corner()
    gam.plot_energyloss_dist()

    #Validation
    gam2 = GammaQuenching(e_cm=2760, rap=2.1)
    gam2.load_sampled_trace(gam2.bayesian_model())
    gam2.plot_RAA_ppc(gam2.bayesian_model(), save_name='Validation2')
if __name__ == '__main__':
    main()