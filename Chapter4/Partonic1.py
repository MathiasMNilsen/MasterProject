from Sampler import JetSampler, in_1sigma
from scipy.special import roots_hermite

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import os


class QuarkGluonQuenching(JetSampler):

    def __init__(self, e_cm=5020, rap=2.8, nPDF=True, name=None, obs='RAA'):
        ''' Initializing the class instance '''

        prms_names = ['μ1', 'μ2', 'σ1', 'σ2', 'δ']
        if not name: name = 'Partonic'
        super().__init__(e_cm, nPDF, rap, name, prms_names, True, obs)
        
    def logNormal(self, x, μ: float, σ: float):
        '''
        Args
        ----------------------------------
        x :  Energy-loss values in D(x)
        μ :  μ-parameter in Lognormal dist
        σ :  σ-parameter in Lognormal dist
        
        Returns
        ----------------------------------
        Value of Lognormal(x|μ,σ)
        '''
        return (np.exp(-(np.log(x)-μ)**2 / (2*σ**2))) / (np.sqrt(2*np.pi)*σ*x)
    
    def raa_model(self, pT, theta: list, n: int = 30): 
        '''
        Computes the convolution D(x)*dN(x+pT)/dpT 
        using Gauss–Laguerre quadrature (or used Gauss–Hermite).
        
        Args
        -------------------------------------------
        pT    : Transvers momenta of jet
        theta : list of parameters [μ1, σ1, μ2, σ2]
        n     : Number of integration points

        Returns
        -------------------------------------------
        values RAA(pT|theta)
        '''
        mQ = theta[0]
        sQ = theta[1]
        mG = theta[2]
        sG = theta[3]
        integral = 0
        x_GH, w_GH = roots_hermite(n)
        for x, w in zip(x_GH, w_GH):
            qrk_pT = np.exp(np.sqrt(2)*sQ*x + mQ) + pT 
            gln_pT = np.exp(np.sqrt(2)*sG*x + mG) + pT
            qrk_eloss = self.vacuum_func(qrk_pT, *self.qrk_prms) 
            gln_eloss = self.vacuum_func(gln_pT, *self.gln_prms)
            integral += w*(qrk_eloss+gln_eloss)/np.sqrt(np.pi)
        
        if self.obs == 'RAA':
            return integral/self.vacuum_func(pT, *self.vac_prms)
        else:
            integral = 0.2212*integral
            return integral
        
    def bayesian_model(self) -> pm.Model:
        '''
        Makes the bayesian model (pymc3 model)

        Returns
        --------------------------------------
        the built pymc3 model
        '''
        c_F = 4/3
        c_A = 3
        a   = 0.3
        with pm.Model() as quark_gluon_qunching:
            pT_data  = pm.Data('RAA_pT', self.raa_pT)
            RAA_data = pm.Data('RAA_data', self.raa)
            
            μ1 = pm.Uniform('μ1', 0, 10)
            μ2 = pm.Uniform('μ2', 0, 10)
            σ1 = pm.Uniform('σ1', 0, 10)
            σ2 = pm.Uniform('σ2', 0, 10)
            δ  = pm.Uniform('δ', 0, 1)    #Used in the likelihood
            
            theta = [μ1, σ1, μ2, σ2]
            raa = pm.Deterministic('Raa', self.raa_model(pT_data, theta))

            pm.Normal('LH', mu=raa, sigma=δ*self.raa_err, observed=RAA_data)
        return quark_gluon_qunching
    
    def plot_energyloss_dist(self):
        #Must be fixed!!
        c_F = 4/3
        c_A = 3
        a = 0.2
        ε_g = np.logspace(-3, 2, 100)
        ε_q = np.logspace(-4, 2, 100)
        s1_tr  = self.trace['σ1']
        s2_tr  = self.trace['σ2']
        m1_tr  = self.trace['μ1']
        m2_tr  = self.trace['μ2']
        s1_mean = np.mean(s1_tr)
        s2_mean = np.mean(s2_tr)
        m1_mean = np.mean(m1_tr)
        m2_mean = np.mean(m2_tr)
        distQ = np.zeros_like(ε_q)
        distG = np.zeros_like(ε_g)  

        plt.figure()
        for m1, s1, m2, s2 in zip(m1_tr, s1_tr, m2_tr, s2_tr): 
            current_Q = self.logNormal(ε_q, m1, s1)
            distQ += current_Q
            if (in_1sigma(m1, m1_tr) and in_1sigma(s1, s1_tr)):
                plt.plot(ε_q, current_Q, color='lightsteelblue',
                         linewidth = 0.05)
            
            current_G = self.logNormal(ε_g, m2, s2)
            distG += current_G
            if (in_1sigma(m2, m2_tr) and in_1sigma(s2, s2_tr)):
                plt.plot(ε_g, current_G, color='rosybrown',
                         linewidth = 0.05)
        
       
        plt.plot(ε_q, self.logNormal(ε_q, m1_mean, s1_mean),
                 color='darkblue',
                 label=f'Posterior expetation for quark jets')
        plt.plot(ε_g, self.logNormal(ε_g, m2_mean, s2_mean),
                 color='darkred',
                 label=f'Posterior expetation for gluon jets')

        plt.xlabel(r'$\varepsilon$ [GeV]')
        plt.ylabel(r'$D(\varepsilon)$')
        #plt.xlim(0.01, 100)
        #plt.ylim(0.001, 0.1)
        plt.xscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()
        directory = f'{self.model_name}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_name = os.path.join(directory, 'Partonic_Eloss.png')
        plt.savefig(path_name)
        return

def main():
    QGQ = QuarkGluonQuenching(e_cm=5020, nPDF=True, name='Partonic1', obs='RAA')
    model = QGQ.bayesian_model()
    QGQ.mcmc_inference(model, 30000, 10000, algo='HMC')
    QGQ.plot_RAA_ppc(model)
    QGQ.plot_energyloss_dist()
    QGQ.plot_corner()


    new_QGQ = QuarkGluonQuenching(e_cm=2760, rap=2.1,
                                            nPDF=True, name='Prediction')
    new_QGQ.plot_vacuum_spectrum()
    new_QGQ.plot_RAA_Data()
    new_QGQ.load_sampled_trace(new_QGQ.bayesian_model(), 'Partonic1')
    new_QGQ.plot_RAA_ppc(new_QGQ.bayesian_model(), 'Partonic1', 'Validation')

main()


