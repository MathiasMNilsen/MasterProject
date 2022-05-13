import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.special import roots_hermite
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
import os
sns.set_theme(style="ticks")

def in_1sigma(var: float, var_array) -> bool:
    var_mean = np.mean(var_array)
    var_std  = np.std(var_array)
    if (var >= var_mean - var_std and 
        var <= var_mean + var_std):
        return True
    else:
        return False
        

class JetSampler:

    def __init__(self,
                 e_cm: float,
                 nPDF: bool,
                 y_max: float = 2.8,
                 name: str = '', 
                 parameters: list = [], 
                 partonic: bool = False,
                 obs: str = 'RAA',
                 ) -> None:
        '''
        JetSampler is a class that loads the data to be used, 
        does the MCMC sampling for Baysesian inference.
        It also makes the posterior predictions

        Args
        ---------------------------------------------------------
        e_cm       : Centre of mass energy [GeV] (for datafile)
        y_max      : Max rapidity, use data for |y| < y_max
        name       : Name of the qunechning model (for saving)
        parameters : List of names of the parameters of the model
        partonic   : If True, partonic energyloss will be used
        '''
        self.e_cm         = e_cm
        self.nPDF         = nPDF
        self.y_max        = y_max
        self.partonic     = partonic
        self.model_name   = name 
        self.parameters   = parameters 
        self.obs          = obs
        self._do_sampling = True
        self.trace        = None

        ''' Load the RAA data '''
        data_raa = self.raa_data(f'Data/RAA/{self.obs}_{self.e_cm}.csv')
        self.raa_pT  = data_raa[0]
        self.pT_bin  = data_raa[1]
        self.raa     = data_raa[2]
        self.raa_err = data_raa[3]

        ''' Load and fit the vacuum spectrum '''
        if not self.partonic:
            pp_data = self.pp_data(f'Data/Vacuum/{self.e_cm}/pp_data.csv')
            self.vac_pT = pp_data[0]
            self.vac_y  = pp_data[1]
            self.vac_prms = self.get_prms(self.vac_pT, self.vac_y)
        else:
            path =f'Data/Vacuum/{self.e_cm}/'
            if self.nPDF: pdf  = '_nPDF'
            else: pdf  = ''
            tagFilename = f'{path}taggedJets_y{self.y_max}{pdf}.csv'
            vacFilename = f'{path}inclusiveJets_y{self.y_max}.csv'
            _, self.qrk_y , self.gln_y = self.pp_data(tagFilename, nCol=3)
            self.vac_pT, self.vac_y = self.pp_data(vacFilename)
            self.qrk_pT = self.gln_pT = self.vac_pT 

            self.vac_prms = self.get_prms(self.vac_pT, self.vac_y)
            self.qrk_prms = self.get_prms(self.qrk_pT, self.qrk_y)
            self.gln_prms = self.get_prms(self.gln_pT, self.gln_y)

    def vacuum_func(self, pT, a, n, B, G, D):
        '''Function to be fitted to the vacuum spectrum'''
        p0 = self.vac_pT[0] 
        m  = n - B*np.log(p0/pT) - G*np.log(p0/pT)**2 - D*np.log(p0/pT)**3
        return a*(self.vac_pT[0]/pT)**m
    
    def get_prms(self, pT, vals):
        prms, _ = curve_fit(self.vacuum_func, pT, vals, 
                            maxfev=20000,
                            sigma=0.05*vals)
        return prms

    def pp_data(self, filename: str, sep: str = '  ', nCol: int = 2) -> tuple:
        '''
        Loads the data for pp spectrum
              
        Args
        -----------------------------------------
        filename : Name of datafile
        sep      : Symbol separating the coloumns

        Returns
        -----------------------------------------
        x, y : (pT [GeV], pectrum) numpy arrays
        '''
        if not self.partonic:
            data = pd.read_csv(filename, skiprows=12)
            x = data.values[:,0]
            y = data.values[:,3]
            self.dpT = (data.values[:,2]-data.values[:,1])/2
            self.pp_err = np.zeros_like(y)
            for i in [4, 6, 8]:
                self.pp_err += data.values[:,i]**2
            self.pp_err  = np.sqrt(self.pp_err)
            return x, y
        else:
            data = pd.read_csv(filename, sep=sep, header=None)
            if nCol == 3:
                x = data.values[:,0]
                y = data.values[:,1]
                z = data.values[:,2]
                return x, y, z
            else:
                x = data.values[:,0]
                y = data.values[:,1]
                return x, y

    def raa_data(self, filename: str, sep: str = '\t') -> tuple:
        '''
        Loads the RAA data:

        Args
        -----------------------------------------
        filename : Name of datafile
        sep      : Symbol separating the coloumns

        Returns
        -----------------------------------------
        RAA_x     :  pT [GeV]
        RAA_x_err :  pT binsize
        RAA_y     :  RAA value
        RAA_y_err :  Error in RAA
        '''
        data = pd.read_csv(filename, sep=sep)
        RAA_x = data.values[:,1]
        RAA_y = data.values[:,3]
        RAA_x_err = data.values[:,2]
        RAA_y_err = data.values[:,4]
        return RAA_x, RAA_x_err, RAA_y, RAA_y_err
    
    def plot_vacuum_spectrum(self, save_name: str = ''):
        '''
        Plots the pp data:   spectrum vs. pT
        '''
        plt.figure()
        m_pT = '{n(p_T)}'
        frac = '\\frac{p_{T,0}}{p_T}'
        text = f'Curve fit: $a\\left({frac}\\right)^{m_pT}$'
        pT   = np.linspace(self.vac_pT[0], self.vac_pT[-1], 1000)
        if self.partonic:
            marker = 'o'
            m_size = 3.5
            if self.nPDF: pdf  = '(nPDF)'
            else: pdf  = ''
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
            fig.set_figheight(6)
            fig.set_figwidth(6)
            fig.subplots_adjust(hspace=0.0)
            ax1.plot(pT, self.vacuum_func(pT, *self.vac_prms),
                     color='slategrey')
            ax1.plot(pT, self.vacuum_func(pT, *self.gln_prms),
                     color = 'darkred')
            ax1.plot(pT, self.vacuum_func(pT, *self.qrk_prms),
                     color='darkblue')
            ax1.plot(self.vac_pT, self.vac_y, marker, markersize=m_size,
                     color='slategrey',label=f'pp -> jets')
            ax1.plot(self.gln_pT, self.gln_y, marker, markersize=m_size,
                     color = 'darkred', label=f'pp -> gluon jets {pdf}')
            ax1.plot(self.qrk_pT, self.qrk_y, marker, markersize=m_size,
                     color='darkblue', label=f'pp -> quark jets {pdf}')
            ax1.set_ylabel('Spectrum')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True, which="both", ls=":")

            qrk_spec = self.vacuum_func(pT, *self.qrk_prms)
            gln_spec = self.vacuum_func(pT, *self.gln_prms)
            vac_spec = self.vacuum_func(pT, *self.vac_prms)
            ax2.plot(pT, qrk_spec/vac_spec, color='darkblue')
            ax2.plot(pT, gln_spec/vac_spec, color='darkred')
            ax2.plot(pT, (qrk_spec+gln_spec)/vac_spec, color='slategrey')
            ax2.set_xlabel('$p_T$ [GeV]')
            ax2.set_ylabel('Fractions')
            ax2.set_xscale('log')
            ax2.grid(True, which="both", ls=":")
        else:
            plt.errorbar(self.vac_pT, self.vac_y, xerr=self.dpT,
                         yerr=self.pp_err, color='black', fmt='.',
                         label=r'pp-Data, $E_{cm} = 5.02$ TeV')
            plt.plot(pT, self.vacuum_func(pT, *self.vac_prms),
                                            label='Fit')
            plt.ylabel(r'$\frac{d\sigma}{dp_Tdy}$ [nb/GeV]')
            plt.text(40, 10e-5, text)

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('$p_T$ [GeV]')
            plt.legend()
            plt.grid(True, which="both", ls=":")

        plt.tight_layout()
        if save_name:
            plt.savefig(f'Data/Vacuum/{save_name}.png')
        if self.partonic:
            plt.savefig(f'Data/Vacuum/Pythia_{self.e_cm}.png')
        else:
            plt.savefig(f'Data/Vacuum/PP_data_{self.e_cm}.png')
        return
    
    def plot_RAA_Data(self):
        '''
        Plots the RAA data:   RAA vs. pT
        '''
        plt.figure()
        plt.errorbar(x=self.raa_pT, y=self.raa,
                     xerr=self.pT_bin, yerr=self.raa_err,
                     capsize=3, fmt='.', color='black')
        plt.ylabel('$R_{AA}$')
        plt.xlabel('$P_T$ [GeV]')
        plt.ylim(0.2,1)
        text1 = 'ATLAS'
        text2 = r'$E_{cm} = 5.02$ TeV'
        text3 = r'$R_{jet}=0.4$, $|y|<2.8$'
        text4 = f'Centrality: 0-10%'
        plt.text(100, 0.95, text1)
        plt.text(100, 0.90, text2)
        plt.text(100, 0.85, text3)
        plt.text(100, 0.80, text4)
        plt.grid()
        plt.savefig(f'Data/RAA/RAA_{self.e_cm}.png')
        return

    def load_sampled_trace(self, model: pm.Model, trace_name=None):
        ''' Tries to load a sampled traced '''
        try:
            with model:
                if trace_name: path = f'Traces/{trace_name}'
                else: path = f'Traces/{self.model_name}' 
                self.trace = pm.load_trace(path)
            self._do_sampling = False
            print('\nLoaded saved trace...\n')
            return
        except:
            print('\nNo trace saved...\n')
            self._do_sampling = True
            return
    
    def mcmc_inference(self,
                       pyModel: pm.Model,
                       n_draws: int,
                       n_tune: int,
                       algo: str = 'HMC',
                       load_trace: bool = False,
                       save_trace: bool = True,
                       conv: bool = False
                       ) -> None:
        '''
        This method perfroms the Bayesian inference using MCMC (via pymc3)

        Args
        ------------------------------------------------------------------
        pyModel    : Model in context to sample from
        n_draws    : Number of samples to be drawn
        n_tune     : Number of samples to be used in tuning
        algo       : Sampling alogrithm, ('HMC', 'NUTS' or 'RWM')
        load_trace : If True, tries to load an existing trace
        save_trace : If True, sampled trace will be saved
        conv       : If True, convergence is checked 
        '''
        if conv: n_chains=3
        else: n_chains = 1
        
        if load_trace == True:
            self.load_sampled_trace(pyModel)
            if not self._do_sampling: save_trace = False
        
        if self._do_sampling:
            with pyModel:
                alogrithm = {'HMC': pm.HamiltonianMC(target_accept=0.97),
                            'NUTS': pm.NUTS(target_accept=0.95),
                             'RWM': pm.Metropolis()}
                self.trace = pm.sample(draws=n_draws,
                                    tune=n_tune,
                                    chains=n_chains,
                                    step = alogrithm[algo],
                                    compute_convergence_checks=conv,
                                    return_inferencedata=False)
                if save_trace == True:
                    save_name = f'Traces/{self.model_name}'
                    pm.save_trace(self.trace, save_name, overwrite=True)

                directory = f'{self.model_name}/'
                if not os.path.exists(directory): os.makedirs(directory)
                path_name = os.path.join(directory, 'Traceplot.png')

                #save trace plot
                axes = az.plot_trace(self.trace, var_names=f'~Raa')
                fig = axes.ravel()[0].figure
                fig.savefig(path_name)

        #Save Summary
        path_name = os.path.join(f'{self.model_name}/', 'Trace_summary.csv')
        df = az.summary(data = self.trace, var_names=self.parameters,
                        round_to=2)
        df.to_csv(path_name, sep='\t')
        return

    def plot_RAA_ppc(self, model: pm.Model, dir_name=None, save_name=None):
        '''
        Samples posterior prdeictive for new RAA points
        
        Args
        -----------------------------------------------
        model : Model in context
        '''
        pT_min = 50
        pT_max = self.raa_pT[-1] + self.pT_bin[-1] + 50

        if self.trace == None:
            print('\nNo trace, call MCMC_infernce to make a trace')
            return

        if self.obs == 'PbPb':
            new_pT = np.logspace(np.log10(pT_min), np.log10(pT_max), 30) 
        else: 
            new_pT = np.linspace(pT_min, pT_max, 50)

        with model:
            pm.set_data({'RAA_pT': new_pT})
            print('\nSampling Posterior Predictive')
            post_pred = pm.sample_posterior_predictive(trace=self.trace,
                                                    samples=5000,
                                                    var_names=['Raa'])
        print('\nPlotting')
        fig, ax = plt.subplots()
        ax.plot(new_pT, post_pred['Raa'].T, color='lightsteelblue', zorder=1)
        ax.plot(new_pT, post_pred['Raa'].mean(axis=0),
                'o-', markersize=4,
                label='Posterior Expectation',
                color='darkblue', zorder=3)
        ax.errorbar(x=self.raa_pT,
                y=self.raa,
                xerr=self.pT_bin,
                yerr=self.raa_err, 
                capsize=4,
                fmt='.',
                color='Black',
                label=r'ATLAS Data: $R_{jet}=0.4$, $y_{max}=$'+str(self.y_max),
                zorder=2)
        ax.set(xlabel='$p_T$ [GeV]', 
                xlim=(pT_min, pT_max))
        if self.obs == 'RAA': ax.set(ylabel='$R_{AA}$', ylim=(0.2, 1))
        else: ax.set(xscale = 'log', yscale = 'log',
                     ylabel=r'$\frac{d \sigma}{d p_T}$ [nb/GeV]',)
        ax.legend()
        ax.grid()

        if dir_name: directory = f'{dir_name}/'
        else: directory = f'{self.model_name}/'

        if save_name: name = save_name
        else: name = 'RAA_ppc.png' 

        if not os.path.exists(directory):
            os.makedirs(directory)
        path_name = os.path.join(directory, name)
        plt.savefig(path_name)
        print('Done')
        return
    
    def plot_corner(self):
        '''
        Makes a corner plot of the model parameters using the 
        sampled trace (self.trace)
        '''
        if self.trace == None:
            print('\nNo trace, call MCMC_infernce to make a trace')
            return

        trace_df = pm.trace_to_dataframe(self.trace, varnames=self.parameters)
        pg = sns.PairGrid(trace_df, despine=False,
                          diag_sharey=False, corner=True)
        pg.map_diag(sns.kdeplot, bw_adjust=0.8)
        pg.map_lower(sns.kdeplot, shade=True, bw_adjust=0.7)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        
        directory = f'{self.model_name}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_name = os.path.join(directory, 'corner_plot.png')
        pg.savefig(path_name)




    
class QuarkGluonQuenching(JetSampler):

    def __init__(self, e_cm=5020, rap=2.8, nPDF=True, name=None, obs='RAA'):
        ''' Initializing the class instance '''

        prms_names = ['ω', 'σ1', 'σ2', 'δ']
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
        with pm.Model() as quark_gluon_qunching:
            pT_data  = pm.Data('RAA_pT', self.raa_pT)
            RAA_data = pm.Data('RAA_data', self.raa)
            
            ω  = pm.Normal('ω', 70, 20)
            σ1 = pm.HalfNormal('σ1', 3)
            σ2 = pm.HalfNormal('σ2', 3)
            δ  = pm.Uniform('δ', 0, 1)    #Used in the likelihood
            
            μ1 = np.log(c_F*ω/2) - 0.5*σ1**2
            μ2 = np.log(c_A*ω/2) - 0.5*σ2**2
            theta = [μ1, σ1, μ2, σ2]
            raa  = pm.Deterministic('Raa', self.raa_model(pT_data, theta))

            pm.Normal('LH', mu=raa, sigma=δ*self.raa_err, observed=RAA_data)
        return quark_gluon_qunching
    
    def plot_energyloss_dist(self):
        #Must be fixed!!
        c_F = 4/3
        c_A = 3
        a = 0.5
        ε_g = np.logspace(-3, 2, 100)
        ε_q = np.logspace(-4, 2, 100)
        σ1_tr  = self.trace['σ1']
        σ2_tr  = self.trace['σ2']
        w_tr = self.trace['ω']
        σ1_mean = np.mean(σ1_tr)
        σ2_mean = np.mean(σ2_tr)
        w_mean  = np.mean(w_tr)
        distQ = np.zeros_like(ε_q)
        distG = np.zeros_like(ε_g)  

        plt.figure()
        for s1, s2, w in zip(σ1_tr, σ2_tr, w_tr): 
            m1 = np.log(w*c_F) - 0.5*s1**2
            current_Q = self.logNormal(ε_q, m1, s1)
            distQ += current_Q
            if (in_1sigma(s1, σ1_tr) and in_1sigma(w, w_tr)):
                plt.plot(ε_q, current_Q, color='lightsteelblue',
                         linewidth = 0.05)
            m2 = np.log(w*c_A) - 0.5*s2**2
            current_G = self.logNormal(ε_g, m2, s2)
            distG += current_G
            if (in_1sigma(s2, σ2_tr) and in_1sigma(w, w_tr)):
                plt.plot(ε_g, current_G, color='rosybrown',
                         linewidth = 0.05)
        
        μ1_mean = np.log(w_mean*c_F) - 0.5*σ1_mean**2
        μ2_mean = np.log(w_mean*c_A) - 0.5*σ2_mean**2
        plt.plot(ε_q, self.logNormal(ε_q, μ1_mean, σ1_mean),
                 color='darkblue',
                 label=f'Posterior expetation for quark jets')
        plt.plot(ε_g, self.logNormal(ε_g, μ2_mean, σ2_mean),
                 color='darkred',
                 label=f'Posterior expetation for gluon jets')

        plt.xlabel(r'$\varepsilon$ [GeV]')
        plt.ylabel(r'$D(\varepsilon)$')
        plt.xlim(0.01, 100)
        plt.ylim(0.001, 0.1)
        #plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()
        directory = f'{self.model_name}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_name = os.path.join(directory, 'Partonic_Eloss.png')
        plt.savefig(path_name)
        return



