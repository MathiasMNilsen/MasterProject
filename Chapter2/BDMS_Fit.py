'''
This program fits a gamma and log-normal to energyloss
distributions for the exact BDMPS spectrumm
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc
from scipy.optimize import curve_fit

rc('font', **{'size': 12, 'family': 'serif', 'serif': ['Computer Modern']})
sns.set_theme(style="ticks")

def analytic(x, a):
    return a*np.sqrt(1/(2*x**3)) * np.exp(-(np.pi*a**2)/(2*x))

def log_normal(x, mu, sigma):
    return np.exp(-(np.log(x)-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma*x)

def main():
    a_s = 0.5
    c_F = 4/3
    c_A = 3 

    #load the points
    dataframe_q = pd.read_csv('BDMS_quark.csv', sep='\t')
    dataframe_g = pd.read_csv('BDMS_gluon.csv', sep='\t')
    x_q = dataframe_q.values[:,1]
    y_q = dataframe_q.values[:,2]
    x_g = dataframe_g.values[:,1]
    y_g = dataframe_g.values[:,2]
    
    #curve fits, we want the curves to fit best around the peak
    prms_q, _ = curve_fit(log_normal, x_q[40:80], y_q[40:80], maxfev=100000)
    prms_g, _ = curve_fit(log_normal, x_g[60:90], y_g[60:90], maxfev=100000)

    #plot curves
    eps = np.linspace(start=0.01, stop=4, num=500)
    lq = f'Log-Normal({format(prms_q[0], ".2f")}, {format(prms_q[1], ".2f")})'
    lg = f'Log-Normal({format(prms_g[0], ".2f")}, {format(prms_g[1], ".2f")})'

    #BDMS estimate
    qBDMS = analytic(eps, 2*a_s*c_F/np.pi)
    gBDMS = analytic(eps, 2*a_s*c_A/np.pi)  

    plt.figure()
    plt.plot(eps, qBDMS, color='lightsteelblue', label='BDMS estimate: quark')
    plt.plot(eps, gBDMS, color='lightcoral', label='BDMS estimate: gluon')
    plt.plot(x_q, y_q, color='darkblue', label=r'Hard quark: $C_R=C_F=4/3$')
    plt.plot(x_g, y_g, color='darkred' , label=r'Hard gluon: $C_R=C_A=3$')
    plt.plot(eps, log_normal(eps, *prms_q), '--', color='darkblue', label=lq)
    plt.plot(eps, log_normal(eps, *prms_g), '--', color='darkred' , label=lg)

    plt.text(1.02, 1.75, r'$\alpha_s=0.5$')
    plt.grid(linewidth=0.2)
    plt.xlim(0, 1.35)
    plt.legend()
    plt.xlabel('$x=\epsilon/\omega_c$')
    plt.ylabel('$\omega_c D(\epsilon)$')
    plt.savefig('BDMS_Fit.png', dpi=400)

if __name__ == '__main__':
    main()