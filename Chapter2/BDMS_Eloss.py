'''
This program calculates the energyloss distribution of the exact 
BDMS spectrum. 

Energyloss ----> ω_c*D(x=ε/ω_c)
[eq(2) in Arleos paper: "Tomography of cold and hot QCD matter"] 
'''

import sympy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre
import numpy as np
import pandas as pd
import sys
from functools import cache

def full_gluon_spectrum(z):
    '''
    The full BDMS induced gluon spectrum (without α)

    Argument
    --------
    z : Gluon energy z = ω/ω_c

    Returns
    -------
    spectrum value α^-1 * dI(ω)/dω
    '''
    spec = np.log(np.cosh(1/np.sqrt(2*z))**2-np.sin(1/np.sqrt(2*z))**2)/(2*z)
    return spec 

@cache
def integ(b: float, trig: int) -> float:
    '''
    Does the integrals I_s or I_c from 0 to infinity
    with integrand N(ω)sin(bω) (or  N(ω)cos(bω))
    for a given b

    Arguments
    ---------
    b    : Paramter in the intergand
    trig : Chooses the trogonometric function 1 for sin(), and 2 for cos()

    Returns
    -------
    I_s(b) or I_c(b)  
    '''
    if trig == 1:
        numpy_trig = np.sin
        sympy_trig = sp.sin
    elif trig == 2:
        numpy_trig = np.cos
        sympy_trig = sp.cos
    
    def integ1(b: float) -> float:
        '''
        Integration of x from 0 to 0.01, could be done
        analyticly in sympy, but is done numerically 
        becasue of time conservation
        '''
        N_lowend   = lambda x: np.sqrt(2/x) + np.log(2)*np.log(x) - 1.44136 
        integrand1 = lambda x: numpy_trig(b*x)*N_lowend(x) 
        return quad(integrand1, 0, 0.01)[0]
    
    def integ2(b: float) -> float:
        '''
        Integration of x from 2 to infinity, is done
        analyticly in sympy
        '''
        x = sp.Symbol('x')
        N_highend  = 1/(24*x**2)
        integrand2 = sympy_trig(b*x)*N_highend
        return float(sp.integrate(integrand2, (x, 2, sp.oo)))

    def integ3(b: float) -> float:
        '''
        Integration of x from 0.01 to 2, this has
        to be done numerically
        '''
        N_full     = lambda x: quad(full_gluon_spectrum, x, np.inf)[0] 
        integrand3 = lambda x: numpy_trig(b*x)*N_full(x)
        return quad(integrand3, 0.01, 2, limit=1000)[0]

    return integ1(b) + integ2(b) + integ3(b)

def printProgressBar(i: int, max: int, postText: str):
    '''
    Prints the progress on for the integration of b
    for a given x

    Arguments
    ---------
    i        : Current intefration step
    max      : Number of total integration step
    postText : text to be displayed to the right the prgresbar 
    '''
    n = 50
    j = i/max
    text = f"[{'=' * int(n*j):{n}s}] {round(float(100 * j), 2)}%  {postText}"
    sys.stdout.write('\r')
    sys.stdout.write(text)
    sys.stdout.flush()

def energyloss(x: float, a: float) -> float:
    '''
    Cumputes the energyloss distribution D(x=ε/ω_c),
    using Gauss-Laguerre quadrature.

    Arguments
    ---------
    x : Energyloss x = ε/ω_c
    a : Parameter  α = 2α_sC_R/π

    Returns
    -------
    D(x) : Probability 
    '''
    def integrand(b):
        return np.exp(-b*a*integ(b, 1))*np.cos(b*(x-a*integ(b, 2)))/np.pi 
    
    integral  = 0
    current_i = 1
    i_steps   = 100
    b, w = roots_laguerre(i_steps)
    for b, w in zip(b, w):
        printProgressBar(current_i, i_steps, f'ε: {round(x, 2)}')
        current_i += 1
        integral  += w*np.exp(b)*integrand(b)
    return integral



def main(C_R: float, save_name: str = 'BDMS_dist'):
    '''
    Calculates the energyloss distribution

    Arguments
    ---------
    C_R       : Casimir for parton (4/3 for quark and 3 for gluon) 
    save_name : Name for plot-file and csv-file  
    '''
    a_s   = 0.5
    a_bar = 2*a_s*C_R/np.pi
    e_bar = np.logspace(-2, 0.176, 100)
    dist  = []
    for e in e_bar:
        dist.append(energyloss(e, a_bar))
    
    #Save data
    dist = np.array(dist)
    data_dict = {'x=ε/ω_c': e_bar, 'ω_c*D(x)': dist}
    data_frame = pd.DataFrame(data_dict) 
    data_frame.to_csv(f'{save_name}.csv', sep='\t')
    return

if __name__ == '__main__':
    main(C_R=4/3, save_name='BDMS_quark')
    main(C_R=3  , save_name='BDMS_gluon')
