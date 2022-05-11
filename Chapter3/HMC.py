'''
This script shows an example of the Hamiltonian Monte Carlo
algorithm with a target distribtuion probortinal to 
0.3exp(-0.2x^2) + 0.7exp(-0.2(x-6)^2).
Inspiration taken from "An Introduction to MCMC for Machine Learning"
'''

import numpy as np
import sympy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys
from functools import cache

def target(q):
    if isinstance(q, sp.Symbol):
        return 0.3*sp.exp(-0.2*q**2) + 0.7*sp.exp(-0.2*(q-6)**2)
    else:
        return 0.3*np.exp(-0.2*q**2) + 0.7*np.exp(-0.2*(q-6)**2)

def printProgressBar(i, max, text):
    j = i/max
    n_bar = 50
    sys.stdout.write('\r')
    txt = f"[{'='*int(n_bar*j):{n_bar}s}] {round(float(100*j), 2)}%  {text}"
    sys.stdout.write(txt)
    sys.stdout.flush()

class HMC:

    def __init__(self, target_func):
        '''
        A simple Hamiltonian Monte Carlo sampler

        Atributes
        ---------
        target : Target function to sample from (must also accept sympy.Symbol)
        n_div  : Number of divergent samples in MCMC run
        n_acc  : Number of accepted samples
        trace  : Chain of samples
        '''
        self.target = target_func
        self.n_div = 0
        self.n_acc = 0
        self.trace = []
    
    def hamiltonian(self, q, p):
        return p**2/2 - np.log(self.target(q))
    
    @cache
    def diff_U(self):
        x = sp.Symbol('x')
        dU = sp.diff(-sp.ln(self.target(x)))
        return dU
    
    def HMC_step(self, q_cur, ε=1.2, L=10):
        p_cur = np.random.normal(0, 1)
        q = q_cur
        p = p_cur

        #Leapfrog integration
        x = sp.Symbol('x')
        for i in range(L):
            p = p - ε/2*float(self.diff_U().evalf(subs={x: q}))
            q = q + ε*p
            p = p - ε/2*float(self.diff_U().evalf(subs={x: q}))
        p = -p

        H_cur  = self.hamiltonian(q_cur, p_cur)
        H_prop = self.hamiltonian(q, p) 
        if abs(H_cur - H_prop) > abs(0.5*H_cur):
            self.n_div += 1
            
        #Accept or Reject
        u = np.random.uniform(0, 1)
        A = min(1, np.exp(H_cur - H_prop))
        if u <= A:
            self.n_acc +=1
            return q
        else:
            return q_cur
    
    def sample(self, n, q_init):
        x = q_init
        for i in range(n):
            x = self.HMC_step(x)
            self.trace.append(x)
            printProgressBar(i+1, n, f'{i+1}/{n} samples, {self.n_div} divs')
        print(f'\n{self.n_div} divegernt samples')
        return

def main():
    n_samples = 10000 
    model = HMC(target)
    model.sample(n=n_samples, q_init=5)
    chain = np.array(model.trace)
    
    #Plot Histogram
    x = np.linspace(-10, 20, 200)
    label = 'Proposal $\\propto 0.3e^{-0.2x^2} + 0.7e^{-0.2(x-6)^2}$'
    target_norm, _ = quad(target, -np.inf, np.inf)
    yMax = 2.8/2*np.max(target(x)/target_norm)
    acc  = round(model.n_acc/n_samples, 2)

    plt.figure()
    plt.hist(chain, bins=60, density=True, color='lightsteelblue')
    plt.plot(x, target(x)/target_norm, color='royalblue', label=label) 
    plt.ylim(0, yMax)
    plt.xlabel('x')
    plt.legend()
    plt.text(x=10, y=yMax/2, s=f'Samples: {n_samples}\nAcceptance rate: {acc}')
    plt.savefig('HMC.png')

    #Plot Trace
    iterations = np.array([i+1 for i in range(len(chain))])
    plt.figure(figsize=(12,4))
    plt.plot(iterations, chain, color='lightsteelblue', label='Trace')
    plt.xlabel('iteration')
    plt.ylabel('values')
    plt.legend()
    plt.savefig('Trace_HMC.png')

if __name__ == '__main__':
    main()