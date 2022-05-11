'''
This script shows an example of the Metropolis-Hastings
algorithm with a Gaussian proposal distribtuion and a target 
distribtuion probortinal to 0.3exp(-0.2x^2) + 0.7exp(-0.2(x-6)^2).
Inspiration taken from "An Introduction to MCMC for Machine Learning"
'''
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def target(x):
    return 0.3*np.exp(-0.2*x**2) + 0.7*np.exp(-0.2*(x-6)**2)

def main():

    target_norm, _ = quad(target, -np.inf, np.inf)
    propsal_width  = 5
    n_samples = 10000
    x_current = 5
    iteration = []
    n_acc = 0
    chain = []

    #Run MCMC (with Random Walk Metropolis)
    for i in range(n_samples):

        x_proposal = np.random.normal(x_current, propsal_width)
        acc_probability = min(1, target(x_proposal)/target(x_current))
        u = np.random.uniform(0, 1)
        iteration.append(i)

        if u <= acc_probability:
            chain.append(x_proposal)
            x_current = x_proposal
            n_acc += 1
        else:
            chain.append(x_current)

    #Plot Histogram
    label = 'Proposal $\\propto 0.3e^{-0.2x^2} + 0.7e^{-0.2(x-6)^2}$'
    txt = f'Samples: {n_samples}\nAcceptance rate: {round(n_acc/n_samples, 2)}'
    x = np.linspace(-10, 20, 200)
    yMax = 2.8/2*np.max(target(x)/target_norm) 
    plt.figure()
    plt.hist(chain, bins=60, density=True, color='lightsteelblue')
    plt.plot(x, target(x)/target_norm, color='royalblue', label=label)
    plt.ylim(0, yMax)
    plt.xlabel('x')
    plt.legend()
    plt.text(x=10, y=yMax/2, s=txt)
    plt.savefig('RWM.png')

    #Plot Trace
    plt.figure(figsize=(12,4))
    plt.plot(iteration, chain, color='lightsteelblue', label='Trace')
    plt.xlabel('iteration')
    plt.ylabel('values')
    plt.legend()
    plt.savefig('Trace_RWM.png')


if __name__ == '__main__':
    main()