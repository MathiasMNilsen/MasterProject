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
    n_samples = 100_000
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
    label = 'target'
    x = np.linspace(-10, 20, 200)
    yMax = 2.8/2*np.max(target(x)/target_norm) 

    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    n = [1000, 10_000, 50_000, 100_000]
    a = [(0,0), (0, 1), (1, 0), (1, 1)]
    for i in range(4):
        txt = f'Samples: {n[i]}'
        axs[a[i]].hist(chain[:n[i]], bins=60,
                       density=True, color='lightsteelblue')
        axs[a[i]].plot(x, target(x)/target_norm,
                       color='royalblue', label=label)
        axs[a[i]].set_ylim(0, yMax)
        axs[a[i]].set_xlabel('x')
        axs[a[i]].legend(fontsize="small")
        if n[i] == n[-1]: 
            txt2 = f'\nAcceptance rate: {round(n_acc/n_samples, 2)}'
            axs[a[i]].text(x=9.5, y=yMax/2, s=txt+txt2 , fontsize="small")
        else:
            axs[a[i]].text(x=9.5, y=yMax/2, s=txt , fontsize="small")
    plt.savefig('RWM.png')

    #Plot Trace
    plt.figure(figsize=(12,4))
    plt.plot(iteration[:10000], chain[:10000],
             color='lightsteelblue', label='10000 samples')
    plt.xlabel('iteration')
    plt.ylabel('x')
    plt.legend()
    plt.savefig('Trace_RWM.png')


if __name__ == '__main__':
    main()