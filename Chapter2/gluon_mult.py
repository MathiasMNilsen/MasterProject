
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

from scipy.integrate import quad

def integrand(x):
    return (np.log(np.cosh(1/np.sqrt(2*x))**2-np.sin(1/np.sqrt(2*x))**2))/(2*x)

def main():

    num_factor, _ = quad(integrand, 0.01, np.inf)
    
    ωInt = sp.Symbol('ωInt')
    ω_c  = sp.Symbol('ω_c')
    α    = sp.Symbol('α')
    ω    = sp.Symbol('ω')  

    dIdω_high = α*ω_c**2/(12*ωInt**3)
    dIdω_low  = α/ωInt*sp.sqrt(ω_c/(2*ωInt))-α/ωInt*sp.ln(2)

    N_high = sp.integrate(dIdω_high, (ωInt, ω, sp.oo))
    N_low  = sp.integrate(dIdω_low, (ωInt, ω, 0.01*ω_c)) + α*num_factor
    N_bdms = α*sp.sqrt(2*ω_c/ω)

    a = 0.42
    x = np.logspace(-3, 1, 200)
    mult_high = np.zeros_like(x)
    mult_low  = np.zeros_like(x)
    mult_full = np.zeros_like(x)
    mult_BDMS = np.zeros_like(x)

    for i, xi in enumerate(x):
        mult_full[i] = a*quad(integrand, xi, np.inf)[0]
        mult_BDMS[i] = N_bdms.subs([(ω, xi), (ω_c, 1), (α, a)]) 
        mult_high[i] = N_high.subs([(ω, xi), (ω_c, 1), (α, a)]) 
        mult_low[i]  = N_low.subs([(ω, xi), (ω_c, 1), (α, a)]) 
    
    high_label = 'Analytic $N(\omega>\omega_c)$'
    low_label  = 'Analytic $N(\omega<\omega_c)$'
    full_label = 'Full numerical result' 
    bdms_label = 'BDMS estimate'
    
    plt.figure()
    plt.loglog(x, mult_BDMS, '--', label=bdms_label, color='darkgrey')
    plt.loglog(x, mult_high, '--', label=high_label, color='darkblue')
    plt.loglog(x[:150], mult_low[:150], '--', label=low_label, color='darkred')
    plt.loglog(x, mult_full, label=full_label, color='black')
    plt.xlabel('$x = \omega/\omega_c$')
    plt.ylabel('N(x)')
    plt.xlim(10e-4, 10e0)
    plt.ylim(10e-4, 10e2)
    plt.legend()
    plt.grid()
    plt.text(1.5, 4, r'$\bar{\alpha}=0.42$')
    plt.savefig('gluon_mult.png')

if __name__ == '__main__':
    main()

    
