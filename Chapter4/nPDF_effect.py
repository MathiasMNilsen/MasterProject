import Sampler as qn
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'size': 12, 'family': 'serif', 'serif': ['Computer Modern']})

nPDF_model = qn.JetSampler(e_cm=5020, nPDF=True , partonic = True, obs='RAA')
pPDF_model = qn.JetSampler(e_cm=5020, nPDF=False, partonic = True, obs='RAA')

pT_min = nPDF_model.raa_pT[0] 
pT_max = nPDF_model.raa_pT[-1] 
pT = np.linspace(pT_min, pT_max, 200)
nPDF_qrk_spec = nPDF_model.vacuum_func(pT, *nPDF_model.qrk_prms) 
nPDF_gln_spec = nPDF_model.vacuum_func(pT, *nPDF_model.gln_prms)
pPDF_qrk_spec = pPDF_model.vacuum_func(pT, *pPDF_model.qrk_prms) 
pPDF_gln_spec = pPDF_model.vacuum_func(pT, *pPDF_model.gln_prms) 

nPDF_spec = nPDF_qrk_spec + nPDF_gln_spec 
pPDF_spec = pPDF_qrk_spec + pPDF_gln_spec

plt.figure(figsize=(6,4))
plt.plot(pT, nPDF_spec/pPDF_spec, color='darkblue', label = 'Fit to PYTHIA data')
plt.plot(pT, nPDF_qrk_spec/pPDF_spec, '--', label='Quark jet contribution')
plt.plot(pT, nPDF_gln_spec/pPDF_spec, '--', label='Gluon jet contribution')
plt.grid()
plt.xlabel(r'$p_T$ [GeV]')
plt.ylabel(r'$\left(\frac{d\sigma}{dp_T}\right)^{nPDF} / \left(\frac{d\sigma}{dp_T}\right)^{PDF}$')
plt.legend()
plt.xlim(pT_min, pT_max)
plt.ylim(0, 1.5)
plt.tight_layout()
plt.savefig('nPDF', format='pdf')