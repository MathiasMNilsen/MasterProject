import Sampler as sm

QGQ = sm.QuarkGluonQuenching(e_cm=5020, nPDF=True, name='Partonic', obs='RAA')
model = QGQ.bayesian_model()
QGQ.mcmc_inference(model, 80000, 5000)
QGQ.plot_RAA_ppc(model)
QGQ.plot_energyloss_dist()
QGQ.plot_corner()


new_QGQ = sm.QuarkGluonQuenching(e_cm=2760, rap=2.1,
                                 nPDF=True, name='Prediction')
new_QGQ.plot_vacuum_spectrum()
new_QGQ.plot_RAA_Data()
new_QGQ.load_sampled_trace(new_QGQ.bayesian_model(), 'Partonic')
new_QGQ.plot_RAA_ppc(new_QGQ.bayesian_model(), 'Partonic', 'Validation2')
