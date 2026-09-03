import numpy as np
from scipy.stats import qmc
import pandas as pd

# Define parameter ranges [min, max]
space = {
    'theta_ej_0': [1.0, 6.0],
    'nu_theta_ej_M': [-0.3, 0.0],
#    'nu_theta_ej_z': [-3.0, 3.0],
#    'mu_beta': [0.01, 1.5]
}

n_samples = 50
sampler = qmc.LatinHypercube(d=len(space))
sample = sampler.random(n=n_samples)

# Scale to actual ranges
l_bounds = [v[0] for v in space.values()]
u_bounds = [v[1] for v in space.values()]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

df = pd.DataFrame(sample_scaled, columns=space.keys())
df.to_csv('validation_samples.csv', index_label='sample_id')
print("Generated "+str(n_samples)+" samples in validation_samples.csv")
