import numpy as np
import healpy as hp
import pickle as pk
import glob
import os
import pandas as pd
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# --- Path Setup ---
ltu_path = '/work/hdd/bdne/aacharya2/ltu-ili'
if ltu_path not in sys.path:
    sys.path.append(ltu_path)

import ili
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
BASE_DIR = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav'
REF_DIR = os.path.join(BASE_DIR, 'reference_run')
# We now load both previous rounds
CSV_FILES = ['lhs_samples.csv', 'round2_samples.csv']
NSIDE = 512
SCALES_ARCMIN = [4.0, 8.0, 16.0, 32.0, 64.0]
CACHE_DIR = 'sample_vector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_data_vector(sample_path, nside=512):
    search_pattern = os.path.join(sample_path, "**", f"allmaps_nside{nside}_z*_split*.pkl")
    files = glob.glob(search_pattern, recursive=True)
    if len(files) < 4: return None

    total_ymap, total_gmap, total_kmap, total_tmap = [np.zeros(12*nside**2, dtype=np.float32) for _ in range(4)]
    for f in files:
        with open(f, 'rb') as h:
            data = pk.load(h)
            total_ymap += np.nan_to_num(data.get('map_ymap', 0))
            total_kmap += np.nan_to_num(data.get('map_kappa', 0))
            total_tmap += np.nan_to_num(data.get('map_tau', 0))
            mock_gals_dict = data.get('mock_gals_all', {})
            for chunk_idx in mock_gals_dict:
                gal_data = mock_gals_dict[chunk_idx]
                if gal_data.size > 0:
                    pix = hp.ang2pix(nside, gal_data[:,0], gal_data[:,1], lonlat=True)
                    total_gmap += np.bincount(pix, minlength=12*nside**2)

    mean_g = np.mean(total_gmap)
    delta_g = (total_gmap / mean_g - 1.0) if mean_g > 0 else np.zeros_like(total_gmap)
    full_vector = []
    for theta in SCALES_ARCMIN:
        fwhm_rad = np.radians(theta/60.)
        g_s = hp.smoothing(delta_g, fwhm=fwhm_rad, verbose=False)
        y_s = hp.smoothing(total_ymap, fwhm=fwhm_rad, verbose=False)
        t_s = hp.smoothing(total_tmap, fwhm=fwhm_rad, verbose=False)
        k_s = hp.smoothing(total_kmap, fwhm=fwhm_rad, verbose=False)
        full_vector.extend([np.mean(g_s**2 * y_s), np.mean(g_s**2 * t_s), np.mean(g_s**2 * k_s)])
    return np.array(full_vector)

# =============================================================================
# 2. DATA AGGREGATION & LOG-RATIO SCALING
# =============================================================================
print("Processing Reference Run for Scaling...")
x_obs_raw = get_data_vector(REF_DIR, nside=NSIDE).astype(np.float32)
# Reference is now Log10(1) = 0
x_obs_scaled = np.zeros_like(x_obs_raw) 
np.save('x_obs_round3.npy', x_obs_scaled)

theta_list, x_list = [], []
for csv in CSV_FILES:
    df = pd.read_csv(csv)
    # Round 2 samples are folders sample_30 to sample_49
    offset = 30 if "round2" in csv else 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {csv}"):
        sid = int(row['sample_id']) + offset
        cache_file = os.path.join(CACHE_DIR, f"x_sample_{sid}.npy")

        if os.path.exists(cache_file):
            x = np.load(cache_file)
        else:
            s_path = os.path.join(BASE_DIR, f"sample_{sid}")
            x = get_data_vector(s_path, nside=NSIDE)
            if x is not None: np.save(cache_file, x)

        if x is not None:
            # APPLY LOG-RATIO SCALING: log10(sim / ref)
            x_scaled = np.log10(np.abs(x / x_obs_raw) + 1e-12)
            theta_list.append([row['theta_ej_0'], row['nu_theta_ej_M'], row['nu_theta_ej_z'], row['mu_beta']])
            x_list.append(x_scaled)

x_train = np.array(x_list).astype(np.float32)
theta_train = np.array(theta_list).astype(np.float32)

np.save('x_train_round3.npy', x_train)
np.save('theta_train_round3.npy', theta_train)

# =============================================================================
# 3. TRAINING & ROUND 3 PROPOSAL
# =============================================================================
loader = StaticNumpyLoader(
    in_dir=os.getcwd(), 
    x_file='x_train_round3.npy', 
    theta_file='theta_train_round3.npy', 
    xobs_file='x_obs_round3.npy'
)

# Ensemble of MAF and NSF for robustness
combined_nets = load_nde_sbi(engine='NPE', model='maf', repeats=2, hidden_features=64, num_transforms=5) + \
                load_nde_sbi(engine='NPE', model='nsf', repeats=2, hidden_features=64, num_transforms=5)

runner = InferenceRunner.load(
    backend='sbi', engine='NPE',
    # Note: nu_theta_ej_M at 0.0 is now handled better by the scaling
    prior=ili.utils.Uniform(low=[1.0, -0.3, -3.0, 0.01], high=[6.0, 0.0, 3.0, 1.5]),
    nets=combined_nets, out_dir=Path(os.getcwd())
)

posterior, _ = runner(loader)

with open('round3_joint_posterior.pkl', 'wb') as f:
    pk.dump(posterior, f)

# Generate 20 samples for Round 3
# Conditioned on x_obs_scaled (the vector of zeros)
x_target = torch.zeros((1, 15))
next_theta = posterior.sample((20,), x=x_target).detach().cpu().numpy()

# Save for next batch of simulations
pd.DataFrame(next_theta, 
             columns=['theta_ej_0', 'nu_theta_ej_M', 'nu_theta_ej_z', 'mu_beta']
            ).to_csv('round3_samples.csv', index_label='sample_id')

print("Successfully generated round3_samples.csv using Log-Ratio Scaling.")
