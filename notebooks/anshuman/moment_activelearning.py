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
LHS_CSV = 'lhs_samples.csv'
NSIDE = 512
SCALES_ARCMIN = [4.0, 8.0, 16.0, 32.0, 64.0]
CACHE_DIR = 'sample_vector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_data_vector(sample_path, nside=512):
    """Reconstructs maps and measures g^2*y, g^2*tau, g^2*kappa."""
    search_pattern = os.path.join(sample_path, "**", f"allmaps_nside{nside}_z*_split*.pkl")
    files = glob.glob(search_pattern, recursive=True)

    if len(files) < 4:
        return None

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
        # Smoothing is CPU-bound but benefits from multiple cores
        g_s = hp.smoothing(delta_g, fwhm=fwhm_rad, verbose=False)
        y_s = hp.smoothing(total_ymap, fwhm=fwhm_rad, verbose=False)
        t_s = hp.smoothing(total_tmap, fwhm=fwhm_rad, verbose=False)
        k_s = hp.smoothing(total_kmap, fwhm=fwhm_rad, verbose=False)

        full_vector.extend([np.mean(g_s**2 * y_s), np.mean(g_s**2 * t_s), np.mean(g_s**2 * k_s)])

    return np.array(full_vector)

# =============================================================================
# 2. DATA AGGREGATION WITH CACHING
# =============================================================================
print(f"Aggregating data from: {BASE_DIR}")
df_lhs = pd.read_csv(LHS_CSV)
theta_list, x_list = [], []

for _, row in tqdm(df_lhs.iterrows(), total=len(df_lhs)):
    sid = int(row['sample_id'])
    cache_file = os.path.join(CACHE_DIR, f"x_sample_{sid}.npy")

    # Check if this sample was already processed in the killed run
    if os.path.exists(cache_file):
        x = np.load(cache_file)
    else:
        s_path = os.path.join(BASE_DIR, f"sample_{sid}")
        x = get_data_vector(s_path, nside=NSIDE)
        if x is not None:
            np.save(cache_file, x) # Save for future runs

    if x is not None and len(x) == 15:
        theta_list.append([row['theta_ej_0'], row['nu_theta_ej_M'], row['nu_theta_ej_z'], row['mu_beta']])
        x_list.append(x)

if not x_list:
    raise ValueError("No valid data found. Check your file paths.")

x_train = np.array(x_list).astype(np.float32)
theta_train = np.array(theta_list).astype(np.float32)

np.save('x_round1.npy', x_train)
np.save('theta_round1.npy', theta_train)

# Process Reference Run
x_obs = get_data_vector(REF_DIR, nside=NSIDE).astype(np.float32)
np.save('x_obs.npy', x_obs)

# =============================================================================
# 3. TRAINING & PROPOSAL (LTU-ILI)
# =============================================================================
loader = StaticNumpyLoader(in_dir=os.getcwd(), x_file='x_round1.npy', theta_file='theta_round1.npy', xobs_file='x_obs.npy')

combined_nets = load_nde_sbi(engine='NPE', model='maf', repeats=1, hidden_features=50, num_transforms=5) + \
                load_nde_sbi(engine='NPE', model='nsf', repeats=1, hidden_features=50, num_transforms=5)



runner = InferenceRunner.load(
    backend='sbi', engine='NPE',
    prior=ili.utils.Uniform(low=[1.0, -0.3, -3.0, 0.01], high=[6.0, 0.0, 3.0, 1.5]),
    nets=combined_nets, out_dir=Path(os.getcwd())
)

posterior, _ = runner(loader)

# 1. SAVE THE POSTERIOR IMMEDIATELY
with open('round1_joint_posterior.pkl', 'wb') as f:
    pk.dump(posterior, f)

# Sampling Round 2
x_obs_tensor = torch.from_numpy(x_obs).reshape(1, -1)
next_theta = posterior.sample((20,), x=x_obs_tensor)


pd.DataFrame(next_theta.detach().cpu().numpy(),

             columns=['theta_ej_0', 'nu_theta_ej_M', 'nu_theta_ej_z', 'mu_beta']).to_csv('round2_samples.csv', index_label='sample_id')

print("Successfully generated round2_samples.csv")
