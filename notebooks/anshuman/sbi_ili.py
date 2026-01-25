import os
import sys
import torch
import numpy as np
import healpy as hp
import pickle as pk
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Path Setup for ltu-ili ---
sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi, Uniform

# =============================================================================
# 1. DATA EXTRACTION & SCALING
# =============================================================================
BASE_DIR = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav'
CSV_FILES = [
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/anshuman/lhs_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/anshuman/round2_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/anshuman/round3/round3_samples.csv'
]
NSIDE = 512
SCALES = [4.0, 8.0, 16.0, 32.0, 64.0]

def extract_moments(path):
    pattern = os.path.join(path, "**", f"allmaps_nside{NSIDE}_z*_split*.pkl")
    files = glob.glob(pattern, recursive=True)
    if len(files) < 4: return None
    ymap, gmap, kmap, tmap = [np.zeros(12*NSIDE**2, dtype=np.float32) for _ in range(4)]
    for f in files:
        with open(f, 'rb') as h:
            data = pk.load(h)
            ymap += np.nan_to_num(data.get('map_ymap', 0)); kmap += np.nan_to_num(data.get('map_kappa', 0))
            tmap += np.nan_to_num(data.get('map_tau', 0))
            for chunk in data.get('mock_gals_all', {}).values():
                if chunk.size > 0:
                    pix = hp.ang2pix(NSIDE, chunk[:,0], chunk[:,1], lonlat=True)
                    gmap += np.bincount(pix, minlength=12*NSIDE**2)
    dg = (gmap / np.mean(gmap) - 1.0) if np.mean(gmap) > 0 else np.zeros_like(gmap)
    vec = []
    for th in SCALES:
        f = np.radians(th/60.)
        gs, ys, ts, ks = [hp.smoothing(m, fwhm=f, verbose=False) for m in [dg, ymap, tmap, kmap]]
        vec.extend([np.mean(gs**2 * ys), np.mean(gs**2 * ts), np.mean(gs**2 * ks)])
    return np.array(vec)

# 1a. Extract Reference and Training Data
print("Extracting reference run...")
x_obs_raw = extract_moments(os.path.join(BASE_DIR, 'reference_run'))

print("Aggregating training data...")
theta_train, x_train_raw = [], []
for csv in CSV_FILES:
    df = pd.read_csv(csv)
    offset = 30 if "round2" in csv else 0
    for _, row in df.iterrows():
        sid = int(row['sample_id']) + offset
        s_path = os.path.join(BASE_DIR, f"sample_{sid}")
        v = extract_moments(s_path)
        if v is not None:
            theta_train.append([row['theta_ej_0'], row['nu_theta_ej_M'], row['nu_theta_ej_z'], row['mu_beta']])
            x_train_raw.append(v)

x_train_raw = np.array(x_train_raw)
theta_train = np.array(theta_train)

# 1b. Apply Ratio Scaling & Log Transform
# Adding a small epsilon to avoid log(0)
eps = 1e-12
x_obs_safe = np.where(x_obs_raw == 0, eps, x_obs_raw)

# Calculate Log-Ratios: log10(Sim / Ref)
# This centers the "Observation" exactly at 0.0 for every element.
x_train_scaled = np.log10(np.abs(x_train_raw / x_obs_safe) + eps)
x_obs_scaled = np.log10(np.abs(x_obs_raw / x_obs_safe) + eps) # This will be a vector of 0s

np.save('theta_all_50.npy', theta_train)

# =============================================================================
# 2. LTU-ILI TRAINING LOOP
# =============================================================================
stat_map = {
    'g2y': [0, 3, 6, 9, 12],
    'g2tau': [1, 4, 7, 10, 13],
    'g2kappa': [2, 5, 8, 11, 14],
    'JOINT': list(range(15))
}

for name, idx in stat_map.items():
    print(f"\n--- LtU-ILI Training (Scaled): {name} ---")

    # Save scaled .npy files
    np.save(f'x_{name}_scaled.npy', x_train_scaled[:, idx])
    np.save(f'xobs_{name}_scaled.npy', x_obs_scaled[idx])

    loader = StaticNumpyLoader(
        in_dir='./',
        x_file=f'x_{name}_scaled.npy',
        theta_file='theta_all_50.npy',
        xobs_file=f'xobs_{name}_scaled.npy'
    )

    # NPE with 3 repeats for better ensemble stability
    nets = load_nde_sbi(engine='NPE', model='maf', repeats=3, hidden_features=64, num_transforms=5)

    runner = InferenceRunner.load(
        backend='sbi',
        engine='NPE',
        prior=Uniform(low=[1.0, -0.3, -3.0, 0.01], high=[6.0, 0.0, 3.0, 1.5]),
        nets=nets,
        out_dir=f'./sbi_logs_{name}_scaled'
    )

    posterior, _ = runner(loader)

    with open(f'ili_posterior_{name}_scaled.pkl', 'wb') as f:
        pk.dump(posterior, f)

print("\nTraining Complete. Use the scaled pkl and xobs_scaled.npy for plotting.")
