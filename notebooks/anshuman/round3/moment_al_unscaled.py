import os
import sys
import torch
import numpy as np
import healpy as hp
import pickle as pk
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- Path Setup ---
sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi, Uniform

# =============================================================================
# 1. DATA AGGREGATION (Total 50 Samples: 30 LHS + 20 Round 2)
# =============================================================================
BASE_DIR = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav'
CSV_FILES = [
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/anshuman/lhs_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/anshuman/round2_samples.csv'
]
NSIDE = 512
SCALES = [4.0, 8.0, 16.0, 32.0, 64.0]
CACHE_DIR = 'sample_vector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_moments(path):
    pattern = os.path.join(path, "**", f"allmaps_nside{NSIDE}_z*_split*.pkl")
    files = glob.glob(pattern, recursive=True)
    if len(files) < 4: return None
    ymap, gmap, kmap, tmap = [np.zeros(12*NSIDE**2, dtype=np.float32) for _ in range(4)]
    for f in files:
        with open(f, 'rb') as h:
            data = pk.load(h)
            ymap += np.nan_to_num(data.get('map_ymap', 0))
            kmap += np.nan_to_num(data.get('map_kappa', 0))
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

# Aggregating all available data
theta_train, x_train = [], []
for csv in CSV_FILES:
    df = pd.read_csv(csv)
    offset = 30 if "round2" in csv else 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {csv}"):
        sid = int(row['sample_id']) + offset
        cache_file = os.path.join(CACHE_DIR, f"x_sample_{sid}.npy")
        
        if os.path.exists(cache_file):
            v = np.load(cache_file)
        else:
            s_path = os.path.join(BASE_DIR, f"sample_{sid}")
            v = extract_moments(s_path)
            if v is not None: np.save(cache_file, v)
            
        if v is not None:
            theta_train.append([row['theta_ej_0'], row['nu_theta_ej_M'], row['nu_theta_ej_z'], row['mu_beta']])
            x_train.append(v)

x_train = np.array(x_train).astype(np.float32)
theta_train = np.array(theta_train).astype(np.float32)
x_obs = extract_moments(os.path.join(BASE_DIR, 'reference_run')).astype(np.float32)

np.save('theta_all_50.npy', theta_train)
np.save('x_obs_final.npy', x_obs)

# =============================================================================
# 2. LTU-ILI TRAINING LOOP (DIRECT/UNSCALED)
# =============================================================================
stat_map = {
    'g2y': [0, 3, 6, 9, 12],
    'g2tau': [1, 4, 7, 10, 13],
    'g2kappa': [2, 5, 8, 11, 14],
    'JOINT': list(range(15))
}

for name, idx in stat_map.items():
    print(f"--- Training: {name} ---")
    np.save(f'x_{name}.npy', x_train[:, idx])
    np.save(f'xobs_{name}.npy', x_obs[idx])

    loader = StaticNumpyLoader(
        in_dir='./',
        x_file=f'x_{name}.npy',
        theta_file='theta_all_50.npy',
        xobs_file=f'xobs_{name}.npy'
    )

    # Ensemble of 2 MAFs and 2 NSFs for better unscaled stability
    nets = load_nde_sbi(engine='NPE', model='maf', repeats=2, hidden_features=64, num_transforms=5) + \
           load_nde_sbi(engine='NPE', model='nsf', repeats=2, hidden_features=64, num_transforms=5)

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=Uniform(low=[1.0, -0.3, -3.0, 0.01], high=[6.0, 0.0, 3.0, 1.5]),
        nets=nets, out_dir=Path(f'./sbi_logs_{name}')
    )

    posterior, _ = runner(loader)

    with open(f'ili_posterior_{name}.pkl', 'wb') as f:
        pk.dump(posterior, f)

# =============================================================================
# 3. ROUND 3 PROPOSAL (Using JOINT Posterior)
# =============================================================================
with open('ili_posterior_JOINT.pkl', 'rb') as f:
    joint_posterior = pk.load(f)

# Sample 20 new points at the direct x_obs values
xo_tensor = torch.from_numpy(x_obs).float().reshape(1, -1)
next_theta = joint_posterior.sample((20,), x=xo_tensor).detach().cpu().numpy()

pd.DataFrame(next_theta, 
             columns=['theta_ej_0', 'nu_theta_ej_M', 'nu_theta_ej_z', 'mu_beta']
            ).to_csv('round3_samples.csv', index_label='sample_id')

print("Round 3 samples generated successfully using unscaled vectors.")
