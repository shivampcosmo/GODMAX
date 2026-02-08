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

# --- Path Setup for ltu-ili ---
sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi, Uniform
from ili.validation.metrics import PosteriorCoverage
# =============================================================================
# 1. DATA AGGREGATION (2-Parameter Version)
# =============================================================================
BASE_DIR = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin'
CSV_FILES = [
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/lhs_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/round2_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/round3_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/round4_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/round5_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/round6_samples.csv'
    ]
NSIDE = 512
LMIN = 50
LMAX = 3*NSIDE - 1
NUM_BINS = 10
BINS = np.logspace(np.log10(LMIN), np.log10(LMAX), NUM_BINS + 1).astype(int)

CACHE_DIR = 'sample_vector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_cls(path):
    # Standard glob pattern from your 4-param script
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
            mock_gals_dict = data.get('mock_gals_all', {})
            for chunk_idx in mock_gals_dict:
                gal_data = mock_gals_dict[chunk_idx]
                if gal_data.size > 0:
                    pix = hp.ang2pix(NSIDE, gal_data[:,0], gal_data[:,1], lonlat=True)
                    gmap += np.bincount(pix, minlength=12*NSIDE**2)

    dg = (gmap / np.mean(gmap) - 1.0) if np.mean(gmap) > 0 else np.zeros_like(gmap)
    
    cl_gy_raw = hp.anafast(dg, ymap, lmax=LMAX)
    cl_gt_raw = hp.anafast(dg, tmap, lmax=LMAX)
    cl_gk_raw = hp.anafast(dg, kmap, lmax=LMAX)
    
    vec = []
    # Binning logic: average Cl within each l-range
    for i in range(NUM_BINS):
        l_start, l_stop = BINS[i], BINS[i+1]
        ell_range = np.arange(l_start, l_stop)
        
        vec.append(np.mean(cl_gy_raw[l_start:l_stop]))
        vec.append(np.mean(cl_gt_raw[l_start:l_stop]))
        vec.append(np.mean(cl_gk_raw[l_start:l_stop]))

    return np.array(vec)
print("Extracting reference run...")
# Reference is in original location
x_obs = extract_cls(os.path.join(BASE_DIR, 'reference_run')).astype(np.float32)
np.save('x_obs.npy', x_obs)

theta_train, x_train = [], []
for csv in CSV_FILES:
    df = pd.read_csv(csv)
    
    # Determine folder offset based on the round
    if "lhs" in csv: offset = 0
    elif "round2" in csv: offset = 30
    elif "round3" in csv: offset = 50
    elif "round4" in csv: offset = 70
    elif "round5" in csv: offset = 90
    elif "round6" in csv: offset =110

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(csv)}"):
        sid = int(row['sample_id']) + offset
        cache_file = os.path.join(CACHE_DIR, f"x_sample_{sid}.npy")

        if os.path.exists(cache_file):
            v = np.load(cache_file)
        else:
            # Note the "twoparams" nesting for this specific run
            s_path = os.path.join(BASE_DIR, "twoparams", f"sample_{sid}")
            v = extract_cls(s_path)
            if v is not None: np.save(cache_file, v)

        if v is not None:
            # ONLY 2 PARAMETERS: theta_ej_0 and mu_beta
            theta_train.append([row['theta_ej_0'], row['mu_beta']])
            x_train.append(v)

x_train = np.array(x_train).astype(np.float32)
theta_train = np.array(theta_train).astype(np.float32)

np.save('theta.npy', theta_train)

# =============================================================================
# 2. LTU-ILI TRAINING LOOP
# =============================================================================
'''
#standardize the statistics
x_mean = np.mean(x_train, axis=0)
x_std = np.std(x_train, axis=0)
x_train = (x_train - x_mean) / (x_std + 1e-6)
x_obs = (x_obs - x_mean) / (x_std + 1e-6)
'''
stat_map = {
    'gy': list(range(0, NUM_BINS * 3, 3)),
    'gtau': list(range(1, NUM_BINS * 3, 3)),
    'gkappa': list(range(2, NUM_BINS * 3, 3)),
    'JOINT': list(range(NUM_BINS * 3)),
}

all_indices = np.arange(len(theta_train))
np.random.shuffle(all_indices)

#val_idx = all_indices[:20]
train_idx = all_indices#[20:]

#theta_val = theta_train[val_idx]
#theta_train_set = theta_train[train_idx]

for name, idx in stat_map.items():
    print(f"\n--- Training NPE: {name} ---")

    # Split the statistics
    xt_full = x_train[:, idx]
    xt_train = xt_full[train_idx]
    #xt_train = xt_train + 0.01 * np.random.randn(*xt_train.shape).astype(np.float32)
    #xt_val = xt_full[val_idx]
    xo = x_obs[idx]

    np.save(f'x_{name}.npy', xt_train)
    np.save(f'xobs_{name}.npy', xo)
    np.save(f'theta_train_{name}.npy', theta_train)

    loader = StaticNumpyLoader(
        in_dir='./',
        x_file=f'x_{name}.npy',
        theta_file=f'theta_train_{name}.npy',
        xobs_file=f'xobs_{name}.npy'
    )

    # STABLE ARCHITECTURE: Reduced transforms to prevent "Peaky" bias and singular matrices
    nets = load_nde_sbi(engine='NPE', model='nsf', repeats=2, hidden_features=32, num_transforms=3) + \
           load_nde_sbi(engine='NPE', model='maf', repeats=6, hidden_features=32, num_transforms=3)

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=Uniform(low=[1.0, 0.01], high=[6.0, 1.5]),
        nets=nets, out_dir=Path(f'./sbi_logs_{name}')
    )

    posterior, _ = runner(loader)

    with open(f'ili_posterior_{name}.pkl', 'wb') as f:
        pk.dump(posterior, f)

