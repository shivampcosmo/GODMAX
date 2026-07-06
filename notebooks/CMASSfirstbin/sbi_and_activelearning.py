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
# 1. DATA AGGREGATION
# =============================================================================
BASE_DIR = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin'
CSV_FILES = [
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/lhs_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/round2_samples.csv'#,
    #'/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/round3_samples.csv'
]
NSIDE = 512
SCALES = [4.0, 8.0, 16.0, 32.0, 64.0]
CACHE_DIR = 'sample_vector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
SEED = 42

def extract_moments(path):
    pattern = os.path.join(path, "**", f"allmaps_nside{NSIDE}_z*_split*.pkl")
    files = glob.glob(pattern, recursive=True)
    if len(files) < 4: return None
    ymap, gmap, kmap, tmap = [np.zeros(12*NSIDE**2, dtype=np.float32) for _ in range(4)]

    LMAX = 3 * NSIDE - 1

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
    vec = []
    for th in SCALES:
        f = np.radians(th/60.)
        gs = hp.smoothing(dg, fwhm=f, lmax=LMAX, verbose=False)
        ys = hp.smoothing(ymap, fwhm=f, lmax=LMAX, verbose=False)
        ts = hp.smoothing(tmap, fwhm=f, lmax=LMAX, verbose=False)
        ks = hp.smoothing(kmap, fwhm=f, lmax=LMAX, verbose=False)
        # 3-point
        vec.extend([np.mean(gs**2 * ys), np.mean(gs**2 * ts), np.mean(gs**2 * ks)])
        # 2-point
        vec.extend([np.mean(gs * ys), np.mean(gs * ts), np.mean(gs * ks)])
        
    return np.array(vec)

print("Extracting reference run...")
x_obs = extract_moments(os.path.join(BASE_DIR, 'reference_run')).astype(np.float32)
np.save('x_obs.npy', x_obs)

theta_train, x_train = [], []
for csv in CSV_FILES:
    df = pd.read_csv(csv)
    # Determine folder offset based on the round
    if "lhs" in csv: offset = 0
    elif "round2" in csv: offset = 30
    elif "round3" in csv: offset = 60
    elif "round4" in csv: offset = 90

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(csv)}"):
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

np.save('theta.npy', theta_train)

# =============================================================================
# 2. LTU-ILI TRAINING LOOP (DIRECT VECTORS)
# =============================================================================
stat_map = {
    'g2y': [0, 3, 6, 9, 12], 'g2tau': [1, 4, 7, 10, 13], 'g2kappa': [2, 5, 8, 11, 14],
    'gy': [15, 18, 21, 24, 27], 'gtau': [16, 19, 22, 25, 28], 'gkappa': [17, 20, 23, 26, 29],
    'JOINT': list(range(30)),
    'y_total': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
    'tau_total': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28],
    'kappa_total': [2, 5, 8, 11, 14, 17, 20, 23, 26, 29],
    'all_3pt': list(range(0, 15)), 'all_2pt': list(range(15, 30)),
}

for name, idx in stat_map.items():
    print(f"\n--- Training Unscaled NPE: {name} ---")
    xt = x_train[:, idx]
    xo = x_obs[idx]
    np.save(f'x_{name}.npy', xt)
    np.save(f'xobs_{name}.npy', xo)

    loader = StaticNumpyLoader(
        in_dir='./',
        x_file=f'x_{name}.npy',
        theta_file='theta.npy',
        xobs_file=f'xobs_{name}.npy'
    )

    nets = load_nde_sbi(engine='NPE', model='nsf', repeats=4, hidden_features=64, num_transforms=3) + \
           load_nde_sbi(engine='NPE', model='maf', repeats=4, hidden_features=64, num_transforms=3)

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=Uniform(low=[1.0, -0.3, -3.0, 0.01], high=[6.0, 0.0, 3.0, 1.5]),
        nets=nets, out_dir=Path(f'./sbi_logs_{name}')
    )

    posterior, _ = runner(loader)

    with open(f'ili_posterior_{name}.pkl', 'wb') as f:
        pk.dump(posterior, f)

    # --- Integrated Coverage Test (Fast Look: 50 random samples) ---
    labels = [r"$\theta_{ej,0}$", r"${\nu_{\theta_{ej}}}^{M}$", r"${\nu_{\theta_{ej}}}^{z}$",
              r"$\mu_{\beta}$"]
    if name != 'JOINT':
        print(f"Running quick coverage check for {name}...")
        val_dir = Path(f'./validation_{name}')
        val_dir.mkdir(exist_ok=True)

        # Select 50 random indices for speed
        #np.random.seed(SEED)
        #idx_val = np.random.choice(len(theta_train), 50, replace=False)

        metric = PosteriorCoverage(num_samples=1000, labels=labels, out_dir=val_dir,
            plot_list=["histogram", "coverage","tarp"],sample_method='direct')

        metric(posterior=posterior,x=xt,theta=theta_train) 
               #x=xt[idx_val], theta=theta_train[idx_val])

# =============================================================================
# 3. NEXT ROUND PROPOSAL (30 New Samples)
# =============================================================================
with open('ili_posterior_JOINT.pkl', 'rb') as f:
    joint_posterior = pk.load(f)

# Sample at the direct observation
xo_tensor = torch.from_numpy(x_obs).float().reshape(1, -1)
next_theta = joint_posterior.sample((30,), x=xo_tensor).detach().cpu().numpy()

pd.DataFrame(next_theta, 
             columns=['theta_ej_0', 'nu_theta_ej_M', 'nu_theta_ej_z', 'mu_beta']
            ).to_csv('round3_samples.csv', index_label='sample_id')

print("\nFinished. Generated round3_samples.csv with 30 new simulation points.")
