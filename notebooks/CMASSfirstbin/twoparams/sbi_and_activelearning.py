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
SCALES = [4.0, 8.0, 16.0, 32.0, 64.0]
CACHE_DIR = 'sample_vector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_moments(path):
    # Standard glob pattern from your 4-param script
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
        # Smoothing with EXPLICIT LMAX for consistency
        gs = hp.smoothing(dg, fwhm=f, lmax=LMAX, verbose=False)
        ys = hp.smoothing(ymap, fwhm=f, lmax=LMAX, verbose=False)
        ts = hp.smoothing(tmap, fwhm=f, lmax=LMAX, verbose=False)
        ks = hp.smoothing(kmap, fwhm=f, lmax=LMAX, verbose=False)

        # 3-point moments <ggTracer> (indices 0-14 in output vector)
        vec.extend([np.mean(gs**2 * ys), np.mean(gs**2 * ts), np.mean(gs**2 * ks)])

    for th in SCALES:
        f = np.radians(th/60.)
        gs = hp.smoothing(dg, fwhm=f, lmax=LMAX, verbose=False)
        ys = hp.smoothing(ymap, fwhm=f, lmax=LMAX, verbose=False)
        ts = hp.smoothing(tmap, fwhm=f, lmax=LMAX, verbose=False)
        ks = hp.smoothing(kmap, fwhm=f, lmax=LMAX, verbose=False)

        # 2-point moments <gTracer> (indices 15-29 in output vector)
        vec.extend([np.mean(gs * ys), np.mean(gs * ts), np.mean(gs * ks)])
    return np.array(vec)
print("Extracting reference run...")
# Reference is in original location
x_obs = extract_moments(os.path.join(BASE_DIR, 'reference_run')).astype(np.float32)
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
            v = extract_moments(s_path)
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
    'g2y': [0, 3, 6, 9, 12],
    'g2tau': [1, 4, 7, 10, 13],
    'g2kappa': [2, 5, 8, 11, 14],
    'gy': [15, 18, 21, 24, 27],
    'gtau': [16, 19, 22, 25, 28],
    'gkappa': [17, 20, 23, 26, 29],
    'JOINT': list(range(30)),
    # Individual Tracer Totals (2pt + 3pt)
    'y_total': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
    'tau_total': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28],
    'kappa_total': [2, 5, 8, 11, 14, 17, 20, 23, 26, 29],

    # Category Totals (All tracers combined)
    'all_3pt': list(range(0, 15)),
    'all_2pt': list(range(15, 30)),
}

np.random.seed(42)
all_indices = np.arange(len(theta_train))
np.random.shuffle(all_indices)

val_idx = all_indices[:20]
train_idx = all_indices#[20:]

theta_val = theta_train[val_idx]
theta_train_set = theta_train[train_idx]

for name, idx in stat_map.items():
    print(f"\n--- Training NPE: {name} ---")

    # Split the statistics
    xt_full = x_train[:, idx]
    xt_train = xt_full[train_idx]
    #xt_train = xt_train + 0.01 * np.random.randn(*xt_train.shape).astype(np.float32)
    xt_val = xt_full[val_idx]
    xo = x_obs[idx]

    np.save(f'x_{name}.npy', xt_train)
    np.save(f'xobs_{name}.npy', xo)
    np.save(f'theta_train_{name}.npy', theta_train_set)

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

    # --- VALIDATION: Using Hold-out Set ---
    '''
    labels = [r"\theta_{ej,0}", r"\mu_{\beta}"]
    val_dir = Path(f'./validation_{name}')
    val_dir.mkdir(exist_ok=True)

    print(f"Running resilient validation for {name}...")
    
    successful_indices = []
    # We loop through the 20 samples in the hold-out set
    for i in range(len(theta_val)):
        xo_single = torch.from_numpy(xt_val[i]).float().reshape(1, -1)
        try:
            # We test if the model can sample from this observation without hanging
            # If it takes more than a few seconds, it's likely a rejection loop
            _ = posterior.sample((100,), x=xo_single, show_progress_bars=False)
            successful_indices.append(i)
        except Exception as e:
            print(f"Skipping Sample {i} for {name}: Model cannot reconcile this simulation.")

    # Only run coverage on the simulations the model successfully sampled
    if len(successful_indices) > 0:
        print(f"Collected valid samples for {len(successful_indices)}/20 validation points.")

        # Filter the validation data to only include the successful ones
        xt_val_successful = torch.from_numpy(xt_val[successful_indices]).float()
        theta_val_successful = torch.from_numpy(theta_val[successful_indices]).float()

        try:
            # Note: sample_method='direct' is key to staying out of the rejection trap
            metric = PosteriorCoverage(
                num_samples=200, 
                labels=labels, 
                out_dir=val_dir,
                plot_list=["histogram", "coverage", "tarp"],
                sample_method='direct'
            )
            
            # The metric now receives only the data points it can handle
            metric(posterior=posterior, x=xt_val_successful, theta=theta_val_successful)
        except Exception as e:
            print(f"Skipping coverage plotting for {name}: {e}")
    '''
# =============================================================================
# 3. ROUND 2 PROPOSAL (Next 20 Samples)
# =============================================================================

with open('ili_posterior_JOINT.pkl', 'rb') as f:
    joint_posterior = pk.load(f)

# Sample at the direct observation
xo_tensor = torch.from_numpy(x_obs).float().reshape(1, -1)
next_theta = joint_posterior.sample((20,), x=xo_tensor).detach().cpu().numpy()

pd.DataFrame(next_theta,
             columns=['theta_ej_0', 'mu_beta']
            ).to_csv('round7_samples.csv', index_label='sample_id')

print("\nFinished. Generated round7_samples.csv for the 2-parameter project.")

