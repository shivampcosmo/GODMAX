import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import pickle as pk
import torch
import torch.nn as nn

# ── Monkey-patch BEFORE any ltu-ili / sbi import ─────────────────────────────
torch.cuda.is_available = lambda: False

_orig_module_to = nn.Module.to
def _patched_module_to(self, *args, **kwargs):
    args = tuple(
        'cpu' if isinstance(a, str) and 'cuda' in a else a
        for a in args
    )
    if 'device' in kwargs and 'cuda' in str(kwargs.get('device', '')):
        kwargs['device'] = 'cpu'
    return _orig_module_to(self, *args, **kwargs)
nn.Module.to   = _patched_module_to
nn.Module.cuda = lambda self, *args, **kwargs: self

_orig_tensor_to = torch.Tensor.to
def _patched_tensor_to(self, *args, **kwargs):
    args = tuple(
        'cpu' if isinstance(a, str) and 'cuda' in a else a
        for a in args
    )
    if 'device' in kwargs and 'cuda' in str(kwargs.get('device', '')):
        kwargs['device'] = 'cpu'
    return _orig_tensor_to(self, *args, **kwargs)
torch.Tensor.to   = _patched_tensor_to
torch.Tensor.cuda = lambda self, *args, **kwargs: self
# ─────────────────────────────────────────────────────────────────────────────

import io
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
from getdist.plots import GetDistPlotSettings
import logging
from pathlib import Path

logging.getLogger('getdist').setLevel(logging.ERROR)

# =============================================================================
# PLOT STYLE
# =============================================================================
mpl.rcParams['xtick.major.size']  = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size']  = 7
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size']  = 10
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size']  = 7
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction']   = 'in'
mpl.rcParams['ytick.direction']   = 'in'
mpl.rcParams['font.size']         = 20
mpl.rcParams['axes.labelsize']    = 22
mpl.rcParams['legend.fontsize']   = 18
mpl.rcParams['xtick.labelsize']   = 16
mpl.rcParams['ytick.labelsize']   = 16

plot_settings = GetDistPlotSettings()
plot_settings.axes_fontsize        = 20
plot_settings.lab_fontsize         = 22
plot_settings.legend_fontsize      = 18
plot_settings.title_limit_fontsize = 18

# =============================================================================
# TUNEABLE PARAMETERS
# =============================================================================
SBI_N_SAMPLES   = 4000
# Simulation Cls colours
HMC_COLOR       = '#1f77b4'   # blue
SBI_COLOR       = '#d62728'   # red
# Theory Cls colours
THEORY_HMC_COLOR = '#2ca02c'  # green
THEORY_SBI_COLOR = '#9467bd'  # purple

HMC_1D_LW      = 3.0
SBI_1D_LW      = 3.0
HMC_ALPHA       = 0.4
SBI_ALPHA       = 0.6
THEORY_ALPHA    = 0.45

PARAM_NAMES     = ['p1', 'p2']
PARAM_LABELS_GD = [r'\theta_{ej,0}', r'\nu_{\theta_{ej}}^{M}']
THEORY_PARAM_NAMES = ['theta_ej_0', 'nu_theta_ej_M']   # keys in theory npz
TRUTH_VALUES    = np.array([2.0, -0.1])
PARAM_LIMITS    = [(1.0, 6.0), (-0.3, 0.0)]

OUTPUT_DIR      = 'plotcontours_hmcvssbi'
os.makedirs(OUTPUT_DIR, exist_ok=True)

HMC_OUTPUT_DIR  = 'hmc_vs_sbi_outputs'

# Theory Cls outputs written by the ported SBI_validate notebook
THEORY_SBI_BASE    = Path('outputs') / 'theory_sbi'
THEORY_HMC_RUN     = 'joint_gg_gy_gtau_gkappa_linearized'
THEORY_SBI_RUN     = 'joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5'
THEORY_HMC_NPZ     = THEORY_SBI_BASE / THEORY_HMC_RUN / 'hmc_samples.npz'
THEORY_SBI_NPZ     = THEORY_SBI_BASE / THEORY_SBI_RUN / 'sbi_posterior_samples.npz'

PROBES = ['gy', 'gtau', 'gkappa', 'all_2pt']

# =============================================================================
# PATH HELPERS  (simulation Cls)
# =============================================================================
def hmc_samples_path(probe):
    return os.path.join(HMC_OUTPUT_DIR, f'hmc_samples_{probe}.npz')

def sbi_posterior_path(probe):
    main = f'ili_posterior_{probe}.pkl'
    own  = f'ili_posterior_2pt_{probe}.pkl'
    return main if Path(main).exists() else own

def sbi_xobs_path(probe):
    main = f'xobs_{probe}.npy'
    own  = f'xobs_2pt_{probe}.npy'
    return main if Path(main).exists() else own

def hmc_diagnostics_path(probe):
    return os.path.join(HMC_OUTPUT_DIR, f'hmc_{probe}',
                        f'hmc_diagnostics_{probe}.json')

# =============================================================================
# CPU LOADING UTILITIES
# =============================================================================
class _CPUUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(
                io.BytesIO(b), map_location='cpu', weights_only=False
            )
        return super().find_class(module, name)


def _recursive_cpu(obj, visited=None):
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)
    if isinstance(obj, nn.Module):
        _orig_module_to(obj, 'cpu')
    elif isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, str):
        return 'cpu' if 'cuda' in obj else obj
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            try:
                obj[k] = _recursive_cpu(obj[k], visited)
            except Exception:
                pass
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)([_recursive_cpu(i, visited) for i in obj])
    if hasattr(obj, '__dict__'):
        for k, v in list(obj.__dict__.items()):
            try:
                new_v = _recursive_cpu(v, visited)
                if new_v is not v:
                    setattr(obj, k, new_v)
            except Exception:
                pass
    return obj


def load_sbi_posterior(path):
    with open(path, 'rb') as f:
        posterior = _CPUUnpickler(f).load()
    _recursive_cpu(posterior)
    return posterior


# =============================================================================
# SAMPLE LOADERS  (simulation Cls)
# =============================================================================
def load_hmc_mcsamples(probe, label='HMC / NUTS (sim)'):
    path = hmc_samples_path(probe)
    if not Path(path).exists():
        print(f'[SKIP] HMC samples not found: {path}')
        return None

    data = np.load(path)
    try:
        samples = np.column_stack([
            data['theta_ej_0'].ravel(),
            data['nu_theta_ej_M'].ravel(),
        ])
    except KeyError as e:
        print(f'[ERROR] Unexpected key structure in {path}: {e}')
        print(f'        Available keys: {list(data.keys())}')
        return None

    print(f'  [{probe}] HMC (sim) samples: {samples.shape}  '
          f'theta_ej_0={samples[:,0].mean():.3f}+/-{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}+/-{samples[:,1].std():.3f}')

    diag_path = hmc_diagnostics_path(probe)
    if Path(diag_path).exists():
        with open(diag_path) as f:
            diag = json.load(f)
        print(f'  [{probe}] HMC (sim) diagnostics — '
              f'r_hat_max: {diag.get("max_rhat", "?")}  '
              f'min_ess_bulk: {diag.get("min_ess_bulk", "?")}')

    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS_GD,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


def load_sbi_mcsamples(probe, label='SBI / NPE+MDN (sim)', n_samples=SBI_N_SAMPLES):
    pkl_path  = sbi_posterior_path(probe)
    xobs_path = sbi_xobs_path(probe)

    if not Path(pkl_path).exists():
        print(f'[SKIP] SBI posterior not found: {pkl_path}')
        return None
    if not Path(xobs_path).exists():
        print(f'[SKIP] SBI xobs not found: {xobs_path}')
        return None

    print(f'  [{probe}] Loading SBI (sim) posterior: {pkl_path}')
    posterior  = load_sbi_posterior(pkl_path)
    x_obs_norm = np.load(xobs_path)

    x_tensor = (torch.from_numpy(np.array(x_obs_norm))
                .float().reshape(1, -1).to('cpu'))

    members = getattr(posterior, 'posteriors', [posterior])
    n_mem   = len(members)
    base    = n_samples // n_mem
    extra   = n_samples - base * n_mem

    all_samples = []
    for i, member in enumerate(members):
        n_draw = base + (extra if i == 0 else 0)
        try:
            s = member.sample((n_draw,), x=x_tensor, show_progress_bars=False)
            all_samples.append(s.detach().cpu().numpy())
        except Exception as e:
            print(f'  [WARN] SBI member {i} failed for {probe}: {e}')

    if not all_samples:
        print(f'[WARN] SBI (sim) sampling failed entirely for {probe}')
        return None

    samples = np.vstack(all_samples)
    print(f'  [{probe}] SBI (sim) samples: {samples.shape}  '
          f'theta_ej_0={samples[:,0].mean():.3f}+/-{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}+/-{samples[:,1].std():.3f}')

    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS_GD,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


# =============================================================================
# SAMPLE LOADERS  (theory Cls)
# =============================================================================

# Prior bounds from default_parameter_specs() — used to filter SBI samples
THEORY_PRIOR_MIN = np.array([0.5, -1.0])
THEORY_PRIOR_MAX = np.array([4.0,  0.0])


def _filter_theory_prior(samples: np.ndarray) -> np.ndarray:
    mask = np.all(
        (samples >= THEORY_PRIOR_MIN[None, :]) &
        (samples <= THEORY_PRIOR_MAX[None, :]),
        axis=1,
    )
    return samples[mask]


def load_theory_hmc_mcsamples(label='HMC / NUTS (theory)'):
    """
    Loads theory HMC samples from hmc_samples.npz written by run_hmc_theory_cls.
    Keys: samples_{name}  for each param name in THEORY_PARAM_NAMES.
    """
    path = THEORY_HMC_NPZ
    if not path.exists():
        print(f'[SKIP] Theory HMC samples not found: {path}')
        return None

    data = np.load(path, allow_pickle=True)
    try:
        samples = np.column_stack([
            data[f'samples_{name}'].ravel()
            for name in THEORY_PARAM_NAMES
        ])
    except KeyError as e:
        print(f'[ERROR] Missing key in {path}: {e}')
        print(f'        Available keys: {list(data.keys())}')
        return None

    print(f'  [theory] HMC samples: {samples.shape}  '
          f'theta_ej_0={samples[:,0].mean():.3f}+/-{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}+/-{samples[:,1].std():.3f}')

    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS_GD,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


def load_theory_sbi_mcsamples(label='SBI / NPE+MDN (theory)'):
    """
    Loads theory SBI samples from sbi_posterior_samples.npz written by
    run_sbi_theory_cls.  Key: 'samples'  shape (n_posterior, n_params)
    in physical (theta) space — already back-transformed from Fisher basis.
    """
    path = THEORY_SBI_NPZ
    if not path.exists():
        print(f'[SKIP] Theory SBI samples not found: {path}')
        return None

    data = np.load(path, allow_pickle=True)
    try:
        samples = np.asarray(data['samples'], dtype=float)
    except KeyError as e:
        print(f'[ERROR] Missing key in {path}: {e}')
        print(f'        Available keys: {list(data.keys())}')
        return None

    n_before = len(samples)
    samples  = _filter_theory_prior(samples)
    print(f'  [theory] SBI samples: {len(samples)} / {n_before} inside prior  '
          f'theta_ej_0={samples[:,0].mean():.3f}+/-{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}+/-{samples[:,1].std():.3f}')

    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS_GD,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


# =============================================================================
# Load theory contours ONCE  (same for every probe — joint over all 4 probes)
# =============================================================================
print('\n── Loading theory Cls contours (joint gg+gy+gtau+gkappa) ──')
theory_hmc_smp = load_theory_hmc_mcsamples()
theory_sbi_smp = load_theory_sbi_mcsamples()


# =============================================================================
# MAIN PLOTTING LOOP
# =============================================================================
for probe in PROBES:
    print(f'\n── Probe: {probe} ──')

    hmc_smp = load_hmc_mcsamples(probe)
    sbi_smp = load_sbi_mcsamples(probe)

    # Collect all available contours in display order:
    #   sim HMC → sim SBI → theory HMC → theory SBI
    entries = [
        (hmc_smp,        HMC_COLOR,        HMC_ALPHA,    HMC_1D_LW,  'HMC / NUTS (sim)'),
        (sbi_smp,        SBI_COLOR,        SBI_ALPHA,    SBI_1D_LW,  'SBI / NPE+MDN (sim)'),
        (theory_hmc_smp, THEORY_HMC_COLOR, THEORY_ALPHA, HMC_1D_LW,  'HMC / NUTS (theory)'),
        (theory_sbi_smp, THEORY_SBI_COLOR, THEORY_ALPHA, SBI_1D_LW,  'SBI / NPE+MDN (theory)'),
    ]
    present = [(s, c, a, lw, lbl) for s, c, a, lw, lbl in entries if s is not None]

    if not present:
        print(f'[SKIP] No data available for {probe}')
        continue

    mcsamples_list = [s   for s, c, a, lw, lbl in present]
    colors         = [c   for s, c, a, lw, lbl in present]
    alphas         = [a   for s, c, a, lw, lbl in present]
    linewidths     = [lw  for s, c, a, lw, lbl in present]
    legend_labels  = [lbl for s, c, a, lw, lbl in present]

    g = plots.get_subplot_plotter(width_inch=8, settings=plot_settings)
    g.triangle_plot(
        mcsamples_list,
        filled=True,
        colors=colors,
        legend_labels=legend_labels,
        contour_args=[{'alpha': a} for a in alphas],
        line_args=[{'lw': lw, 'color': c} for lw, c in zip(linewidths, colors)],
    )

    # ── Axis limits and truth lines ───────────────────────────────────────────
    for i, (lo_i, hi_i) in enumerate(PARAM_LIMITS):
        ax_diag = g.subplots[i, i]
        ax_diag.set_xlim(lo_i, hi_i)
        ax_diag.axvline(TRUTH_VALUES[i], color='black', ls='--', lw=2, zorder=10)

        for j in range(i):
            lo_j, hi_j = PARAM_LIMITS[j]
            ax_off = g.subplots[i, j]
            ax_off.set_xlim(lo_j, hi_j)
            ax_off.set_ylim(lo_i, hi_i)
            ax_off.axvline(TRUTH_VALUES[j], color='black', ls='--', lw=2, zorder=10)
            ax_off.axhline(TRUTH_VALUES[i], color='black', ls='--', lw=2, zorder=10)

    # ── r_hat warning annotation ──────────────────────────────────────────────
    diag_path = hmc_diagnostics_path(probe)
    if Path(diag_path).exists():
        with open(diag_path) as f:
            diag = json.load(f)
        max_rhat = float(diag.get('max_rhat', 1.0))
        if max_rhat > 1.01:
            g.subplots[0, 0].text(
                0.97, 0.95,
                rf'$\hat{{R}}_\mathrm{{max}}={max_rhat:.2f}$ ⚠️',
                transform=g.subplots[0, 0].transAxes,
                ha='right', va='top',
                fontsize=13, color='darkorange',
            )

    out_path = os.path.join(OUTPUT_DIR, f'hmc_vs_sbi_{probe}.png')
    g.export(out_path, dpi=200)
    print(f'[DONE] Saved: {out_path}')
    plt.close('all')

print('\nAll plots complete.')
os._exit(0)
