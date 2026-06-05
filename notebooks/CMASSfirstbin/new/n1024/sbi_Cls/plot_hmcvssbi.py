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
HMC_COLOR       = '#1f77b4'
SBI_COLOR       = '#d62728'
HMC_1D_LW      = 3.0
SBI_1D_LW      = 3.0
HMC_ALPHA       = 0.4
SBI_ALPHA       = 0.6

PARAM_NAMES     = ['p1', 'p2']
PARAM_LABELS_GD = [r'\theta_{ej,0}', r'\nu_{\theta_{ej}}^{M}']
TRUTH_VALUES    = np.array([2.0, -0.1])
PARAM_LIMITS    = [(1.0, 6.0), (-0.3, 0.0)]

OUTPUT_DIR      = 'plotcontours_hmcvssbi'
os.makedirs(OUTPUT_DIR, exist_ok=True)

#HMC outputs live in hmc_vs_sbi_outputs/, not the current directory.
#SBI posteriors and xobs scalers live in the current directory (sbi_Cls/).
HMC_OUTPUT_DIR  = 'hmc_vs_sbi_outputs'

PROBES = ['gy', 'gtau', 'gkappa', 'all_2pt']

def hmc_samples_path(probe):
    return os.path.join(HMC_OUTPUT_DIR, f'hmc_samples_{probe}.npz')

def sbi_posterior_path(probe):
    # Main pipeline posteriors live in sbi_Cls/ (current dir).
    # Fall back to 2pt-specific posteriors if main ones are absent.
    main = f'ili_posterior_{probe}.pkl'
    own  = f'ili_posterior_2pt_{probe}.pkl'
    return main if Path(main).exists() else own

def sbi_xobs_path(probe):
    # Main pipeline saves xobs_{probe}.npy; fallback is xobs_2pt_{probe}.npy.
    main = f'xobs_{probe}.npy'
    own  = f'xobs_2pt_{probe}.npy'
    return main if Path(main).exists() else own

#Diagnostics are inside hmc_vs_sbi_outputs/hmc_{probe}/.
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
# SAMPLE LOADERS
# =============================================================================
def load_hmc_mcsamples(probe, label='HMC / NUTS'):
    """
    Loads HMC samples saved by np.savez_compressed with named arrays:
        theta_ej_0    : shape (num_chains * num_samples,)
        nu_theta_ej_M : shape (num_chains * num_samples,)
    """
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

    print(f'  [{probe}] HMC samples loaded: {samples.shape}  '
          f'theta_ej_0={samples[:,0].mean():.3f}+/-{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}+/-{samples[:,1].std():.3f}')

    # Print diagnostics summary if available
    diag_path = hmc_diagnostics_path(probe)
    if Path(diag_path).exists():
        with open(diag_path) as f:
            diag = json.load(f)
        print(f'  [{probe}] HMC diagnostics — '
              f'r_hat_max: {diag.get("max_rhat", "?")}  '
              f'min_ess_bulk: {diag.get("min_ess_bulk", "?")}')

    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS_GD,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


def load_sbi_mcsamples(probe, label='SBI / NPE+MDN', n_samples=SBI_N_SAMPLES):
    pkl_path  = sbi_posterior_path(probe)
    xobs_path = sbi_xobs_path(probe)

    if not Path(pkl_path).exists():
        print(f'[SKIP] SBI posterior not found: {pkl_path}')
        return None
    if not Path(xobs_path).exists():
        print(f'[SKIP] SBI xobs not found: {xobs_path}')
        return None

    print(f'  [{probe}] Loading SBI posterior: {pkl_path}')
    print(f'  [{probe}] Loading SBI xobs:      {xobs_path}')

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
        print(f'[WARN] SBI sampling failed entirely for {probe}')
        return None

    samples = np.vstack(all_samples)
    print(f'  [{probe}] SBI samples drawn: {samples.shape}  '
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
# MAIN PLOTTING LOOP
# =============================================================================
for probe in PROBES:
    print(f'\n── Probe: {probe} ──')

    hmc_smp = load_hmc_mcsamples(probe)
    sbi_smp = load_sbi_mcsamples(probe)

    if hmc_smp is None and sbi_smp is None:
        print(f'[SKIP] No data for {probe}')
        continue

    mcsamples_list = [s for s in [hmc_smp, sbi_smp] if s is not None]
    colors         = [HMC_COLOR, SBI_COLOR][:len(mcsamples_list)]
    legend_labels  = ['HMC / NUTS', 'SBI / NPE+MDN'][:len(mcsamples_list)]

    g = plots.get_subplot_plotter(width_inch=8, settings=plot_settings)
    g.triangle_plot(
        mcsamples_list,
        filled=True,
        colors=colors,
        legend_labels=legend_labels,
        contour_args=[
            {'alpha': HMC_ALPHA},
            {'alpha': SBI_ALPHA},
        ][:len(mcsamples_list)],
        line_args=[
            {'lw': HMC_1D_LW, 'color': HMC_COLOR},
            {'lw': SBI_1D_LW, 'color': SBI_COLOR},
        ][:len(mcsamples_list)],
    )

    # ── Apply axis limits and truth lines ─────────────────────────────────────
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
