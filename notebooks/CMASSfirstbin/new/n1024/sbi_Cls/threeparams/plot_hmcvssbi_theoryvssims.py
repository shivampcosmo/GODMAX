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
# TUNEABLE PARAMETERS — simulation-based chains
# =============================================================================
SBI_N_SAMPLES = 4000
HMC_COLOR     = '#1f77b4'   # blue
SBI_COLOR     = '#d62728'   # red
HMC_1D_LW    = 3.0
SBI_1D_LW    = 3.0
HMC_ALPHA     = 0.40
SBI_ALPHA     = 0.60

# =============================================================================
# TUNEABLE PARAMETERS — theory-based chains
# =============================================================================
THEORY_HMC_COLOR  = '#2ca02c'   # green
THEORY_SBI_COLOR  = '#9467bd'   # purple
THEORY_HMC_1D_LW  = 3.0
THEORY_SBI_1D_LW  = 3.0
THEORY_ALPHA      = 0.45

# Keys stored inside the theory HMC npz as  samples_{name}
THEORY_PARAM_NAMES = ['theta_ej_0', 'nu_theta_ej_M', 'nu_theta_ej_z']
THEORY_PRIOR_MIN   = np.array([1.0, -0.3, -3.0])
THEORY_PRIOR_MAX   = np.array([6.0,  0.0, 3.0])

# =============================================================================
# SHARED PARAMETER METADATA
# =============================================================================
PARAM_NAMES     = ['p1', 'p2', 'p3']
PARAM_LABELS_GD = [r'\theta_{ej,0}', r'\nu_{\theta_{ej}}^{M}', r'\nu_{\theta_{ej}}^{z}']
TRUTH_VALUES    = np.array([2.0, -0.1, 0.0])
PARAM_LIMITS    = [(1.0, 6.0), (-0.3, 0.0), (-3.0, 3.0)]

# =============================================================================
# PATHS
# =============================================================================
OUTPUT_DIR      = 'plotcontours_hmcvssbi_3params'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation-based outputs
HMC_OUTPUT_DIR  = 'hmc_vs_sbi_outputs'

# Theory-based outputs  (produced by run_hmcvssbi_theory.py)
THEORY_SBI_BASE = Path('outputs/theory_sbi')

PROBES = ['gy', 'gtau', 'gkappa', 'all_2pt']

# =============================================================================
# PATH HELPERS
# =============================================================================

# ── simulation-based ─────────────────────────────────────────────────────────
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

# ── theory-based ─────────────────────────────────────────────────────────────
def theory_hmc_path(probe):
    return THEORY_SBI_BASE / f'{probe}_linearized' / 'hmc_samples.npz'

def theory_sbi_path(probe):
    return THEORY_SBI_BASE / f'{probe}_linearized_fisher_mdn5' / 'sbi_posterior_samples.npz'

def theory_hmc_diagnostics_path(probe):
    return THEORY_SBI_BASE / f'{probe}_linearized' / 'hmc_diagnostics.json'

# =============================================================================
# CPU LOADING UTILITIES  (shared by both pipelines)
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
# PRIOR FILTER  (theory chains only — sim chains are already within bounds)
# =============================================================================
def _filter_theory_prior(samples):
    samples = np.asarray(samples, dtype=float)
    mask    = np.all(
        (samples >= THEORY_PRIOR_MIN[None, :]) &
        (samples <= THEORY_PRIOR_MAX[None, :]),
        axis=1,
    )
    n_before = len(samples)
    filtered = samples[mask]
    n_after  = len(filtered)
    if n_after < n_before:
        print(f'    prior filter: {n_before} → {n_after} '
              f'({n_before - n_after} rejected)')
    return filtered

# =============================================================================
# SHARED MCSAMPLES FACTORY
# =============================================================================
def _make_mcsamples(samples, label):
    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS_GD,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )

# =============================================================================
# SIMULATION-BASED LOADERS
# =============================================================================
def load_hmc_mcsamples(probe, label='HMC / NUTS (sim)'):
    path = hmc_samples_path(probe)
    if not Path(path).exists():
        print(f'  [SKIP] Sim HMC not found: {path}')
        return None

    data = np.load(path)
    try:
        samples = np.column_stack([
            data['theta_ej_0'].ravel(),
            data['nu_theta_ej_M'].ravel(),
        ])
    except KeyError as e:
        print(f'  [ERROR] Unexpected key in {path}: {e}  '
              f'(available: {list(data.keys())})')
        return None

    print(f'  [{probe}] Sim HMC: {samples.shape}  '
          f'theta_ej_0={samples[:,0].mean():.3f}±{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}±{samples[:,1].std():.3f}')

    diag_path = hmc_diagnostics_path(probe)
    if Path(diag_path).exists():
        with open(diag_path) as f:
            diag = json.load(f)
        print(f'  [{probe}] Sim HMC diagnostics — '
              f'r_hat_max: {diag.get("max_rhat", "?")}  '
              f'min_ess_bulk: {diag.get("min_ess_bulk", "?")}')

    return _make_mcsamples(samples, label)


def load_sbi_mcsamples(probe, label='SBI / NPE+MDN (sim)', n_samples=SBI_N_SAMPLES):
    pkl_path  = sbi_posterior_path(probe)
    xobs_path = sbi_xobs_path(probe)

    if not Path(pkl_path).exists():
        print(f'  [SKIP] Sim SBI posterior not found: {pkl_path}')
        return None
    if not Path(xobs_path).exists():
        print(f'  [SKIP] Sim SBI xobs not found: {xobs_path}')
        return None

    print(f'  [{probe}] Loading Sim SBI posterior: {pkl_path}')
    print(f'  [{probe}] Loading Sim SBI xobs:      {xobs_path}')

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
            print(f'  [WARN] Sim SBI member {i} failed for {probe}: {e}')

    if not all_samples:
        print(f'  [WARN] Sim SBI sampling failed entirely for {probe}')
        return None

    samples = np.vstack(all_samples)
    print(f'  [{probe}] Sim SBI: {samples.shape}  '
          f'theta_ej_0={samples[:,0].mean():.3f}±{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}±{samples[:,1].std():.3f}')

    return _make_mcsamples(samples, label)


# =============================================================================
# THEORY-BASED LOADERS
# =============================================================================
def load_theory_hmc_mcsamples(probe, label='HMC / NUTS (theory)'):
    path = theory_hmc_path(probe)
    if not path.exists():
        print(f'  [SKIP] Theory HMC not found: {path}')
        return None

    data = np.load(path, allow_pickle=True)

    # theory HMC npz stores arrays as  samples_{param_name}
    try:
        samples = np.column_stack(
            [data[f'samples_{n}'].ravel() for n in THEORY_PARAM_NAMES]
        )
    except KeyError as e:
        print(f'  [ERROR] Unexpected key in {path}: {e}  '
              f'(available: {list(data.keys())})')
        return None

    print(f'  [{probe}] Theory HMC: {samples.shape}  '
          f'theta_ej_0={samples[:,0].mean():.3f}±{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}±{samples[:,1].std():.3f}')

    diag_path = theory_hmc_diagnostics_path(probe)
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)
        print(f'  [{probe}] Theory HMC diagnostics — '
              f'r_hat_max: {diag.get("max_rhat", "?")}  '
              f'min_ess_bulk: {diag.get("min_ess_bulk", "?")}')

    return _make_mcsamples(samples, label)


def load_theory_sbi_mcsamples(probe, label='SBI / NPE+MDN (theory)'):
    path = theory_sbi_path(probe)
    if not path.exists():
        print(f'  [SKIP] Theory SBI not found: {path}')
        return None

    data    = np.load(path, allow_pickle=True)
    # theory SBI npz stores all samples under the key 'samples'
    raw     = np.asarray(data['samples'], dtype=float)
    samples = _filter_theory_prior(raw)

    if len(samples) == 0:
        print(f'  [WARN] Theory SBI for {probe}: all samples rejected by prior filter')
        return None

    print(f'  [{probe}] Theory SBI: {samples.shape}  '
          f'theta_ej_0={samples[:,0].mean():.3f}±{samples[:,0].std():.3f}  '
          f'nu={samples[:,1].mean():.3f}±{samples[:,1].std():.3f}')

    return _make_mcsamples(samples, label)


# =============================================================================
# RHAT ANNOTATION HELPER
# =============================================================================
def _annotate_rhat(ax, diag_path):
    """Adds a max-R̂ warning to `ax` if the value exceeds 1.01."""
    if not Path(diag_path).exists():
        return
    with open(diag_path) as f:
        diag = json.load(f)
    max_rhat = float(diag.get('max_rhat', 1.0))
    if max_rhat > 1.01:
        ax.text(
            0.97, 0.95,
            rf'$\hat{{R}}_\mathrm{{max}}={max_rhat:.2f}$ ⚠️',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=13, color='darkorange',
        )


# =============================================================================
# MAIN PLOTTING LOOP
# =============================================================================
for probe in PROBES:
    print(f'\n{"─"*60}')
    print(f'  Probe: {probe}')
    print(f'{"─"*60}')

    sim_hmc_smp    = load_hmc_mcsamples(probe)
    sim_sbi_smp    = load_sbi_mcsamples(probe)
    theory_hmc_smp = load_theory_hmc_mcsamples(probe)
    theory_sbi_smp = load_theory_sbi_mcsamples(probe)

    # Build the ordered list of (MCSamples, color, alpha, lw, label) tuples,
    # skipping any that failed to load.
    entries = [
        (sim_hmc_smp,    HMC_COLOR,        HMC_ALPHA,    HMC_1D_LW,       'HMC / NUTS (sim)'),
        (sim_sbi_smp,    SBI_COLOR,        SBI_ALPHA,    SBI_1D_LW,       'SBI / NPE+MDN (sim)'),
        (theory_hmc_smp, THEORY_HMC_COLOR, THEORY_ALPHA, THEORY_HMC_1D_LW,'HMC / NUTS (theory)'),
        (theory_sbi_smp, THEORY_SBI_COLOR, THEORY_ALPHA, THEORY_SBI_1D_LW,'SBI / NPE+MDN (theory)'),
    ]
    present = [(s, c, a, lw, lbl) for s, c, a, lw, lbl in entries if s is not None]

    if not present:
        print(f'  [SKIP] No data available for {probe}')
        continue

    mcsamples_list = [s   for s, c, a, lw, lbl in present]
    colors         = [c   for s, c, a, lw, lbl in present]
    legend_labels  = [lbl for s, c, a, lw, lbl in present]
    contour_args   = [{'alpha': a}              for s, c, a, lw, lbl in present]
    line_args      = [{'lw': lw, 'color': c}   for s, c, a, lw, lbl in present]

    print(f'\n  Plotting {len(present)} chain(s): '
          + '  |  '.join(legend_labels))

    g = plots.get_subplot_plotter(width_inch=8, settings=plot_settings)
    g.triangle_plot(
        mcsamples_list,
        filled=True,
        colors=colors,
        legend_labels=legend_labels,
        contour_args=contour_args,
        line_args=line_args,
    )

    # ── Axis limits + truth lines ─────────────────────────────────────────────
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

    # ── R̂ warnings (sim HMC + theory HMC) ────────────────────────────────────
    _annotate_rhat(g.subplots[0, 0], hmc_diagnostics_path(probe))
    _annotate_rhat(g.subplots[0, 0], theory_hmc_diagnostics_path(probe))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, f'hmc_vs_sbi_{probe}.png')
    g.export(out_path, dpi=200)
    print(f'  [DONE] Saved: {out_path}')
    plt.close('all')

print('\nAll plots complete.')
os._exit(0)
