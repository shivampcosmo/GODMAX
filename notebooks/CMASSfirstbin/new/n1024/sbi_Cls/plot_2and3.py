import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # hide GPUs before any torch import

import sys
import pickle as pk
import torch
import torch.nn as nn

# ?~T~@?~T~@ Monkey-patch BEFORE any ltu-ili / sbi import ?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@
# Redirect any .to('cuda*') or .cuda() calls to CPU so that posteriors
# trained on GPU can be used on a CPU-only PyTorch build.

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
nn.Module.to = _patched_module_to
nn.Module.cuda = lambda self, *args, **kwargs: self   # no-op

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
torch.Tensor.cuda = lambda self, *args, **kwargs: self  # no-op
# ?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@?~T~@

import io
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import logging
from pathlib import Path

logging.getLogger('getdist').setLevel(logging.ERROR)

import matplotlib as mpl
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
mpl.rcParams['legend.fontsize']   = 20
mpl.rcParams['xtick.labelsize']   = 16
mpl.rcParams['ytick.labelsize']   = 16

sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.validation.metrics import PlotSinglePosterior

from getdist.plots import GetDistPlotSettings
plot_settings = GetDistPlotSettings()
plot_settings.axes_fontsize        = 20
plot_settings.lab_fontsize         = 22
plot_settings.legend_fontsize      = 20
plot_settings.title_limit_fontsize = 18

# =============================================================================
# CPU LOADING UTILITIES
# =============================================================================

class _CPUUnpickler(pk.Unpickler):
    """Remap all torch tensors to CPU at unpickle time."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(
                io.BytesIO(b), map_location='cpu', weights_only=False
            )
        return super().find_class(module, name)


def _recursive_cpu(obj, visited=None):
    """
    Walk every attribute of obj recursively and:
      - move nn.Module instances to CPU
      - move Tensor instances to CPU
      - replace any string containing 'cuda' with 'cpu'
    Uses a visited-set to avoid infinite loops on circular references.
    """
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if isinstance(obj, nn.Module):
        _orig_module_to(obj, 'cpu')   # use the original .to(), not the patched one
        # still recurse into its __dict__ to fix string attributes
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
        # tuples are immutable; return a new one only if something changed
        new_items = [_recursive_cpu(item, visited) for item in obj]
        return type(obj)(new_items)

    if hasattr(obj, '__dict__'):
        for k, v in list(obj.__dict__.items()):
            try:
                new_v = _recursive_cpu(v, visited)
                if new_v is not v:
                    setattr(obj, k, new_v)
            except Exception:
                pass

    return obj


def load_posterior(path):
    with open(path, 'rb') as f:
        posterior = _CPUUnpickler(f).load()
    _recursive_cpu(posterior)
    return posterior


# =============================================================================
# 0. SANITY CHECK
# =============================================================================
try:
    theta_check = np.load('theta_train_full.npy')
    x_check     = np.load('x_JOINT.npy')
    print("--- SBI Dataset Verification ---")
    print(f"Parameters shape: {theta_check.shape}")
    print(f"Statistics shape: {x_check.shape}")
    print(f"Loaded {theta_check.shape[0]} samples.")
    print("--------------------------------")
except FileNotFoundError as e:
    print(f"Initialization Error: {e}")


# =============================================================================
# 1. SETUP
# =============================================================================
def sample_ensemble_direct(posterior, x_obs_norm, n_samples=5000):
    """
    Sample from each member of an EnsemblePosterior individually and
    concatenate, avoiding the rejection-sampling hang on the full ensemble.
    """
    x_tensor = (torch.from_numpy(np.array(x_obs_norm))
                .float()
                .reshape(1, -1)
                .to('cpu'))
    members = getattr(posterior, 'posteriors', [posterior])
    n_mem   = len(members)
    base    = n_samples // n_mem
    extra   = n_samples - base * n_mem

    all_samples = []
    for i, member in enumerate(members):
        n_draw = base + (extra if i == 0 else 0)
        try:
            s = member.sample((n_draw,), x=x_tensor)
            all_samples.append(s.detach().cpu().numpy())
        except Exception as e:
            print(f'  [WARN] Member {i} sampling failed: {e}')

    if not all_samples:
        return None
    return np.vstack(all_samples)

def apply_axis_limits_to_figure(fig_or_grid, param_names, param_limits):
    """
    Works for both a matplotlib Figure and a seaborn PairGrid,
    which is what PlotSinglePosterior actually returns.
    """
    # PairGrid stores axes as a 2D numpy array in .axes
    # matplotlib Figure requires .get_axes() and reshaping
    if hasattr(fig_or_grid, 'axes') and isinstance(fig_or_grid.axes, np.ndarray):
        axes = fig_or_grid.axes   # already (n_params, n_params)
    else:
        axes = np.array(fig_or_grid.get_axes()).reshape(
            len(param_names), len(param_names))

    for col, pname_col in enumerate(param_names):
        lo_col, hi_col = param_limits[pname_col]
        for row in range(col, len(param_names)):
            axes[row, col].set_xlim(lo_col, hi_col)
            if row != col:
                pname_row = param_names[row]
                lo_row, hi_row = param_limits[pname_row]
                axes[row, col].set_ylim(lo_row, hi_row)

# PlotSinglePosterior uses $...$ format; GetDist uses raw strings without $
labels_psp  = [r'$\theta_{ej,0}$', r'${\nu_{\theta_{ej}}}^{M}$']
labels_gd   = [r'\theta_{ej,0}',   r'{\nu_{\theta_{ej}}}^{M}']
param_names  = ['p1', 'p2']
truth_values = np.array([2.0, -0.1])

comparison_colors = ['#1f77b4', '#d62728', '#2ca02c']

stat_map_keys = [
    'g2y', 'g2tau', 'g2kappa',
    'gy',  'gtau',  'gkappa',
    'JOINT', 'y_total', 'tau_total', 'kappa_total',
    'all_3pt', 'all_2pt',
]

tracer_groups = [
    {
        'title': 'y_correlations',
        'pairs': [('g2y', '3-point'), ('gy', '2-point'), ('y_total', 'all_y')],
        'legend_labels': [r'$g^2 y$', r'$gy$', r'$g^2 y + gy$'],
    },
    {
        'title': 'tau_correlations',
        'pairs': [('g2tau', '3-point'), ('gtau', '2-point'), ('tau_total', 'all_tau')],
        'legend_labels': [r'$g^2 \tau$', r'$g\tau$', r'$g^2 \tau + g\tau$'],
    },
    {
        'title': 'kappa_correlations',
        'pairs': [('g2kappa', '3-point'), ('gkappa', '2-point'), ('kappa_total', 'all_kappa')],
        'legend_labels': [r'$g^2 \kappa$', r'$g\kappa$', r'$g^2 \kappa + g\kappa$'],
    },
]

os.makedirs('plotcontours_Cls', exist_ok=True)

# List indexed parallel to param_names ?~@~T avoids the p1/p2 key mismatch
PARAM_LIMITS = [
    (1.0,  6.0),   # p1 : theta_ej_0
    (-0.3, 0.0),   # p2 : nu_theta_ej_M
    ]
# =============================================================================
# 2. INDIVIDUAL POSTERIOR PLOTS ?~@~T PlotSinglePosterior
# =============================================================================
def apply_axis_limits_to_figure(fig, param_names, param_limits):
    """
    For a 2-parameter triangle plot the axes are arranged as:
      ax[0,0] : 1D marginal of param 0  (x = param 0)
      ax[1,0] : 2D joint               (x = param 0, y = param 1)
      ax[1,1] : 1D marginal of param 1  (x = param 1)
    """
    axes = np.array(fig.get_axes()).reshape(len(param_names), len(param_names))
    for col, pname_col in enumerate(param_names):
        lo_col, hi_col = param_limits[pname_col]
        for row in range(col, len(param_names)):
            axes[row, col].set_xlim(lo_col, hi_col)
            if row != col:                          # 2D panel: also fix y-axis
                pname_row = param_names[row]
                lo_row, hi_row = param_limits[pname_row]
                axes[row, col].set_ylim(lo_row, hi_row)

for name in stat_map_keys:
    pkl_path  = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'

    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f'[SKIP] Missing files for {name}')
        continue

    posterior  = load_posterior(pkl_path)
    x_obs_norm = np.load(xobs_path)
    out_dir    = Path(f'plotcontours_Cls/{name}')
    out_dir.mkdir(exist_ok=True)

    metric = PlotSinglePosterior(
        num_samples=5000,
        sample_method='direct',
        labels=labels_psp,
        out_dir=out_dir,
    )

    try:
        grid = metric(
            posterior=posterior,
            x_obs=x_obs_norm,
            theta_fid=truth_values,
            plot_kws=dict(fill=True),
        )
        apply_axis_limits_to_figure(grid, param_names, PARAM_LIMITS)
        # PairGrid.fig is the underlying matplotlib Figure
        out_path = out_dir / 'posterior.png'
        grid.fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'[DONE] {name} --> {out_path}')
    except Exception as e:
        print(f'[WARN] PlotSinglePosterior failed for {name}: {e}')

    plt.close('all')


# =============================================================================
# 3. GROUP COMPARISON PLOTS ?~@~T GetDist
# =============================================================================

def get_mcsamples(name, legend_label, n_samples=5000):
    pkl_path  = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'

    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f'[SKIP] Missing files for {name}')
        return None

    posterior  = load_posterior(pkl_path)
    x_obs_norm = np.load(xobs_path)
    samples    = sample_ensemble_direct(posterior, x_obs_norm, n_samples=n_samples)

    if samples is None:
        print(f'[WARN] Sampling failed for {name}')
        return None

    return MCSamples(
        samples=samples,
        names=param_names,
        labels=labels_gd,
        label=legend_label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )

for group in tracer_groups:
    print(f"\nGenerating comparison plot for {group['title']}...")
    mcsamples_list = []

    for tracer_id, legend_tag in group['pairs']:
        smp = get_mcsamples(tracer_id, f'{tracer_id} ({legend_tag})')
        if smp is not None:
            mcsamples_list.append(smp)

    if len(mcsamples_list) < 2:
        print(f'[SKIP] Fewer than 2 valid posteriors for {group["title"]}')
        continue

    n = len(mcsamples_list)
    g = plots.get_subplot_plotter(width_inch=8, settings=plot_settings)
    g.triangle_plot(
        mcsamples_list,
        filled=True,
        colors=comparison_colors[:n],
        legend_labels=group['legend_labels'][:n],
        contour_args=[{'alpha': 0.6}] * n,
        line_args=[{'lw': 2.5, 'color': c} for c in comparison_colors[:n]],
    )
    # Apply axis limits and truth lines together after rendering
    for i in range(len(param_names)):
        lo_i, hi_i = PARAM_LIMITS[i]

        g.subplots[i, i].set_xlim(lo_i, hi_i)
        g.subplots[i, i].axvline(
            truth_values[i], color='black', ls='--',lw=2, zorder=10)

        for j in range(i):
            lo_j, hi_j = PARAM_LIMITS[j]

            g.subplots[i, j].set_xlim(lo_j, hi_j)
            g.subplots[i, j].set_ylim(lo_i, hi_i)
            g.subplots[i, j].axvline(
                truth_values[j], color='black', ls='--', lw=2, zorder=10)
            g.subplots[i, j].axhline(
                truth_values[i], color='black', ls='--', lw=2, zorder=10)
    output_fn = f'plotcontours_Cls/contour_comparison_{group["title"]}.png'
    g.export(output_fn)
    print(f'[DONE] Saved: {output_fn}')

plt.close('all')
os._exit(0)
