import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import pickle as pk
import torch
import torch.nn as nn
import io
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import logging
from pathlib import Path

logging.getLogger('getdist').setLevel(logging.ERROR)

# =============================================================================
# CPU MONKEY-PATCH
# =============================================================================
torch.cuda.is_available = lambda: False

_orig_module_to = nn.Module.to
def _patched_module_to(self, *args, **kwargs):
    args = tuple('cpu' if isinstance(a, str) and 'cuda' in a else a for a in args)
    if 'device' in kwargs and 'cuda' in str(kwargs.get('device', '')):
        kwargs['device'] = 'cpu'
    return _orig_module_to(self, *args, **kwargs)
nn.Module.to   = _patched_module_to
nn.Module.cuda = lambda self, *args, **kwargs: self

_orig_tensor_to = torch.Tensor.to
def _patched_tensor_to(self, *args, **kwargs):
    args = tuple('cpu' if isinstance(a, str) and 'cuda' in a else a for a in args)
    if 'device' in kwargs and 'cuda' in str(kwargs.get('device', '')):
        kwargs['device'] = 'cpu'
    return _orig_tensor_to(self, *args, **kwargs)
torch.Tensor.to   = _patched_tensor_to
torch.Tensor.cuda = lambda self, *args, **kwargs: self

# =============================================================================
# PLOT SETTINGS
# =============================================================================
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
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(
                io.BytesIO(b), map_location='cpu', weights_only=False)
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
# SAMPLING
# =============================================================================

def sample_ensemble_direct(posterior, x_obs_norm, n_samples=5000):
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
            s = member.sample((n_draw,), x=x_tensor)
            all_samples.append(s.detach().cpu().numpy())
        except Exception as e:
            print(f'  [WARN] Member {i} sampling failed: {e}')

    if not all_samples:
        return None
    return np.vstack(all_samples)


# =============================================================================
# SETUP
# =============================================================================
labels_psp   = [r'$\theta_{ej,0}$', r'${\nu_{\theta_{ej}}}^{M}$']
labels_gd    = [r'\theta_{ej,0}',   r'{\nu_{\theta_{ej}}}^{M}']
param_names  = ['p1', 'p2']
truth_values = np.array([2.0, -0.1])
PARAM_LIMITS = [(1.0, 6.0), (-0.3, 0.0)]

# Four statistics to compare
STATS = [
    ('g2kappa',    r'$g^2\kappa$'),
    ('k2g',        r'$\kappa^2 g$'),
    ('gky',        r'$g\kappa y$'),
    ('kappa_break', 'all'),
]

# One distinct colour per statistic
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

os.makedirs('plotcontours_kappa_break', exist_ok=True)


# =============================================================================
# HELPER: axis limits + truth lines
# =============================================================================

def apply_limits_and_truth(g, truth_values, param_limits):
    for i in range(len(param_names)):
        lo_i, hi_i = param_limits[i]
        g.subplots[i, i].set_xlim(lo_i, hi_i)
        g.subplots[i, i].axvline(
            truth_values[i], color='black', ls='--', lw=2, zorder=10)
        for j in range(i):
            lo_j, hi_j = param_limits[j]
            g.subplots[i, j].set_xlim(lo_j, hi_j)
            g.subplots[i, j].set_ylim(lo_i, hi_i)
            g.subplots[i, j].axvline(
                truth_values[j], color='black', ls='--', lw=2, zorder=10)
            g.subplots[i, j].axhline(
                truth_values[i], color='black', ls='--', lw=2, zorder=10)


# =============================================================================
# 1. INDIVIDUAL PlotSinglePosterior for each statistic
# =============================================================================

def apply_axis_limits_to_figure(fig, param_names, param_limits):
    axes = np.array(fig.get_axes()).reshape(len(param_names), len(param_names))
    for col in range(len(param_names)):
        lo_col, hi_col = param_limits[col]
        for row in range(col, len(param_names)):
            axes[row, col].set_xlim(lo_col, hi_col)
            if row != col:
                lo_row, hi_row = param_limits[row]
                axes[row, col].set_ylim(lo_row, hi_row)


for name, legend_label in STATS:
    pkl_path  = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'

    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f'[SKIP] Missing files for {name}')
        continue

    posterior  = load_posterior(pkl_path)
    x_obs_norm = np.load(xobs_path)
    out_dir    = Path(f'plotcontours_kappa_break/{name}')
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
        apply_axis_limits_to_figure(grid.fig, param_names, PARAM_LIMITS)
        out_path = out_dir / 'posterior.png'
        grid.fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'[DONE] {name} --> {out_path}')
    except Exception as e:
        print(f'[WARN] PlotSinglePosterior failed for {name}: {e}')

    plt.close('all')


# =============================================================================
# 2. COMBINED GetDist comparison: all four on one triangle plot
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


print('\nGenerating combined kappa degeneracy-breaking comparison plot...')

mcsamples_list  = []
legend_labels   = []
colors_used     = []

for (name, legend_label), color in zip(STATS, COLORS):
    smp = get_mcsamples(name, legend_label)
    if smp is not None:
        mcsamples_list.append(smp)
        legend_labels.append(legend_label)
        colors_used.append(color)

if len(mcsamples_list) >= 2:
    g = plots.get_subplot_plotter(width_inch=8, settings=plot_settings)
    g.triangle_plot(
        mcsamples_list,
        filled=True,
        colors=colors_used,
        legend_labels=legend_labels,
        contour_args=[{'alpha': 0.6}] * len(mcsamples_list),
        line_args=[{'lw': 2.5, 'color': c} for c in colors_used],
    )
    apply_limits_and_truth(g, truth_values, PARAM_LIMITS)
    out_fn = 'plotcontours_kappa_break/contour_kappa_degeneracy_break.png'
    g.export(out_fn)
    print(f'[DONE] Saved: {out_fn}')
else:
    print('[SKIP] Fewer than 2 valid posteriors, skipping combined plot.')

plt.close('all')
os._exit(0)
