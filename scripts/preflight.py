#!/usr/bin/env python
"""Pre-flight consistency checks for a GODMAX configuration, before a fit.

Builds the model at the fiducial point of a parameter file and, if a priors file is given, with
each varied parameter set in turn to the lower and upper edge of its prior (all others at
fiducial). At every point it checks, read-only (godmax.validation):

  shells     CLM enclosed mass is monotonic (no shell more negative than 1e-4 of M(r_max))
  closure    rho_clm on r_array recovers the CLM mass to 1%
  coverage   CLM, gas and total (DMB) mass on the radial grid within 2% of their targets
  zeta       relaxation root inside its trial bracket and accurate to 1e-3
  u(k->0)    raw FFTLog u at the lowest k within 1% of 1 (DMB, NFW, n_e); shell u(0) = 1
  finite     no NaN/inf in profiles, u(k) and P(k)
  onoff      (with --onoff) baryonification ON/OFF P_mm, P_gm, P_gg within 1% at k <= 0.01

Exit status 0 if every check passes at every point, 1 otherwise. Failures are reported with
the worst value and where it occurs. Fix a failure by changing the grid or the prior, never
by clipping inside the likelihood.

Examples
--------
  python scripts/preflight.py param_files/params_default.yaml
  python scripts/preflight.py ../tSZxEuclid/params_godmax/params_default.yaml \\
         --override ../tSZxEuclid/params_godmax/Pge/params.yaml \\
         --priors ../tSZxEuclid/params_godmax/Pge/priors.yaml --onoff
"""
import argparse
import contextlib
import copy
import io
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import yaml

TOL = dict(shell_neg=1e-4, closure=0.01, coverage=0.02, zeta=1e-3, u0=0.01, onoff=0.01)
ON = dict(model_matter="DMB", backreaction=True, baryonification_tSZ=True)
OFF = dict(model_matter="halofit", backreaction=False, baryonification_tSZ=False)


# ----------------------------------------------------------------------------- parameters
def numeric(x):
    """YAML 1.1 reads bare '5e-4' as a string: convert numeric-looking strings."""
    if isinstance(x, dict):
        return {k: numeric(v) for k, v in x.items()}
    if isinstance(x, list):
        return [numeric(v) for v in x]
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x


def merge(base, over):
    """Recursive merge; None in the override keeps the base value (commented-out YAML sections)."""
    if not isinstance(base, dict) or not isinstance(over, dict):
        return base if over is None else over
    out = dict(base)
    for k, v in over.items():
        if v is None:
            continue
        out[k] = merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_params(path, override=None):
    p = yaml.safe_load(open(path)) or {}
    if override:
        p = merge(p, yaml.safe_load(open(override)) or {})
    p = numeric(p)
    sim, halo, ana, other = (copy.deepcopy(p.get(k) or {}) for k in ("sim_params", "halo_params", "analysis", "other_params"))
    nb = ana.get("nbar_gal_comoving_val", 5e-4)
    if isinstance(nb, list) and len(nb) == 1:
        ana["nbar_gal_comoving_val"] = float(nb[0])
    for key in ("yy_noise_ell_fname", "yy_total_ell_fname"):
        ana.pop(key, None)
    return sim, halo, ana, other


def load_priors(path):
    """{name: (lo, hi)} from 'prior_uniform: {name: "lo hi"}' (tSZxEuclid format) or {name: [lo, hi]}."""
    d = yaml.safe_load(open(path)) or {}
    d = d.get("prior_uniform", d)
    out = {}
    for k, v in d.items():
        vals = [float(x) for x in (v.split() if isinstance(v, str) else v)]
        if len(vals) == 2:
            out[k] = tuple(vals)
    return out


def set_param(sim, name, value):
    """Set a sampled parameter in sim_params (cosmology sub-dict, scalar, or per-bin list)."""
    if name in sim.get("cosmo", {}):
        sim["cosmo"][name] = value
        return True
    for key in (name, name + "_array"):
        if key in sim:
            sim[key] = [value] * len(sim[key]) if isinstance(sim[key], list) else value
            return True
    return False


# ----------------------------------------------------------------------------- checks
def build(root, sim, halo, ana, other, switches):
    sys.path.insert(0, str(root / "src"))
    from godmax.get_Pkzs import get_Pkz
    a = dict(copy.deepcopy(ana), **switches)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return get_Pkz(copy.deepcopy(sim), copy.deepcopy(halo), a, copy.deepcopy(other))


def worst_location(arr, dev, obj):
    i = np.unravel_index(int(np.nanargmax(dev)), dev.shape)
    z, M = np.asarray(obj.z_array), np.asarray(obj.M_array)
    return f"{float(arr[i]):.4g} at z={z[i[-2]]:.2f}, log10M={np.log10(M[i[-1]]):.2f}"


def run_checks(obj, off=None):
    from godmax import validation as V
    res = {}

    rep = V.shell_report(obj)
    res["shells"] = (rep["min_shell_over_Mend"] > -TOL["shell_neg"],
                     f"{rep['n_negative_shells']} negative, min {rep['min_shell_over_Mend']:.2e} of M(r_max)")

    c = V.rho_clm_closure(obj)
    d = np.abs(c - 1.0)
    res["closure"] = (d.max() < TOL["closure"], f"worst {worst_location(c, d, obj)}")

    cov = V.coverage_report(obj)
    msgs, ok = [], True
    for name, arr in cov.items():
        d = np.abs(arr - 1.0)
        if d.max() >= TOL["coverage"]:
            ok = False
            msgs.append(f"{name} {worst_location(arr, d, obj)}")
    res["coverage"] = (ok, "; ".join(msgs) or "all within %.0f%%" % (100 * TOL["coverage"]))

    zr = V.zeta_report(obj)
    if zr is not None:
        res["zeta"] = (zr["n_at_bracket_edge"] == 0 and zr["zeta_error_abs_max"] < TOL["zeta"],
                       f"error {zr['zeta_error_abs_max']:.2e}, {zr['n_at_bracket_edge']} at bracket edge, range {zr['zeta_range'][0]:.3f}-{zr['zeta_range'][1]:.3f}")

    u = V.u_k0_report(obj)
    msgs, ok = [], True
    for name in ("uk_dmb_tointp", "uk_nfw_tointp", "uk_ne_tointp"):
        if name in u:
            s = u[name]
            if max(abs(s["min"] - 1), abs(s["max"] - 1)) >= TOL["u0"]:
                ok = False
                msgs.append(f"{name} {s['min']:.3f}-{s['max']:.3f}")
    if "shell_u_at_k0_max_abs_err" in u and u["shell_u_at_k0_max_abs_err"] > 1e-12:
        ok = False
        msgs.append(f"shell u(0) err {u['shell_u_at_k0_max_abs_err']:.1e}")
    res["u(k->0)"] = (ok, "; ".join(msgs) or "all within %.0f%%" % (100 * TOL["u0"]))

    bad = V.finite_report(obj, ["rho_dmb_mat", "rho_gas_mat", "rho_clm_mat", "Mdmb_mat", "Pe_mat_physical", "y3d_mat",
                                "uk_dmb", "uk_nfw", "uk_clm", "uk_y", "Pmm_tot_mat", "Pgg_tot_mat", "Pgm_tot_mat",
                                "Pyy_tot_kz_mat"])
    res["finite"] = (not bad, ", ".join(bad) or "ok")

    if off is not None:
        msgs, ok = [], True
        for name in ("Pmm_tot_mat", "Pgm_tot_mat", "Pgg_tot_mat"):
            if hasattr(obj, name):
                d = V.max_lowk_deviation(obj, off, name)
                ok &= d < TOL["onoff"]
                msgs.append(f"{name[:3]} {d:.2e}")
        res["onoff"] = (ok, ", ".join(msgs))
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("params", help="parameter YAML (e.g. params_default.yaml)")
    ap.add_argument("--override", help="YAML merged on top of params")
    ap.add_argument("--priors", help="priors YAML; each parameter is checked at both prior edges")
    ap.add_argument("--only", nargs="*", help="restrict prior-edge checks to these parameters")
    ap.add_argument("--onoff", action="store_true", help="also check the large-scale baryonification ON/OFF ratio")
    ap.add_argument("--godmax-root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    ap.add_argument("--json", help="write all results to this file")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    root = pathlib.Path(args.godmax_root)
    sim0, halo0, ana0, other0 = load_params(args.params, args.override)

    points = [("fiducial", sim0)]
    skipped = []
    if args.priors:
        for name, (lo, hi) in load_priors(args.priors).items():
            if args.only and name not in args.only:
                continue
            for tag, val in (("lo", lo), ("hi", hi)):
                s = copy.deepcopy(sim0)
                if set_param(s, name, val):
                    points.append((f"{name}={val:g} ({tag})", s))
                elif tag == "lo":
                    skipped.append(name)
    if skipped:
        print("not in sim_params, skipped:", ", ".join(skipped))

    all_ok, out = True, {}
    print(f"godmax: {root}   points: {len(points)}   grid: nr={halo0.get('nr')} rmax={halo0.get('rmax')} "
          f"nM={halo0.get('nM')} M=10^{halo0.get('lg10_Mmin')}-10^{halo0.get('lg10_Mmax')}")
    for label, sim in points:
        t0 = time.time()
        try:
            on = build(root, sim, halo0, ana0, other0, ON)
            off = build(root, sim, halo0, ana0, other0, OFF) if args.onoff else None
            res = run_checks(on, off)
        except Exception as exc:                       # a crash is a failure, not a skip
            res = {"build": (False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}")}
        ok = all(v[0] for v in res.values())
        all_ok &= ok
        out[label] = {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in res.items()}
        print(f"\n[{'PASS' if ok else 'FAIL'}] {label}  ({time.time() - t0:.1f} s)")
        for k, (passed, detail) in res.items():
            print(f"    {'ok  ' if passed else 'FAIL'} {k:9s} {detail}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} ({len(points)} points)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
