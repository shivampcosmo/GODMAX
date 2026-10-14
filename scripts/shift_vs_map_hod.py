"""Results shift of this branch relative to a reference commit of the old flat-layout code
(default: upstream/map_hod, 14ae4e4), with identical parameters and everything ON
(DMB matter, backreaction, baryonified tSZ).

For each set-up it builds:
  old             the reference code (extracted with `git archive <ref> src` into a temporary directory)
  new             this branch
  revert_all      this branch with the three fixes reverted (old get_rho_clm, old gas normalisation,
                  num_points_halofit = 64); must reproduce `old` exactly (null control)
  revert_B7 / revert_gas / revert_halofit   one fix reverted at a time

and prints max |A/B - 1| for 10 spectra (P(k) at k <= 10 h/Mpc, all z; C(ell) at all ell).

Set-ups:
  default   param_files/params_default.yaml with the test grid of tests/physics
  pge       tSZxEuclid params_default.yaml + Pge/params.yaml (nu_theta_ej_M = -0.33), same n(z)
            and ell range as `default`; needs --tszx (skipped if not found)

Run (from the repository root):
    python scripts/shift_vs_map_hod.py
    python scripts/shift_vs_map_hod.py --ref upstream/map_hod --tszx ../tSZxEuclid/params_godmax --out shift_out
"""
import argparse
import contextlib
import io
import pathlib
import subprocess
import sys
import tempfile
import warnings
from functools import partial

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
ON = dict(model_matter="DMB", backreaction=True, baryonification_tSZ=True)

SPECTRA = ["Pmm_tot_mat", "Pgm_tot_mat", "Pgg_tot_mat", "Pyy_tot_kz_mat",
           "Cl_kappa_kappa_tot_mat", "Cl_gal_gal_tot_mat", "Cl_gal_kappa_tot_mat",
           "Cl_kappa_y_tot_mat", "Cl_gal_y_tot_mat", "Cl_y_y"]
SHORT = ["P_mm", "P_gm", "P_gg", "P_yy", "C_kk", "C_gg", "C_gk", "C_ky", "C_gy", "C_yy"]


def dicts(setup, tszx):
    sys.path.insert(0, str(REPO / "tests/physics"))
    sys.path.insert(0, str(REPO / "scripts"))
    import conftest as TC
    import preflight as PF
    if setup == "default":
        s, h, a, o = TC.base_dicts()
    else:
        s, h, a, o = PF.load_params(tszx / "params_default.yaml", tszx / "Pge/params.yaml")
        _, _, a1, _ = TC.base_dicts()
        for k in ("nz_source_info_dict", "nz_lens_info_dict", "nbar_lens_bins", "is_cmb_lensing"):
            a[k] = a1[k]
        h = dict(h, ellmin=10.0, ellmax=3000.0, nell=12, nz=6, nM=16)
    a.update(ON)
    return s, h, a, o


def collect(obj, yy_name):
    out = {name: np.asarray(getattr(obj, yy_name if name == "Cl_y_y" else name)) for name in SPECTRA}
    out["kPk_array"] = np.asarray(obj.kPk_array)
    return out


def build_old(setup, tszx, out, old_src):
    """Reference code. On map_hod the yy spectra live in get_cov: run only its yy block."""
    sys.path.insert(0, str(old_src))
    import jax
    jax.config.update("jax_enable_x64", True)
    import interpax
    import jax.numpy as jnp
    from jax import vmap
    from base_class import get_vmapped_func, get_vmapped_func_warg
    from get_Cls import get_Cl
    from get_covs import get_cov
    s, h, a, o = dicts(setup, tszx)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        obj = get_Cl(s, h, a, o)
    obj.__class__ = get_cov
    P1 = get_vmapped_func_warg(obj.get_P_1h, 2, 4)(jnp.arange(obj.nk), jnp.arange(obj.nz), 3, 3).T
    obj.Pyy_tot_kz_mat = P1 + obj.by_kz_mat * obj.by_kz_mat * obj.plin_kz_mat
    obj.Pkyy_lz_mat = get_vmapped_func(obj.get_Pkyy_lz, 2)(jnp.arange(obj.nell), jnp.arange(obj.nz)).T
    obj.logPkyylz_2d_interp = interpax.Interpolator2D(jnp.log(obj.ell_array), obj.z_array, jnp.log(obj.Pkyy_lz_mat), extrap=True)
    obj.Cl_y_y_tot_mat = vmap(obj.get_Cl_y_y_tot)(jnp.arange(obj.nell))
    np.savez(out / f"{setup}_old.npz", **collect(obj, "Cl_y_y_tot_mat"))


def build_new(setup, tszx, out):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import jit
    import godmax.get_radial_profiles as grp
    from godmax import get_Cl

    @partial(jit, static_argnums=(0,))
    def old_rho_clm(self, jz, jM, r_array_here=None):
        """get_rho_clm as on map_hod (14ae4e4): jax.grad of a linear interp at its own knots."""
        if r_array_here is None:
            r_array_here = self.r_array
            Mclm_here = self.Mclm_mat[:, jz, jM]
        else:
            Mclm_here = jnp.exp(jnp.interp(jnp.log(r_array_here), jnp.log(self.r_array), jnp.log(self.Mclm_mat[:, jz, jM])))
        dlnMclm_dr = jax.vmap(jax.grad(lambda x_val: jnp.interp(x_val, jnp.log(r_array_here), jnp.log(jnp.clip(Mclm_here, 0, None) + 1e-30))))(jnp.log(r_array_here))
        dMclm_dr = dlnMclm_dr * Mclm_here / r_array_here
        rho_clm = dMclm_dr / (4 * jnp.pi * r_array_here**2)
        return jnp.clip(rho_clm, 0, 1e30)

    @partial(jit, static_argnums=(0,))
    def old_gas_norm(self, jz, jM, rmax_r200c=grp.RMAX_R200c):
        """get_rho_gas_norm as on map_hod (14ae4e4): normalised within rmax_r200c * r200c."""
        r200c = self.r200c_mat[jz, jM]
        logx = jnp.linspace(jnp.log(0.01 * r200c), jnp.log(rmax_r200c * r200c), self.num_points_trapz_int)
        integral = self.logspace_trapezoidal_integral(self.get_rho_gas_unnorm, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        return self.fgas_mat[jz, jM] * self.Mtot_mat[jz, jM] / integral

    new_rho, new_gas = grp.Profiles.get_rho_clm, grp.Profiles.get_rho_gas_norm
    variants = {  # name: (old get_rho_clm, old gas normalisation, num_points_halofit)
        "new": (False, False, 512),
        "revert_all": (True, True, 64),
        "revert_B7": (True, False, 512),
        "revert_gas": (False, True, 512),
        "revert_halofit": (False, False, 64),
    }
    for name, (orho, ogas, nhf) in variants.items():
        s, h, a, o = dicts(setup, tszx)
        a["num_points_halofit"] = nhf
        grp.Profiles.get_rho_clm = old_rho_clm if orho else new_rho
        grp.Profiles.get_rho_gas_norm = old_gas_norm if ogas else new_gas
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                obj = get_Cl(s, h, a, o)
        finally:
            grp.Profiles.get_rho_clm, grp.Profiles.get_rho_gas_norm = new_rho, new_gas
        np.savez(out / f"{setup}_{name}.npz", **collect(obj, "Cl_y_y_signal_mat"))
        print(f"  {setup}: {name} built", flush=True)


def maxdev(a, b, name, k):
    a, b = a[name], b[name]
    if name.startswith("P"):
        sel = k <= 10.0
        a, b = a[sel], b[sel]
    return np.max(np.abs(a / b - 1.0))


def compare(setups, out, ref):
    ok = True
    for setup in setups:
        R = {p.stem.split("_", 1)[1]: dict(np.load(p)) for p in sorted(out.glob(f"{setup}_*.npz"))}
        k = R["old"]["kPk_array"]
        print(f"\n=== {setup}: max |A/B - 1|   (P: k <= 10 h/Mpc, all z; C: all ell)")
        print(f"{'A vs B':44s}" + "".join(f"{s:>9s}" for s in SHORT))
        rows = [(f"new vs {ref} (total shift)", "new", "old"),
                (f"revert_all vs {ref} (null control)", "revert_all", "old"),
                ("B7 alone: new vs revert_B7", "new", "revert_B7"),
                ("gas alone: new vs revert_gas", "new", "revert_gas"),
                ("halofit alone: new vs revert_halofit", "new", "revert_halofit")]
        for label, A, B in rows:
            print(f"{label:44s}" + "".join(f"{maxdev(R[A], R[B], n, k):9.2e}" for n in SPECTRA))
        null = max(maxdev(R["revert_all"], R["old"], n, k) for n in SPECTRA)
        lowk = np.max(np.abs(R["new"]["Pmm_tot_mat"][k <= 0.01] / R["old"]["Pmm_tot_mat"][k <= 0.01] - 1.0))
        print(f"null control (must be 0): {null:.1e};  P_mm at k <= 0.01: {lowk:.1e}")
        ok &= null == 0.0
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="upstream/map_hod", help="reference commit with the flat src/ layout")
    ap.add_argument("--tszx", type=pathlib.Path, default=REPO.parent / "tSZxEuclid" / "params_godmax",
                    help="tSZxEuclid params_godmax directory (for the pge set-up)")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="directory for the .npz outputs (default: temporary)")
    ap.add_argument("--_worker", nargs=3, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:  # child process: one code version per process, so the two layouts never mix
        kind, setup, old_src = args._worker
        out = args.out
        (build_old(setup, args.tszx, out, pathlib.Path(old_src)) if kind == "old" else build_new(setup, args.tszx, out))
        return

    setups = ["default"] + (["pge"] if (args.tszx / "Pge/params.yaml").exists() else [])
    if "pge" not in setups:
        print(f"pge set-up skipped: {args.tszx / 'Pge/params.yaml'} not found")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        out = args.out or tmp / "out"
        out.mkdir(parents=True, exist_ok=True)
        old_src = tmp / "old"
        old_src.mkdir()
        archive = subprocess.run(["git", "-C", str(REPO), "archive", args.ref, "src"], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(old_src)], input=archive, check=True)
        for setup in setups:
            for kind in ("old", "new"):
                print(f"building {kind} code, {setup} set-up", flush=True)
                subprocess.run([sys.executable, "-W", "ignore", __file__, "--tszx", str(args.tszx), "--out", str(out),
                                "--_worker", kind, setup, str(old_src / "src")], check=True)
        ok = compare(setups, out, args.ref)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
