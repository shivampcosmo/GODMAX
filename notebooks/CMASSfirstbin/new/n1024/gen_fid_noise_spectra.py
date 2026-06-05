"""
gen_fid_noise_spectra.py
========================
Run ONCE to extract instrumental/reconstruction noise power spectra
from the fiducial theory product and save them for use in extract_Cls.

Noise saved (on the theory ell grid):
    nl_yy     -> N_ell^{yy}         instrumental noise for y map
    nl_tautau -> N_ell^{tautau}     instrumental noise for tau map
    nl_kk     -> N_ell^{kappakappa} CMB lensing reconstruction noise

Shot noise for gg is NOT stored here. It is computed per-simulation
from the actual galaxy counts in each pkl file inside extract_Cls.

Usage
-----
    python gen_fid_noise_spectra.py
    python gen_fid_noise_spectra.py --force
    python gen_fid_noise_spectra.py --theory-path /path/to/fiducial.npz

Load in SBI pipeline:
    from gen_fid_noise_spectra import load_noise_pkg
    noise_pkg = load_noise_pkg()
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

THIS_DIR   = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "outputs"
NOISE_PATH = OUTPUT_DIR / "sbi_noise_spectra.npz"

DEFAULT_THEORY_PATH = OUTPUT_DIR / "fiducial_theory_datavector.npz"


def build_noise_pkg(
    theory_path: pathlib.Path | str = DEFAULT_THEORY_PATH,
    output_path: pathlib.Path | str = NOISE_PATH,
    force: bool = False,
) -> dict:
    """
    Extract noise spectra from the fiducial theory product and save.

    The theory product stores noise under field-name keys:
        noise["y"]     -> N_ell^{yy}
        noise["tau"]   -> N_ell^{tautau}
        noise["kappa"] -> N_ell^{kappakappa}

    Returns the noise package dict (same format as load_noise_pkg).
    """
    output_path = pathlib.Path(output_path)
    if output_path.exists() and not force:
        print(f"[gen_noise] Found existing: {output_path}  (use --force to recompute)")
        return load_noise_pkg(output_path)

    theory_path = pathlib.Path(theory_path)
    if not theory_path.exists():
        raise FileNotFoundError(
            f"Fiducial theory product not found: {theory_path}\n"
            "Run fiducial_theory_datavector.py first."
        )

    # ── Load theory product ───────────────────────────────────────────────────
    # Use the loader from fiducial_theory_datavector if available,
    # otherwise load the npz directly.
    try:
        _ftd_dir = str(theory_path.parent)
        if _ftd_dir not in sys.path:
            sys.path.insert(0, _ftd_dir)
        from fiducial_theory_datavector import load_validation_product
        theory = load_validation_product(theory_path)
        noise_dict = theory["noise"]   # keys: "g", "y", "tau", "kappa"
        ell       = np.asarray(theory["ell"],       dtype=float)
        delta_ell = np.asarray(theory["delta_ell"], dtype=float)
        meta      = theory["metadata"]
    except ImportError:
        data      = np.load(theory_path, allow_pickle=True)
        ell       = np.asarray(data["ell"],       dtype=float)
        delta_ell = np.asarray(data["delta_ell"], dtype=float)
        noise_dict = {
            key[6:]: data[key]
            for key in data.files
            if key.startswith("noise_")
        }
        meta = {}

    # ── Extract the three instrumental / reconstruction noise spectra ─────────
    def _get(field: str) -> np.ndarray:
        """Return N_ell for a field, zero-padded if missing."""
        for key in (field, f"noise_{field}"):
            if key in noise_dict:
                arr = np.asarray(noise_dict[key], dtype=float)
                if arr.shape == ell.shape:
                    return arr
        print(f"  [WARN] noise key '{field}' not found in theory product — using zeros")
        return np.zeros_like(ell)

    nl_yy     = _get("y")
    nl_tautau = _get("tau")
    nl_kk     = _get("kappa")

    # ── fsky from covariance metadata ─────────────────────────────────────────
    fsky = float(
        meta.get("covariance", {}).get("fsky",
        meta.get("fsky", 0.4))
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ell       = ell,
        delta_ell = delta_ell,
        fsky      = np.float64(fsky),
        nl_yy     = nl_yy,
        nl_tautau = nl_tautau,
        nl_kk     = nl_kk,
    )
    print(f"[gen_noise] Saved -> {output_path}")
    print(f"  ell range  : {ell[0]:.1f} – {ell[-1]:.1f}  ({len(ell)} bins)")
    print(f"  fsky       : {fsky:.3f}")
    print(f"  nl_yy      : [{nl_yy.min():.3e}, {nl_yy.max():.3e}]")
    print(f"  nl_tautau  : [{nl_tautau.min():.3e}, {nl_tautau.max():.3e}]")
    print(f"  nl_kk      : [{nl_kk.min():.3e}, {nl_kk.max():.3e}]")

    return dict(ell=ell, delta_ell=delta_ell, fsky=fsky,
                nl_yy=nl_yy, nl_tautau=nl_tautau, nl_kk=nl_kk)


def load_noise_pkg(path: pathlib.Path | str = NOISE_PATH) -> dict:
    """
    Load the noise package saved by build_noise_pkg.

    Returns
    -------
    dict with keys:
        ell       : (n_ell,)  theory multipole bin centres
        delta_ell : (n_ell,)  theory bin widths
        fsky      : float
        nl_yy     : (n_ell,)  N_ell^{yy}
        nl_tautau : (n_ell,)  N_ell^{tautau}
        nl_kk     : (n_ell,)  N_ell^{kappakappa}
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Noise package not found: {path}\n"
            "Run:  python gen_fid_noise_spectra.py"
        )
    data = np.load(path)
    return dict(
        ell       = data["ell"].astype(float),
        delta_ell = data["delta_ell"].astype(float),
        fsky      = float(data["fsky"]),
        nl_yy     = data["nl_yy"].astype(float),
        nl_tautau = data["nl_tautau"].astype(float),
        nl_kk     = data["nl_kk"].astype(float),
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theory-path", default=str(DEFAULT_THEORY_PATH),
        help="Path to fiducial_theory_datavector.npz",
    )
    parser.add_argument(
        "--output-path", default=str(NOISE_PATH),
        help="Where to save sbi_noise_spectra.npz",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if output already exists",
    )
    args = parser.parse_args()
    build_noise_pkg(
        theory_path = args.theory_path,
        output_path = args.output_path,
        force       = args.force,
    )


if __name__ == "__main__":
    main()
