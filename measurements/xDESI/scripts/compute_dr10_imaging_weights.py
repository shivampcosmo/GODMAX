#!/usr/bin/env python3
"""Approximate DR10 imaging weights for the transferred DESI Extended LRGs.

This script replaces the older DR9 example that read row-matched
``dr9_*_more_2.fits`` tables.  The transferred DR10 velocity catalog does not
carry object-level imaging quantities, so this script builds HEALPix template
maps from the official Legacy Survey DR10 imaging randoms, samples those maps
at the galaxy positions, and applies the precomputed Extended-LRG DR9 linear
coefficients.

Important scientific caveat
---------------------------
The output is a useful first-pass systematic-weight estimate, not a refit of
the DR10 selection function.  For a production DR10 galaxy auto spectrum, the
preferred approach is to fit DR10 galaxy/random density versus DR10 imaging
templates in each tomographic bin, then validate that the weighted field has
small residual coupling to depth, seeing, and extinction templates.

Typical use from the repository root:

    /global/homes/s/spandey/.conda/envs/myenv_conda/bin/python \
        DESI/compute_imaging_weights.py --nside 1024 --overwrite

The same script is also copied into the transfer package under ``scripts/``;
there it uses package-relative default paths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np


TEMPLATE_COLUMNS = (
    "EBV",
    "GALDEPTH_G",
    "GALDEPTH_R",
    "GALDEPTH_Z",
    "PSFSIZE_G",
    "PSFSIZE_R",
    "PSFSIZE_Z",
)

DEPTH_COLUMNS = ("GALDEPTH_G", "GALDEPTH_R", "GALDEPTH_Z")
NOBS_COLUMNS = ("NOBS_G", "NOBS_R", "NOBS_Z")
EXTINCTION_COEFF = {"G": 3.214, "R": 2.165, "Z": 1.211}


def find_transfer_root() -> Path:
    """Return the ACT/DESI transfer package root if it can be found."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "act_desi_ksz_transfer",
        here.parent.parent,
        here.parent,
        Path.cwd(),
        Path.cwd() / "DESI" / "act_desi_ksz_transfer",
        Path.cwd() / "act_desi_ksz_transfer",
    ]
    for candidate in candidates:
        if (candidate / "manifest.json").exists() and (candidate / "data").exists():
            return candidate
    return here.parent / "act_desi_ksz_transfer"


def default_coeff_path(root: Path) -> Path:
    return (
        root
        / "data"
        / "desi_dr10_imaging_weights"
        / "dr9_coefficients"
        / "extended_lrg_linear_coeffs_pz.yaml"
    )


def parse_args() -> argparse.Namespace:
    root = find_transfer_root()
    parser = argparse.ArgumentParser(
        description=(
            "Build DR10 random-derived imaging-template maps and apply the "
            "old Extended-LRG DR9 linear imaging-weight coefficients to the "
            "transferred DESI DR10 compact catalog."
        )
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=root,
        help="Transfer package root. Defaults to auto-detection from this script path.",
    )
    parser.add_argument(
        "--catalog-h5",
        type=Path,
        default=None,
        help="Input combined compact DESI HDF5 catalog. Default is package-relative.",
    )
    parser.add_argument(
        "--random-fits",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more Legacy Survey DR10 random FITS files. Default is the "
            "transferred randoms-1-0.fits file."
        ),
    )
    parser.add_argument(
        "--coeffs-yaml",
        type=Path,
        default=None,
        help=(
            "YAML file with linear coefficients. Default is the transferred "
            "Extended-LRG DR9 coefficient file."
        ),
    )
    parser.add_argument(
        "--template-map-h5",
        type=Path,
        default=None,
        help="Output/input HDF5 file for DR10 random-derived template maps.",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=None,
        help="Output HDF5 file for row-matched galaxy imaging weights.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=1024,
        help="RING HEALPix NSIDE for random-derived imaging templates.",
    )
    parser.add_argument(
        "--field",
        choices=("north", "south"),
        default="south",
        help=(
            "Coefficient block to use. The transferred Extended LRG velocity "
            "catalog is expected to be in the south/DECaLS footprint."
        ),
    )
    parser.add_argument(
        "--photsys",
        choices=("N", "S", "any"),
        default="S",
        help="PHOTSYS cut applied to randoms before building template maps.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of random rows read per chunk.",
    )
    parser.add_argument(
        "--min-randoms-per-pixel",
        type=int,
        default=1,
        help="Minimum accepted random count for a valid template pixel.",
    )
    parser.add_argument(
        "--no-require-nobs",
        action="store_false",
        dest="require_nobs",
        help="Do not require NOBS_G/R/Z > 0 for randoms.",
    )
    parser.set_defaults(require_nobs=True)
    parser.add_argument(
        "--allow-nonzero-maskbits",
        action="store_true",
        help="Do not require MASKBITS == 0 for randoms.",
    )
    parser.add_argument(
        "--rebuild-template-maps",
        action="store_true",
        help="Rebuild template maps even if --template-map-h5 already exists.",
    )
    parser.add_argument(
        "--allow-high-nside",
        action="store_true",
        help=(
            "Allow NSIDE > 1024. With only one transferred random file this is "
            "usually sparse and not recommended for raw masks/weights."
        ),
    )
    parser.add_argument(
        "--renormalize-by-bin",
        action="store_true",
        help="Renormalize finite positive weights to mean 1 within each pz bin.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def fill_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.transfer_root.resolve()
    args.transfer_root = root
    if args.catalog_h5 is None:
        args.catalog_h5 = (
            root
            / "data"
            / "desi_dr10_extended_velocity_catalogs"
            / "desi_dr10_extended_all_pz_compact.h5"
        )
    if args.random_fits is None:
        args.random_fits = [
            root / "data" / "desi_dr10_imaging_randoms" / "randoms-1-0.fits"
        ]
    if args.coeffs_yaml is None:
        args.coeffs_yaml = default_coeff_path(root)
    if args.template_map_h5 is None:
        args.template_map_h5 = (
            root
            / "data"
            / "desi_dr10_imaging_weights"
            / f"dr10_random_imaging_templates_nside{args.nside}.h5"
        )
    if args.output_h5 is None:
        args.output_h5 = (
            root
            / "data"
            / "desi_dr10_imaging_weights"
            / f"desi_dr10_extended_lrg_imaging_weights_dr9coeffs_nside{args.nside}.h5"
        )
    return args


def import_runtime_modules():
    try:
        import h5py
        from astropy.io import fits
        import healpy as hp
        import yaml
    except Exception as exc:  # pragma: no cover - dependency guidance.
        raise RuntimeError(
            "This script needs h5py, astropy, healpy, and PyYAML. On this "
            "NERSC account, /global/homes/s/spandey/.conda/envs/myenv_conda/bin/python "
            "has those modules available."
        ) from exc
    return h5py, fits, hp, yaml


def ensure_inputs(args: argparse.Namespace) -> None:
    paths = [args.catalog_h5, args.coeffs_yaml] + list(args.random_fits)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input path(s):\n" + "\n".join(missing))
    if args.nside <= 0 or args.nside & (args.nside - 1) != 0:
        raise ValueError("HEALPix NSIDE must be a positive power of two.")
    if args.nside > 1024 and not args.allow_high_nside:
        raise ValueError(
            "Refusing NSIDE > 1024 by default. One transferred DR10 random file "
            "is sparse at high NSIDE. Re-run with --allow-high-nside only if you "
            "understand this caveat or have supplied many random files."
        )
    if args.output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_h5} exists; pass --overwrite to replace it.")
    if (
        args.template_map_h5.exists()
        and args.rebuild_template_maps
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{args.template_map_h5} exists; pass --overwrite to rebuild it."
        )


def read_coefficients(path: Path, yaml_module) -> dict:
    with path.open("r") as handle:
        coeffs = yaml_module.safe_load(handle)
    required = [f"{field}_bin_{i}" for field in ("north", "south") for i in range(1, 5)]
    missing = [key for key in required if key not in coeffs]
    if missing:
        raise KeyError(f"Coefficient YAML is missing block(s): {missing}")
    return coeffs


def depth_to_mag_ebv(galdepth: np.ndarray, ebv: np.ndarray, band: str) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * (np.log10(5.0 / np.sqrt(galdepth)) - 9.0)
    return mag - EXTINCTION_COEFF[band] * ebv


def radec_to_ring_pixels(hp_module, nside: int, ra_deg: np.ndarray, dec_deg: np.ndarray):
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(np.mod(ra_deg, 360.0))
    return hp_module.ang2pix(nside, theta, phi, nest=False)


def require_columns(column_names, required, path: Path) -> None:
    missing = [name for name in required if name not in column_names]
    if missing:
        raise KeyError(f"{path} is missing required FITS columns: {missing}")


def build_template_maps(args: argparse.Namespace, fits_module, hp_module, h5py_module) -> None:
    npix = hp_module.nside2npix(args.nside)
    counts = np.zeros(npix, dtype=np.uint32)
    sums = {name: np.zeros(npix, dtype=np.float64) for name in TEMPLATE_COLUMNS}

    for path in args.random_fits:
        print(f"Reading randoms from {path}", flush=True)
        with fits_module.open(path, memmap=True) as hdul:
            table_hdu = hdul[1]
            nrows = int(table_hdu.header["NAXIS2"])
            columns = set(table_hdu.columns.names)
            required = {"RA", "DEC", "PHOTSYS", "MASKBITS", *TEMPLATE_COLUMNS}
            if args.require_nobs:
                required |= set(NOBS_COLUMNS)
            require_columns(columns, required, path)
            data = table_hdu.data

            for start in range(0, nrows, args.chunk_size):
                stop = min(start + args.chunk_size, nrows)
                slc = slice(start, stop)

                ra = np.asarray(data["RA"][slc], dtype=np.float64)
                dec = np.asarray(data["DEC"][slc], dtype=np.float64)
                good = np.isfinite(ra) & np.isfinite(dec)

                if args.photsys != "any":
                    photsys = np.asarray(data["PHOTSYS"][slc]).astype(str)
                    good &= np.char.strip(photsys) == args.photsys
                if not args.allow_nonzero_maskbits:
                    good &= np.asarray(data["MASKBITS"][slc]) == 0
                if args.require_nobs:
                    for col in NOBS_COLUMNS:
                        good &= np.asarray(data[col][slc]) > 0

                values = {}
                for col in TEMPLATE_COLUMNS:
                    values[col] = np.asarray(data[col][slc], dtype=np.float64)
                    good &= np.isfinite(values[col])
                for col in DEPTH_COLUMNS:
                    good &= values[col] > 0.0

                n_good = int(np.count_nonzero(good))
                if n_good == 0:
                    continue

                pix = radec_to_ring_pixels(hp_module, args.nside, ra[good], dec[good])
                counts += np.bincount(pix, minlength=npix).astype(np.uint32)
                for col in TEMPLATE_COLUMNS:
                    sums[col] += np.bincount(
                        pix, weights=values[col][good], minlength=npix
                    )

                if start == 0 or ((start // args.chunk_size + 1) % 10 == 0):
                    frac = stop / nrows
                    print(
                        f"  rows {stop:,}/{nrows:,} ({frac:.1%}); "
                        f"accepted this chunk: {n_good:,}",
                        flush=True,
                    )

    valid = counts >= args.min_randoms_per_pixel
    args.template_map_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py_module.File(args.template_map_h5, "w") as h5:
        h5.attrs["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        h5.attrs["nside"] = args.nside
        h5.attrs["ordering"] = "RING"
        h5.attrs["random_fits"] = json.dumps([str(path) for path in args.random_fits])
        h5.attrs["photsys_cut"] = args.photsys
        h5.attrs["require_nobs_grz_gt_zero"] = bool(args.require_nobs)
        h5.attrs["require_maskbits_zero"] = not bool(args.allow_nonzero_maskbits)
        h5.attrs["min_randoms_per_pixel"] = args.min_randoms_per_pixel
        h5.attrs["caveat"] = (
            "Template maps are built from imaging randoms. With only one "
            "randoms-1-* file, high-NSIDE maps are sparse and should be smoothed, "
            "apodized, or rebuilt with more random realizations for production."
        )
        group = h5.create_group("maps")
        group.create_dataset("count", data=counts, compression="gzip", shuffle=True)
        group.create_dataset("valid", data=valid, compression="gzip", shuffle=True)
        for col in TEMPLATE_COLUMNS:
            mean = np.full(npix, np.nan, dtype=np.float32)
            mean[valid] = (sums[col][valid] / counts[valid]).astype(np.float32)
            group.create_dataset(col, data=mean, compression="gzip", shuffle=True)

    print(f"Wrote template maps to {args.template_map_h5}", flush=True)


def load_catalog(path: Path, h5py_module):
    with h5py_module.File(path, "r") as h5:
        ra = h5["catalog/ra_deg"][:].astype(np.float64)
        dec = h5["catalog/dec_deg"][:].astype(np.float64)
        if "catalog/pz_bin" not in h5:
            raise KeyError("Input catalog must contain catalog/pz_bin for this script.")
        pz_bin = h5["catalog/pz_bin"][:].astype(np.int16)
    return ra, dec, pz_bin


def load_template_maps(path: Path, h5py_module):
    with h5py_module.File(path, "r") as h5:
        nside = int(h5.attrs["nside"])
        maps = {name: h5[f"maps/{name}"][:] for name in TEMPLATE_COLUMNS}
        counts = h5["maps/count"][:]
        valid = h5["maps/valid"][:].astype(bool)
    return nside, maps, counts, valid


def compute_weights(args: argparse.Namespace, hp_module, h5py_module, yaml_module) -> None:
    coeffs = read_coefficients(args.coeffs_yaml, yaml_module)
    key0 = f"{args.field}_bin_1"
    feature_names = [name for name in coeffs[key0] if name != "intercept"]

    ra, dec, pz_bin = load_catalog(args.catalog_h5, h5py_module)
    nside, maps, counts_map, valid_map = load_template_maps(
        args.template_map_h5, h5py_module
    )
    if nside != args.nside:
        raise ValueError(
            f"Template map NSIDE={nside}, but requested NSIDE={args.nside}."
        )

    pix = radec_to_ring_pixels(hp_module, args.nside, ra, dec)
    sampled = {name: maps[name][pix].astype(np.float32) for name in TEMPLATE_COLUMNS}
    count_at_pixel = counts_map[pix].astype(np.uint32)
    template_pixel_valid = valid_map[pix]

    sampled["galdepth_gmag_ebv"] = depth_to_mag_ebv(
        sampled["GALDEPTH_G"], sampled["EBV"], "G"
    ).astype(np.float32)
    sampled["galdepth_rmag_ebv"] = depth_to_mag_ebv(
        sampled["GALDEPTH_R"], sampled["EBV"], "R"
    ).astype(np.float32)
    sampled["galdepth_zmag_ebv"] = depth_to_mag_ebv(
        sampled["GALDEPTH_Z"], sampled["EBV"], "Z"
    ).astype(np.float32)

    nobj = len(ra)
    predicted = np.full(nobj, np.nan, dtype=np.float32)
    weight = np.zeros(nobj, dtype=np.float32)
    valid_weight = np.zeros(nobj, dtype=bool)
    invalid_reason = np.zeros(nobj, dtype=np.int16)
    invalid_reason[~template_pixel_valid] = 1

    diagnostics = {}
    for bin_index in range(1, 5):
        key = f"{args.field}_bin_{bin_index}"
        if key not in coeffs:
            raise KeyError(f"Missing coefficient block {key}")
        in_bin = pz_bin == bin_index
        finite = in_bin & template_pixel_valid
        for name in feature_names:
            finite &= np.isfinite(sampled[name])
        bad_template = in_bin & ~finite
        invalid_reason[bad_template & template_pixel_valid] = 2

        idx = np.where(finite)[0]
        if len(idx) == 0:
            diagnostics[f"pz{bin_index}"] = {
                "n_total": int(np.count_nonzero(in_bin)),
                "n_valid": 0,
                "mean_weight": None,
            }
            continue

        pred = np.full(len(idx), coeffs[key]["intercept"], dtype=np.float64)
        for name in feature_names:
            pred += float(coeffs[key][name]) * sampled[name][idx]
        good_pred = np.isfinite(pred) & (pred > 0.0)
        predicted[idx] = pred.astype(np.float32)
        valid_idx = idx[good_pred]
        invalid_reason[idx[~good_pred]] = 3
        weight[valid_idx] = (1.0 / pred[good_pred]).astype(np.float32)

        if args.renormalize_by_bin and len(valid_idx) > 0:
            mean_weight = float(np.mean(weight[valid_idx]))
            if np.isfinite(mean_weight) and mean_weight > 0.0:
                weight[valid_idx] /= mean_weight

        valid_weight[valid_idx] = True
        diagnostics[f"pz{bin_index}"] = {
            "n_total": int(np.count_nonzero(in_bin)),
            "n_valid": int(len(valid_idx)),
            "n_invalid_template": int(np.count_nonzero(in_bin & ~template_pixel_valid)),
            "n_invalid_features": int(np.count_nonzero(bad_template)),
            "n_invalid_prediction": int(np.count_nonzero(finite) - len(valid_idx)),
            "mean_weight": float(np.mean(weight[valid_idx])) if len(valid_idx) else None,
            "std_weight": float(np.std(weight[valid_idx])) if len(valid_idx) else None,
        }

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py_module.File(args.output_h5, "w") as h5:
        h5.attrs["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        h5.attrs["method"] = "DR10 random-template maps sampled at galaxy positions; DR9 Extended-LRG linear coefficients applied"
        h5.attrs["catalog_h5"] = str(args.catalog_h5)
        h5.attrs["template_map_h5"] = str(args.template_map_h5)
        h5.attrs["coeffs_yaml"] = str(args.coeffs_yaml)
        h5.attrs["field"] = args.field
        h5.attrs["photsys"] = args.photsys
        h5.attrs["nside"] = args.nside
        h5.attrs["ordering"] = "RING"
        h5.attrs["row_matching"] = (
            "All 1D datasets are row-matched to catalog/ra_deg in the input "
            "combined DESI compact HDF5 catalog."
        )
        h5.attrs["caveat"] = (
            "These are approximate weights obtained by applying DR9-trained "
            "coefficients to DR10 random-derived imaging templates. Refit on "
            "DR10 galaxy/random density for production galaxy auto spectra."
        )
        h5.attrs["feature_names_used_by_coefficients"] = json.dumps(feature_names)
        h5.attrs["diagnostics_by_pz_bin"] = json.dumps(diagnostics, indent=2)
        h5.attrs["invalid_reason_codes"] = json.dumps(
            {
                "0": "not assigned invalid, or object outside pz1..pz4",
                "1": "template pixel invalid or below min random count",
                "2": "sampled imaging feature not finite",
                "3": "linear model prediction not finite or <= 0",
            }
        )

        catalog = h5.create_group("catalog")
        catalog.create_dataset("ra_deg", data=ra.astype(np.float32), compression="gzip", shuffle=True)
        catalog.create_dataset("dec_deg", data=dec.astype(np.float32), compression="gzip", shuffle=True)
        catalog.create_dataset("pz_bin", data=pz_bin, compression="gzip", shuffle=True)

        weights = h5.create_group("weights")
        weights.create_dataset("weight", data=weight, compression="gzip", shuffle=True)
        weights.create_dataset(
            "predicted_density", data=predicted, compression="gzip", shuffle=True
        )
        weights.create_dataset(
            "valid", data=valid_weight, compression="gzip", shuffle=True
        )
        weights.create_dataset(
            "invalid_reason", data=invalid_reason, compression="gzip", shuffle=True
        )

        features = h5.create_group("sampled_imaging_features")
        for name, values in sampled.items():
            features.create_dataset(name, data=values, compression="gzip", shuffle=True)

        diagnostics_group = h5.create_group("diagnostics")
        diagnostics_group.create_dataset(
            "template_pixel_ring", data=pix.astype(np.int64), compression="gzip", shuffle=True
        )
        diagnostics_group.create_dataset(
            "random_count_at_template_pixel",
            data=count_at_pixel,
            compression="gzip",
            shuffle=True,
        )
        diagnostics_group.create_dataset(
            "template_pixel_valid",
            data=template_pixel_valid,
            compression="gzip",
            shuffle=True,
        )

    print(f"Wrote row-matched imaging weights to {args.output_h5}", flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)


def main() -> int:
    args = fill_default_paths(parse_args())
    h5py_module, fits_module, hp_module, yaml_module = import_runtime_modules()
    ensure_inputs(args)

    if args.rebuild_template_maps or not args.template_map_h5.exists():
        if args.template_map_h5.exists() and args.overwrite:
            args.template_map_h5.unlink()
        build_template_maps(args, fits_module, hp_module, h5py_module)
    else:
        print(f"Using existing template maps: {args.template_map_h5}", flush=True)

    if args.output_h5.exists() and args.overwrite:
        args.output_h5.unlink()
    compute_weights(args, hp_module, h5py_module, yaml_module)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
