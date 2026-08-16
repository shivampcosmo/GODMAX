#!/usr/bin/env python3
"""Prepare DR9 Extended LRG velocity, imaging-weight, and random products.

This script builds DR9 products parallel to the DR10 transfer products.  It
uses the kSZ ASCII velocity catalogs for the four tomographic bins, joins them
to the public DESI LRG x-correlation DR9 Extended LRG catalog by exact sky
position, attaches the public precomputed imaging weights, and applies the DR9
LRG quality-cut logic from ``randoms_quality_cuts.py`` to one or more exactly
paired random/LRG-mask files.

The generated HDF5 catalogs keep all input velocity rows but provide
``catalog/valid_for_cl`` as the recommended default selection for harmonic-space
auto and cross spectra.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/act_desi_ksz_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/act_desi_ksz_xdgcache")

import h5py
import healpy as hp
import matplotlib
import fitsio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree


C_KM_S = 3.0e5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_CATALOG_DIR = Path("data/desi_dr9_extended_velocity_catalogs")
OUT_RANDOM_DIR = Path("data/desi_dr9_imaging_randoms")
OUT_DOC_DIR = Path("docs")
OUT_FIG_DIR = Path("quicklook_figures")

DR9_PAPER_ROOT = Path("/global/cfs/cdirs/lsst/www/shivamp/desi/lrg_xcorr_2023/v1")
DR9_CATALOG_PATH = DR9_PAPER_ROOT / "catalogs/dr9_extended_lrg_pzbins.fits"
DR9_WEIGHT_PATH = DR9_PAPER_ROOT / "catalogs/more/dr9_extended_lrg_pzbins-weights.fits"
DR9_WEIGHT_NO_EBV_PATH = (
    DR9_PAPER_ROOT / "catalogs/more/dr9_extended_lrg_pzbins-weights_no_ebv.fits"
)
DR9_RANDOM_SOURCE_PATH = Path(
    "/global/cfs/cdirs/desi/public/ets/target/catalogs/dr9/0.49.0/"
    "randoms/resolve/randoms-1-0.fits"
)
DR9_RANDOM_LRGMASK_PATH = (
    DR9_PAPER_ROOT / "catalogs/lrgmask_v1.1/randoms-1-0-lrgmask_v1.1.fits.gz"
)
DR9_STARDENS_PATH = DR9_PAPER_ROOT / "misc/pixweight-dr7.1-0.22.0_stardens_64_ring.fits"

LOCAL_DR9_PAPER_RELATIVE = Path("zhou-lrg-xcorr-2023-v1")
RANDOM_REQUIRED_COLUMNS = (
    "RA",
    "DEC",
    "NOBS_G",
    "NOBS_R",
    "NOBS_Z",
    "MASKBITS",
    "EBV",
)
RANDOM_MAP_NSIDES = (1024, 2048, 4096)
RANDOM_IDENTITY_SAMPLE_SIZE = 33
DR9_RANDOM_EXPECTED_ROWS = 51_738_616
RANDOM_PRODUCT_SCHEMA_VERSION = "desi-dr9-extended-lrg-random-mask-v3"
LEGACY_RANDOM_SUMMARY_NAME = "desi_dr9_randoms_1_0_lrg_quality_cut_compact.h5"
LEGACY_RANDOM_MAP_NAME = "desi_dr9_randoms_1_0_lrg_quality_count_maps_nside1024_4096.h5"
RANDOM_FILENAME_RE = re.compile(r"randoms-1-(\d+)\.fits")
LRGMASK_FILENAME_RE = re.compile(r"randoms-1-(\d+)-lrgmask_v1\.1\.fits\.gz")

RANDOM_INITIAL_MASKBITS = (1, 12, 13)
MIN_TARGET_NOBS = 1
MIN_LRG_NOBS = 2
MAX_EBV = 0.15
MAX_STARDENS = 2500.0
STARDENS_NSIDE = 64
MATCH_TOLERANCE_ARCSEC = 1.0e-4
MATCH_Z_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class BinSpec:
    label: str
    pz_bin: int
    path: Path


@dataclass(frozen=True)
class RandomPairSpec:
    index: int
    random_path: Path
    lrgmask_path: Path


@dataclass(frozen=True)
class ValidatedRandomPair:
    spec: RandomPairSpec
    n_rows: int
    random_size_bytes: int
    random_mtime_ns: int
    lrgmask_size_bytes: int
    lrgmask_mtime_ns: int
    sample_identity_sha256: str
    random_sha256: str
    lrgmask_sha256: str


@dataclass(frozen=True)
class ValidatedRandomInputs:
    pairs: tuple[ValidatedRandomPair, ...]
    stardens_path: Path
    stardens_size_bytes: int
    stardens_mtime_ns: int
    stardens_identity_sha256: str
    cuts_source_path: Path | None
    full_source_sha256_by_relative_path: dict[str, str]
    sha256_ledger_path: Path | None
    sha256_ledger_sha256: str
    full_source_sha256_verified: bool
    input_identity_sha256: str


BIN_SPECS = (
    BinSpec(
        "pz1",
        1,
        Path(
            "/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz1/"
            "extended_catalog_allfoot_perbin_sigmaz0.0500.txt"
        ),
    ),
    BinSpec(
        "pz2",
        2,
        Path(
            "/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz2/"
            "extended_catalog_allfoot_perbin_sigmaz0.0500.txt"
        ),
    ),
    BinSpec(
        "pz3",
        3,
        Path(
            "/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz3/"
            "extended_catalog_allfoot_perbin_sigmaz0.0500.txt"
        ),
    ),
    BinSpec(
        "pz4",
        4,
        Path(
            "/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz4/"
            "extended_catalog_allfoot_perbin_sigmaz0.0500.txt"
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def package_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def h5_kwargs(dtype) -> dict:
    return {
        "dtype": dtype,
        "chunks": True,
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }


def atomic_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def set_common_attrs(h5: h5py.File, product_type: str) -> None:
    h5.attrs["product_type"] = product_type
    h5.attrs["created_utc"] = utc_now()
    h5.attrs["created_by"] = Path(__file__).name
    source_sha256 = os.environ.get("XDESI_RANDOM_RUNTIME_SOURCE_SHA256")
    if not source_sha256:
        source_sha256 = sha256_file(Path(__file__).resolve())
    h5.attrs["created_by_sha256"] = source_sha256
    h5.attrs["hostname"] = socket.gethostname()
    h5.attrs["python"] = sys.version
    h5.attrs["platform"] = platform.platform()
    h5.attrs["transfer_package_root"] = "."
    h5.attrs["path_convention"] = (
        "Package-internal paths are relative to the transfer package root "
        "containing README.md."
    )
    h5.attrs["nersc_source_path_note"] = (
        "Absolute paths are NERSC provenance only and are not required after "
        "transferring this package."
    )


def random_cuts_definition() -> dict:
    """Return the public Zhou et al. random cuts without interpretation changes."""
    return {
        "initial_geometry": {
            "nobs_grz_min": MIN_TARGET_NOBS,
            "maskbits_veto": list(RANDOM_INITIAL_MASKBITS),
        },
        "lrg_quality": {
            "remove_ngc_islands": "not (DEC < -10.5 and 120 < RA < 260)",
            "nobs_grz_min": MIN_LRG_NOBS,
            "lrg_mask": 0,
            "ebv_max": MAX_EBV,
            "stardens_max": MAX_STARDENS,
            "stardens_nside_ring": STARDENS_NSIDE,
        },
    }


def parse_random_indices(tokens: list[str] | None) -> tuple[int, ...] | None:
    if tokens is None:
        return None
    parsed = []
    for token in tokens:
        for item in token.split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    if not parsed:
        raise ValueError("--random-indices did not contain any indices.")
    if any(index < 0 for index in parsed):
        raise ValueError(f"Random indices must be non-negative, got {parsed}.")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate random indices are not allowed: {parsed}.")
    return tuple(sorted(parsed))


def _indexed_paths(directory: Path, pattern: str, regex: re.Pattern) -> dict[int, Path]:
    indexed = {}
    for path in sorted(directory.glob(pattern)):
        match = regex.fullmatch(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Duplicate files for random index {index}: {indexed[index]}, {path}")
        indexed[index] = path.resolve()
    return indexed


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return a streaming SHA256 digest without loading a source FITS into memory."""
    stat_before = path.stat()
    identity = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            identity.update(chunk)
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise RuntimeError(f"Source changed while computing full SHA256: {path}")
    return identity.hexdigest()


def read_sha256_ledger(path: Path) -> tuple[dict[str, str], str]:
    """Parse a sha256sum-compatible ledger and bind its exact bytes."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw-source SHA256 ledger: {path}")
    raw = path.read_bytes()
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed SHA256 ledger line {line_number}: {raw_line!r}")
        digest, relative = fields
        relative = relative.lstrip("*").strip()
        relative_path = Path(relative)
        if (
            not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in {"", "."}
        ):
            raise ValueError(f"Unsafe SHA256 ledger line {line_number}: {raw_line!r}")
        key = relative_path.as_posix()
        if key in entries:
            raise ValueError(f"Duplicate SHA256 ledger path on line {line_number}: {key}")
        entries[key] = digest.lower()
    if not entries:
        raise ValueError(f"SHA256 ledger contains no source entries: {path}")
    return entries, hashlib.sha256(raw).hexdigest()


def ledger_entry_for_path(
    source_path: Path,
    ledger_path: Path,
    ledger_entries: dict[str, str],
) -> tuple[str, str]:
    """Resolve a source to its exact ledger-relative path and expected digest."""
    try:
        relative = source_path.resolve().relative_to(ledger_path.resolve().parent).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Raw source {source_path} is outside SHA256 ledger root {ledger_path.parent}."
        ) from error
    if relative not in ledger_entries:
        raise ValueError(f"SHA256 ledger {ledger_path} has no entry for {relative}.")
    return relative, ledger_entries[relative]


def resolve_random_pair_specs(
    source_root: Path | None,
    requested_indices: tuple[int, ...] | None,
) -> tuple[tuple[RandomPairSpec, ...], Path, Path | None]:
    """Resolve exact random/mask pairs, either locally or from the legacy pair-0 paths."""
    if source_root is None:
        indices = requested_indices or (0,)
        if indices != (0,):
            raise ValueError(
                "Indices other than 0 require --random-source-root so filenames can be resolved locally."
            )
        return (
            (RandomPairSpec(0, DR9_RANDOM_SOURCE_PATH, DR9_RANDOM_LRGMASK_PATH),),
            DR9_STARDENS_PATH,
            DR9_PAPER_ROOT / "catalogs/randoms_quality_cuts.py",
        )

    source_root = source_root.resolve()
    random_dir = source_root / "randoms/resolve"
    paper_root = source_root / LOCAL_DR9_PAPER_RELATIVE
    mask_dir = paper_root / "catalogs/lrgmask_v1.1"
    if not random_dir.is_dir():
        raise FileNotFoundError(f"Missing DR9 random directory: {random_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing Zhou LRG-mask directory: {mask_dir}")

    random_by_index = _indexed_paths(random_dir, "randoms-1-*.fits", RANDOM_FILENAME_RE)
    mask_by_index = _indexed_paths(
        mask_dir,
        "randoms-1-*-lrgmask_v1.1.fits.gz",
        LRGMASK_FILENAME_RE,
    )
    if requested_indices is None:
        random_only = sorted(set(random_by_index) - set(mask_by_index))
        mask_only = sorted(set(mask_by_index) - set(random_by_index))
        if random_only or mask_only:
            raise ValueError(
                f"Unpaired local inputs: random_without_mask={random_only}, "
                f"mask_without_random={mask_only}."
            )
        indices = tuple(sorted(random_by_index))
    else:
        indices = requested_indices
    if not indices:
        raise ValueError(f"No random/mask pairs found below {source_root}.")

    missing_randoms = [index for index in indices if index not in random_by_index]
    missing_masks = [index for index in indices if index not in mask_by_index]
    if missing_randoms or missing_masks:
        raise FileNotFoundError(
            f"Requested random inputs are incomplete: missing_randoms={missing_randoms}, "
            f"missing_masks={missing_masks}."
        )
    pairs = tuple(
        RandomPairSpec(index, random_by_index[index], mask_by_index[index])
        for index in indices
    )
    return (
        pairs,
        paper_root / "misc/pixweight-dr7.1-0.22.0_stardens_64_ring.fits",
        paper_root / "catalogs/randoms_quality_cuts.py",
    )


def validate_stardens_input(path: Path) -> tuple[int, int, str]:
    """Fully validate the small nside-64 RING stellar-density lookup table."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Zhou stellar-density map: {path}")
    stat_before = path.stat()
    with fitsio.FITS(str(path)) as hdul:
        columns = set(hdul[1].get_colnames())
        required = {"HPXPIXEL", "STARDENS"}
        if not required.issubset(columns):
            raise ValueError(f"{path} lacks stellar-density columns {sorted(required - columns)}.")
        table = hdul[1].read(columns=["HPXPIXEL", "STARDENS"])
    pixels = np.asarray(table["HPXPIXEL"], dtype=np.int64)
    values = np.asarray(table["STARDENS"], dtype=np.float32)
    expected_pixels = np.arange(hp.nside2npix(STARDENS_NSIDE), dtype=np.int64)
    if not np.array_equal(np.sort(pixels), expected_pixels):
        raise ValueError(
            f"{path} must contain each nside={STARDENS_NSIDE} RING pixel exactly once."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite STARDENS values.")
    identity = hashlib.sha256()
    identity.update(pixels.tobytes(order="C"))
    identity.update(values.tobytes(order="C"))
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise RuntimeError(f"Stellar-density input changed during validation: {path}")
    return int(stat_after.st_size), int(stat_after.st_mtime_ns), identity.hexdigest()


def validate_random_inputs(
    pair_specs: tuple[RandomPairSpec, ...],
    stardens_path: Path,
    cuts_source_path: Path | None,
    sha256_ledger_path: Path,
    verify_full_sha256: bool = False,
) -> ValidatedRandomInputs:
    """Validate pair structure and bind every used source to a full-file SHA256 ledger."""
    sha256_ledger_path = sha256_ledger_path.resolve()
    ledger_entries, ledger_sha256 = read_sha256_ledger(sha256_ledger_path)
    full_source_sha256_by_relative_path: dict[str, str] = {}
    validated = []
    required = set(RANDOM_REQUIRED_COLUMNS)
    for spec in pair_specs:
        if not spec.random_path.is_file():
            raise FileNotFoundError(spec.random_path)
        if not spec.lrgmask_path.is_file():
            raise FileNotFoundError(spec.lrgmask_path)
        random_stat_before = spec.random_path.stat()
        mask_stat_before = spec.lrgmask_path.stat()
        with fitsio.FITS(str(spec.random_path)) as random_hdul, fitsio.FITS(
            str(spec.lrgmask_path)
        ) as mask_hdul:
            random_columns = set(random_hdul[1].get_colnames())
            mask_columns = list(mask_hdul[1].get_colnames())
            missing = sorted(required - random_columns)
            if missing:
                raise ValueError(f"{spec.random_path} lacks required columns {missing}.")
            if mask_columns != ["lrg_mask"]:
                raise ValueError(
                    f"{spec.lrgmask_path} must contain only ['lrg_mask']; got {mask_columns}."
                )
            n_random = int(random_hdul[1].get_nrows())
            n_mask = int(mask_hdul[1].get_nrows())
            if n_random != n_mask:
                raise ValueError(
                    f"Pair {spec.index} row mismatch: random={n_random}, lrgmask={n_mask}."
                )
            if n_random != DR9_RANDOM_EXPECTED_ROWS:
                raise ValueError(
                    f"Pair {spec.index} has {n_random} rows; expected DR9 count "
                    f"{DR9_RANDOM_EXPECTED_ROWS}."
                )
            sample_rows = np.unique(
                np.linspace(
                    0,
                    n_random - 1,
                    RANDOM_IDENTITY_SAMPLE_SIZE,
                    dtype=np.int64,
                )
            )
            random_sample = random_hdul[1].read(
                rows=sample_rows,
                columns=list(RANDOM_REQUIRED_COLUMNS),
            )
            mask_sample = mask_hdul[1].read(rows=sample_rows, columns=["lrg_mask"])

        ra = np.asarray(random_sample["RA"])
        dec = np.asarray(random_sample["DEC"])
        ebv = np.asarray(random_sample["EBV"])
        if not (
            np.all(np.isfinite(ra))
            and np.all((ra >= 0.0) & (ra <= 360.0))
            and np.all(np.isfinite(dec))
            and np.all((dec >= -90.0) & (dec <= 90.0))
            and np.all(np.isfinite(ebv))
        ):
            raise ValueError(f"Pair {spec.index} has invalid sampled RA/DEC/EBV values.")
        for name in ("NOBS_G", "NOBS_R", "NOBS_Z"):
            if np.any(np.asarray(random_sample[name]) < 0):
                raise ValueError(f"Pair {spec.index} has negative sampled {name} values.")
        mask_values = np.asarray(mask_sample["lrg_mask"])
        if mask_values.dtype.kind != "u" or mask_values.dtype.itemsize != 1:
            raise ValueError(
                f"Pair {spec.index} lrg_mask must be uint8; got {mask_values.dtype}."
            )

        sample_identity = hashlib.sha256()
        sample_identity.update(np.asarray(sample_rows, dtype="<i8").tobytes(order="C"))
        sample_identity.update(random_sample.tobytes(order="C"))
        sample_identity.update(mask_sample.tobytes(order="C"))
        random_stat = spec.random_path.stat()
        mask_stat = spec.lrgmask_path.stat()
        if (random_stat_before.st_size, random_stat_before.st_mtime_ns) != (
            random_stat.st_size,
            random_stat.st_mtime_ns,
        ):
            raise RuntimeError(f"Random input changed during validation: {spec.random_path}")
        if (mask_stat_before.st_size, mask_stat_before.st_mtime_ns) != (
            mask_stat.st_size,
            mask_stat.st_mtime_ns,
        ):
            raise RuntimeError(f"LRG-mask input changed during validation: {spec.lrgmask_path}")
        random_relative, random_sha256 = ledger_entry_for_path(
            spec.random_path, sha256_ledger_path, ledger_entries
        )
        mask_relative, lrgmask_sha256 = ledger_entry_for_path(
            spec.lrgmask_path, sha256_ledger_path, ledger_entries
        )
        full_source_sha256_by_relative_path[random_relative] = random_sha256
        full_source_sha256_by_relative_path[mask_relative] = lrgmask_sha256
        if verify_full_sha256:
            observed_random_sha256 = sha256_file(spec.random_path)
            if observed_random_sha256 != random_sha256:
                raise ValueError(
                    f"Full SHA256 mismatch for {random_relative}: "
                    f"observed={observed_random_sha256}, expected={random_sha256}."
                )
            observed_lrgmask_sha256 = sha256_file(spec.lrgmask_path)
            if observed_lrgmask_sha256 != lrgmask_sha256:
                raise ValueError(
                    f"Full SHA256 mismatch for {mask_relative}: "
                    f"observed={observed_lrgmask_sha256}, expected={lrgmask_sha256}."
                )
        validated.append(
            ValidatedRandomPair(
                spec=spec,
                n_rows=n_random,
                random_size_bytes=int(random_stat.st_size),
                random_mtime_ns=int(random_stat.st_mtime_ns),
                lrgmask_size_bytes=int(mask_stat.st_size),
                lrgmask_mtime_ns=int(mask_stat.st_mtime_ns),
                sample_identity_sha256=sample_identity.hexdigest(),
                random_sha256=random_sha256,
                lrgmask_sha256=lrgmask_sha256,
            )
        )

    for nside in RANDOM_MAP_NSIDES:
        if not hp.isnsideok(nside):
            raise ValueError(f"Unsupported HEALPix nside in builder: {nside}.")
    stardens_size, stardens_mtime, stardens_identity = validate_stardens_input(stardens_path)
    if cuts_source_path is not None and not cuts_source_path.is_file():
        raise FileNotFoundError(f"Missing Zhou random-cut source: {cuts_source_path}")
    stardens_relative, stardens_sha256 = ledger_entry_for_path(
        stardens_path, sha256_ledger_path, ledger_entries
    )
    full_source_sha256_by_relative_path[stardens_relative] = stardens_sha256
    cuts_relative = None
    cuts_sha256 = None
    if cuts_source_path is not None:
        cuts_relative, cuts_sha256 = ledger_entry_for_path(
            cuts_source_path, sha256_ledger_path, ledger_entries
        )
        full_source_sha256_by_relative_path[cuts_relative] = cuts_sha256
    if verify_full_sha256:
        observed_stardens_sha256 = sha256_file(stardens_path)
        if observed_stardens_sha256 != stardens_sha256:
            raise ValueError(
                f"Full SHA256 mismatch for {stardens_relative}: "
                f"observed={observed_stardens_sha256}, expected={stardens_sha256}."
            )
        if cuts_source_path is not None:
            observed_cuts_sha256 = sha256_file(cuts_source_path)
            if observed_cuts_sha256 != cuts_sha256:
                raise ValueError(
                    f"Full SHA256 mismatch for {cuts_relative}: "
                    f"observed={observed_cuts_sha256}, expected={cuts_sha256}."
                )

    identity_payload = {
        "schema_version": RANDOM_PRODUCT_SCHEMA_VERSION,
        "random_map_nsides": list(RANDOM_MAP_NSIDES),
        "cuts": random_cuts_definition(),
        "pairs": [
            {
                "index": pair.spec.index,
                "random_name": pair.spec.random_path.name,
                "lrgmask_name": pair.spec.lrgmask_path.name,
                "n_rows": pair.n_rows,
                "random_size_bytes": pair.random_size_bytes,
                "lrgmask_size_bytes": pair.lrgmask_size_bytes,
                "sample_identity_sha256": pair.sample_identity_sha256,
                "random_sha256": pair.random_sha256,
                "lrgmask_sha256": pair.lrgmask_sha256,
            }
            for pair in validated
        ],
        "stardens_name": stardens_path.name,
        "stardens_size_bytes": stardens_size,
        "stardens_identity_sha256": stardens_identity,
        "stardens_sha256": stardens_sha256,
        "cuts_name": cuts_source_path.name if cuts_source_path is not None else None,
        "cuts_sha256": cuts_sha256,
        "sha256_ledger_sha256": ledger_sha256,
        "full_source_sha256_by_relative_path": dict(
            sorted(full_source_sha256_by_relative_path.items())
        ),
    }
    input_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ValidatedRandomInputs(
        pairs=tuple(validated),
        stardens_path=stardens_path.resolve(),
        stardens_size_bytes=stardens_size,
        stardens_mtime_ns=stardens_mtime,
        stardens_identity_sha256=stardens_identity,
        cuts_source_path=cuts_source_path.resolve() if cuts_source_path is not None else None,
        full_source_sha256_by_relative_path=dict(
            sorted(full_source_sha256_by_relative_path.items())
        ),
        sha256_ledger_path=sha256_ledger_path,
        sha256_ledger_sha256=ledger_sha256,
        full_source_sha256_verified=bool(verify_full_sha256),
        input_identity_sha256=input_identity,
    )


def random_validation_summary(inputs: ValidatedRandomInputs) -> dict:
    return {
        "schema_version": RANDOM_PRODUCT_SCHEMA_VERSION,
        "input_identity_sha256": inputs.input_identity_sha256,
        "identity_method": "full-file SHA256 ledger plus structural/sample validation",
        "sha256_ledger_path": str(inputs.sha256_ledger_path),
        "sha256_ledger_sha256": inputs.sha256_ledger_sha256,
        "full_source_sha256_verified": inputs.full_source_sha256_verified,
        "full_source_sha256_by_relative_path": (
            inputs.full_source_sha256_by_relative_path
        ),
        "indices": [pair.spec.index for pair in inputs.pairs],
        "n_pairs": len(inputs.pairs),
        "n_random_realizations": len(inputs.pairs),
        "random_realization_count": len(inputs.pairs),
        "n_source_rows_total": int(sum(pair.n_rows for pair in inputs.pairs)),
        "nsides": list(RANDOM_MAP_NSIDES),
        "stardens": {
            "path": str(inputs.stardens_path),
            "size_bytes": inputs.stardens_size_bytes,
            "mtime_ns": inputs.stardens_mtime_ns,
            "identity_sha256": inputs.stardens_identity_sha256,
        },
        "cuts_source_path": (
            str(inputs.cuts_source_path) if inputs.cuts_source_path is not None else None
        ),
        "pairs": [
            {
                "index": pair.spec.index,
                "random_path": str(pair.spec.random_path),
                "lrgmask_path": str(pair.spec.lrgmask_path),
                "n_rows": pair.n_rows,
                "random_size_bytes": pair.random_size_bytes,
                "random_mtime_ns": pair.random_mtime_ns,
                "lrgmask_size_bytes": pair.lrgmask_size_bytes,
                "lrgmask_mtime_ns": pair.lrgmask_mtime_ns,
                "sample_identity_sha256": pair.sample_identity_sha256,
                "random_sha256": pair.random_sha256,
                "lrgmask_sha256": pair.lrgmask_sha256,
            }
            for pair in inputs.pairs
        ],
    }


def ensure_dirs(root: Path) -> None:
    for subdir in (OUT_CATALOG_DIR, OUT_RANDOM_DIR, OUT_DOC_DIR, OUT_FIG_DIR):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cos_dec = np.cos(dec)
    return np.column_stack(
        (cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec))
    )


def chord_to_arcsec(chord: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    return angle * 206264.80624709636


def load_stardens_bad_pixels(path: Path = DR9_STARDENS_PATH) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        tab = hdul[1].data
        bad = np.asarray(tab["HPXPIXEL"][np.asarray(tab["STARDENS"]) >= MAX_STARDENS])
    return np.asarray(np.sort(bad), dtype=np.int64)


def stardens_for_positions(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    with fits.open(DR9_STARDENS_PATH, memmap=True) as hdul:
        tab = hdul[1].data
        values = np.zeros(hp.nside2npix(STARDENS_NSIDE), dtype=np.float32)
        pixels = np.asarray(tab["HPXPIXEL"], dtype=np.int64)
        values[pixels] = np.asarray(tab["STARDENS"], dtype=np.float32)
    pix = hp.ang2pix(STARDENS_NSIDE, ra_deg, dec_deg, lonlat=True, nest=False)
    return values[pix]


def quality_mask_for_catalog(
    ra: np.ndarray,
    dec: np.ndarray,
    nobs_g: np.ndarray,
    nobs_r: np.ndarray,
    nobs_z: np.ndarray,
    ebv: np.ndarray,
    lrg_mask: np.ndarray,
    bad_stardens_pixels: np.ndarray,
) -> np.ndarray:
    good = ~((dec < -10.5) & (ra > 120.0) & (ra < 260.0))
    good &= (nobs_g >= MIN_LRG_NOBS) & (nobs_r >= MIN_LRG_NOBS) & (nobs_z >= MIN_LRG_NOBS)
    good &= lrg_mask == 0
    good &= ebv < MAX_EBV
    pix = hp.ang2pix(STARDENS_NSIDE, ra, dec, lonlat=True, nest=False)
    good &= ~np.isin(pix, bad_stardens_pixels)
    return good


def load_velocity_ascii(path: Path, load_all_columns: bool) -> tuple[np.ndarray, int]:
    if load_all_columns:
        cat = np.loadtxt(path)
        if cat.ndim != 2 or cat.shape[1] <= 18:
            raise ValueError(f"{path} has unexpected shape {cat.shape}; need >=19 columns.")
        use = cat[:, [0, 1, 2, 15, 18]].copy()
        n_cols = int(cat.shape[1])
        del cat
        gc.collect()
        return use, n_cols

    use = np.loadtxt(path, usecols=(0, 1, 2, 15, 18))
    return use, 24


def build_public_bin_arrays(pz_bin: int, bad_stardens_pixels: np.ndarray) -> dict:
    log(f"Loading public DR9 Extended LRG rows for pz_bin={pz_bin}")
    with fits.open(DR9_CATALOG_PATH, memmap=True) as cat_hdul:
        tab = cat_hdul[1].data
        all_pz = np.asarray(tab["pz_bin"])
        idx = np.where(all_pz == pz_bin)[0]
        ra = np.asarray(tab["RA"][idx], dtype=np.float64)
        dec = np.asarray(tab["DEC"][idx], dtype=np.float64)
        out = {
            "source_row": idx.astype(np.int64),
            "targetid": np.asarray(tab["TARGETID"][idx], dtype=np.int64),
            "ra": ra,
            "dec": dec,
            "z_phot_median": np.asarray(tab["Z_PHOT_MEDIAN"][idx], dtype=np.float32),
            "ebv": np.asarray(tab["EBV"][idx], dtype=np.float32),
            "pixel_nobs_g": np.asarray(tab["PIXEL_NOBS_G"][idx], dtype=np.int16),
            "pixel_nobs_r": np.asarray(tab["PIXEL_NOBS_R"][idx], dtype=np.int16),
            "pixel_nobs_z": np.asarray(tab["PIXEL_NOBS_Z"][idx], dtype=np.int16),
            "maskbits": np.asarray(tab["MASKBITS"][idx], dtype=np.int16),
            "photsys": np.asarray(tab["PHOTSYS"][idx]).astype("S1"),
            "lrg_mask": np.asarray(tab["lrg_mask"][idx], dtype=np.uint8),
        }
    with fits.open(DR9_WEIGHT_PATH, memmap=True) as w_hdul:
        out["weight_imaging"] = np.asarray(
            w_hdul[1].data["weight"][out["source_row"]], dtype=np.float32
        )
    with fits.open(DR9_WEIGHT_NO_EBV_PATH, memmap=True) as w_hdul:
        out["weight_imaging_no_ebv"] = np.asarray(
            w_hdul[1].data["weight"][out["source_row"]], dtype=np.float32
        )

    out["quality_cut"] = quality_mask_for_catalog(
        out["ra"],
        out["dec"],
        out["pixel_nobs_g"],
        out["pixel_nobs_r"],
        out["pixel_nobs_z"],
        out["ebv"],
        out["lrg_mask"],
        bad_stardens_pixels,
    )
    out["stardens"] = stardens_for_positions(out["ra"], out["dec"])
    return out


def write_catalog_bin(root: Path, spec: BinSpec, bad_stardens_pixels: np.ndarray, args) -> Path:
    final_path = root / OUT_CATALOG_DIR / f"desi_dr9_extended_{spec.label}_compact_with_weights.h5"
    if final_path.exists() and not args.force:
        log(f"Skipping existing {final_path}")
        return final_path

    tmp_path = atomic_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()

    log(f"Loading DR9 velocity ASCII for {spec.label}: {spec.path}")
    vel, n_cols = load_velocity_ascii(spec.path, load_all_columns=not args.usecols_only)
    ra_match = vel[:, 0].astype(np.float64, copy=True)
    dec_match = vel[:, 1].astype(np.float64, copy=True)
    ra = ra_match.astype(np.float32)
    dec = dec_match.astype(np.float32)
    z = vel[:, 2].astype(np.float32)
    v_los = vel[:, 3].astype(np.float32)
    vr_over_c = (vel[:, 3] / C_KM_S).astype(np.float32)
    mass = vel[:, 4].astype(np.float64)
    n_rows = int(vel.shape[0])
    del vel
    gc.collect()

    public = build_public_bin_arrays(spec.pz_bin, bad_stardens_pixels)
    tree = cKDTree(radec_to_unit(public["ra"], public["dec"]))
    dist, nearest = tree.query(radec_to_unit(ra_match, dec_match), k=1)
    match_arcsec = chord_to_arcsec(dist).astype(np.float32)
    z_diff = np.abs(public["z_phot_median"][nearest].astype(np.float64) - z.astype(np.float64))
    matched = (match_arcsec <= MATCH_TOLERANCE_ARCSEC) & (z_diff <= MATCH_Z_TOLERANCE)

    matched_public = {name: values[nearest] for name, values in public.items() if name not in ("ra", "dec")}
    matched_public["source_ra_deg"] = public["ra"][nearest].astype(np.float64)
    matched_public["source_dec_deg"] = public["dec"][nearest].astype(np.float64)
    valid_weight = np.isfinite(matched_public["weight_imaging"]) & (matched_public["weight_imaging"] > 0.0)
    valid_for_cl = matched & matched_public["quality_cut"] & valid_weight

    mean_weight = float(np.mean(matched_public["weight_imaging"][valid_for_cl])) if np.any(valid_for_cl) else np.nan
    if np.isfinite(mean_weight) and mean_weight > 0.0:
        weight_mean1 = (matched_public["weight_imaging"] / mean_weight).astype(np.float32)
    else:
        weight_mean1 = np.full(n_rows, np.nan, dtype=np.float32)

    log(
        f"Writing {final_path}: n={n_rows:,}, matched={np.count_nonzero(matched):,}, "
        f"valid_for_cl={np.count_nonzero(valid_for_cl):,}"
    )
    with h5py.File(tmp_path, "w", track_order=True) as h5:
        set_common_attrs(h5, "DESI DR9 Extended compact velocity catalog with imaging weights")
        h5.attrs["nersc_source_ascii_path"] = str(spec.path)
        h5.attrs["photo_z_bin_label"] = spec.label
        h5.attrs["photo_z_bin_number"] = spec.pz_bin
        h5.attrs["source_n_rows"] = n_rows
        h5.attrs["source_n_columns"] = n_cols
        h5.attrs["velocity_convention"] = "vr_over_c = input column 15 / 3.0e5; no sign flip applied"
        h5.attrs["public_dr9_extended_lrg_catalog_provenance"] = str(DR9_CATALOG_PATH)
        h5.attrs["public_dr9_imaging_weight_provenance"] = str(DR9_WEIGHT_PATH)
        h5.attrs["public_dr9_no_ebv_weight_provenance"] = str(DR9_WEIGHT_NO_EBV_PATH)
        h5.attrs["match_method"] = "Nearest-neighbor match in unit-vector sky coordinates within pz_bin, checked against Z_PHOT_MEDIAN."
        h5.attrs["match_tolerance_arcsec"] = MATCH_TOLERANCE_ARCSEC
        h5.attrs["match_z_tolerance"] = MATCH_Z_TOLERANCE
        h5.attrs["n_matched"] = int(np.count_nonzero(matched))
        h5.attrs["n_valid_for_cl"] = int(np.count_nonzero(valid_for_cl))
        h5.attrs["mean_imaging_weight_valid_for_cl"] = mean_weight
        h5.attrs["quality_cut_definition"] = json.dumps(
            {
                "remove_ngc_islands": "not (DEC < -10.5 and 120 < RA < 260)",
                "pixel_nobs_grz_min": MIN_LRG_NOBS,
                "lrg_mask": 0,
                "ebv_max": MAX_EBV,
                "stardens_max": MAX_STARDENS,
                "stardens_nside_ring": STARDENS_NSIDE,
            },
            indent=2,
        )

        g = h5.create_group("catalog")
        datasets = {
            "ra_deg": (ra, "deg", "Right ascension from kSZ velocity ASCII catalog."),
            "dec_deg": (dec, "deg", "Declination from kSZ velocity ASCII catalog."),
            "z": (z, "dimensionless", "Photometric redshift from kSZ velocity ASCII catalog."),
            "v_los_km_s": (v_los, "km s^-1", "Line-of-sight reconstructed velocity."),
            "vr_over_c": (vr_over_c, "dimensionless", "Line-of-sight velocity divided by c."),
            "mass_msun": (mass, "Msun", "Stellar mass estimate from source column 18."),
            "targetid": (matched_public["targetid"], "none", "DESI TARGETID from matched public DR9 catalog."),
            "dr9_source_row": (matched_public["source_row"], "none", "Zero-indexed row in public dr9_extended_lrg_pzbins.fits."),
            "z_phot_median_public": (matched_public["z_phot_median"], "dimensionless", "Public DR9 Z_PHOT_MEDIAN used for matching."),
            "ebv": (matched_public["ebv"], "mag", "SFD E(B-V) from public DR9 catalog."),
            "pixel_nobs_g": (matched_public["pixel_nobs_g"], "none", "DR9 PIXEL_NOBS_G."),
            "pixel_nobs_r": (matched_public["pixel_nobs_r"], "none", "DR9 PIXEL_NOBS_R."),
            "pixel_nobs_z": (matched_public["pixel_nobs_z"], "none", "DR9 PIXEL_NOBS_Z."),
            "maskbits": (matched_public["maskbits"], "none", "DR9 MASKBITS."),
            "lrg_mask": (matched_public["lrg_mask"], "none", "LRG veto mask; clean objects have 0."),
            "stardens": (matched_public["stardens"], "deg^-2", "Stellar density from pixweight DR7.1 map."),
            "weight_imaging": (matched_public["weight_imaging"], "none", "Public precomputed DR9 Extended LRG imaging systematic weight."),
            "weight_imaging_no_ebv": (matched_public["weight_imaging_no_ebv"], "none", "Public precomputed imaging weight excluding E(B-V)."),
            "weight_imaging_mean1": (weight_mean1, "none", "weight_imaging renormalized to mean 1 among valid_for_cl rows in this bin."),
            "match_arcsec": (match_arcsec, "arcsec", "Angular distance to matched public DR9 catalog row."),
            "match_z_absdiff": (z_diff.astype(np.float32), "dimensionless", "Absolute redshift difference to matched public row."),
            "matched_to_public_dr9": (matched, "bool", "True if the public DR9 match passes angular and redshift tolerances."),
            "dr9_lrg_quality_cut": (matched_public["quality_cut"], "bool", "True if object passes the DR9 LRG quality/footprint cuts."),
            "valid_imaging_weight": (valid_weight, "bool", "True if imaging weight is finite and positive."),
            "valid_for_cl": (valid_for_cl, "bool", "Recommended default object selection for auto and cross spectra."),
        }
        for name, (data, units, desc) in datasets.items():
            ds = g.create_dataset(name, data=data, **h5_kwargs(data.dtype))
            ds.attrs["units"] = units
            ds.attrs["description"] = desc
        ph = g.create_dataset("photsys", data=matched_public["photsys"], **h5_kwargs("S1"))
        ph.attrs["description"] = "N for BASS/MzLS; S for DECaLS."

        hist = h5.create_group("histograms")
        hist.create_dataset("z_edges", data=np.linspace(0.35, 1.05, 71))
        hist.create_dataset("z_counts_all", data=np.histogram(z, bins=hist["z_edges"][:])[0])
        hist.create_dataset("z_counts_valid_for_cl", data=np.histogram(z[valid_for_cl], bins=hist["z_edges"][:])[0])

    os.replace(tmp_path, final_path)
    del public, tree, nearest, matched_public
    gc.collect()
    return final_path


def combine_catalogs(root: Path, paths: list[Path], force: bool) -> Path:
    final_path = root / OUT_CATALOG_DIR / "desi_dr9_extended_all_pz_compact_with_weights.h5"
    if final_path.exists() and not force:
        log(f"Skipping existing {final_path}")
        return final_path
    tmp_path = atomic_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()

    sizes = []
    dataset_names = None
    for path in paths:
        with h5py.File(path, "r") as h5:
            sizes.append(int(h5.attrs["source_n_rows"]))
            names = list(h5["catalog"].keys())
            dataset_names = names if dataset_names is None else dataset_names
    total = int(np.sum(sizes))
    starts = np.cumsum([0] + sizes[:-1]).astype(np.int64)
    stops = np.cumsum(sizes).astype(np.int64)

    log(f"Writing combined DR9 catalog: n={total:,}")
    with h5py.File(tmp_path, "w", track_order=True) as out:
        set_common_attrs(out, "Combined DESI DR9 Extended compact velocity catalog with imaging weights")
        out.attrs["source_catalog_hdf5_json"] = json.dumps([package_relative(p, root) for p in paths], indent=2)
        out.attrs["n_objects"] = total
        out.attrs["velocity_convention"] = "vr_over_c = input column 15 / 3.0e5; no sign flip applied"
        g = out.create_group("catalog")

        created = {}
        for path, start, stop, pz in zip(paths, starts, stops, [1, 2, 3, 4]):
            with h5py.File(path, "r") as h5:
                src = h5["catalog"]
                if not created:
                    for name in dataset_names:
                        ds = src[name]
                        created[name] = g.create_dataset(name, shape=(total,), **h5_kwargs(ds.dtype))
                        for key, value in ds.attrs.items():
                            created[name].attrs[key] = value
                    created["pz_bin"] = g.create_dataset("pz_bin", shape=(total,), **h5_kwargs("u1"))
                    created["pz_bin"].attrs["description"] = "Tomographic bin number, 1..4."
                for name in dataset_names:
                    created[name][start:stop] = src[name][:]
                created["pz_bin"][start:stop] = np.uint8(pz)
        out.attrs["n_valid_for_cl"] = int(np.count_nonzero(created["valid_for_cl"][:]))

    os.replace(tmp_path, final_path)
    return final_path


def add_counts(counts: np.ndarray, pix: np.ndarray) -> None:
    unique, n = np.unique(pix, return_counts=True)
    current = counts[unique].astype(np.uint64)
    updated = current + n.astype(np.uint64)
    if np.any(updated > np.iinfo(counts.dtype).max):
        raise OverflowError(f"Random-count overflow for dtype {counts.dtype}.")
    counts[unique] = updated.astype(counts.dtype)


def assert_pair_sources_unchanged(pair: ValidatedRandomPair) -> None:
    random_stat = pair.spec.random_path.stat()
    mask_stat = pair.spec.lrgmask_path.stat()
    if (random_stat.st_size, random_stat.st_mtime_ns) != (
        pair.random_size_bytes,
        pair.random_mtime_ns,
    ):
        raise RuntimeError(f"Random input changed after validation: {pair.spec.random_path}")
    if (mask_stat.st_size, mask_stat.st_mtime_ns) != (
        pair.lrgmask_size_bytes,
        pair.lrgmask_mtime_ns,
    ):
        raise RuntimeError(f"LRG-mask input changed after validation: {pair.spec.lrgmask_path}")


def assert_aux_sources_unchanged(inputs: ValidatedRandomInputs) -> None:
    """Recheck the small shared inputs and exact ledger around the streamed build."""
    if sha256_file(inputs.sha256_ledger_path) != inputs.sha256_ledger_sha256:
        raise RuntimeError(f"SHA256 ledger changed after validation: {inputs.sha256_ledger_path}")
    for source_path in (inputs.stardens_path, inputs.cuts_source_path):
        if source_path is None:
            continue
        relative, expected = ledger_entry_for_path(
            source_path,
            inputs.sha256_ledger_path,
            inputs.full_source_sha256_by_relative_path,
        )
        observed = sha256_file(source_path)
        if observed != expected:
            raise RuntimeError(
                f"Bound auxiliary source changed after validation: {relative}; "
                f"observed={observed}, expected={expected}."
            )


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    identity = hashlib.sha256()
    identity.update(memoryview(contiguous).cast("B"))
    return identity.hexdigest()


def sha256_h5_dataset(dataset: h5py.Dataset) -> str:
    identity = hashlib.sha256()
    for selection in dataset.iter_chunks():
        chunk = np.ascontiguousarray(dataset[selection])
        identity.update(memoryview(chunk).cast("B"))
    return identity.hexdigest()


def check_legacy_pair0_reference(
    root: Path,
    counts: dict[int, np.ndarray],
    stats: dict,
) -> None:
    """Require pair 0 to reproduce the transferred legacy products when available."""
    legacy_summary = root / OUT_RANDOM_DIR / LEGACY_RANDOM_SUMMARY_NAME
    legacy_map = root / OUT_RANDOM_DIR / LEGACY_RANDOM_MAP_NAME
    present = (legacy_summary.exists(), legacy_map.exists())
    stats["legacy_pair0_reference_checked"] = int(all(present))
    if not any(present):
        return
    if not all(present):
        raise RuntimeError(
            f"Partial legacy pair-0 reference: summary={present[0]}, map={present[1]}."
        )

    with h5py.File(legacy_summary, "r") as reference:
        expected = {
            "n_source_rows": int(reference.attrs["n_source_rows"]),
            "n_after_initial_geometry_cuts": int(
                reference.attrs["n_after_initial_geometry_cuts"]
            ),
            "n_quality_cut_randoms": int(reference.attrs["n_quality_cut_randoms"]),
        }
    observed = {key: int(stats[key]) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"Pair-0 cut-count null failed: observed={observed}, expected={expected}."
        )

    with h5py.File(legacy_map, "r") as reference:
        if reference.attrs.get("ordering", "RING") != "RING":
            raise RuntimeError(f"Legacy pair-0 map is not RING ordered: {legacy_map}")
        for nside in (1024, 4096):
            dataset = reference[f"nside{nside}/random_count"]
            if dataset.shape != counts[nside].shape:
                raise RuntimeError(
                    f"Legacy nside={nside} shape {dataset.shape} != {counts[nside].shape}."
                )
            equal = True
            for selection in dataset.iter_chunks():
                if not np.array_equal(dataset[selection], counts[nside][selection]):
                    equal = False
                    break
            stats[f"legacy_pair0_nside{nside}_bitwise_equal"] = int(equal)
            if not equal:
                raise RuntimeError(f"Pair-0 nside={nside} map null failed against {legacy_map}.")


def random_output_paths(root: Path, inputs: ValidatedRandomInputs) -> tuple[Path, Path]:
    index_label = "-".join(str(pair.spec.index) for pair in inputs.pairs)
    identity_short = inputs.input_identity_sha256[:12]
    prefix = f"desi_dr9_randoms_i{index_label}_{identity_short}_lrg_quality"
    summary_path = root / OUT_RANDOM_DIR / f"{prefix}_provenance.h5"
    map_path = (
        root
        / OUT_RANDOM_DIR
        / f"{prefix}_count_maps_nside1024_2048_4096.h5"
    )
    return summary_path, map_path


def _existing_random_products_match(
    summary_path: Path,
    map_path: Path,
    inputs: ValidatedRandomInputs,
) -> bool:
    present = (summary_path.exists(), map_path.exists())
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError(
            f"Partial random product exists for identity {inputs.input_identity_sha256}: "
            f"summary={present[0]}, map={present[1]}. Inspect it before rebuilding."
        )
    expected_indices = json.dumps([pair.spec.index for pair in inputs.pairs])
    expected_index_array = np.asarray(
        [pair.spec.index for pair in inputs.pairs], dtype=np.int32
    )
    expected_n_realizations = len(inputs.pairs)
    expected_cuts = json.dumps(random_cuts_definition(), sort_keys=True)
    expected_full_sha256 = json.dumps(
        inputs.full_source_sha256_by_relative_path, sort_keys=True
    )
    try:
        with h5py.File(summary_path, "r") as summary:
            if summary.attrs.get("schema_version", "") != RANDOM_PRODUCT_SCHEMA_VERSION:
                raise ValueError("summary schema_version mismatch")
            if summary.attrs.get("input_identity_sha256", "") != inputs.input_identity_sha256:
                raise ValueError("summary input identity mismatch")
            if summary.attrs.get("random_indices_json", "") != expected_indices:
                raise ValueError("summary random-index mismatch")
            if int(summary.attrs.get("n_random_realizations", -1)) != expected_n_realizations:
                raise ValueError("summary realization-count mismatch")
            if not np.array_equal(
                np.asarray(summary.attrs.get("random_realization_indices", [])),
                expected_index_array,
            ):
                raise ValueError("summary realization-index array mismatch")
            if summary.attrs.get("cuts_json", "") != expected_cuts:
                raise ValueError("summary cut-definition mismatch")
            if int(summary.attrs.get("full_source_sha256_verified", 0)) != 1:
                raise ValueError("summary does not attest full source SHA256 verification")
            if summary.attrs.get("sha256_ledger_sha256", "") != inputs.sha256_ledger_sha256:
                raise ValueError("summary SHA256-ledger identity mismatch")
            if summary.attrs.get("full_source_sha256_json", "") != expected_full_sha256:
                raise ValueError("summary full-source SHA256 inventory mismatch")
            for pair in inputs.pairs:
                group_name = f"pairs/index_{pair.spec.index}"
                if group_name not in summary:
                    raise ValueError(f"summary lacks {group_name}")
                group = summary[group_name]
                if group.attrs.get("random_sha256", "") != pair.random_sha256:
                    raise ValueError(f"summary {group_name} random SHA256 mismatch")
                if group.attrs.get("lrgmask_sha256", "") != pair.lrgmask_sha256:
                    raise ValueError(f"summary {group_name} LRG-mask SHA256 mismatch")
        with h5py.File(map_path, "r") as maps:
            if maps.attrs.get("schema_version", "") != RANDOM_PRODUCT_SCHEMA_VERSION:
                raise ValueError("map schema_version mismatch")
            if maps.attrs.get("input_identity_sha256", "") != inputs.input_identity_sha256:
                raise ValueError("map input identity mismatch")
            if maps.attrs.get("random_indices_json", "") != expected_indices:
                raise ValueError("map random-index mismatch")
            if int(maps.attrs.get("n_random_realizations", -1)) != expected_n_realizations:
                raise ValueError("map realization-count mismatch")
            if int(maps.attrs.get("random_realization_count", -1)) != expected_n_realizations:
                raise ValueError("map realization-count alias mismatch")
            if not np.array_equal(
                np.asarray(maps.attrs.get("random_realization_indices", [])),
                expected_index_array,
            ):
                raise ValueError("map realization-index array mismatch")
            if maps.attrs.get("cuts_json", "") != expected_cuts:
                raise ValueError("map cut-definition mismatch")
            if int(maps.attrs.get("full_source_sha256_verified", 0)) != 1:
                raise ValueError("map does not attest full source SHA256 verification")
            if maps.attrs.get("sha256_ledger_sha256", "") != inputs.sha256_ledger_sha256:
                raise ValueError("map SHA256-ledger identity mismatch")
            if maps.attrs.get("full_source_sha256_json", "") != expected_full_sha256:
                raise ValueError("map full-source SHA256 inventory mismatch")
            for nside in RANDOM_MAP_NSIDES:
                dataset = f"nside{nside}/random_count"
                if dataset not in maps:
                    raise ValueError(f"map lacks {dataset}")
                if maps[dataset].shape != (hp.nside2npix(nside),):
                    raise ValueError(f"{dataset} shape mismatch")
                if maps[dataset].dtype != np.dtype("uint32"):
                    raise ValueError(f"{dataset} dtype mismatch")
                expected_sha256 = maps[f"nside{nside}"].attrs.get(
                    "random_count_sha256", ""
                )
                if not expected_sha256:
                    raise ValueError(f"{dataset} lacks random_count_sha256")
                if sha256_h5_dataset(maps[dataset]) != expected_sha256:
                    raise ValueError(f"{dataset} content checksum mismatch")
    except Exception as error:
        raise RuntimeError(
            f"Existing random product does not match its encoded identity: {error}"
        ) from error
    return True


def load_stardens_values(path: Path) -> np.ndarray:
    with fitsio.FITS(str(path)) as hdul:
        table = hdul[1].read(columns=["HPXPIXEL", "STARDENS"])
    values = np.empty(hp.nside2npix(STARDENS_NSIDE), dtype=np.float32)
    values[np.asarray(table["HPXPIXEL"], dtype=np.int64)] = np.asarray(
        table["STARDENS"], dtype=np.float32
    )
    return values


def _write_random_summary(
    path: Path,
    root: Path,
    inputs: ValidatedRandomInputs,
    pair_stats: list[dict],
    total_stats: dict,
) -> None:
    with h5py.File(path, "w", track_order=True) as out:
        set_common_attrs(out, "DESI DR9 Extended LRG random-mask provenance summary")
        out.attrs["schema_version"] = RANDOM_PRODUCT_SCHEMA_VERSION
        out.attrs["input_identity_sha256"] = inputs.input_identity_sha256
        out.attrs["identity_method"] = (
            "Full-file SHA256 ledger verified before build, plus FITS schema, file size, "
            "row count, and 33 evenly spaced paired rows"
        )
        out.attrs["sha256_ledger_path"] = package_relative(
            inputs.sha256_ledger_path, root
        )
        out.attrs["sha256_ledger_sha256"] = inputs.sha256_ledger_sha256
        out.attrs["full_source_sha256_verified"] = int(
            inputs.full_source_sha256_verified
        )
        out.attrs["full_source_sha256_json"] = json.dumps(
            inputs.full_source_sha256_by_relative_path, sort_keys=True
        )
        out.attrs["random_indices_json"] = json.dumps(
            [pair.spec.index for pair in inputs.pairs]
        )
        out.attrs["random_realization_indices"] = np.asarray(
            [pair.spec.index for pair in inputs.pairs], dtype=np.int32
        )
        out.attrs["n_random_realizations"] = len(inputs.pairs)
        out.attrs["random_realization_count"] = len(inputs.pairs)
        out.attrs["random_map_nsides_json"] = json.dumps(list(RANDOM_MAP_NSIDES))
        out.attrs["cuts_json"] = json.dumps(random_cuts_definition(), sort_keys=True)
        out.attrs["randoms_quality_cuts_source"] = (
            str(inputs.cuts_source_path) if inputs.cuts_source_path is not None else ""
        )
        out.attrs["stardens_source"] = str(inputs.stardens_path)
        out.attrs["stardens_identity_sha256"] = inputs.stardens_identity_sha256
        out.attrs["stardens_size_bytes"] = inputs.stardens_size_bytes
        out.attrs["stardens_mtime_ns"] = inputs.stardens_mtime_ns
        for key, value in total_stats.items():
            out.attrs[key] = int(value)

        pairs_group = out.create_group("pairs")
        validation_by_index = {pair.spec.index: pair for pair in inputs.pairs}
        for stats in pair_stats:
            pair = validation_by_index[stats["index"]]
            group = pairs_group.create_group(f"index_{pair.spec.index}")
            group.attrs["index"] = pair.spec.index
            group.attrs["random_source_path"] = package_relative(pair.spec.random_path, root)
            group.attrs["lrgmask_source_path"] = package_relative(pair.spec.lrgmask_path, root)
            group.attrs["n_source_rows"] = pair.n_rows
            group.attrs["random_size_bytes"] = pair.random_size_bytes
            group.attrs["random_mtime_ns"] = pair.random_mtime_ns
            group.attrs["lrgmask_size_bytes"] = pair.lrgmask_size_bytes
            group.attrs["lrgmask_mtime_ns"] = pair.lrgmask_mtime_ns
            group.attrs["sample_identity_sha256"] = pair.sample_identity_sha256
            group.attrs["random_sha256"] = pair.random_sha256
            group.attrs["lrgmask_sha256"] = pair.lrgmask_sha256
            for key, value in stats.items():
                if key != "index":
                    group.attrs[key] = value


def _write_random_maps(
    path: Path,
    root: Path,
    summary_path: Path,
    inputs: ValidatedRandomInputs,
    counts: dict[int, np.ndarray],
    n_quality: int,
) -> None:
    with h5py.File(path, "w", track_order=True) as h5:
        set_common_attrs(h5, "DESI DR9 quality-cut random-count HEALPix maps")
        h5.attrs["schema_version"] = RANDOM_PRODUCT_SCHEMA_VERSION
        h5.attrs["input_identity_sha256"] = inputs.input_identity_sha256
        h5.attrs["identity_method"] = (
            "Full-file SHA256 ledger verified before build, plus structural/sample validation"
        )
        h5.attrs["sha256_ledger_path"] = package_relative(
            inputs.sha256_ledger_path, root
        )
        h5.attrs["sha256_ledger_sha256"] = inputs.sha256_ledger_sha256
        h5.attrs["full_source_sha256_verified"] = int(
            inputs.full_source_sha256_verified
        )
        h5.attrs["full_source_sha256_json"] = json.dumps(
            inputs.full_source_sha256_by_relative_path, sort_keys=True
        )
        h5.attrs["random_indices_json"] = json.dumps(
            [pair.spec.index for pair in inputs.pairs]
        )
        h5.attrs["random_realization_indices"] = np.asarray(
            [pair.spec.index for pair in inputs.pairs], dtype=np.int32
        )
        h5.attrs["n_random_realizations"] = len(inputs.pairs)
        h5.attrs["random_realization_count"] = len(inputs.pairs)
        h5.attrs["cuts_json"] = json.dumps(random_cuts_definition(), sort_keys=True)
        h5.attrs["randoms_hdf5"] = package_relative(summary_path, root)
        h5.attrs["ordering"] = "RING"
        h5.attrs["coordinate_precision"] = (
            "RA/DEC cast to float32 after cuts and before ang2pix, preserving legacy builder"
        )
        h5.attrs["n_quality_cut_randoms"] = int(n_quality)
        for nside in RANDOM_MAP_NSIDES:
            array = counts[nside]
            count_sum = int(np.sum(array, dtype=np.uint64))
            if count_sum != n_quality:
                raise RuntimeError(
                    f"nside={nside} map sum {count_sum} != quality count {n_quality}."
                )
            group = h5.create_group(f"nside{nside}")
            group.attrs["nside"] = nside
            group.attrs["count_sum"] = count_sum
            group.attrs["n_nonzero_pixels"] = int(np.count_nonzero(array))
            group.attrs["max_count"] = int(np.max(array))
            group.attrs["random_count_sha256"] = sha256_array(array)
            group.create_dataset("random_count", data=array, **h5_kwargs("u4"))


def process_randoms(
    root: Path,
    args,
    inputs: ValidatedRandomInputs,
) -> tuple[Path, Path]:
    summary_path, map_path = random_output_paths(root, inputs)
    existing_match = _existing_random_products_match(summary_path, map_path, inputs)
    if existing_match and not args.force:
        log(f"Skipping validated existing DR9 random products: {summary_path}, {map_path}")
        return summary_path, map_path
    if not inputs.full_source_sha256_verified:
        raise RuntimeError(
            "Refusing to build a new v3 random-mask product without full source "
            "verification; rerun with --verify-full-sha256."
        )
    assert_aux_sources_unchanged(inputs)

    tmp_summary = atomic_path(summary_path)
    tmp_map = atomic_path(map_path)
    stale_temps = [path for path in (tmp_summary, tmp_map) if path.exists()]
    if stale_temps:
        raise RuntimeError(f"Stale temporary products require inspection: {stale_temps}")

    stardens_values = load_stardens_values(inputs.stardens_path)
    bad_stardens_pixels = np.flatnonzero(stardens_values >= MAX_STARDENS).astype(np.int64)
    counts = {
        nside: np.zeros(hp.nside2npix(nside), dtype=np.uint32)
        for nside in RANDOM_MAP_NSIDES
    }
    pair_stats = []
    total_stats = {
        "n_source_rows": 0,
        "n_after_initial_geometry_cuts": 0,
        "n_after_ngc_island_cut": 0,
        "n_after_lrg_nobs_cut": 0,
        "n_after_lrg_mask_cut": 0,
        "n_after_ebv_cut": 0,
        "n_quality_cut_randoms": 0,
    }
    previous_support = {nside: 0 for nside in RANDOM_MAP_NSIDES}
    previous_count_sum = {nside: 0 for nside in RANDOM_MAP_NSIDES}

    for pair in inputs.pairs:
        assert_pair_sources_unchanged(pair)
        stats = {"index": pair.spec.index, **{key: 0 for key in total_stats}}
        log(
            f"Streaming DR9 random pair index={pair.spec.index}: "
            f"{pair.spec.random_path.name} + {pair.spec.lrgmask_path.name}"
        )
        with fitsio.FITS(str(pair.spec.random_path)) as random_hdul, fitsio.FITS(
            str(pair.spec.lrgmask_path)
        ) as mask_hdul:
            random_table = random_hdul[1]
            mask_table = mask_hdul[1]
            for start in range(0, pair.n_rows, args.random_chunk_size):
                stop = min(start + args.random_chunk_size, pair.n_rows)
                rows = range(start, stop)
                chunk = random_table.read(rows=rows, columns=list(RANDOM_REQUIRED_COLUMNS))
                lrg_mask = np.asarray(
                    mask_table.read(rows=rows, columns=["lrg_mask"])["lrg_mask"],
                    dtype=np.uint8,
                )
                ra = np.asarray(chunk["RA"], dtype=np.float64)
                dec = np.asarray(chunk["DEC"], dtype=np.float64)
                ebv = np.asarray(chunk["EBV"], dtype=np.float32)
                nobs_g = np.asarray(chunk["NOBS_G"], dtype=np.int16)
                nobs_r = np.asarray(chunk["NOBS_R"], dtype=np.int16)
                nobs_z = np.asarray(chunk["NOBS_Z"], dtype=np.int16)
                maskbits = np.asarray(chunk["MASKBITS"], dtype=np.int16)

                good = (
                    (nobs_g >= MIN_TARGET_NOBS)
                    & (nobs_r >= MIN_TARGET_NOBS)
                    & (nobs_z >= MIN_TARGET_NOBS)
                )
                for bit in RANDOM_INITIAL_MASKBITS:
                    good &= (maskbits & (2**bit)) == 0
                stats["n_after_initial_geometry_cuts"] += int(np.count_nonzero(good))

                good &= ~((dec < -10.5) & (ra > 120.0) & (ra < 260.0))
                stats["n_after_ngc_island_cut"] += int(np.count_nonzero(good))
                good &= (
                    (nobs_g >= MIN_LRG_NOBS)
                    & (nobs_r >= MIN_LRG_NOBS)
                    & (nobs_z >= MIN_LRG_NOBS)
                )
                stats["n_after_lrg_nobs_cut"] += int(np.count_nonzero(good))
                good &= lrg_mask == 0
                stats["n_after_lrg_mask_cut"] += int(np.count_nonzero(good))
                good &= ebv < MAX_EBV
                stats["n_after_ebv_cut"] += int(np.count_nonzero(good))
                pix64 = hp.ang2pix(STARDENS_NSIDE, ra, dec, lonlat=True, nest=False)
                good &= ~np.isin(pix64, bad_stardens_pixels)
                n_good = int(np.count_nonzero(good))
                stats["n_quality_cut_randoms"] += n_good
                stats["n_source_rows"] += stop - start

                if n_good:
                    # Preserve the legacy map builder's coordinate precision so pair 0
                    # remains an exact null control against the transferred product.
                    ra_good = ra[good].astype(np.float32)
                    dec_good = dec[good].astype(np.float32)
                    for nside in RANDOM_MAP_NSIDES:
                        pix = hp.ang2pix(
                            nside,
                            ra_good,
                            dec_good,
                            lonlat=True,
                            nest=False,
                        )
                        add_counts(counts[nside], pix)

                if stop == pair.n_rows or (stop // args.random_chunk_size) % 10 == 0:
                    log(
                        f"  pair {pair.spec.index} rows {stop:,}/{pair.n_rows:,}; "
                        f"quality kept so far {stats['n_quality_cut_randoms']:,}"
                    )

        if stats["n_source_rows"] != pair.n_rows:
            raise RuntimeError(
                f"Pair {pair.spec.index} processed {stats['n_source_rows']} rows; "
                f"expected {pair.n_rows}."
            )
        assert_pair_sources_unchanged(pair)
        expected_cumulative_count = (
            total_stats["n_quality_cut_randoms"] + stats["n_quality_cut_randoms"]
        )
        for nside, array in counts.items():
            support = int(np.count_nonzero(array))
            count_sum = int(np.sum(array, dtype=np.uint64))
            if support < previous_support[nside]:
                raise RuntimeError(
                    f"nside={nside} cumulative support decreased after pair "
                    f"{pair.spec.index}: {support} < {previous_support[nside]}."
                )
            if count_sum < previous_count_sum[nside]:
                raise RuntimeError(
                    f"nside={nside} cumulative count decreased after pair "
                    f"{pair.spec.index}: {count_sum} < {previous_count_sum[nside]}."
                )
            if count_sum != expected_cumulative_count:
                raise RuntimeError(
                    f"nside={nside} cumulative sum {count_sum} != expected "
                    f"{expected_cumulative_count} after pair {pair.spec.index}."
                )
            stats[f"cumulative_nside{nside}_nonzero_pixels"] = support
            stats[f"cumulative_nside{nside}_count_sum"] = count_sum
            stats[f"cumulative_nside{nside}_mean_count_nonzero"] = (
                float(count_sum / support) if support else 0.0
            )
            previous_support[nside] = support
            previous_count_sum[nside] = count_sum
        if pair.spec.index == 0:
            if pair_stats:
                raise RuntimeError("Pair 0 must be processed first for the legacy null control.")
            check_legacy_pair0_reference(root, counts, stats)
        pair_stats.append(stats)
        for key in total_stats:
            total_stats[key] += stats[key]

    n_quality = total_stats["n_quality_cut_randoms"]
    for nside, array in counts.items():
        count_sum = int(np.sum(array, dtype=np.uint64))
        if count_sum != n_quality:
            raise RuntimeError(
                f"nside={nside} accumulated sum {count_sum} != quality count {n_quality}."
            )
    assert_aux_sources_unchanged(inputs)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_random_summary(tmp_summary, root, inputs, pair_stats, total_stats)
        _write_random_maps(tmp_map, root, summary_path, inputs, counts, n_quality)
        os.replace(tmp_summary, summary_path)
        os.replace(tmp_map, map_path)
    except Exception:
        for path in (tmp_summary, tmp_map):
            if path.exists():
                path.unlink()
        raise
    log(
        f"Finished {len(inputs.pairs)}-pair random products: "
        f"n_quality={n_quality:,}, identity={inputs.input_identity_sha256}"
    )
    return summary_path, map_path


def make_quicklooks(root: Path, combined_path: Path, force: bool) -> None:
    out_nz = root / OUT_FIG_DIR / "desi_dr9_extended_nz_by_pz.png"
    out_weight = root / OUT_FIG_DIR / "desi_dr9_extended_imaging_weight_hist.png"
    if out_nz.exists() and out_weight.exists() and not force:
        return
    with h5py.File(combined_path, "r") as h5:
        z = h5["catalog/z"][:]
        pz = h5["catalog/pz_bin"][:]
        valid = h5["catalog/valid_for_cl"][:].astype(bool)
        w = h5["catalog/weight_imaging_mean1"][:]

    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(0.35, 1.05, 71)
    for b in range(1, 5):
        m = valid & (pz == b)
        plt.hist(z[m], bins=bins, histtype="step", linewidth=1.5, label=f"pz{b}")
    plt.xlabel("z")
    plt.ylabel("valid_for_cl galaxy count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_nz, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(w[valid & np.isfinite(w)], bins=100, histtype="stepfilled", alpha=0.75)
    plt.xlabel("DR9 imaging weight, mean-normalized per bin")
    plt.ylabel("valid_for_cl galaxy count")
    plt.tight_layout()
    plt.savefig(out_weight, dpi=160)
    plt.close()


def write_doc(root: Path, paths: dict) -> Path:
    doc_path = root / OUT_DOC_DIR / "DESI_DR9_EXTENDED_LRG_PRODUCTS.md"
    text = f"""# DESI DR9 Extended LRG Products

This directory documents the DR9 Extended LRG products prepared for harmonic
space galaxy auto and cross correlations.

## Products

Galaxy/velocity catalogs:

```text
data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_pz1_compact_with_weights.h5
data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_pz2_compact_with_weights.h5
data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_pz3_compact_with_weights.h5
data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_pz4_compact_with_weights.h5
data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_all_pz_compact_with_weights.h5
```

Random products:

```text
{package_relative(paths['randoms'], root)}
{package_relative(paths['random_maps'], root)}
```

## Galaxy Catalog Contents

Each catalog stores the kSZ velocity columns:

```text
catalog/ra_deg
catalog/dec_deg
catalog/z
catalog/v_los_km_s
catalog/vr_over_c
catalog/mass_msun
```

and public DR9 LRG metadata joined by exact sky-position/redshift matching:

```text
catalog/targetid
catalog/ebv
catalog/pixel_nobs_g
catalog/pixel_nobs_r
catalog/pixel_nobs_z
catalog/maskbits
catalog/photsys
catalog/lrg_mask
catalog/stardens
catalog/weight_imaging
catalog/weight_imaging_no_ebv
catalog/weight_imaging_mean1
catalog/dr9_lrg_quality_cut
catalog/valid_for_cl
```

Use `catalog/valid_for_cl` as the default selection for galaxy auto and cross
spectra.  It requires a successful public DR9 match, the DR9 LRG quality cuts,
and a finite positive imaging weight.  `catalog/weight_imaging_mean1` is the
public precomputed imaging weight renormalized to mean 1 within each tomographic
bin among `valid_for_cl` rows.

## Random Cuts

The random product follows the public `randoms_quality_cuts.py` logic:

```text
Initial geometry:
  NOBS_G/R/Z >= 1
  MASKBITS bits 1, 12, and 13 are vetoed

LRG quality footprint:
  remove DEC < -10.5 and 120 < RA < 260 islands
  NOBS_G/R/Z >= 2
  lrg_mask == 0
  EBV < 0.15
  stellar density < 2500 using the nside=64 DR7.1 stardens map
```

Use `{package_relative(paths['random_maps'], root)}` to build NaMaster spin-0
galaxy masks.  Its filename and metadata encode the exact random-realization
set and sampled input identity; the companion provenance summary records the
per-realization cut counts.

## NERSC Provenance

These absolute paths describe source locations on NERSC only:

```text
{DR9_CATALOG_PATH}
{DR9_WEIGHT_PATH}
{DR9_WEIGHT_NO_EBV_PATH}
{DR9_RANDOM_SOURCE_PATH}
{DR9_RANDOM_LRGMASK_PATH}
{DR9_STARDENS_PATH}
/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz{{1,2,3,4}}/extended_catalog_allfoot_perbin_sigmaz0.0500.txt
```

The public catalog directory is also available at:

```text
https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/
```
"""
    doc_path.write_text(text)
    return doc_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--usecols-only",
        action="store_true",
        help="Load only needed ASCII columns with np.loadtxt(usecols=...). Default loads the full ASCII files to follow the original processing instructions.",
    )
    parser.add_argument("--skip-randoms", action="store_true")
    parser.add_argument("--skip-catalogs", action="store_true")
    parser.add_argument("--random-chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--random-source-root",
        type=Path,
        help=(
            "Local legacy-survey-0.49.0 root containing randoms/resolve and "
            "zhou-lrg-xcorr-2023-v1. Omit for the legacy NERSC pair-0 paths."
        ),
    )
    parser.add_argument(
        "--random-indices",
        nargs="+",
        help=(
            "Exact random indices, separated by spaces and/or commas. If omitted with "
            "--random-source-root, discover all complete pairs and reject unpaired files."
        ),
    )
    parser.add_argument(
        "--random-stardens-path",
        type=Path,
        help="Override the nside-64 RING stellar-density FITS path.",
    )
    parser.add_argument(
        "--random-cuts-path",
        type=Path,
        help="Override the Zhou randoms_quality_cuts.py provenance path.",
    )
    parser.add_argument(
        "--random-sha256-ledger",
        type=Path,
        help=(
            "sha256sum-compatible raw-source ledger. Defaults to "
            "<random-source-root>/SHA256SUMS.raw.txt."
        ),
    )
    parser.add_argument(
        "--verify-full-sha256",
        action="store_true",
        help=(
            "Stream every selected random, LRG mask, stellar-density map, and cut source "
            "and require its full SHA256 to match the ledger. Required to build a new "
            "v3 product; a validated existing v3 product may be reused without rehashing."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate pair discovery, FITS schemas/rows/samples and the full small "
            "stellar-density table, print JSON, and write nothing."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.transfer_root.resolve()
    if args.random_chunk_size <= 0:
        raise ValueError(f"--random-chunk-size must be positive; got {args.random_chunk_size}.")

    random_inputs = None
    if not args.skip_randoms or args.validate_only:
        requested_indices = parse_random_indices(args.random_indices)
        pair_specs, stardens_path, cuts_source_path = resolve_random_pair_specs(
            args.random_source_root,
            requested_indices,
        )
        if args.random_stardens_path is not None:
            stardens_path = args.random_stardens_path.resolve()
        if args.random_cuts_path is not None:
            cuts_source_path = args.random_cuts_path.resolve()
        if args.random_sha256_ledger is not None:
            sha256_ledger_path = args.random_sha256_ledger.resolve()
        elif args.random_source_root is not None:
            sha256_ledger_path = args.random_source_root.resolve() / "SHA256SUMS.raw.txt"
        else:
            raise ValueError(
                "The v3 random-mask contract requires --random-sha256-ledger when "
                "--random-source-root is omitted."
            )
        random_inputs = validate_random_inputs(
            pair_specs,
            stardens_path,
            cuts_source_path,
            sha256_ledger_path,
            verify_full_sha256=args.verify_full_sha256,
        )
        log(
            f"Validated random indices {[pair.spec.index for pair in random_inputs.pairs]} "
            f"with identity {random_inputs.input_identity_sha256}"
        )

    if args.validate_only:
        summary = random_validation_summary(random_inputs)
        summary_path, map_path = random_output_paths(root, random_inputs)
        summary["proposed_outputs"] = {
            "provenance_summary": str(summary_path),
            "count_maps": str(map_path),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    ensure_dirs(root)
    catalog_paths = []
    combined_path = root / OUT_CATALOG_DIR / "desi_dr9_extended_all_pz_compact_with_weights.h5"
    if not args.skip_catalogs:
        for path in (
            DR9_CATALOG_PATH,
            DR9_WEIGHT_PATH,
            DR9_WEIGHT_NO_EBV_PATH,
            DR9_STARDENS_PATH,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        for spec in BIN_SPECS:
            if not spec.path.exists():
                raise FileNotFoundError(spec.path)
        bad_stardens_pixels = load_stardens_bad_pixels()
        for spec in BIN_SPECS:
            catalog_paths.append(write_catalog_bin(root, spec, bad_stardens_pixels, args))
        combined_path = combine_catalogs(root, catalog_paths, args.force)
        make_quicklooks(root, combined_path, args.force)

    random_path = root / OUT_RANDOM_DIR / "desi_dr9_randoms_1_0_lrg_quality_cut_compact.h5"
    random_map_path = root / OUT_RANDOM_DIR / "desi_dr9_randoms_1_0_lrg_quality_count_maps_nside1024_4096.h5"
    if not args.skip_randoms:
        random_path, random_map_path = process_randoms(root, args, random_inputs)

    if not args.skip_catalogs:
        write_doc(
            root,
            {
                "combined": combined_path,
                "randoms": random_path,
                "random_maps": random_map_path,
            },
        )
    log("DR9 preparation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
