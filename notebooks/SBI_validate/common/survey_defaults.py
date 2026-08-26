"""Survey and noise defaults for the first SBI validation step.

The numbers here are deliberately centralized so the fiducial datavector,
Gaussian covariance, and map-validation notebook use the same assumptions.
They are effective survey-level defaults, not a replacement for final
experiment-specific noise curves.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, Mapping

import numpy as np


ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)
SO_NOISE_DIR = pathlib.Path(__file__).resolve().parent / "noise_curves" / "simons_observatory"
SO_TSZ_NOISE_PATH = SO_NOISE_DIR / "SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"
SO_KAPPA_NOISE_PATH = (
    SO_NOISE_DIR
    / "nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat"
)
SO_TSZ_DEPROJ2_COLUMN = 3
SO_KAPPA_MV_COLUMN = 7
SO_NOISE_MODEL = "so_lat_v3_1_tsz_deproj2_cmbkappa_mv"
SO_NOISE_UPSTREAM_COMMIT = "fac881eb5ee012673d8994443caa3c6ad7fac2b6"
SO_NOISE_UPSTREAM_REPOSITORY = "https://github.com/simonsobs/so_noise_models"


@dataclass(frozen=True)
class SurveyDefaults:
    """Compact survey assumptions used by the validation products."""

    # Approximate sky fractions.
    fsky_so: float = 0.40
    fsky_g: float = 0.34
    fsky_k: float = 0.40

    # LSST-like weak-lensing shape noise.
    sigma_e: float = 0.26
    neff_lsst_arcmin2: float = 26.0

    # Effective SO-like map noise in y and tau map units per arcmin.
    # These are intentionally easy to change once a preferred SO noise curve
    # or kSZ optical-depth reconstruction forecast is chosen.
    delta_y_arcmin: float = 2.0e-6
    delta_tau_arcmin: float = 1.0e-5

    # The pasting config uses this beam for smoothed y/tau/kappa profiles.
    beam_fwhm_arcmin: float = 6.87

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def overlap_fsky(self) -> Dict[str, float]:
        """Return overlap sky fractions for the spectra used here."""

        return {
            "gg": self.fsky_g,
            "gy": min(self.fsky_g, self.fsky_so),
            "gtau": min(self.fsky_g, self.fsky_so),
            "gkappa": min(self.fsky_g, self.fsky_k),
            "yy": self.fsky_so,
            "tautau": self.fsky_so,
            "kappakappa": self.fsky_k,
            "ytau": self.fsky_so,
            "ykappa": min(self.fsky_so, self.fsky_k),
            "taukappa": min(self.fsky_so, self.fsky_k),
        }


def white_noise_from_arcmin(delta_arcmin: float, ell: np.ndarray,
                            beam_fwhm_arcmin: float = 0.0,
                            deconvolved: bool = True) -> np.ndarray:
    """Return white map noise N_l from an arcmin map-depth.

    Parameters
    ----------
    delta_arcmin
        White-noise amplitude in map units times arcmin.
    ell
        Multipole centers.
    beam_fwhm_arcmin
        Gaussian beam FWHM. If ``deconvolved`` is true, the returned noise is
        beam-deconvolved by multiplying by ``1 / B_l^2``.
    deconvolved
        Whether to return noise in deconvolved theory space.
    """

    ell = np.asarray(ell, dtype=float)
    noise = np.full_like(ell, (delta_arcmin * ARCMIN_TO_RAD) ** 2, dtype=float)
    if deconvolved and beam_fwhm_arcmin > 0:
        sigma = beam_fwhm_arcmin * ARCMIN_TO_RAD / np.sqrt(8.0 * np.log(2.0))
        beam = np.exp(-0.5 * ell * (ell + 1.0) * sigma ** 2)
        noise = noise / np.clip(beam ** 2, 1.0e-30, np.inf)
    return noise


def shape_noise_kappa(sigma_e: float, neff_arcmin2: float,
                      ell: np.ndarray) -> np.ndarray:
    """Return LSST-like convergence shape noise."""

    ell = np.asarray(ell, dtype=float)
    neff_rad2 = neff_arcmin2 / (ARCMIN_TO_RAD ** 2)
    return np.full_like(ell, sigma_e ** 2 / neff_rad2, dtype=float)


def galaxy_shot_noise_from_nbar_sr(nbar_sr: float, ell: np.ndarray) -> np.ndarray:
    """Return angular galaxy shot noise for an overdensity map."""

    ell = np.asarray(ell, dtype=float)
    if nbar_sr <= 0:
        return np.full_like(ell, np.inf, dtype=float)
    return np.full_like(ell, 1.0 / nbar_sr, dtype=float)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tabulated_noise_table(
    path: pathlib.Path,
    column: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one contiguous integer-ell noise table."""

    table = np.loadtxt(path)
    if table.ndim != 2 or table.shape[1] <= column:
        raise ValueError(f"Malformed {label} table {path}: shape={table.shape}")
    source_ell = np.asarray(table[:, 0], dtype=float)
    source_noise = np.asarray(table[:, column], dtype=float)
    if (
        not np.all(np.isfinite(table))
        or not np.all(np.diff(source_ell) > 0.0)
        or not np.array_equal(source_ell, np.arange(source_ell[0], source_ell[-1] + 1.0))
        or np.any(source_noise <= 0.0)
    ):
        raise ValueError(
            f"{label} table must be finite, positive, and contiguous at integer ell"
        )
    return source_ell, source_noise


def integer_ell_for_bin(center: float, width: float) -> np.ndarray:
    """Return integer multipoles in ``[center-width/2, center+width/2)``."""

    lower = float(center) - 0.5 * float(width)
    upper = float(center) + 0.5 * float(width)
    if not np.isfinite(lower + upper) or width <= 0.0:
        raise ValueError(f"Invalid ell bin center={center}, width={width}")
    values = np.arange(int(np.ceil(lower)), int(np.ceil(upper)), dtype=int)
    if values.size == 0:
        raise ValueError(f"Ell bin center={center}, width={width} contains no integer ell")
    return values


def integer_mode_counts(ell: np.ndarray, delta_ell: np.ndarray) -> np.ndarray:
    """Return exact full-sky mode counts, ``sum_L (2L+1)``, for each bin."""

    ell = np.asarray(ell, dtype=float)
    delta_ell = np.asarray(delta_ell, dtype=float)
    if ell.shape != delta_ell.shape:
        raise ValueError("ell and delta_ell must have the same shape")
    return np.asarray(
        [np.sum(2.0 * integer_ell_for_bin(c, w) + 1.0) for c, w in zip(ell, delta_ell)],
        dtype=float,
    )


def band_average_tabulated_noise(
    path: pathlib.Path,
    column: int,
    ell: np.ndarray,
    delta_ell: np.ndarray,
    label: str,
) -> np.ndarray:
    """Band-average integer-ell N_ell with mode-count weights.

    The SO files are evaluated only at their tabulated integer multipoles.  A
    bin is rejected unless every integer multipole in it is supported; neither
    interpolation nor extrapolation is used.
    """

    source_ell, source_noise = _load_tabulated_noise_table(path, column, label)
    lookup = {int(multipole): value for multipole, value in zip(source_ell, source_noise)}
    averaged = []
    for center, width in zip(np.asarray(ell, dtype=float), np.asarray(delta_ell, dtype=float)):
        multipoles = integer_ell_for_bin(center, width)
        if multipoles[0] < source_ell[0] or multipoles[-1] > source_ell[-1]:
            raise ValueError(
                f"Ell bin [{center - width / 2}, {center + width / 2}) lies outside "
                f"{label} support [{source_ell[0]}, {source_ell[-1]}]; "
                "partial bins and extrapolation are forbidden"
            )
        values = np.asarray([lookup[int(multipole)] for multipole in multipoles])
        weights = 2.0 * multipoles + 1.0
        averaged.append(float(np.average(values, weights=weights)))
    return np.asarray(averaged, dtype=float)


def so_noise_supported_bins(ell: np.ndarray, delta_ell: np.ndarray) -> np.ndarray:
    """Select bins whose complete integer-ell support exists in both SO tables."""

    support_min, support_max = so_noise_common_ell_support()
    return np.asarray(
        [
            integer_ell_for_bin(center, width)[0] >= support_min
            and integer_ell_for_bin(center, width)[-1] <= support_max
            for center, width in zip(ell, delta_ell)
        ],
        dtype=bool,
    )


def so_noise_common_ell_support() -> tuple[float, float]:
    """Return the intersection of the vendored SO y and kappa supports."""

    tsz_ell = np.loadtxt(SO_TSZ_NOISE_PATH, usecols=(0,))
    kappa_ell = np.loadtxt(SO_KAPPA_NOISE_PATH, usecols=(0,))
    return (
        float(max(tsz_ell[0], kappa_ell[0])),
        float(min(tsz_ell[-1], kappa_ell[-1])),
    )


def so_noise_provenance() -> Mapping[str, object]:
    """Return immutable source, column, support, and observable-basis metadata."""

    ell_min, ell_max = so_noise_common_ell_support()
    return {
        "model": SO_NOISE_MODEL,
        "upstream_repository": SO_NOISE_UPSTREAM_REPOSITORY,
        "upstream_commit": SO_NOISE_UPSTREAM_COMMIT,
        "observable_basis": "beam_deconvolved_sky_fields",
        "common_ell_support": [ell_min, ell_max],
        "bandpower_policy": (
            "direct integer-ell lookup and (2ell+1)-weighted average over each "
            "complete bin; no interpolation, extrapolation, or partial bins"
        ),
        "tsz": {
            "source": (
                f"{SO_NOISE_UPSTREAM_REPOSITORY}/blob/{SO_NOISE_UPSTREAM_COMMIT}/"
                "LAT_comp_sep_noise/v3.1.0/"
                "SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"
            ),
            "path": str(SO_TSZ_NOISE_PATH),
            "sha256": _sha256(SO_TSZ_NOISE_PATH),
            "column_zero_based": SO_TSZ_DEPROJ2_COLUMN,
            "column_label": "Deproj-2 (fiducial CIB SED deprojection)",
            "units": "dimensionless Compton-y N_ell",
        },
        "kappa": {
            "source": (
                f"{SO_NOISE_UPSTREAM_REPOSITORY}/blob/{SO_NOISE_UPSTREAM_COMMIT}/"
                "LAT_lensing_noise/lensing_v3_1_1/"
                "nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat"
            ),
            "path": str(SO_KAPPA_NOISE_PATH),
            "sha256": _sha256(SO_KAPPA_NOISE_PATH),
            "column_zero_based": SO_KAPPA_MV_COLUMN,
            "column_label": "N_lensing_MV (all), iterative, deproj0, SENS1",
            "units": "dimensionless convergence N_L",
        },
        "galaxy": {
            "model": "DESI-like Poisson shot noise",
            "formula": "N_ell^gg = 1 / nbar_gal_sr",
        },
        "tau": {
            "model": "legacy effective white tau map depth",
            "units": "dimensionless tau N_ell",
        },
    }


def default_noise_dict(ell: np.ndarray, nbar_gal_sr: float,
                       survey: SurveyDefaults | None = None,
                       noise_model: str = "legacy_effective",
                       delta_ell: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """Return auto-noise curves for the fields used in the covariance."""

    survey = survey or SurveyDefaults()
    if noise_model == SO_NOISE_MODEL:
        if delta_ell is None:
            raise ValueError("SO tabulated noise requires delta_ell for band averaging")
        y_noise = band_average_tabulated_noise(
            SO_TSZ_NOISE_PATH,
            SO_TSZ_DEPROJ2_COLUMN,
            ell,
            delta_ell,
            "SO LAT tSZ Deproj-2 noise",
        )
        kappa_noise = band_average_tabulated_noise(
            SO_KAPPA_NOISE_PATH,
            SO_KAPPA_MV_COLUMN,
            ell,
            delta_ell,
            "SO LAT CMB-lensing MV noise",
        )
    elif noise_model == "legacy_effective":
        y_noise = white_noise_from_arcmin(
            survey.delta_y_arcmin,
            ell,
            beam_fwhm_arcmin=survey.beam_fwhm_arcmin,
            deconvolved=True,
        )
        kappa_noise = shape_noise_kappa(
            survey.sigma_e, survey.neff_lsst_arcmin2, ell
        )
    else:
        raise ValueError(f"Unknown covariance noise model: {noise_model}")
    return {
        "g": galaxy_shot_noise_from_nbar_sr(nbar_gal_sr, ell),
        "y": y_noise,
        "tau": white_noise_from_arcmin(
            survey.delta_tau_arcmin,
            ell,
            beam_fwhm_arcmin=survey.beam_fwhm_arcmin,
            deconvolved=True,
        ),
        "kappa": kappa_noise,
    }
