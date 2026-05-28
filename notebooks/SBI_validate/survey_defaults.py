"""Survey and noise defaults for the first SBI validation step.

The numbers here are deliberately centralized so the fiducial datavector,
Gaussian covariance, and map-validation notebook use the same assumptions.
They are effective survey-level defaults, not a replacement for final
experiment-specific noise curves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)


@dataclass(frozen=True)
class SurveyDefaults:
    """Compact survey assumptions used by the validation products."""

    # Approximate sky fractions.
    fsky_so: float = 0.40
    fsky_g: float = 0.34
    fsky_k: float = 0.44

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


def default_noise_dict(ell: np.ndarray, nbar_gal_sr: float,
                       survey: SurveyDefaults | None = None) -> Dict[str, np.ndarray]:
    """Return auto-noise curves for the fields used in the covariance."""

    survey = survey or SurveyDefaults()
    return {
        "g": galaxy_shot_noise_from_nbar_sr(nbar_gal_sr, ell),
        "y": white_noise_from_arcmin(
            survey.delta_y_arcmin,
            ell,
            beam_fwhm_arcmin=survey.beam_fwhm_arcmin,
            deconvolved=True,
        ),
        "tau": white_noise_from_arcmin(
            survey.delta_tau_arcmin,
            ell,
            beam_fwhm_arcmin=survey.beam_fwhm_arcmin,
            deconvolved=True,
        ),
        "kappa": shape_noise_kappa(survey.sigma_e, survey.neff_lsst_arcmin2, ell),
    }

