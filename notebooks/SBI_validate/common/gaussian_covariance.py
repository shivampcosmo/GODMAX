"""Gaussian covariance utilities for the SBI validation datavector."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from survey_defaults import (
    SO_NOISE_MODEL,
    SurveyDefaults,
    default_noise_dict,
    integer_mode_counts,
    so_noise_provenance,
)


FieldPair = Tuple[str, str]


TARGET_SPECTRA: Tuple[str, ...] = ("gg", "gy", "gtau", "gkappa")
FIELD_ALIAS = {"g": "g", "y": "y", "tau": "tau", "kappa": "kappa"}


def spectrum_to_fields(name: str) -> FieldPair:
    """Map a spectrum label to field names."""

    if name == "gg":
        return ("g", "g")
    if name == "gy":
        return ("g", "y")
    if name == "gtau":
        return ("g", "tau")
    if name == "gkappa":
        return ("g", "kappa")
    if name == "yy":
        return ("y", "y")
    if name == "tautau":
        return ("tau", "tau")
    if name == "kappakappa":
        return ("kappa", "kappa")
    if name == "ytau":
        return ("y", "tau")
    if name == "ykappa":
        return ("y", "kappa")
    if name == "taukappa":
        return ("tau", "kappa")
    raise KeyError(f"Unknown spectrum label: {name}")


def canonical_pair(a: str, b: str) -> str:
    """Return the canonical key used in the Cl dictionaries."""

    if a == b:
        if a == "g":
            return "gg"
        if a == "y":
            return "yy"
        if a == "tau":
            return "tautau"
        if a == "kappa":
            return "kappakappa"
    pairs = {
        frozenset(("g", "y")): "gy",
        frozenset(("g", "tau")): "gtau",
        frozenset(("g", "kappa")): "gkappa",
        frozenset(("y", "tau")): "ytau",
        frozenset(("y", "kappa")): "ykappa",
        frozenset(("tau", "kappa")): "taukappa",
    }
    key = pairs.get(frozenset((a, b)))
    if key is None:
        raise KeyError(f"No canonical key for field pair ({a}, {b})")
    return key


def get_cl(cl_signal: Mapping[str, np.ndarray], a: str, b: str) -> np.ndarray:
    return np.asarray(cl_signal[canonical_pair(a, b)], dtype=float)


def get_noise(noise: Mapping[str, np.ndarray], a: str, b: str) -> np.ndarray:
    if a == b:
        return np.asarray(noise.get(a, 0.0), dtype=float)
    ref = next(iter(noise.values()))
    return np.zeros_like(np.asarray(ref, dtype=float))


def cl_plus_noise(cl_signal: Mapping[str, np.ndarray],
                  noise: Mapping[str, np.ndarray],
                  a: str, b: str) -> np.ndarray:
    return get_cl(cl_signal, a, b) + get_noise(noise, a, b)


def effective_fsky(spec1: str, spec2: str, survey: SurveyDefaults) -> float:
    """Geometric-mean overlap sky fraction for two spectra."""

    overlaps = survey.overlap_fsky()
    return float(np.sqrt(overlaps[spec1] * overlaps[spec2]))


def build_datavector(cl_signal: Mapping[str, np.ndarray],
                     spectra_order: Sequence[str] = TARGET_SPECTRA) -> Tuple[np.ndarray, List[str]]:
    """Concatenate target Cl arrays and return element labels."""

    data_parts: List[np.ndarray] = []
    labels: List[str] = []
    nell = len(np.asarray(cl_signal[spectra_order[0]]))
    for spec in spectra_order:
        arr = np.asarray(cl_signal[spec], dtype=float)
        if len(arr) != nell:
            raise ValueError(f"Spectrum {spec} has length {len(arr)} but expected {nell}")
        data_parts.append(arr)
        labels.extend([f"{spec}[{i}]" for i in range(nell)])
    return np.concatenate(data_parts), labels


def build_gaussian_covariance(
    ell: np.ndarray,
    delta_ell: np.ndarray,
    cl_signal: Mapping[str, np.ndarray],
    nbar_gal_sr: float,
    survey: SurveyDefaults | None = None,
    spectra_order: Sequence[str] = TARGET_SPECTRA,
    noise_model: str = "legacy_effective",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """Build a full Gaussian covariance for the target spectra.

    The covariance is diagonal in ell bins but includes all cross-covariance
    blocks among the spectra in ``spectra_order``.
    """

    survey = survey or SurveyDefaults()
    ell = np.asarray(ell, dtype=float)
    delta_ell = np.asarray(delta_ell, dtype=float)
    if ell.shape != delta_ell.shape:
        raise ValueError("ell and delta_ell must have the same shape")

    noise = default_noise_dict(
        ell,
        nbar_gal_sr,
        survey=survey,
        noise_model=noise_model,
        delta_ell=delta_ell,
    )
    modes_per_bin = integer_mode_counts(ell, delta_ell)
    nell = len(ell)
    nspec = len(spectra_order)
    cov = np.zeros((nspec * nell, nspec * nell), dtype=float)

    for i, spec1 in enumerate(spectra_order):
        a, b = spectrum_to_fields(spec1)
        for j, spec2 in enumerate(spectra_order):
            c, d = spectrum_to_fields(spec2)
            fsky = effective_fsky(spec1, spec2, survey)
            denom = fsky * modes_per_bin
            term = (
                cl_plus_noise(cl_signal, noise, a, c)
                * cl_plus_noise(cl_signal, noise, b, d)
                + cl_plus_noise(cl_signal, noise, a, d)
                * cl_plus_noise(cl_signal, noise, b, c)
            )
            block_diag = term / np.clip(denom, 1.0e-30, np.inf)
            rows = slice(i * nell, (i + 1) * nell)
            cols = slice(j * nell, (j + 1) * nell)
            cov[rows, cols] = np.diag(block_diag)

    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    corr = cov / np.sqrt(np.clip(np.outer(diag, diag), 1.0e-300, np.inf))
    meta = {
        "spectra_order": list(spectra_order),
        "survey": survey.as_dict(),
        "overlap_fsky": survey.overlap_fsky(),
        "nbar_gal_sr": float(nbar_gal_sr),
        "full_sky_modes_per_bin": modes_per_bin.tolist(),
        "mode_count_policy": "exact sum of (2ell+1) over integer multipoles in each bin",
        "noise_model": noise_model,
        "noise_provenance": (
            so_noise_provenance() if noise_model == SO_NOISE_MODEL else {
                "model": "legacy_effective",
                "validation_status": "not validated for native-full SO covariance",
            }
        ),
    }
    return cov, corr, noise, meta


def covariance_quality_checks(cov: np.ndarray) -> Dict[str, float | bool]:
    """Return small numerical checks for notebook display."""

    cov = np.asarray(cov, dtype=float)
    eig = np.linalg.eigvalsh(cov)
    return {
        "finite": bool(np.all(np.isfinite(cov))),
        "symmetric_max_abs": float(np.max(np.abs(cov - cov.T))),
        "positive_diagonal": bool(np.all(np.diag(cov) > 0)),
        "min_eigenvalue": float(np.min(eig)),
        "max_eigenvalue": float(np.max(eig)),
        "condition_number": float(np.max(eig) / np.clip(np.min(eig), 1.0e-300, np.inf)),
    }


def regularize_covariance(cov: np.ndarray,
                          jitter_fraction: float = 1.0e-10) -> Tuple[np.ndarray, float]:
    """Return a positive-definite covariance and correlation-space jitter.

    Scaling first prevents one absolute floor from replacing the physically
    small diagonal blocks of a heterogeneous Cl data vector.
    """

    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    if not np.all(np.isfinite(cov)) or np.any(diag <= 0.0):
        raise ValueError("Covariance must be finite with a strictly positive diagonal")
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    corr = 0.5 * (corr + corr.T)
    jitter = 0.0
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eig_min = float(np.min(np.linalg.eigvalsh(corr)))
        jitter = max(float(jitter_fraction), -eig_min + float(jitter_fraction))
        corr = corr + np.eye(corr.shape[0]) * jitter
        np.linalg.cholesky(corr)
    return corr * np.outer(scale, scale), jitter


def invert_covariance(cov: np.ndarray, jitter_fraction: float = 1.0e-10) -> Tuple[np.ndarray, float]:
    """Return a precision matrix using correlation-scaled linear algebra."""

    cov, jitter = regularize_covariance(cov, jitter_fraction=jitter_fraction)
    scale = np.sqrt(np.diag(cov))
    corr = cov / np.outer(scale, scale)
    precision_corr = np.linalg.solve(corr, np.eye(corr.shape[0]))
    return precision_corr / np.outer(scale, scale), jitter
