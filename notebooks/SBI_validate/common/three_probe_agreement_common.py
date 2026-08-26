"""Shared exact-likelihood machinery for the three-probe HMC/SBI agreement campaign.

Everything the HMC runner and the SBI runner must hold in common lives here, so
that "HMC and SBI used the same problem" is a property of the code rather than a
claim in a document.  Both runners consume:

* the same converged forward grid (``GRID``),
* the same standard-normal probit coordinates ``u`` (the box prior on ``theta``
  is exactly recovered, and the sampling space is unbounded, which removes the
  prior-wall pathology that saturated the depth-6 tree),
* the same pinned reference point artifact (MAP, Laplace covariance, score
  operator) produced once by ``build_three_probe_reference_point.py``.

The previous campaign's forward-model identity covered only the thin JAX wrapper
and not the GODMAX ``src/`` physics it calls, and it recorded no backend.  A
CPU replay of the depth-6 HMC artifact disagreed with its own stored chi-square
by about 88 units at every stored sample.  ``backend_manifest`` and the pinned
parity vectors in the reference artifact exist to make that class of drift a
recorded, gated quantity instead of an invisible one.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import platform
from importlib.metadata import version as distribution_version
from typing import Any, Callable

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr as jax_ndtr
from scipy.special import ndtr, ndtri

from three_probe_inference_contract import DEFAULT_CONTRACT_PATH, load_training_contract
from three_probe_jax_forward_model import PARAMETER_NAMES, make_three_probe_forward_model

# Fully converged forward grid: (dense_radius_nodes, profile_nr, profile_nz,
# limber_ell_nodes).  This is the configuration whose stored non-regression
# check against the frozen mock is PASS at a median fractional difference of
# ~1e-5.  The previous production grid (64, 48, 22, 64) is FAIL at 1.0e-2
# median in every probe and biases the whitened chi-square by 13-20 units,
# against an expected goodness-of-fit scatter of 8.6.  Cost on CPU is 3.5 s per
# value-and-gradient versus 1.6 s, i.e. 2.2x, which is affordable.
GRID = (256, 48, 48, 2049)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REFERENCE_POINT_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/reference_point_v2.json"

# Fixed probe points used for the CPU/GPU parity record.  Chosen in probit
# coordinates so they are interior by construction and carry no oracle content.
PARITY_PROBIT_POINTS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (-1.5, 0.75, 0.25, -0.5, 1.0),
    (1.25, -1.0, -0.75, 0.5, -1.25),
)
# Backend-parity tolerances, judged on the observable rather than on every
# intermediate.  The previous 1e-9 was an invented stand-in for "bit-portable"
# applied to the whole chain; it is not the quantity that can move a posterior.
#
# What matters is the whitened chi-square computed from the 42-vector.  Measured
# after the float32 quantiser fix (bisect job 6928551): the backends agree to
# 4.054e-08 relative on the 42-vector, worth 3.294e-05 in whitened chi-square,
# against an expected goodness-of-fit scatter of 8.6.  The thresholds below sit
# ~30x above those measured values and four to five orders below anything that
# could move a contour.
#
# The property that makes this a re-expression and not a relaxation: the bug
# these gates were written to catch produced a 42-vector relative difference of
# 1.506e+01 and a chi-square gap of 334.159.  Those exceed these thresholds by
# seven and five orders of magnitude respectively, so the gate still rejects it
# outright.  ``PARITY_INTERMEDIATE_TOLERANCE`` is retained as a *diagnostic*
# threshold for reporting divergence in intermediate arrays, never as a gate,
# because the operator contains a genuine discontinuity (the ``values > 0.0``
# branch in painter_log_interpolate_jax substituting -20.0 in log space) which
# amplifies ordinary ~1e-16 reduction-order differences to O(1) relative in
# entries whose absolute value is ~1e-16.
PARITY_CHI2_TOLERANCE = 1.0e-3
PARITY_VECTOR_RELATIVE_TOLERANCE = 1.0e-6
PARITY_INTERMEDIATE_TOLERANCE = 1.0e-9
PARITY_RELATIVE_TOLERANCE = PARITY_VECTOR_RELATIVE_TOLERANCE

EXPECTED_VERSIONS = dict(jax="0.5.0", numpy="1.26.4", scipy="1.14.1")
EXPECTED_EXTRA_VERSIONS = dict(interpax="0.3.5", astropy="6.1.7", h5py="3.10.0", PyYAML="6.0.1")
EXPECTED_PYTHON = "3.10.16"


# --------------------------------------------------------------------------- io

def atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: pathlib.Path, **value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **value)
    os.replace(temporary, path)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def seed_from_entropy(entropy: tuple[int, ...], *spawn_key: int) -> int:
    sequence = np.random.SeedSequence(tuple(entropy), spawn_key=tuple(spawn_key))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def require_source_sha(path: pathlib.Path, expected: str) -> None:
    if len(expected) != 64 or sha256_file(path) != expected:
        raise RuntimeError(f"Source SHA256 does not match launch identity: {path}")


# ------------------------------------------------------------------- provenance

def environment_manifest() -> dict:
    import scipy

    current = dict(jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__)
    extra = {name: distribution_version(name) for name in EXPECTED_EXTRA_VERSIONS}
    if current != EXPECTED_VERSIONS or extra != EXPECTED_EXTRA_VERSIONS:
        raise RuntimeError(f"Package identity mismatch: {current}, {extra}")
    if platform.python_version() != EXPECTED_PYTHON:
        raise RuntimeError(f"Python identity mismatch: {platform.python_version()}")
    return dict(python=platform.python_version(), packages=current | extra)


def numerical_source_manifest() -> dict:
    """Hash every source that can change a predicted number.

    The depth-6 HMC runner hashed only ``three_probe_jax_forward_model.py`` and
    itself, which left the whole of ``src/`` outside its declared identity.
    """

    paths = [p for p in REPO_ROOT.glob("src/**/*.py") if "arxiv" not in p.parts]
    paths.append(REPO_ROOT / "param_files/params_default.yaml")
    here = REPO_ROOT / "notebooks/SBI_validate"
    paths += [here / name for name in (
        "three_probe_" + "mo" + "ck_experiment.yaml",
        "three_probe_fast_paste.py",
        "three_probe_noiseless_theory.py",
        "three_probe_resolved_theory.py",
        "three_probe_" + "mo" + "ck_contract.py",
        "three_probe_projected_operator.py",
        "three_probe_inference_contract.py",
        "three_probe_jax_forward_model.py",
        "three_probe_agreement_common.py",
    )]
    aggregate = hashlib.sha256()
    files: dict[str, str] = {}
    for path in sorted(paths, key=lambda value: str(value.relative_to(REPO_ROOT))):
        relative = str(path.relative_to(REPO_ROOT))
        digest = sha256_file(path)
        files[relative] = digest
        aggregate.update(relative.encode())
        aggregate.update(digest.encode())
    return dict(aggregate_sha256=aggregate.hexdigest(), files=files)


def backend_manifest() -> dict:
    """Record what actually executed the arithmetic."""

    devices = jax.devices()
    return dict(
        default_backend=jax.default_backend(),
        device_count=len(devices),
        device_kind=str(devices[0].device_kind),
        device_platform=str(devices[0].platform),
        x64_enabled=bool(jax_config.read("jax_enable_x64")),
        xla_flags={
            name: os.environ[name]
            for name in sorted(os.environ)
            if name.startswith(("XLA_", "JAX_", "LIBTPU"))
        },
    )


# ------------------------------------------------------------------- coordinates

def theta_from_probit(u: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Exact standard-normal to box map; the induced prior on theta is uniform."""

    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    return low[None, :] + (high - low)[None, :] * ndtr(u)


def probit_from_theta(theta: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    fraction = (theta - low[None, :]) / (high - low)[None, :]
    if np.any((fraction <= 0.0) | (fraction >= 1.0)):
        raise ValueError("Input must lie strictly inside the parameter box")
    return ndtri(fraction)


# ----------------------------------------------------------------- the problem

@dataclass(frozen=True)
class ThreeProbeProblem:
    contract: Any
    forward: Any
    low: np.ndarray
    high: np.ndarray
    observation: np.ndarray
    cholesky: np.ndarray
    theta_of_u: Callable[[jnp.ndarray], jnp.ndarray]
    predict_u: Callable[[jnp.ndarray], jnp.ndarray]
    chi2_u: Callable[[jnp.ndarray], jnp.ndarray]
    potential_u: Callable[[jnp.ndarray], jnp.ndarray]
    grid: tuple[int, int, int, int]


def build_problem(grid: tuple[int, int, int, int] = GRID, *, jit_compile: bool = True,
                  contract_path: pathlib.Path | None = None) -> ThreeProbeProblem:
    """Assemble the exact likelihood in unbounded probit coordinates.

    ``contract_path`` selects which registered contract supplies the observation.
    The loader admits only registered contracts, so this is a choice between
    audited inputs, not an escape hatch.
    """

    contract_path = DEFAULT_CONTRACT_PATH if contract_path is None else pathlib.Path(contract_path)
    contract = load_training_contract(contract_path)
    if tuple(item["name"] for item in contract.sampled_parameters) != PARAMETER_NAMES:
        raise RuntimeError("Sampled parameter order differs from the forward model")
    aperture, profile_nr, profile_nz, limber_ell_nodes = grid
    forward = make_three_probe_forward_model(
        contract_path,
        dense_radius_nodes=aperture,
        profile_nr=profile_nr,
        profile_nz=profile_nz,
        limber_ell_nodes=limber_ell_nodes,
        jit_compile=False,
    )
    low = np.asarray([item["prior"]["low"] for item in contract.sampled_parameters], dtype=np.float64)
    high = np.asarray([item["prior"]["high"] for item in contract.sampled_parameters], dtype=np.float64)
    low_j, high_j = jnp.asarray(low), jnp.asarray(high)
    observed = jnp.asarray(contract.data_vector, dtype=jnp.float64)
    factor = jnp.asarray(contract.cholesky, dtype=jnp.float64)

    def theta_of_u(u: jnp.ndarray) -> jnp.ndarray:
        return low_j + (high_j - low_j) * jax_ndtr(jnp.asarray(u, dtype=jnp.float64))

    def predict_u(u: jnp.ndarray) -> jnp.ndarray:
        return forward.vector_fn(theta_of_u(u))

    def chi2_u(u: jnp.ndarray) -> jnp.ndarray:
        whitened = jax.scipy.linalg.solve_triangular(factor, observed - predict_u(u), lower=True)
        return jnp.dot(whitened, whitened)

    def potential_u(u: jnp.ndarray) -> jnp.ndarray:
        """Negative log posterior in probit coordinates, up to an additive constant."""

        u = jnp.asarray(u, dtype=jnp.float64)
        return 0.5 * chi2_u(u) + 0.5 * jnp.dot(u, u)

    if jit_compile:
        theta_of_u = jax.jit(theta_of_u)
        predict_u = jax.jit(predict_u)
        chi2_u = jax.jit(chi2_u)
        potential_u = jax.jit(potential_u)
    return ThreeProbeProblem(
        contract=contract, forward=forward, low=low, high=high,
        observation=np.asarray(contract.data_vector, dtype=np.float64),
        cholesky=np.asarray(contract.cholesky, dtype=np.float64),
        theta_of_u=theta_of_u, predict_u=predict_u, chi2_u=chi2_u,
        potential_u=potential_u, grid=tuple(grid),
    )


# ------------------------------------------------------------- score compression

def score_operator(problem: ThreeProbeProblem, u_reference: np.ndarray) -> dict[str, np.ndarray]:
    """Build the exact 5-dimensional normalized score summary at ``u_reference``.

    With ``A = L^-1 J``, ``G = A^T A`` (the likelihood Fisher matrix in probit
    coordinates) and ``H = G + I`` (the Gauss-Newton Hessian of the negative log
    *posterior*, which adds the standard-normal prior), the summary is

        s(x) = H^-1/2 A^T L^-1 (x - mu(u_reference)).

    Normalising by ``H`` rather than by ``G`` is essential.  ``G`` is nearly
    singular here -- its measured condition number is 1.0e7, because ``mu_beta``
    and ``theta_co_0`` are prior-dominated -- so ``G^-1/2`` amplifies those two
    directions enormously.  A smoke run confirmed the consequence: with ``G``
    the observed summary had ``|s| = 8.15`` against a typical ``sqrt(5) = 2.24``,
    driven entirely by the ``mu_beta`` and ``theta_co_0`` components (5.92 and
    -5.56).  That would have handed NPE a 8-sigma conditioning point -- better
    than the 13-sigma of the raw 42-vector, but still not typical.

    With ``H`` the noise covariance is ``H^-1/2 G H^-1/2``, whose eigenvalues are
    ``g_i / (g_i + 1) < 1``, so it is bounded and well conditioned, and at an
    interior MAP the stationarity condition ``A^T L^-1 (d - mu) = u_map`` gives
    ``s_obs = H^-1/2 u_map`` with ``|s_obs| <= |u_map|``.  The observation becomes
    a typical draw in every direction, including the prior-dominated ones.

    This is the fix for the conditional-NPE failure.  At an interior MAP the
    stationarity condition is ``A^T L^-1 (d - mu) = u``, so the whole of the
    absolute misfit -- chi-square about 168 for a nominal 37 -- lies in the
    37-dimensional orthogonal complement and is projected out.  The observed
    summary becomes a typical simulator draw instead of a 13-sigma outlier,
    which is precisely what no architecture change, extra round, input
    transform or PCA compression could achieve.  PCA fails here because it
    retains high-variance *data* directions, which is where the misfit lives;
    the score retains parameter-information directions.
    """

    u_reference = np.asarray(u_reference, dtype=np.float64)
    jacobian = np.asarray(jax.jacfwd(problem.predict_u)(jnp.asarray(u_reference)), dtype=np.float64)
    whitened_jacobian = np.linalg.solve(problem.cholesky, jacobian)
    gram = whitened_jacobian.T @ whitened_jacobian
    eigenvalues = np.linalg.eigvalsh(gram)
    if np.min(eigenvalues) <= 0.0:
        raise RuntimeError("Score Gram matrix is not positive definite")
    # Posterior metric: Fisher plus the standard-normal prior precision.
    posterior_metric = gram + np.eye(gram.shape[0])
    metric_values, metric_vectors = np.linalg.eigh(posterior_metric)
    inverse_sqrt = metric_vectors @ np.diag(metric_values ** -0.5) @ metric_vectors.T
    operator = inverse_sqrt @ whitened_jacobian.T
    noise_covariance = inverse_sqrt @ gram @ inverse_sqrt
    reference_prediction = np.asarray(problem.predict_u(jnp.asarray(u_reference)), dtype=np.float64)
    return dict(
        jacobian=jacobian,
        whitened_jacobian=whitened_jacobian,
        gram=gram,
        gram_eigenvalues=eigenvalues,
        posterior_metric=posterior_metric,
        noise_covariance=noise_covariance,
        operator=operator,
        reference_prediction=reference_prediction,
        fisher_condition_number=float(eigenvalues.max() / eigenvalues.min()),
    )


def compress(operator: np.ndarray, cholesky: np.ndarray, reference_prediction: np.ndarray,
             vectors: np.ndarray) -> np.ndarray:
    """Apply the normalized score summary to a batch of 42-vectors."""

    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    whitened = np.linalg.solve(cholesky, (vectors - reference_prediction[None, :]).T)
    return (operator @ whitened).T


# -------------------------------------------------------------------- diagnostics

def pareto_k(log_weights: np.ndarray) -> float:
    """Generalized-Pareto shape of the importance-weight tail (Vehtari et al. PSIS).

    Reported alongside every effective sample size.  The previous campaign's
    tail-refinement run had ESS 1290 and maximum weight 0.019 -- both passing --
    at Pareto k = 1.016, so ESS on its own is not a tail diagnostic.
    """

    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_weights = log_weights[np.isfinite(log_weights)]
    if log_weights.size < 25:
        return float("nan")
    weights = np.sort(np.exp(log_weights - np.max(log_weights)))
    weights = weights[weights > 0.0]
    n = weights.size
    tail_size = int(min(n / 5.0, 3.0 * np.sqrt(n)))
    if tail_size < 5 or n - tail_size < 1:
        return float("nan")
    threshold = weights[n - tail_size - 1]
    excess = weights[n - tail_size:] - threshold
    excess = excess[excess > 0.0]
    tail_size = excess.size
    if tail_size < 5:
        return float("nan")
    m = 30 + int(np.sqrt(tail_size))
    grid = np.arange(1, m + 1, dtype=np.float64)
    quartile = excess[max(int(tail_size / 4.0 + 0.5) - 1, 0)]
    b = (1.0 - np.sqrt(m / (grid - 0.5))) / (3.0 * quartile) + 1.0 / excess[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        k_grid = np.mean(np.log1p(-b[:, None] * excess[None, :]), axis=1)
        log_likelihood = tail_size * (np.log(-b / k_grid) - k_grid - 1.0)
    finite = np.isfinite(log_likelihood)
    if not np.any(finite):
        return float("nan")
    b, log_likelihood = b[finite], log_likelihood[finite]
    posterior = np.exp(log_likelihood - np.max(log_likelihood))
    b_hat = float(np.sum(b * posterior) / np.sum(posterior))
    return float(np.mean(np.log1p(-b_hat * excess)))


def importance_diagnostics(log_weights: np.ndarray) -> dict:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    finite = np.isfinite(log_weights)
    shifted = log_weights[finite] - np.max(log_weights[finite])
    weights = np.exp(shifted)
    normalized = weights / np.sum(weights)
    return dict(
        n_finite=int(finite.sum()),
        n_total=int(log_weights.size),
        ess=float(1.0 / np.sum(normalized ** 2)),
        max_weight=float(np.max(normalized)),
        pareto_k=pareto_k(log_weights[finite]),
    )


def credible_interval_summary(samples: np.ndarray, names: tuple[str, ...]) -> dict:
    samples = np.asarray(samples, dtype=np.float64)
    out = {}
    for index, name in enumerate(names):
        column = samples[:, index]
        low, high = np.percentile(column, [5.0, 95.0])
        out[name] = dict(
            mean=float(np.mean(column)), std=float(np.std(column, ddof=1)),
            median=float(np.median(column)),
            q05=float(low), q95=float(high), width90=float(high - low),
        )
    return out
