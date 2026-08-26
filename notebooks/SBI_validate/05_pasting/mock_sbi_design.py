#!/usr/bin/env python3
"""Draw a mock-SBI paste design and write one fail-closed paste config per point.

The design is drawn IID from a frozen normalized mixture in probit coordinates and
never post-filtered.  Candidate ranking, minimum-separation rejection, rounding or
any post-draw selection would change the sampling density and invalidate the stored
``log q``, which is what a sequential NPE correction and any importance weight
depend on.  So the draws go straight from the mixture into the manifest, together
with their component label and their exact mixture log-density.

Each point gets its own paste config: a copy of the frozen experiment YAML with
only ``pasting.run_name`` and the gas-parameter override changed, plus
``require_gas_parameter_overrides: true`` so the run cannot silently fall back to
the ``params_default.yaml`` values -- which are the point the frozen mock was
painted at.  Run names are content-addressed by the canonical hash of the resolved
parameters, so two configs can never collide and one map can never be reused for
two different parameter points.
"""

from __future__ import annotations

# --- keep imports working from a theme subfolder: common/ holds the
# --- modules shared by more than one stage.
import pathlib as _pl, sys as _sys
_ROOT = _pl.Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "common"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import copy
import json
import os
import pathlib
import sys

import numpy as np
import yaml
from scipy.stats import multivariate_normal

THIS_DIR = pathlib.Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mock_sbi_common as msc

PARAMETER_NAMES = ("theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M")
PRIOR_LOW = np.asarray([0.5, 0.0, 0.005, 0.001, -1.0])
PRIOR_HIGH = np.asarray([8.0, 0.5, 1.5, 0.5, 1.0])

BASE_CONFIG = THIS_DIR / "three_probe_mock_experiment.yaml"

MIXTURE_RECIPES = {
    # The design that the Stage-0 oracle actually validated.  Use this one for the
    # campaign: the NLE result (0.050 sigma drift, 6.6% width at 512 points, pooled over
    # 4 seeds) was obtained with these exact weights and this broadening factor, and a
    # different design is a different experiment.  See
    # notebooks/SBI_validate/07_mock_sbi_snle/oracle_mock_selfconsistent_test.py::RECIPE.
    "oracle_validated": {"weights": {"guide": 0.55, "broadened": 0.25, "prior": 0.20},
                         "broaden_covariance_factor": 4.0},
    "defensive": {"weights": {"guide": 0.50, "broadened": 0.30, "prior": 0.20},
                  "broaden_covariance_factor": 4.0},
    "concentrated": {"weights": {"guide": 0.60, "broadened": 0.30, "prior": 0.10},
                     "broaden_covariance_factor": 2.25},
}
# Round-scoped namespaces so a later round can never reuse an earlier round's draws.
ROUND_ENTROPY = {1: (20260821, 201, 1), 2: (20260821, 201, 2), 3: (20260821, 201, 3),
                 4: (20260821, 201, 4),
                 "flatness": (20260821, 201, 0), "holdout": (20260821, 301, 0)}


def guide_from_samples(path: pathlib.Path, key: str = "u") -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of a posterior sample set, in probit coordinates.

    Used for the two guides that are sample sets rather than Laplace fits: the
    self-consistent theory-SBI posterior for round 1, and the previous round's mock
    posterior for rounds 2+.  Only the first two moments are taken, deliberately: the
    design mixture must have an analytic, normalized ``log q`` for every draw, because
    that density is what the NPE reweighting divides out.  A KDE or a resampled cloud
    has no such density and would silently invalidate the correction.
    """

    payload = np.load(path)
    if key not in payload.files:
        raise SystemExit(f"{path} has no '{key}' array (found {payload.files})")
    samples = np.asarray(payload[key], dtype=np.float64).reshape(-1, len(PARAMETER_NAMES))
    if samples.shape[0] < 100:
        raise SystemExit(f"{path} has only {samples.shape[0]} draws; too few for a covariance")
    return samples.mean(axis=0), np.cov(samples, rowvar=False)


def repo_relative(path: pathlib.Path) -> str:
    """Repo-relative when inside the tree, absolute otherwise (test output dirs)."""

    path = pathlib.Path(path).resolve()
    try:
        return str(path.relative_to(msc.REPO_ROOT))
    except ValueError:
        return str(path)


def theta_from_u(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * ndtr(np.asarray(u, dtype=np.float64))


def mixture_components(u_map, covariance, recipe):
    return [
        ("guide", recipe["weights"]["guide"], np.asarray(u_map), np.asarray(covariance)),
        ("broadened", recipe["weights"]["broadened"], np.asarray(u_map),
         recipe["broaden_covariance_factor"] * np.asarray(covariance)),
        ("prior", recipe["weights"]["prior"], np.zeros(len(u_map)), np.eye(len(u_map))),
    ]


def mixture_log_prob(u, u_map, covariance, recipe) -> np.ndarray:
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    total = np.zeros(u.shape[0], dtype=np.float64)
    for _, weight, mean, cov in mixture_components(u_map, covariance, recipe):
        total += weight * multivariate_normal(mean=mean, cov=cov).pdf(u)
    return np.log(total)


def draw_design(n: int, u_map, covariance, recipe, entropy):
    rng = np.random.default_rng(np.random.SeedSequence(tuple(int(v) for v in entropy)))
    components = mixture_components(u_map, covariance, recipe)
    names = [c[0] for c in components]
    weights = [c[1] for c in components]
    labels = rng.choice(names, size=n, p=weights)
    u = np.empty((n, len(u_map)), dtype=np.float64)
    for name, _, mean, cov in components:
        mask = labels == name
        if np.any(mask):
            u[mask] = rng.multivariate_normal(mean, cov, size=int(mask.sum()))
    return u, labels, mixture_log_prob(u, u_map, covariance, recipe)


def write_paste_config(theta: np.ndarray, *, output_dir: pathlib.Path, nside: int,
                       skip_galaxy_map: bool, num_splits: int,
                       run_name_suffix: str = "") -> dict:
    with BASE_CONFIG.open() as handle:
        config = yaml.safe_load(handle)
    override = {name: float(value) for name, value in zip(PARAMETER_NAMES, theta)}
    digest = msc.canonical_json_sha256(override)
    # Content-addressed by the resolved parameters, so one map can never be reused
    # for two different points.  The suffix exists only for deliberate variants of
    # the SAME point (a null test needs the galaxy-on and galaxy-off runs side by
    # side, and they would otherwise collide on one output directory).
    run_name = f"mocksbi_{digest[:16]}" + (f"_{run_name_suffix}" if run_name_suffix else "")

    paste = config["pasting"]
    paste["run_name"] = run_name
    paste["nside"] = int(nside)
    paste["num_splits"] = int(num_splits)
    paste["gas_parameter_overrides"] = override
    paste["require_gas_parameter_overrides"] = True
    if skip_galaxy_map:
        # The galaxy catalog is frozen and none of the five sampled parameters enters
        # the HOD, so the galaxy map is identical at every design point and already
        # exists.  Painting it again is 66% of the chunk loop for no information.
        paste["get_galmap"] = False

    path = output_dir / f"{run_name}.yaml"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(config, sort_keys=False))
    os.replace(tmp, path)
    return {"run_name": run_name, "config_path": repo_relative(path),
            "config_sha256": msc.sha256_file(path), "theta": override,
            "theta_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True,
                        help="1, 2, 3, 'flatness' or 'holdout'; selects the PRNG namespace")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--recipe", choices=sorted(MIXTURE_RECIPES), default="concentrated")
    parser.add_argument("--guide", type=pathlib.Path,
                        default=msc.REPO_ROOT / "data/SBI_validate/mock_sbi/transfer_and_guide.json")
    parser.add_argument("--guide-key", default="guide",
                        choices=("guide", "raw_theory_guide"),
                        help="'guide' is the transfer-corrected one; ignored when "
                             "--guide-samples is given")
    parser.add_argument("--guide-samples", type=pathlib.Path, default=None,
                        help="npz of posterior draws in probit coordinates to use as the "
                             "guide instead of --guide: the self-consistent theory-SBI "
                             "posterior for round 1, or the previous round's mock "
                             "posterior for a later round. Only its mean and covariance "
                             "are used, so log q stays analytic and normalized.")
    parser.add_argument("--guide-samples-key", default="u")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=msc.REPO_ROOT / "data/SBI_validate/mock_sbi/paste_plan")
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--num-splits", type=int, default=32)
    parser.add_argument("--keep-galaxy-map", action="store_true",
                        help="Paint the galaxy map too. Only needed if the Stage-1 "
                             "galaxy-skip null test fails.")
    parser.add_argument("--run-name-suffix", default="",
                        help="Distinguish deliberate variants of the same parameter point "
                             "(null tests). Never use it to paste the same point twice as "
                             "if it were two design points.")
    parser.add_argument("--include-reference-point", action="store_true",
                        help="Force the frozen paste's own parameters in as a forced, "
                             "non-importance-eligible anchor")
    args = parser.parse_args()

    key = int(args.round) if args.round.isdigit() else args.round
    if key not in ROUND_ENTROPY:
        raise SystemExit(f"--round must be one of {sorted(map(str, ROUND_ENTROPY))}")

    if args.guide_samples is not None:
        u_map, covariance = guide_from_samples(args.guide_samples, args.guide_samples_key)
        guide_description = {"kind": "posterior_samples",
                             "path": repo_relative(args.guide_samples),
                             "sha256": msc.sha256_file(args.guide_samples),
                             "key": args.guide_samples_key}
    else:
        guide_payload = json.loads(args.guide.read_text())
        u_map = np.asarray(guide_payload[args.guide_key]["u_map"], dtype=np.float64)
        covariance = np.asarray(guide_payload[args.guide_key]["covariance"], dtype=np.float64)
        guide_description = {"kind": "laplace_fit", "path": repo_relative(args.guide),
                             "sha256": msc.sha256_file(args.guide), "key": args.guide_key}
    recipe = MIXTURE_RECIPES[args.recipe]

    plan_dir = args.output_dir / f"round_{args.round}"
    plan_dir.mkdir(parents=True, exist_ok=True)

    u, labels, log_q = draw_design(args.count, u_map, covariance, recipe, ROUND_ENTROPY[key])
    theta = theta_from_u(u)
    if not np.all((theta > PRIOR_LOW) & (theta < PRIOR_HIGH)):
        raise RuntimeError("A drawn theta fell outside the prior box; the probit map is broken")

    entries = []
    for index in range(args.count):
        record = write_paste_config(theta[index], output_dir=plan_dir, nside=args.nside,
                                    skip_galaxy_map=not args.keep_galaxy_map,
                                    num_splits=args.num_splits,
                                    run_name_suffix=args.run_name_suffix)
        record.update({"index": index, "u": u[index].tolist(), "component": str(labels[index]),
                       "log_q": float(log_q[index]), "sampling_role": "iid_mixture_draw",
                       "importance_eligible": True})
        entries.append(record)

    if args.include_reference_point:
        with (msc.REPO_ROOT / "param_files/params_default.yaml").open() as handle:
            sim = yaml.safe_load(handle)["sim_params"]
        anchor = np.asarray([float(sim[name]) for name in PARAMETER_NAMES])
        record = write_paste_config(anchor, output_dir=plan_dir, nside=args.nside,
                                    skip_galaxy_map=not args.keep_galaxy_map,
                                    num_splits=args.num_splits,
                                    run_name_suffix=args.run_name_suffix)
        # A forced anchor is not an IID draw from the mixture.  Recording a proposal
        # density for it would be a fabrication, so it is excluded from any
        # importance-weighted use rather than given a fake log_q.
        # The frozen paste IS this parameter point, so re-pasting it would spend a
        # GPU-hour reproducing a map that already exists and is already validated.
        # Point the entry at the cached product; the paste array skips any entry that
        # declares one, and the measurement step reads it directly.
        record.update({"index": len(entries), "u": None, "component": "forced_anchor",
                       "log_q": None, "sampling_role": "forced_or_diagnostic",
                       "importance_eligible": False,
                       "cached_map": repo_relative(msc.FROZEN_MAP_PATH),
                       "cached_map_sha256": msc.sha256_file(msc.FROZEN_MAP_PATH),
                       "note": "the frozen paste's own parameters; its cached map is reused, "
                               "so this consumes no paste slot"})
        entries.append(record)

    duplicates = [name for name, count in
                  zip(*np.unique([e["run_name"] for e in entries], return_counts=True))
                  if count > 1]
    if duplicates:
        raise RuntimeError(f"Content-addressed run names collided: {duplicates}")

    manifest = {
        "schema_version": "godmax.mock_sbi.paste_plan.v1",
        "round": args.round,
        "count": len(entries),
        "recipe_name": args.recipe,
        "recipe": recipe,
        "guide": guide_description,
        "guide_key": guide_description["key"],
        "guide_path": guide_description["path"],
        "guide_sha256": guide_description["sha256"],
        "guide_u_map": u_map.tolist(),
        "guide_covariance": covariance.tolist(),
        "prng_namespace": list(ROUND_ENTROPY[key]),
        "nside": args.nside,
        "num_splits": args.num_splits,
        "galaxy_map_painted": bool(args.keep_galaxy_map),
        "run_name_suffix": args.run_name_suffix,
        "parameter_names": list(PARAMETER_NAMES),
        "prior_low": PRIOR_LOW.tolist(),
        "prior_high": PRIOR_HIGH.tolist(),
        "base_config": repo_relative(BASE_CONFIG),
        "base_config_sha256": msc.sha256_file(BASE_CONFIG),
        "entries": entries,
    }
    out = plan_dir / "paste_plan.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)

    print(f"round {args.round}: wrote {len(entries)} paste configs to {plan_dir}")
    print(f"  recipe {args.recipe} on the '{guide_description['key']}' "
          f"({guide_description['kind']}); components "
          f"{ {n: int(np.sum(labels == n)) for n in recipe['weights']} }")
    print(f"  galaxy map painted: {bool(args.keep_galaxy_map)}   nside {args.nside}   "
          f"num_splits {args.num_splits}")
    print(f"  manifest {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
