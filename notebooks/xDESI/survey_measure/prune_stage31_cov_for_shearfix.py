#!/usr/bin/env python
"""Delete only the products that the shear shape-noise fix actually changes.

The shear shape-noise fix only changes the shear-auto C(s_i, s_i) covariance input
(its BB component). A covariance group therefore changes ONLY if one of its four
gaussian_covariance input spectra is a same-shear-field auto C(s_i, s_i).  All other
groups (10 scalar + the cross-input spin2 groups) are byte-identical and can be reused.

This script removes, for the midres2048 true-nz product:
  * the spectra product (so the data vector regenerates with the fixed shear autos),
  * the assembled cls_cov product (so it re-assembles),
  * ONLY the covariance shards of the affected spin2 groups,
leaving the map product and all reusable covariance shards in place.

Then re-run the pipeline WITHOUT --force: prepare reuses maps, spectra regenerates,
unaffected/scalar shards are skipped, affected shards recompute, assemble/validate/plot
regenerate.

Dry-run by default. Pass --apply to actually delete. Pass --all-spin2 to conservatively
recompute every spin2 group instead of only the affected subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
DEFAULT_DIR = REPO / "data/xDESI/processed/multiprobe_namaster_true_nz/midres2048"
TAG = "nside2048_ell128_lmax3000_nbin13_log_apo1deg_C2_pairmean"


def is_shear(field: str) -> bool:
    return field.startswith("s") and field[1:].isdigit()


def uses_shear_auto_input(reps) -> bool:
    a1, a2, b1, b2 = reps
    inputs = [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
    return any(x == y and is_shear(x) for x, y in inputs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run).")
    ap.add_argument("--all-spin2", action="store_true", help="Recompute every spin2 group (conservative).")
    ap.add_argument("--keep-spectra", action="store_true",
                    help="Do NOT delete the spectra product (for PATCH_SHEAR_SPECTRA=1 reruns, which "
                         "patch the 4 shear autos into the existing product instead of full regen).")
    args = ap.parse_args()

    d = Path(args.dir)
    manifest = json.loads((d / f"covariance_manifest_{TAG}.json").read_text())
    block_dir = d / f"covariance_blocks_{TAG}"

    spin2 = [g for g in manifest["groups"] if g["class"] == "spin2"]
    if args.all_spin2:
        affected = spin2
    else:
        affected = [g for g in spin2 if uses_shear_auto_input(g["representative_fields"])]

    targets = [d / f"xdesi_multiprobe_cls_cov_{TAG}.h5"]
    if not args.keep_spectra:
        targets.insert(0, d / f"xdesi_multiprobe_spectra_{TAG}.h5")
    targets += [block_dir / f"cov_group_{int(g['index']):04d}_spin2.h5" for g in affected]

    n_groups_total = len(manifest["groups"])
    print(f"groups total={n_groups_total}  spin2={len(spin2)}  recompute={len(affected)}  reuse={n_groups_total - len(affected)}")
    print(f"mode={'ALL spin2' if args.all_spin2 else 'affected-only (shear-auto input)'}")
    print(f"\n{'DELETING' if args.apply else 'WOULD DELETE'} {len(targets)} files:")
    missing = 0
    for t in targets:
        exists = t.exists()
        missing += (not exists)
        tag = "" if exists else "  [absent]"
        print(f"  {t.name}{tag}")
        if args.apply and exists:
            t.unlink()
    print(f"\n{'Deleted' if args.apply else 'Dry-run; pass --apply to delete'}. "
          f"(map product and {n_groups_total - len(affected)} reusable shards left in place.)")
    if missing:
        print(f"note: {missing} target(s) already absent.")


if __name__ == "__main__":
    main()
