#!/usr/bin/env python
"""Collect status for Abacus paste scaling jobs from Slurm logs and JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--repo", default="/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
    parser.add_argument("--run", default="stage31_pz3_cap600")
    parser.add_argument("--tail-lines", type=int, default=40)
    args = parser.parse_args()

    repo = Path(args.repo)
    log_dir = repo / "notebooks/xDESI/abacus_paste/slurm_logs"
    meas_dir = repo / f"data/xDESI/processed/abacus_backlight/{args.run}/measurements"
    out_logs = sorted(log_dir.glob(f"*_{args.job_id}.out"))
    err_logs = sorted(log_dir.glob(f"*_{args.job_id}.err"))
    jsons = sorted(meas_dir.glob(f"*job{args.job_id}.json"))

    print(json.dumps({
        "job_id": str(args.job_id),
        "out_logs": [str(path) for path in out_logs],
        "err_logs": [str(path) for path in err_logs],
        "json_outputs": [str(path) for path in jsons],
    }, indent=2))
    for path in out_logs + err_logs:
        print(f"\n===== {path} =====")
        print(tail(path, args.tail_lines))
    if jsons:
        print("\n===== JSON summaries =====")
        for path in jsons:
            payload = json.loads(path.read_text())
            print(json.dumps({
                "path": str(path),
                "nside": payload.get("nside"),
                "n_halos": payload.get("n_halos"),
                "n_pairs": payload.get("n_pairs"),
                "rows": [
                    {
                        "fused": row.get("fused"),
                        "runtime_s": row.get("runtime_s"),
                        "profile_total_execution": (row.get("timing_results") or {}).get("total_execution"),
                    }
                    for row in payload.get("rows", [])
                ],
                "diffs": payload.get("diffs_fused_minus_unfused", {}),
            }, indent=2))


if __name__ == "__main__":
    main()
