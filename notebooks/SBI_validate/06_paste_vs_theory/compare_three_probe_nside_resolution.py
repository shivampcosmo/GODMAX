#!/usr/bin/env python3
"""Compare immutable nside-512 and nside-1024 paste/theory residual products."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np


SPECTRA = ("gg", "gy", "gtau", "gkappa")
COLORS = {"gg": "#1f77b4", "gy": "#d62728", "gtau": "#9467bd", "gkappa": "#2ca02c"}


def sha256_file(path: pathlib.Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_resolution_change(
    residual_512: dict[str, np.ndarray], residual_1024: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for name in SPECTRA:
        low = np.asarray(residual_512[name], dtype=np.float64)
        high = np.asarray(residual_1024[name], dtype=np.float64)
        if low.shape != (12,) or high.shape != (12,):
            raise ValueError(f"{name} residuals do not use the frozen 12 bands")
        summary[name] = {
            "residual_percent_nside512": (100.0 * low).tolist(),
            "residual_percent_nside1024": (100.0 * high).tolist(),
            "change_percentage_points_1024_minus_512": (100.0 * (high - low)).tolist(),
            "last_band_residual_percent_nside512": float(100.0 * low[-1]),
            "last_band_residual_percent_nside1024": float(100.0 * high[-1]),
            "last_band_change_percentage_points": float(100.0 * (high[-1] - low[-1])),
        }
    return summary


def run(path_512: pathlib.Path, path_1024: pathlib.Path, output_dir: pathlib.Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "nside512_vs_nside1024_residual_comparison"
    json_path = output_dir / f"{stem}.json"
    npz_path = output_dir / f"{stem}.npz"
    png_path = output_dir / f"{stem}.png"
    if any(path.exists() for path in (json_path, npz_path, png_path)):
        raise FileExistsError("Resolution comparison outputs are immutable")
    with h5py.File(path_512, "r") as first, h5py.File(path_1024, "r") as second:
        ell_512 = np.asarray(first["ell_effective"], dtype=np.float64)
        ell_1024 = np.asarray(second["ell_effective"], dtype=np.float64)
        if not np.array_equal(ell_512, ell_1024):
            raise ValueError("Resolution products do not use identical saved bands")
        residual_512 = {name: np.asarray(first[f"{name}/fractional_residual"]) for name in SPECTRA}
        residual_1024 = {name: np.asarray(second[f"{name}/fractional_residual"]) for name in SPECTRA}
        provenance_512 = json.loads(str(first.attrs["provenance_json"]))
        provenance_1024 = json.loads(str(second.attrs["provenance_json"]))
    if provenance_512.get("nside", 512) != 512 or provenance_1024.get("nside") != 1024:
        raise ValueError("Input products do not form the registered 512/1024 pair")
    summary = summarize_resolution_change(residual_512, residual_1024)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, name in zip(axes.flat, SPECTRA):
        axis.axhspan(-10.0, 10.0, color="#2ca02c", alpha=0.10)
        axis.axhline(0.0, color="black", lw=0.8)
        axis.plot(ell_512, 100.0 * residual_512[name], "o--", color=COLORS[name], alpha=0.55, label="nside 512")
        axis.plot(ell_512, 100.0 * residual_1024[name], "o-", color=COLORS[name], label="nside 1024")
        axis.set_xscale("log"); axis.grid(alpha=0.25); axis.set_title(name); axis.legend()
    axes[1, 0].set_xlabel(r"$\ell_{\rm eff}$"); axes[1, 1].set_xlabel(r"$\ell_{\rm eff}$")
    axes[0, 0].set_ylabel("mock/theory - 1 [%]"); axes[1, 0].set_ylabel("mock/theory - 1 [%]")
    fig.suptitle("Resolution control: matched nside 512 vs 1024")
    fig.tight_layout(); fig.savefig(png_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    np.savez_compressed(
        npz_path,
        ell_effective=ell_512,
        **{f"{name}_residual_nside512": residual_512[name] for name in SPECTRA},
        **{f"{name}_residual_nside1024": residual_1024[name] for name in SPECTRA},
    )
    result: dict[str, object] = {
        "status": "DIAGNOSTIC_COMPLETE",
        "summary": summary,
        "inputs": {
            "nside512": str(path_512.resolve()), "nside512_sha256": sha256_file(path_512),
            "nside1024": str(path_1024.resolve()), "nside1024_sha256": sha256_file(path_1024),
        },
        "script_sha256": sha256_file(pathlib.Path(__file__)),
        "npz": str(npz_path.resolve()), "plot": str(png_path.resolve()),
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside512", type=pathlib.Path, required=True)
    parser.add_argument("--nside1024", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.nside512, args.nside1024, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
