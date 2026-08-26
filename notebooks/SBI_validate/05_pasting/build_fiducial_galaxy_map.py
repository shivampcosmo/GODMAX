"""Cache a HEALPix galaxy count/overdensity map from the fiducial paste's frozen catalog.

The per-point pastes run with ``get_galmap: false`` -- the galaxy field is frozen across
the five sampled gas parameters -- so no galaxy MAP is stored anywhere.  What is stored is
the 62.9M-row galaxy catalog inside the fiducial paste.  This builds the map from it once
and caches it, so a notebook can display the galaxy field without re-reading 1.8 GB.

Only rows with valid > 0.5 are counted, matching the selection combine_partial_maps applies
when it forms the realized n(z).
"""

from __future__ import annotations

import argparse
import pathlib

import h5py
import healpy as hp
import numpy as np

RA, DEC, VALID = 0, 1, 5


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paste", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=4_000_000)
    args = p.parse_args()

    npix = hp.nside2npix(args.nside)
    counts = np.zeros(npix, dtype=np.int64)
    n_read = n_valid = 0
    n_clipped, worst = [0], [0.0]
    with h5py.File(args.paste, "r") as handle:
        gal = handle["galaxies"]
        total = gal.shape[0]
        for start in range(0, total, args.chunk):
            block = np.asarray(gal[start:start + args.chunk, :], dtype=np.float64)
            keep = block[:, VALID] > 0.5
            ra, dec = block[keep, RA], block[keep, DEC]
            # 4 of the 62.9M galaxies land at |dec| just past 90 deg (max overshoot
            # 0.127 deg) -- float32 rounding on the satellite displacement at the poles.
            # healpy rejects theta outside [0, pi], so clip to the pole and COUNT it
            # rather than dropping the rows or letting ang2pix raise.
            over = int((np.abs(dec) > 90.0).sum())
            if over:
                n_clipped[0] += over
                worst[0] = max(worst[0], float(np.abs(dec).max()))
                dec = np.clip(dec, -90.0, 90.0)
            pix = hp.ang2pix(args.nside, np.radians(90.0 - dec), np.radians(ra))
            counts += np.bincount(pix, minlength=npix)
            n_read += block.shape[0]
            n_valid += int(keep.sum())
            print(f"   {n_read:,} / {total:,} rows", flush=True)

    if counts.sum() != n_valid:
        raise RuntimeError(f"binned {counts.sum()} galaxies but selected {n_valid}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(""), counts=counts.astype(np.int32),
                        nside=args.nside, n_valid=n_valid, n_rows=total,
                        source=str(args.paste), n_pole_clipped=n_clipped[0],
                        worst_abs_dec_deg=worst[0])
    print(f"wrote {args.output}  ({n_valid:,} valid of {total:,} rows, "
          f"mean {n_valid / npix:.2f} per pixel)")
    print(f"  clipped {n_clipped[0]} galaxies to the poles "
          f"({n_clipped[0] / max(n_valid, 1):.1e} of the sample; "
          f"worst |dec| = {worst[0]:.3f} deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
