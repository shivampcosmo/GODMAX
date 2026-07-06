"""Paste one Abacus xDESI halo split into HEALPix maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from abacus_pasting_helpers import run_paste_split, xdesi_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(xdesi_dir() / "abacus_pasting_config.yaml"))
    parser.add_argument("--catalog", default="zlt1p0_logMgt12p5")
    parser.add_argument("--split-index", type=int, required=True)
    parser.add_argument("--num-splits", type=int, default=4)
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_paste_split(
        Path(args.config),
        args.catalog,
        args.split_index,
        args.num_splits,
        args.nside,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
