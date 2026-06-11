#!/usr/bin/env python
"""CLI wrapper for the xDESI GODMAX stage-31 HMC fit."""

from __future__ import annotations

import os
import sys


def _preconfigure_jax_from_argv(argv: list[str]) -> None:
    """Set JAX platform env vars before importing modules that import JAX."""

    platform = None
    for i, arg in enumerate(argv):
        if arg == "--platform" and i + 1 < len(argv):
            platform = argv[i + 1]
            break
        if arg.startswith("--platform="):
            platform = arg.split("=", 1)[1]
            break

    if platform == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    elif platform == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    os.environ.setdefault("JAX_ENABLE_X64", "True")


_preconfigure_jax_from_argv(sys.argv[1:])

from godmax_multiprobe_hmc_stage31 import main


if __name__ == "__main__":
    raise SystemExit(main())
