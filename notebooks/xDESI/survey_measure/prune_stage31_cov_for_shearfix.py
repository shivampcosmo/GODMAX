#!/usr/bin/env python
"""Refuse the obsolete partial covariance-reuse workflow.

Pipeline v2 changes masks, mode-coupling windows, noise spectra, transfer functions, and
covariance inputs.  Consequently, the old assumption that only four shear-auto spectra and
a subset of spin-2 covariance shards need rebuilding is false.  Keeping this command as a
hard failure gives existing runbooks a clear migration error without deleting any products.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "The partial shear-fix prune workflow is unsafe and has been retired. "
        "Run the full pipeline-v2 spectra and covariance phases; versioned _pipev2 "
        "products will not reuse legacy maps, workspaces, or shards. No files were changed."
    )


if __name__ == "__main__":
    main()
