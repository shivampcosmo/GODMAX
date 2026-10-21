#!/usr/bin/env python3
"""Small adapter for the CURRENT run_mock_sbi_all256_snle.py.

Place beside that runner. All original CLI arguments, probes, compression,
data policies, four-network ensemble and MCMC settings remain supported.

Additional arguments:
  --ablation-case baseline|batch128|lr2e4|patience100|epochs2000
  --seed-offset N        0 preserves the existing seed namespace exactly.
  --ablation-help        show this adapter's help without importing the runner.

Each training case changes ONE setting relative to that probe's defaults.
The cases refer to literal values: gy normally has batch size 64 and patience
100, so batch128 increases its batch size and patience100 is a null for gy.
For gy_g2y the defaults are batch512, lr5e-4, patience25, max500 epochs.
The effective settings and seed namespace are saved by the original runner.
"""
import argparse
import importlib
import sys


CASES = {
    'baseline': {},
    'batch128': {'training_batch_size': 128},
    'lr2e4': {'learning_rate': 2.0e-4},
    'patience100': {'stop_after_epochs': 100},
    'epochs2000': {'max_num_epochs': 2000},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument('--ablation-case', choices=tuple(CASES), default='baseline')
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--ablation-help', action='store_true')
    args, forwarded = parser.parse_known_args(argv)
    if args.ablation_help:
        parser.print_help()
        return 0
    if args.seed_offset < 0:
        parser.error('--seed-offset must be non-negative')

    runner = importlib.import_module('run_mock_sbi_all256_snle')
    overrides = CASES[args.ablation_case]
    runner.TRAINING = {**runner.TRAINING, **overrides}
    runner.GY_TRAINING = {**runner.GY_TRAINING, **overrides}
    if args.seed_offset:
        runner.NAMESPACE = (*runner.NAMESPACE, args.seed_offset)
    print(f'ABLATION: case={args.ablation_case}; overrides={overrides}; '
          f'seed offset={args.seed_offset}; namespace={runner.NAMESPACE}', flush=True)

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *forwarded]
        return runner.main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    raise SystemExit(main())

