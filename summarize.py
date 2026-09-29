"""Aggregate multi-seed runs produced by train.py.

    python summarize.py                              # mean ± std table on the test split
    python summarize.py --compare ResMUNet ResUNet   # + Welch t-test on F1

With a handful of seeds a t-test has little power: treat a non-significant
difference as "not shown to differ", not as "equal".
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

METRICS = ['f1_score', 'precision', 'recall', 'iou_landslide', 'miou', 'kappa']


def load_runs(runs_dir, split):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(runs_dir, '*', 'seed*', 'results.json'))):
        with open(path) as f:
            res = json.load(f)
        if split in res:
            runs[res['config']['model']].append(res[split])
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs-dir', default='runs')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    p.add_argument('--metric', default='f1_score', choices=METRICS)
    args = p.parse_args()

    runs = load_runs(args.runs_dir, args.split)
    if not runs:
        raise SystemExit(f'No results.json with a {args.split!r} entry under {args.runs_dir}/')

    header = '| Model | n | ' + ' | '.join(METRICS) + ' |'
    print(header)
    print('|' + '---|' * (len(METRICS) + 2))
    for model, results in runs.items():
        cells = []
        for m in METRICS:
            v = np.array([r[m] for r in results])
            cells.append(f'{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}')
        print(f'| {model} | {len(results)} | ' + ' | '.join(cells) + ' |')

    if args.compare:
        a, b = ([r[args.metric] for r in runs[name]] for name in args.compare)
        t, pval = stats.ttest_ind(a, b, equal_var=False)
        print(f'\n{args.compare[0]} vs {args.compare[1]} ({args.metric}): '
              f'diff of means {np.mean(a) - np.mean(b):+.4f}, Welch t = {t:.2f}, p = {pval:.3f}')


if __name__ == '__main__':
    main()
