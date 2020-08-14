# -*- coding: utf-8 -*-

# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0

import sys

if (__name__ == '__main__') and (len(sys.argv) >= 3) and (sys.argv[-1] != '-'):
    import matplotlib
    matplotlib.use('Agg')

import os
import getopt
import glob
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plot
import plot_data

def main():
    opt, arg = getopt.getopt(sys.argv[1:], 'f:b:i:s:')
    opt = dict(opt)

    ffilter = (('-f' in opt) and opt['-f']) or None

    if ('-h' in opt):
        print('usage %s [-help] [-filter] [INPUT.. [OUTPUT]]' % (sys.argv[0]))
        sys.exit()

    if (len(arg) < 2):
        arg.append('-')
    if (len(arg) < 2):
        arg.append('-')

    out = arg.pop()
    arg = (glob.glob(a) if a != '-' else [a] for a in arg)
    log = pd.concat([plot_data.get_log(f, ffilter) for g in arg for f in g])
    log = log.dropna()
    log = log[log['kernel'] != '']
    log = log[log['graph'] != '']
    #log = log[log['dataset'] != 'real-world']

    if ('-s' in opt):
        log = plot_data.merge_stats(opt['-s'], log)

    print(log)
    print(log.groupby('kernel')['PS'].mean())
    print(log.groupby('kernel')['PS'].std())

    print(log[log['PS'] > 30])
    
    sns.set('paper', 'ticks', color_codes=True)
    plot_data.filter_markers(log['SOLVER'])

    if ('-b' in opt):
        log = log[log['|B|'] == int(opt['-b'])]
    if ('-i' in opt):
        log = log[log['it'] == int(opt['-i'])]

    plt.figure(figsize=(6,3))

    with sns.plotting_context(), sns.axes_style():
        ax = sns.boxplot(
            x="kernel",
            y="EPS",
            hue="dataset",
            # color="b",
            order=['OMP1', 'OMP2', 'OMP3', 'OCL1', 'OCL2', 'CUD1', 'CUD2', 'CUD3', 'MKL', 'CSP'],
            data=log,
            fliersize=1,
        )

        hatches = itertools.cycle(['/////', '.....'])
        for _, bar in enumerate(ax.artists):
            bar.set_hatch(next(hatches))
        for _, bar in enumerate(ax.patches):
            bar.set_hatch(next(hatches))

        plt.ylim(-2, 30)

        ax.grid(axis='y', linestyle='-', linewidth='0.1')
        ax.legend(loc='upper left', title='Dataset', framealpha=1)

    plt.tight_layout()

    if out == '-':
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()
