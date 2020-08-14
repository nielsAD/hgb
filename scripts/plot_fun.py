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
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

import plot
import plot_data

def main():
    opt, arg = getopt.getopt(sys.argv[1:], 'hx:y:r:c:g:f:b:i:s:')
    opt = dict(opt)

    layoutx = (('-x' in opt) and opt['-x']) or '|B|'
    layouty = (('-y' in opt) and opt['-y']) or '|E|/sec'
    layoutr = (('-r' in opt) and opt['-r']) or 'FRAMEWORK'
    layoutc = (('-c' in opt) and opt['-c']) or 'DATAGROUP'
    layoutg = (('-g' in opt) and opt['-g']) or None
    ffilter = (('-f' in opt) and opt['-f']) or None

    if ('-h' in opt):
        print('usage %s [-help] [-filter] [-x datacol] [-y datacol] [-row datacol] [-col datacol] [-group datacol] [INPUT.. [OUTPUT]]' % (sys.argv[0]))
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
    log = log[log['|B|'] > 1]
    # log = log[(log['kernel'] == 'OMP3') | (log['kernel'] == 'CUD3')]
    # log = log[(log['dataset'] == 'real-world') | log['DATASET'].str.contains('_4M_32')]

    if ('-s' in opt):
        log = plot_data.merge_stats(opt['-s'], log)
        log.to_csv('out.csv', index=False)

    log.loc[(log['kernel'] == 'OMP1') | (log['kernel'] == 'OMP2') | (log['kernel'] == 'OMP2'), 'kernel'] = 'OMP'
    log.loc[(log['kernel'] == 'CUD1') | (log['kernel'] == 'CUD2') | (log['kernel'] == 'CUD3'), 'kernel'] = 'CUD'

    log = log.groupby(['graph', 'DATASET', '|V|', '$\\overline{D}$', 'kernel', 'FLOW'], as_index=False)['EPS'].max()

    log = log.sort_values(by=[layoutx])
    print(log)

    sns.set('paper', 'darkgrid', color_codes=True)

    if ('-b' in opt):
        log = log[log['|B|'] == int(opt['-b'])]
    if ('-i' in opt):
        log = log[log['it'] == int(opt['-i'])]

    set_order = set(log[layoutc])
    if (layoutc == 'FRAMEWORK') or (layoutc == 'SOLVER') or (layoutc == 'METHOD'):
        col_order = filter(lambda z: z[0] in z[1], itertools.product(plot_data.fw_order, sorted(set_order)))
        _, col_order = zip(*list(col_order))
    else:
        col_order = sorted(set_order)

    set_order = set(log[layoutr])
    if (layoutr == 'FRAMEWORK') or (layoutr == 'SOLVER') or (layoutr == 'METHOD'):
        row_order = filter(lambda z: z[0] in z[1], itertools.product(plot_data.fw_order, sorted(set_order)))
        _, row_order = zip(*list(row_order))
    else:
        row_order = sorted(set_order)


    row_order = ['OMP', 'CUD', 'SPU1', 'SPU2']

    if (layoutg == None):
        palette = None
        regplotm = sns.regplot
    elif (layoutg in ['SOLVER', 'kernel', 'graph', 'device']):
        palette = (v[0] for k, v in sorted(plot_data.markers.items()))

        def regplotm(x, y, label=None, color=None, **kwargs):
            return plt.plot(x, y,
                # color=plot_data.markers[label][0],
                marker='s',
                # marker=plot_data.markers[label][1],
                label=label,
                # logx=True,
                # ci=None,
                # linewidth=0.5,
                **kwargs
            )
    else:
        palette = sns.light_palette('teal', len(set(log[layoutg])), input='xkcd')

        def regplotm(x, y, data=None, **kwargs):
            sns.regplot(x, y, scatter=False, data=data, ci=None, **kwargs)
            for f, g in data.groupby('SOLVER'):
                sns.regplot(x, y, fit_reg=False, marker=plot_data.markers[f][1], data=g, **kwargs)

    minx, miny = min(log[layoutx]), min(log[layouty])
    maxx, maxy = max(log[layoutx]), max(log[layouty])

    with sns.plotting_context(), sns.axes_style():
        p = sns.FacetGrid(
            row=layoutr,
            col=layoutc,
            hue=layoutg,
            row_order=row_order,
            col_order=col_order,
            hue_order=sorted(set(log[layoutg])) if layoutg is not None else None,
            # hue_order=["ER", "REG (stride dst)", "REG (stride src+dst)", "REG"],
            palette=palette,
            xlim=(minx - 0.1*(maxx - minx), maxx + 0.1*(maxx - minx)),
            #ylim=(miny - 0.1*(maxy - miny), maxy + 0.1*(maxy - miny)),
            margin_titles=True,
            legend_out=True,
            sharey=False,
            height=2,
            data=log,
        )

        # p.map_dataframe(regplotm, layoutx, layouty, line_kws={'lw': 1.0, 'ls': ':', 'color': 'r'}, scatter_kws={'s': 10, 'color': 'b', 'alpha': 1, 'edgecolors': 'none'})
        p.map_dataframe(regplotm, layoutx, layouty)
        p.axes[0,0].set_xscale('log', basex=2)
        p.axes[0,0].set_xlim(2 ** (math.log2(minx)-.5), 2**(math.log2(maxx)+.5))
        # p.axes[0,0].set_xlim(-.1, 1.1)

        for ax in p.axes.reshape(-1): 
            ax.set_ylim(0)
            ax.set_ylabel('')
            ax.set_xlabel('')

        minx = math.ceil(math.log2(minx)-.5)
        maxx = math.floor(math.log2(maxx)+.5)
        # p.axes[0,0].set_xticks([2 ** minx, 2 ** ((minx+maxx)//2), 2 ** maxx])
        p.axes[0,0].set_xticks([2, 8, 32, 128])
        p.axes[0,0].set_xticklabels(['2', '8', '32', '128'])
        if (layoutg is not None):
            p.add_legend()

    # plt.tight_layout()

    if out == '-':
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight', dpi=300)

        # (base, ext) = os.path.splitext(out)
        # for r in range(len(p.axes)):
        #     for c in range(len(p.axes[r])):
        #         plt.savefig('%s.sub%d%d%s' % (base, r, c, ext), dpi=300, bbox_inches=plot.full_extent(p.axes[r][c]).transformed(p.fig.dpi_scale_trans.inverted()))


if __name__ == '__main__':
    main()
