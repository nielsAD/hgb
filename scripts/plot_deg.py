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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plot
import plot_data

def main():
    opt, arg = getopt.getopt(sys.argv[1:], 'hpy:f:d:s:L:P:')
    opt = dict(opt)

    layout    = (('-y' in opt) and opt['-y']) or 'mbhHc'
    logfile   = (('-d' in opt) and opt['-d']) or sys.stdin
    statsfile = (('-s' in opt) and opt['-s']) or None
    ffilter   = (('-f' in opt) and opt['-f']) or None
    parts     = int(('-P' in opt) and opt['-P']) or 1

    if ('-h' in opt):
        print('usage %s [-help] [-data csvfile] [-LogBase int] [-Parts int] [INPUT.. [OUTPUT]]' % (sys.argv[0]))
        sys.exit()

    if (len(arg) < 2):
        arg.append('-')
    if (len(arg) < 2):
        arg.append('-')

    out = arg.pop()
    arg = (glob.glob(a) if a != '-' else [a] for a in arg)
    arg = [f for g in arg for f in g]

    sns.set('paper', color_codes=True)
    plot_data.filter_markers(ffilter)

    with sns.plotting_context(), sns.axes_style():
        row = len(layout)
        col = len(arg)*parts
        fig, axes = plt.subplots(row, col, figsize=(len(arg)*4,len(layout)*4), sharex='row', sharey='row')
        axes = axes.reshape(row, col) if row*col > 1 else [[axes]]

        for f in range(len(arg)):
            print('Plotting', arg[f])

            for p in range(parts):
                name = plot_data.get_name(arg[f], p, parts)
                deg  = None
                hist = None
                heat = None
                log  = None

                for i, c in enumerate(layout):
                    ax = axes[i][f*parts + p]
                    plt.sca(ax)

                    ax.tick_params(labelleft=True, labelbottom=True)

                    if (i == 0):
                        # column header
                        ax.set_title(os.path.basename(name))

                    if (f + p == 0):
                        # force nice fit
                        plt.xlim(0, 0)
                        plt.ylim(0, 0)

                    if (c == 'b'):
                        deg = deg if deg is not None else plot_data.get_deg(name)
                        plot.plot_bars(deg, np.int32, color='g')
                        ax.set_xlabel('Vertex ID')
                        ax.set_ylabel('Degree per vertex')
                    elif (c == 'c') or (c == 'C'):
                        hist = hist if hist is not None else (plot_data.get_hist(name) if deg is None else np.bincount(deg))

                        w = np.repeat(np.arange(0, len(hist)), hist)
                        y = np.cumsum(w / np.sum(w)) #+ 1E-3
                        plt.plot(np.arange(0, len(y)) / len(y), y, 'b-')

                        lim = 1 #+ 1E-3
                        plt.xlim(0, lim)
                        plt.ylim(0, lim)
                        ax.set_xlabel('V (%)')
                        ax.set_ylabel('E (%)')

                        if (c == 'c'):
                            half = np.searchsorted(y, lim/2)
                            plt.plot([0, half/len(y)], [y[half], y[half]], 'r:')
                            plt.axvline(x=half/len(y), ymax=y[half]/lim, color='r', linestyle=":")

                            # plt.axvline(x=lim/2, ymax=y[len(y)//2]/lim, color='m', linestyle=":")
                            # plt.plot([0, 0.5], [y[len(y)//2], y[len(y)//2]], 'm:')
                        if (c == 'C'):
                            ax.set_xscale('symlog', linthreshx=.005, linscalex=0.5, basex=int(('-L' in opt) and opt['-L']) or 2)
                            ax.set_yscale('symlog', linthreshy=.005, linscaley=0.5, basey=int(('-L' in opt) and opt['-L']) or 2)
                    elif (c == 'h') or (c == 'H'):
                        hist = hist if hist is not None else (plot_data.get_hist(name) if deg is None else np.bincount(deg))

                        plot.plot_bars(hist, np.int32, color='g')
                        # plot.plot_fit(hist, '-p' in opt)

                        ax.set_xlabel('Degree')
                        ax.set_ylabel('Count')
                        if (c == 'H'):
                            ax.set_xscale('symlog', linthreshx=1, linscalex=0.5, basex=int(('-L' in opt) and opt['-L']) or 2)
                            ax.set_yscale('symlog', linthreshy=1, linscaley=0.5, basey=int(('-L' in opt) and opt['-L']) or 2)

                        # plt.legend()
                    elif (c == 'm'):
                        heat = heat if heat is not None else plot_data.get_heat(name)
                        plot.plot_heat(heat)

                        ax.patch.set_edgecolor('black')
                        ax.patch.set_linewidth('1')

                        ax.set_xlabel('Destination')
                        ax.set_ylabel('Source')

                    elif (c == 't') or (c == 'e'):
                        if log is None:
                            log = plot_data.get_log(logfile, ffilter)
                            if statsfile is not None:
                                log = plot_data.merge_stats(statsfile, log)

                        d = log[log['DATASET'] == os.path.basename(arg[f])]

                        if (c == 't'):
                            y = 'time/it'
                            lbl = 'ms per iteration'
                            loc = 1
                        elif (c == 'e'):
                            y = '|E|/sec'
                            lbl = 'edges per second'
                            loc = 2

                        ax.set_xlabel('Blocks')
                        plt.xlim(min(plt.xlim()[0], min(d['|B|'])-1), max(plt.xlim()[1], max(d['|B|'])+1))
                        plt.ylim(0,                                   max(plt.ylim()[1], max(d[y])*1.1))

                        for name, group in d.groupby('SOLVER'):
                            sns.regplot('|B|', y, x_jitter=0.25, color=plot_data.markers[name][0], marker=plot_data.markers[name][1], line_kws={'lw': 1.0, 'ls': ':'}, data=group, label=name)

                        ax.set_xlabel('Blocks')
                        ax.set_ylabel(lbl)

                        plt.legend(loc=loc, ncol=max(6 - col, 2), fontsize='xx-small', columnspacing=0.1, handletextpad=0.01)
                    else:
                        print('Invalid layout option "%s"' % (c))
                        assert(False)

        fig.tight_layout()

        if out == '-':
            plt.show()
        else:
            plt.savefig(out, dpi=300, bbox_inches='tight')

            (base, ext) = os.path.splitext(out)
            for r in range(row):
                for c in range(col):
                    plt.savefig('%s.sub%d%d%s' % (base, r, c, ext), dpi=300, bbox_inches=plot.full_extent(axes[r][c]).transformed(fig.dpi_scale_trans.inverted()))


if __name__ == '__main__':
    main()
