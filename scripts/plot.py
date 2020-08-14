# -*- coding: utf-8 -*-

# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0

import sys

if (__name__ == '__main__') and (len(sys.argv) >= 3) and (sys.argv[-1] != '-'):
    import matplotlib
    matplotlib.use('Agg')

import math
import getopt
import colorsys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import seaborn as sns
# import plfit

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def darken(color, mod=0.2):
    c = colorsys.rgb_to_hls(*colors.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, mod * c[1])), c[2])

def full_extent(ax):
    ax.figure.canvas.draw()
    items = [ax, ax.xaxis.label, ax.yaxis.label]
    # items = [ax]
    bbox = transforms.Bbox.union([item.get_window_extent() for item in items])
    return bbox.padded(3)

def plot_bars(input, dt=None, **kwargs):
    x = np.repeat(np.arange(len(input)+1, dtype=dt),2)[1:-1]
    y = np.repeat(input, 2)

    plt.xlim(0, max(plt.xlim()[1], len(input)+1))
    plt.ylim(0, max(plt.ylim()[1], np.max(input)*1.1))

    return plt.fill_between(x,0,y,**kwargs)

def plot_heat(input, **kwargs):
    perca = input.flatten()
    perca = sorted(perca[np.flatnonzero(perca)].tolist())
    percf = lambda p: perca[round((len(perca)-1) * (p / 100.0))]

    labels = list(range(0,101,5))
    bounds = [percf(p) for p in labels]
    cmap   = sns.color_palette("Blues", 20)

    # mark low outliers in purple
    cut = percf(25) - (percf(75)-percf(25))*3
    idx = 0
    while percf(idx) < cut:
        idx += 1
    if idx > 0:
        count = (idx+4)//5
        cmap  = sns.color_palette("Purples", count) + cmap[count:]

    # mark high outliers in red
    cut = percf(75) + (percf(75)-percf(25))*3
    idx = 0
    while percf(100-idx) > cut:
        idx += 1
    if idx > 0:
        count = (idx+4)//5
        cmap  = cmap[:-count] + sns.color_palette("Reds", count)

    # determine labels
    ticks = bounds.copy()
    if np.min(input) == 0:
        bounds = [0] + bounds
        cmap   = ["white"] + cmap

    i = 1
    while i < len(ticks):
        if ticks[i-1] == ticks[i] or labels[i-1]+5 == labels[i] or labels[i-1]+2 > labels[i]:
            ticks.pop(i)
            labels.pop(i)
            continue
        i += 1

    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds)-1, clip=True)
    sns.heatmap(
        input,
        cmap=cmap,
        norm=norm,
        rasterized=True, square=True, xticklabels=False, yticklabels=False,
        cbar_kws=dict(
            ticks=ticks,
            format=ticker.FuncFormatter(lambda x, p: "$P_{%d}=$%s" % (labels[p], ("%.2g" if x < 1 else "%.2f") % (x)))
        ),
        **kwargs
    )

def plot_fit(input, do_print=False):
    ly = np.log2(input+1)
    lx = np.log2(np.arange(len(ly))+0.5)

    # try:
    #     pl = plfit.plfit(input,silent=True)

    #     if do_print:
    #         print('alpha =', -pl._alpha, 'xmin =', pl._xmin)
    #         pl.test_pl(100)

    #         pl.lognormal(doprint=False)
    #         print('powerlaw' if pl._likelihood + pl.lognormal_likelihood > 0 else 'lognormal')

    #     if (pl._alpha > 0) and (pl._xmin < len(input)):
    #         pf = (-pl._alpha, ly[pl._xmin] + pl._alpha*lx[pl._xmin])
    #         lf = plt.plot(np.exp2(lx[pl._xmin:]), np.exp2(np.poly1d(pf)(lx[pl._xmin:])), 'm--', alpha=.5, label="x^%.3f (for x > %d)" % (pf[0], pl._xmin))
    # except Exception as e:
    #     print(e)

    p1 = np.polyfit(lx, ly, 1, w=ly)
    if (p1[0] < 0):
        l1 = plt.plot(np.exp2(lx), np.exp2(np.poly1d(p1)(lx))-1, 'b--', alpha=.5, label="x^%.3f" % (p1[0]))

    if do_print:
        print('alpha(fit,1d) =', p1[0])
        # print('alpha(fit,3d) =', p3[:-1])
        if len(input) >= 8 :
            print('p(normal) =', stats.mstats.normaltest(input)[1])

def print_stats(input):
    x = np.arange(len(input))
    y = input

    w_avg = np.average(x, weights=y)
    w_std = math.sqrt(np.average((x-w_avg)**2, weights=y))
    print(w_avg, w_std)

    y.sort()
    print(y[0],y[int(len(y)*.05)],y[int(len(y)*.25)],y[int(len(y)*.5)],y[int(len(y)*.75)],y[int(len(y)*.95)],y[-1])
    print(np.average(y), np.std(y))

def main():
    opt, arg = getopt.getopt(sys.argv[1:], 'hfplL:')
    opt = dict(opt)

    if ('-h' in opt):
        print('usage %s [-help] [-fit] [-print] [-log] [-LogBase int] [INPUT.. [OUTPUT]]' % (sys.argv[0]))
        sys.exit()

    if (len(arg) < 2):
        arg.append('-')
    if (len(arg) < 2):
        arg.append('-')

    dt = np.int32
    plt.figure(figsize=(11,8),dpi=100)

    row = int(math.sqrt(len(arg)-1))
    col = 1
    if (row > 1) and (math.sqrt(len(arg)-1) == row):
        col = row
    else:
        row = len(arg)-1

    for f in range(len(arg)-1):
        print('Plotting', arg[f])
        input = np.loadtxt(sys.stdin if (f >= len(arg)) or (arg[f] == '-') else arg[f], dtype=dt, ndmin=1)

        plt.subplot(row,col,f+1)
        plot_bars(input, dt, color='g')

        if ('-l' in opt) or ('-L' in opt):
            plt.xscale('symlog', basex=int(('-L' in opt) and opt['-L']) or 2)
            plt.yscale('symlog', basey=int(('-L' in opt) and opt['-L']) or 2)

        if ('-f' in opt):
            plot_fit(input, '-f' in opt)

        if ('-p' in opt):
            print_stats(input)

    if (len(arg) < 2) or (arg[-1] == '-'):
        plt.show()
    else:
        plt.savefig(arg[-1], bbox_inches='tight')

if __name__ == '__main__':
    main()
