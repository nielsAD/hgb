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

    id_vars = set(log.columns)

    # log.sort_values(by=['DATASET', 'SOLVER', 'FLOW'])
    # log.loc[log['FLOW'] == 'pull', 'trans_ms'] = log.loc[log['FLOW'] == 'push', 'trans_ms'].values

    # log.loc[log['kernel'] == 'CUD3', 'rank_us_omp'] =  log.loc[log['kernel'] == 'OMP3', 'rank_us'].values
    # log['transfer/it_gpu'] = (log['trans_ms'] * 1e3) / log['rank_us']
    # log['transfer/it_cpu'] = (log['trans_ms'] * 1e3) / log['rank_us_omp']
    # log = log.melt(id_vars=id_vars, value_vars=['transfer/it_gpu', 'transfer/it_cpu'], var_name='transdev', value_name='transfer/it')
    # log['device'] = log['transdev'].map({
    #     'transfer/it_gpu': 'GPU',
    #     'transfer/it_cpu': 'CPU',
    # })

    # log = log.dropna()
    # log = log[log['SOLVER'] == 'pcsc_mpi_stp']
    # log = log[(log['kernel'] == 'SPU1')]
    # log = log[
    #           (log['kernel'] == 'SPU1')
    #         | (log['kernel'] == 'SPU2')
    #         | (log['kernel'] == 'OMP1')
    #         | (log['kernel'] == 'OMP2')
    #         | (log['kernel'] == 'OMP3')
    #         | (log['kernel'] == 'CUD1')
    #         | (log['kernel'] == 'CUD2')
    #         | (log['kernel'] == 'CUD3')
    # ]
    log = log[(log['SOLVER'] == 'bcsc_mpi_stp')
            | (log['SOLVER'] == 'bcsc_mpi_rdx')
            | (log['SOLVER'] == 'pcsc_mpi_stp')
            | (log['SOLVER'] == 'rndpcsc_mpi_stp')
            | (log['SOLVER'] == 'blkpcsc_mpi_stp')
        ]
    # log = log[log['FLOW'] == 'pull']
    log = log[log['graph'] != '']
    # log = log[log['|B|'] == 4]
    log = log[(log['|B|'] == 1) | ((log['SOLVER'] != 'bcsc_mpi_rdx') & (log['|B|'] == 16)) | ((log['SOLVER'] == 'bcsc_mpi_rdx') & (log['|B|'] == 4))]
    log = log[(log['dataset'] == 'real-world') | log['DATASET'].str.contains('_4M_32')]
    # log = log[log['dataset'] == 'real-world']

    log.loc[log['SOLVER'] == 'bcsc_mpi_rdx', '|B|'] *= log[log['SOLVER'] == 'bcsc_mpi_rdx']['|B|'].values
    print(log)

    if ('-s' in opt):
        log = plot_data.merge_stats(opt['-s'], log)

    # log.loc[(log['kernel'] == 'OMP1') | (log['kernel'] == 'OMP2') | (log['kernel'] == 'OMP3'), 'kernel'] = 'OMP'
    # log.loc[(log['kernel'] == 'CUD1') | (log['kernel'] == 'CUD2') | (log['kernel'] == 'CUD3'), 'kernel'] = 'CUD'
    # log = log.groupby(['graph', 'DATASET', 'kernel'], as_index=False)['EPS'].max()
    # log.loc[log['kernel'] == 'OMP', 'EPS'] = log.loc[log['kernel'] == 'CUD', 'EPS'].values / log.loc[log['kernel'] == 'OMP', 'EPS'].values
    # log.loc[log['kernel'] == 'OMP', 'EPS'] = (log.loc[log['kernel'] == 'OMP', 'EPS'] + 1) / log.loc[log['kernel'] == 'OMP', 'EPS'].values

    # log.loc[(log['kernel'] == 'OMP1') | (log['kernel'] == 'OMP2') | (log['kernel'] == 'OMP3'), 'kernel'] = 'Best'
    # log.loc[(log['kernel'] == 'CUD1') | (log['kernel'] == 'CUD2') | (log['kernel'] == 'CUD3'), 'kernel'] = 'Best'

    # print(log)

    # log = log.groupby(['graph', 'DATASET', 'device'], as_index=False)['EPS'].max()
    # log = log.groupby(['graph', 'DATASET', 'device', 'FLOW'], as_index=False)['EPS'].max()
    # log = log.groupby(['graph', 'DATASET', 'device'], as_index=False).aggregate(lambda data: data.iloc[0] / data.iloc[1])
    # log = log.sort_values(by=['device', 'EPS'])

    # log = log.groupby(['graph', 'DATASET', 'kernel'], as_index=False)['EPS'].max()
    # print(log)

    # log.loc[log['device'] == 'GPU', 'EPS'] = log.loc[log['device'] == 'GPU', 'EPS'] / log.loc[log['device'] == 'CPU', 'EPS'].values
    # log = log[log['device'] == 'GPU']

    # ref = log[log['kernel'] == 'CUD']['EPS'].values
    # log.loc[log['kernel'] == 'SPU1', 'EPS'] = log.loc[log['kernel'] == 'SPU1', 'EPS'] / ref
    # log.loc[log['kernel'] == 'SPU2', 'EPS'] = log.loc[log['kernel'] == 'SPU2', 'EPS'] / ref

    # ref = log[log['kernel'] == 'OMP']['EPS'].values
    # log.loc[log['kernel'] == 'SPU1', 'EPS'] = log.loc[log['kernel'] == 'SPU1', 'EPS'] - ref
    # log.loc[log['kernel'] == 'SPU2', 'EPS'] = log.loc[log['kernel'] == 'SPU2', 'EPS'] - ref

    # print(log)
    # log.loc[log['kernel'] == 'SPU1_2', 'EPS'] = log.loc[log['kernel'] == 'SPU1_2', 'EPS'] / log.loc[log['kernel'] == 'SPU1', 'EPS'].values
    # log.loc[log['kernel'] == 'SPU2_2', 'EPS'] = log.loc[log['kernel'] == 'SPU2_2', 'EPS'] / log.loc[log['kernel'] == 'SPU2', 'EPS'].values

    # log.loc[log['kernel'] == 'SPU1_2', 'kernel'] = 'SPU1$_{2\/GPU}$'
    # log.loc[log['kernel'] == 'SPU2_2', 'kernel'] = 'SPU2$_{2\/GPU}$'

    ref = log[log['|B|'] == 1]['EPS'].values
    # log.loc[log['|B|'] == 2, 'EPS'] = log.loc[log['|B|'] == 2, 'EPS'] / ref
    # log.loc[log['|B|'] == 4, 'EPS'] = log.loc[log['|B|'] == 4, 'EPS'] / ref
    # log.loc[log['|B|'] == 8, 'EPS'] = log.loc[log['|B|'] == 8, 'EPS'] / ref
    log.loc[log['|B|'] == 16,'EPS'] = log.loc[log['|B|'] == 16,'EPS'] / ref

    log.loc[log['kernel'] == 'MPI3', 'kernel'] = 'MPI3$_{METIS}$'

    # log['B1'] = log['EPS']
    # log.loc[log['|B|'] == 1, 'B2'] = log.loc[log['|B|'] == 2, 'EPS'].values
    # log.loc[log['|B|'] == 1, 'B4'] = log.loc[log['|B|'] == 4, 'EPS'].values
    # log.loc[log['|B|'] == 1, 'B8'] = log.loc[log['|B|'] == 8, 'EPS'].values
    # log.loc[log['|B|'] == 1, 'B16']= log.loc[log['|B|'] == 16,'EPS'].values

    log = log.loc[log['|B|'] == 16]
    print(log)

    # log['base_ms'] = (log['base_us'] * log['it']) / 1e3
    # log['rank_ms'] = (log['rank_us'] * log['it']) / 1e3
    # log['diff_ms'] = (log['diff_us'] * log['it']) / 1e3

    # log['total_ms'] = log['init_ms'] + log['base_ms'] + log['rank_ms'] + log['diff_ms'] + log['trans_ms']
    # log['Init'] = log['init_ms'] / log['total_ms']
    # # log['Base'] = log['base_ms'] / log['total_ms']
    # log['Traversal'] = log['rank_ms'] / log['total_ms']
    # log['Update'] = (log['base_ms']+log['diff_ms']) / log['total_ms']
    # log['Sync'] = log['trans_ms'] / log['total_ms']

    # log = log[['graph', 'Init', 'Traversal', 'Update', 'Sync']]
    # print(log)

    # log['|B|'] = log['|B|']*log['|B|']

    # log['1'] = log['trans_ms'] + (log['rank_us'] / 1e3)
    # log['5'] = log['trans_ms'] + (log['rank_us'] / 1e3)*5
    # log['20'] = log['trans_ms'] + (log['rank_us'] / 1e3)*20
    # log = log.melt(id_vars=id_vars, value_vars=['1', '5', '20'], var_name='exec_it', value_name='exec_ms')

    # log = log.groupby(['graph', 'DATASET', 'exec_it', 'device'], as_index=False)['exec_ms'].min()
    # log = log.groupby(['graph', 'DATASET', 'exec_it'], as_index=False).aggregate(lambda data: data.iloc[0] / data.iloc[1])
    # log = log.sort_values(by=['exec_ms'])

    # log['$P_C$'] = 8.39 * log['Clustering'] + 1.63 # OMP3
    # log['$P_H$'] = 5.71 * log['cache_time'] + 1.17 # OMP3
    # log['$P_C$'] = 48.98 * log['Clustering'] + 6.84 # CUD3
    # log['$P_H$'] = 32.63 * log['cache_time'] + 4.37 # CUD3
    # log['$P_M$'] = log['EPS'].mean()
    # log['REF']   = log['EPS'].values

    # log['$P_C$'] = ((log['$P_C$'] - log['REF']).abs() / log['REF'])
    # log['$P_H$'] = ((log['$P_H$'] - log['REF']).abs() / log['REF'])
    # log['$P_M$'] = ((log['$P_M$'] - log['REF']).abs() / log['REF'])

    # print('RMSD for P_M', 
    #     log['EPS'].mean(),
    #     (log['$P_M$'] - log['REF']).abs().mean(),
    #     ((log['$P_M$'] - log['REF']) / log['REF']).abs().mean() * 100,
    #     ((log['$P_M$'] - log['REF']) ** 2).mean() ** .5,
    #     ((((log['$P_M$'] - log['REF']) / log['REF']) ** 2).mean() ** .5) * 100
    #     )
    # log = log.melt(id_vars=id_vars, value_vars=['$P_H$', '$P_C$', '$P_M$'], var_name='Predictor', value_name='pEPS')

    # print(log.groupby(['device'], as_index=False)['EPS'].mean())
    # print(log)

    sns.set('paper', 'ticks', color_codes=True)

    if ('-b' in opt):
        log = log[log['|B|'] == int(opt['-b'])]
    if ('-i' in opt):
        log = log[log['it'] == int(opt['-i'])]

    plt.figure(figsize=(3,3))

    # graphs = ['CIT', 'WWW', 'WIKI', 'EDU', 'SOC', 'COL', 'AS', 'HOL', 'BUB', 'OSM']
    # graphs = ['CIT', 'ER', 'TER', 'PA', 'WWW', 'WIKI', 'KRO', 'EDU', 'SOC', 'COL', 'AS', 'HOL', 'BUB', 'REG', 'OSM']
    graphs = ['WWW', 'SOC', 'OSM']

    with sns.plotting_context(), sns.axes_style():
        ax = sns.barplot(
            x="graph",
            y="EPS",
            hue="kernel",
            hue_order=['MPI1', 'MPI2', 'MPI3$_{METIS}$', 'MPI3$_{block}$', 'MPI3$_{rand}$'],
            # hue_order=['SPU1','SPU2'],
            # hue_order=[4, 16, 64, 256],
            # color="b",
            # palette=sns.color_palette(),
            order=graphs,
            data=log,
            # edgecolor=sns.color_palette("pastel"),
            linewidth=0.5,
        )

        # ax = log.set_index('graph').reindex(index = graphs).plot.bar(stacked=True, figsize=(4,3))

        plt.xlabel("")
        # plt.ylabel("$\\frac{gpu\\_transfer\\_time}{avg\\_iteration\\_time}$")
        plt.ylabel(" ")

        # for i, g in enumerate(graphs):
        #     r = log[log['graph'] == g].iloc[1]['EPS']
        #     plt.plot([-.5 + i, .5 + i], [r, r], 'k-', linewidth=0.8)

        plt.plot([-1, len(graphs)], [1, 1], 'r:')
        # plt.plot([-1, len(graphs)], [0, 0], 'k-')
        plt.xlim(-1, len(graphs))
        # plt.ylim(0, 1)

        colors = itertools.cycle(sns.color_palette())
        # colors = itertools.cycle(sns.light_palette("b"))
        # hatches = itertools.cycle(['///', '...', '\\\\\\', '---', '||||', '++++', ''])
        hatches = itertools.cycle(['///', '....', '\\\\\\', '--', '', '|||', '+++'])
        # next(colors)
        # next(hatches)
        for i, bar in enumerate(ax.patches):
            if i % len(graphs) == 0:
                hatch = next(hatches)
                color = next(colors)
            bar.set_facecolor(plot.darken(color, 1.5))
            bar.set_edgecolor(plot.darken(color, 0.8))
            bar.set_hatch(hatch)

        ax.grid(axis='y', linestyle='-', linewidth='0.1')
        ax.legend(loc='upper left', title='Kernel', framealpha=1)
        # ax.legend(loc='center left', title='Kernel', bbox_to_anchor=(1, 0.5), frameon=False)

        ax.get_legend().remove()

    plt.tight_layout()

    # print(log['EPS'].mean())
    # print(log.groupby(['device'])['EPS'].mean())

    if out == '-':
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()


# CUD3  ->  EPS = 48.98 * Clustering + 6.84
#           Correlation coefficient                  0.5157
#           Mean absolute error                      8.5639
#           Root mean squared error                 16.2897
#           Relative absolute error                 92.2462 %
#           Root relative squared error            123.648  %

#           EPS = 32.63 * cache_time + 4.37
#           Correlation coefficient                  0.6069
#           Mean absolute error                      7.0312
#           Root mean squared error                 15.0509
#           Relative absolute error                 75.7373 %
#           Root relative squared error            114.2449 %

#           EPS = 11.83 (MEAN)
#           Root mean squared error                 12.6503
#           Mean absolute error                      9.8032
#           Relative absolute error                189.0455 %
#           Root relative squared error            250.8872 %

# OMP3  ->  EPS = 8.39 * Clustering + 1.63
#           Correlation coefficient                  0.6403
#           Mean absolute error                      1.2339
#           Root mean squared error                  2.1779
#           Relative absolute error                 78.7713 %
#           Root relative squared error             99.8186 %

#           EPS = 5.71 * cache_time + 1.17
#           Correlation coefficient                  0.9029
#           Mean absolute error                      0.6685
#           Root mean squared error                  0.9242
#           Relative absolute error                 42.6783 %
#           Root relative squared error             42.3568 %

#           EPS = 2.42 (MEAN)
#           Root mean squared error                  2.0722
#           Mean absolute error                      1.7680
#           Relative absolute error                116.2252 %
#           Root relative squared error            138.7935 %

# % OSM
# % DIMACS10/europe_osm/europe_osm.mtx.el_gz
# %     2   51.955  426 (0%)
# %     4   50.359  1168 (0%)
# %     8   47.315  1883 (0%)
# %     16  53.046  3060 (0%)

# % BUB
# % DIMACS10/hugebubbles-00020/hugebubbles-00020.mtx.el_gz
# %     2   30.259  3234 (0%)
# %     4   30.250  13274 (0%)
# %     8   24.620  22086 (0%)
# %     16  27.136  37114 (0%)

# % CIT
# % SNAP/cit-Patents/cit-Patents.mtx.el_gz
# %     2   21.293  367808 (2%)
# %     4   23.667  656978 (4%)
# %     8   27.351  970154 (6%)
# %     16  37.722  1263040 (8%)

# % EDU
# % Gleich/wb-edu/wb-edu.mtx.el_gz
# %     2   13.945  54059 (0%)
# %     4   13.392  93564 (0%)
# %     8   17.270  125718 (0%)
# %     16  15.490  148717 (0%)

# % WIKI
# % Gleich/wikipedia-20070206/wikipedia-20070206.mtx.el_gz
# %     2   55.167  643204 (1%)
# %     4   72.153  1353636 (3%)
# %     8   70.196  2102437 (5%)
# %     16  93.239  3062527 (7%)

# % AS
# % SNAP/as-Skitter/as-Skitter.mtx.el_gz
# %     2   6.010   148403 (1%)
# %     4   6.052   292079 (1%)
# %     8   7.083   492317 (2%)
# %     16  7.006   692542 (3%)

# % SOC
# % SNAP/soc-LiveJournal1/soc-LiveJournal1.mtx.el_gz
# %     2   44.156  1075113 (2%)
# %     4   59.849  2129921 (3%)
# %     8   65.289  3725991 (5%)
# %     16  77.342  5133052 (7%)

# % WWW
# % LAW/arabic-2005/arabic-2005.mtx.el_gz
# %     2   69.086  83865 (0%)
# %     4   76.051  195494 (0%)
# %     8   77.451  287442 (0%)
# %     16  67.286  348675 (0%)

# % COL
# % DIMACS10/coPapersCiteseer/coPapersCiteseer.mtx.el_gz
# %     2   1.171   116671 (0%)
# %     4   1.559   224829 (1%)
# %     8   1.262   308266 (1%)
# %     16  1.417   409115 (1%)

# % HOL
# % LAW/hollywood-2009/hollywood-2009.mtx.el_gz
# %     2   16.448  567763 (1%)
# %     4   19.841  1412722 (1%)
# %     8   23.889  2156512 (2%)
# %     16  22.230  3443400 (3%)