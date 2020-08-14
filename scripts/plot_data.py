# -*- coding: utf-8 -*-

# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0

import sys
import os
import re
import math
import subprocess
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

kernels = {
    'kern_rank_omp_def_row': '',
    'kern_rank_omp_stp_row': 'OMP1',
    'kern_rank_omp_bin_row': 'OMP3',
    'kern_rank_cpu_lib_row': 'MKL',
    'kern_rank_cud_def_row': '',
    'kern_rank_cud_stp_row': 'CUD1',
    'kern_rank_cud_wrp_row': 'CUD2',
    'kern_rank_cud_dyn_row': 'CUD3',
    'kern_rank_cud_lib_row': 'CSP',
    'kern_rank_ocl_def_row': '',
    'kern_rank_ocl_stp_row': 'OCL1',
    'kern_rank_ocl_wrp_row': 'OCL2',
    'kern_rank_omp_def_col': '',
    'kern_rank_omp_stp_col': 'OMP1',
    'kern_rank_omp_bin_col': 'OMP3',
    'kern_rank_cpu_lib_col': 'MKL',
    'kern_rank_cud_def_col': '',
    'kern_rank_cud_stp_col': 'CUD1',
    'kern_rank_cud_wrp_col': 'CUD2',
    'kern_rank_cud_dyn_col': 'CUD3',
    'kern_rank_cud_lib_col': 'CSP',
    'kern_rank_ocl_def_col': '',
    'kern_rank_ocl_stp_col': 'OCL1',
    'kern_rank_ocl_wrp_col': 'OCL2',

    'csr_omp_bin': 'OMP3',
    'csr_mkl_lib': 'MKL',
    'csr_cud_lib': 'CSP',

    'bcsr_ref_def': '',
    'bcsr_ref_stp': '',
    'bcsr_ref_map': '',
    'bcsr_omp_stp': 'OMP1',
    'bcsr_omp_map': 'OMP1',
    'bcsr_omp_rdx': 'OMP2',
    'bcsr_mpi_stp': 'MPI1',
    'bcsr_mpi_map': 'MPI1',
    'bcsr_mpi_rdx': 'MPI2',
    'pcsr_mpi_map': 'MPI3',
    'pcsr_mpi_stp': 'MPI3',
    'bcsr_spu_stp': 'SPU1',
    'bcsr_spu_map': 'SPU1',
    'bcsr_spu_rdx': 'SPU2',
    'bcsr_spu2_stp': 'SPU1_2',
    'bcsr_spu2_map': 'SPU1_2',
    'bcsr_spu2_rdx': 'SPU2_2',
    'bcsr_ocl_mix': '',
    'bcsr_ocl_stp': 'OCL1',
    'bcsr_ocl_map': 'OCL1',
    'bcsr_ocl_wrp': 'OCL2',
    'bcsr_cud_mix': '',
    'bcsr_cud_stp': 'CUD1',
    'bcsr_cud_map': 'CUD1',
    'bcsr_cud_wrp': 'CUD2',
    'bcsr_cud_dyn': 'CUD3',

    'csc_omp_bin': 'OMP3',
    'csc_mkl_lib': 'MKL',
    'csc_cud_lib': 'CSP',

    'bcsc_ref_def': '',
    'bcsc_ref_stp': '',
    'bcsc_ref_map': '',
    'bcsc_omp_stp': 'OMP1',
    'bcsc_omp_map': 'OMP1',
    'bcsc_omp_rdx': 'OMP2',
    'bcsc_mpi_stp': 'MPI1',
    'bcsc_mpi_map': 'MPI1',
    'bcsc_mpi_rdx': 'MPI2',
    'pcsc_mpi_map': 'MPI3',
    'pcsc_mpi_stp': 'MPI3',
    'bcsc_spu_stp': 'SPU1',
    'bcsc_spu_map': 'SPU1',
    'bcsc_spu_rdx': 'SPU2',
    'bcsc_spu2_stp': 'SPU1_2',
    'bcsc_spu2_map': 'SPU1_2',
    'bcsc_spu2_rdx': 'SPU2_2',
    'bcsc_ocl_mix': '',
    'bcsc_ocl_stp': 'OCL1',
    'bcsc_ocl_map': 'OCL1',
    'bcsc_ocl_wrp': 'OCL2',
    'bcsc_cud_mix': '',
    'bcsc_cud_stp': 'CUD1',
    'bcsc_cud_map': 'CUD1',
    'bcsc_cud_wrp': 'CUD2',
    'bcsc_cud_dyn': 'CUD3',

    'rndpcsr_mpi_stp': 'MPI3$_{rand}$',
    'blkpcsr_mpi_stp': 'MPI3$_{block}$',
    'rndpcsc_mpi_stp': 'MPI3$_{rand}$',
    'blkpcsc_mpi_stp': 'MPI3$_{block}$',
}

device = {
    'OMP1': 'CPU',
    'OMP2': 'CPU',
    'OMP3': 'CPU',
    'MKL':  'CPU',
    'OCL1': 'GPU',
    'OCL2': 'GPU',
    'CUD1': 'GPU',
    'CUD2': 'GPU',
    'CUD3': 'GPU',
    'CSP':  'GPU',
    'SPU1': 'CPU+GPU',
    'SPU2': 'CPU+GPU',
    'SPU1_2': 'CPU+GPU',
    'SPU2_2': 'CPU+GPU',
    'MPI1': 'CPUs',
    'MPI2': 'CPUs',
    'MPI3': 'CPUs',
}

flows = {
    'row':  'push',
    'csr':  'push',
    'bcsr': 'push',
    'pcsr': 'push',
    'col':  'pull',
    'csc':  'pull',
    'bcsc': 'pull',
    'pcsc': 'pull',

    'rndpcsr': 'push',
    'blkpcsr': 'push',
    'rndpcsc': 'pull',
    'blkpcsc': 'pull',
}

graphs = {
    'europe_osm':         'OSM',
    'hugebubbles-00020':  'BUB',
    'cit-Patents':        'CIT',
    'wb-edu':             'EDU',
    'wikipedia-20070206': 'WIKI',
    'as-Skitter':         'AS',
    'soc-LiveJournal1':   'SOC',
    'arabic-2005':        'WWW',
    'coPapersCiteseer':   'COL',
    'hollywood-2009':     'HOL',
    'regular':            'REG',
    'regular_stride':     'REG (stride dst)',
    'regular_stride_sd':  'REG (stride src+dst)',
    'erdos_renyi':        'ER',
    'triangular_erdos_renyi': 'TER',
    'trans_powerlaw' :    'PA',
    'kronecker':          'KRO'
}

graph_types = {
    'OSM':  'real-world',
    'BUB':  'real-world',
    'CIT':  'real-world',
    'EDU':  'real-world',
    'WIKI': 'real-world',
    'AS':   'real-world',
    'SOC':  'real-world',
    'WWW':  'real-world',
    'COL':  'real-world',
    'HOL':  'real-world',
    'REG':  'synthetic',
    'ER':   'synthetic',
    'TER':  'synthetic',
    'PA':   'synthetic',
    'KRO':  'synthetic',
    'REG (stride dst)': 'synthetic',
    'REG (stride src+dst)': 'synthetic',
}

markers = {
    'OMP1': ('light blue', 'o'),
    'OMP2': ('light blue', 'D'),
    'OMP3': ('light blue', '^'),
    'MKL':  ('light blue', '*'),
    'OCL1': ('light red',  'o'),
    'OCL2': ('light red',  'v'),
    'CUD1': ('light red',  'o'),
    'CUD2': ('light red',  'v'),
    'CUD3': ('light red',  '^'),
    'CSP':  ('light red',  '*'),

    'OSM':  ('light blue', 'o'),
    'BUB':  ('light blue', 'o'),
    'CIT':  ('light blue', 'o'),
    'EDU':  ('light blue', 'o'),
    'WIKI': ('light blue', 'o'),
    'AS':   ('light blue', 'o'),
    'SOC':  ('light blue', 'o'),
    'WWW':  ('light blue', 'o'),
    'COL':  ('light blue', 'o'),
    'HOL':  ('light blue', 'o'),

    'REG': ('blue',    'o'),
    'ER':  ('green',   '^'),
    'TER': ('cyan',    'v'),
    'PA':  ('magenta', 's'),
    'KRO': ('yellow',  'D'),

    'bcsc_ref_def': ('light tan',    '*'),
    'bcsc_ref_map': ('light tan',    's'),
    'bcsc_ref_stp': ('light tan',    'o'),
    'bcsc_omp_def': ('light blue',   '*'),
    'bcsc_omp_map': ('light blue',   's'),
    'bcsc_omp_rdx': ('light blue',   'D'),
    'bcsc_omp_stp': ('light blue',   'o'),
    'bcsc_mpi_def': ('light olive',  '*'),
    'bcsc_mpi_map': ('light olive',  's'),
    'bcsc_mpi_rdx': ('light olive',  'D'),
    'bcsc_mpi_stp': ('light olive',  'o'),
    'pcsc_mpi_def': ('light teal',   '*'),
    'pcsc_mpi_map': ('light teal',   's'),
    'pcsc_mpi_stp': ('light teal',   'o'),
    'bcsc_spu_map': ('light green',  's'),
    'bcsc_spu_rdx': ('light green',  'D'),
    'bcsc_spu_stp': ('light green',  'o'),
    'bcsc_ocl_map': ('light purple', 's'),
    'bcsc_ocl_mix': ('light purple', '+'),
    'bcsc_ocl_stp': ('light purple', 'o'),
    'bcsc_ocl_wrp': ('light purple', 'v'),
    'bcsc_cud_map': ('light red',    's'),
    'bcsc_cud_mix': ('light red',    '+'),
    'bcsc_cud_stp': ('light red',    'o'),
    'bcsc_cud_wrp': ('light red',    'v'),
    'bcsc_cud_dyn': ('light red',    '^'),

    'kern_rank_cpu_lib_col': ('light blue',   '*'),
    'kern_rank_omp_def_col': ('light blue',   '*'),
    'kern_rank_omp_map_col': ('light blue',   's'),
    'kern_rank_omp_stp_col': ('light blue',   'o'),
    'kern_rank_omp_bin_col': ('light blue',   '^'),
    'kern_rank_ocl_def_col': ('light purple', 's'),
    'kern_rank_ocl_stp_col': ('light purple', 'o'),
    'kern_rank_ocl_wrp_col': ('light purple', 'v'),
    'kern_rank_cud_def_col': ('light red',    's'),
    'kern_rank_cud_stp_col': ('light red',    'o'),
    'kern_rank_cud_wrp_col': ('light red',    'v'),
    'kern_rank_cud_dyn_col': ('light red',    '^'),
    'kern_rank_cud_lib_col': ('light red',    '*'),

    'bcsr_ref_def': ('dark tan',    '*'),
    'bcsr_ref_map': ('dark tan',    's'),
    'bcsr_ref_stp': ('dark tan',    'o'),
    'bcsr_omp_def': ('dark blue',   '*'),
    'bcsr_omp_map': ('dark blue',   's'),
    'bcsr_omp_rdx': ('dark blue',   'D'),
    'bcsr_omp_stp': ('dark blue',   'o'),
    'bcsr_mpi_def': ('dark olive',  '*'),
    'bcsr_mpi_map': ('dark olive',  's'),
    'bcsr_mpi_rdx': ('dark olive',  'D'),
    'bcsr_mpi_stp': ('dark olive',  'o'),
    'pcsr_mpi_def': ('dark teal',   '*'),
    'pcsr_mpi_map': ('dark teal',   's'),
    'pcsr_mpi_stp': ('dark teal',   'o'),
    'bcsr_spu_map': ('dark green',  's'),
    'bcsr_spu_rdx': ('dark green',  'D'),
    'bcsr_spu_stp': ('dark green',  'o'),
    'bcsr_ocl_map': ('dark purple', 's'),
    'bcsr_ocl_mix': ('dark purple', '+'),
    'bcsr_ocl_stp': ('dark purple', 'o'),
    'bcsr_ocl_wrp': ('dark purple', 'v'),
    'bcsr_cud_map': ('dark red',    's'),
    'bcsr_cud_mix': ('dark red',    '+'),
    'bcsr_cud_stp': ('dark red',    'o'),
    'bcsr_cud_wrp': ('dark red',    'v'),
    'bcsr_cud_dyn': ('dark red',    '^'),

    'kern_rank_cpu_lib_row': ('dark blue',   '*'),
    'kern_rank_omp_def_row': ('dark blue',   '*'),
    'kern_rank_omp_map_row': ('dark blue',   's'),
    'kern_rank_omp_stp_row': ('dark blue',   'o'),
    'kern_rank_omp_bin_row': ('dark blue',   '^'),
    'kern_rank_ocl_def_row': ('dark purple', 's'),
    'kern_rank_ocl_stp_row': ('dark purple', 'o'),
    'kern_rank_ocl_wrp_row': ('dark purple', 'v'),
    'kern_rank_cud_def_row': ('dark red',    's'),
    'kern_rank_cud_stp_row': ('dark red',    'o'),
    'kern_rank_cud_wrp_row': ('dark red',    'v'),
    'kern_rank_cud_dyn_row': ('dark red',    '^'),
    'kern_rank_cud_lib_row': ('dark red',    '*'),
}

fw_order = ['ref', 'omp', 'mpi', 'spu', 'spu2', 'ocl', 'cud']

colours = {colour: itertools.cycle(sns.light_palette(colour, len(list(count))+2, input='xkcd')[1:-1]) for colour, count in itertools.groupby(sorted(m[0] for m in markers.values()))}
markers = {f: (tuple(next(colours[c])), m) for f, (c, m) in sorted(markers.items())}

def filter_markers(filter):
    global markers
    if isinstance(filter, str):
        r = re.compile(filter)
        markers = {key: val for key, val in markers.items() if r.search(key) is not None}
    elif (filter is not None):
        filter = set(filter)
        markers = {key: val for key, val in markers.items() if key in filter}

def get_name(fname, part, parts):
    if parts > 1:
        fname = fname.rsplit('.', 1)
        fname = fname[0] + ('.p%d_%d' % (parts, part)) + ((len(fname) > 1 and '.'+fname[1]) or '')
    return fname

def get_deg(fname):
    if fname == '-':
        input = sys.stdin
    elif fname.endswith('.deg'):
        input = fname
    elif os.path.isfile(fname + '.deg'):
        input = fname + '.deg'
    else:
        input = subprocess.Popen(['%s/../bin/graphdeg' % os.path.dirname(os.path.abspath(__file__)), '-mOUT', fname], stdout=subprocess.PIPE).stdout
    return np.loadtxt(input, dtype=np.int32, ndmin=1)

def get_hist(fname):
    if fname.endswith('.hist'):
        input = fname
    elif os.path.isfile(fname + '.hist'):
        input = fname + '.hist'
    else:
        return np.bincount(get_deg(fname))
    return np.loadtxt(input, dtype=np.int32, ndmin=1)

def get_heat(fname):
    if fname == '-':
        input = sys.stdin
    elif fname.endswith('.heat'):
        input = fname
    elif os.path.isfile(fname + '.heat'):
        input = fname + '.heat'
    else:
        input = subprocess.Popen(['%s/../bin/graphheat' % os.path.dirname(os.path.abspath(__file__)), '-b32', fname], stdout=subprocess.PIPE).stdout
    heat = np.loadtxt(input, dtype=np.single, ndmin=1)
    return np.reshape(heat, (-1, int(math.sqrt(heat.size))))

def get_log(fname, filter = None):
    data = pd.read_csv(fname, delim_whitespace=True, comment='#')

    if (filter is not None):
        data = data[data['SOLVER'].str.contains(filter)]

    data = data.groupby(['DATASET', '|B|', 'it', 'SOLVER'], as_index=False).mean()

    data['all'] = 1
    data['DATAGROUP'] = data['DATASET'].str.extract('(.+?)(?:_s\\d+\\w*)?(?:_\\d+\\w*_\\d+\\w*)?\\.(?:[^.]+).el_gz')
    data['FRAMEWORK'] = data['SOLVER'].str.extract('(?:kern_)?[^_]+_([^_]+)')
    data['METHOD']    = data['SOLVER'].str.extract('(?:kern_)?[^_]+_[^_]+_([^_]+)')
    data['FLOW']      = data['SOLVER'].str.extract('((?:^.*(?:cs[rc]))|(?:(?:row|col)$))')
    data['FLOW']      = data['FLOW'].map(flows)
    data['kernel']    = data['SOLVER'].map(kernels)
    data['graph']     = data['DATAGROUP'].map(graphs)
    data['dataset']   = data['graph'].map(graph_types)
    data['device']    = data['kernel'].map(device)

    data['DATAGROUP'] += data.groupby(['DATAGROUP'])['DATASET'].transform('nunique').map(' ({0})'.format)
    data['FRAMEWORK'] += data.groupby(['FRAMEWORK'])['SOLVER'].transform('nunique').map(' ({0})'.format)
    data['METHOD']    += data.groupby(['METHOD'])['SOLVER'].transform('nunique').map(' ({0})'.format)

    data['time/it'] = data['time_ms'] / data['it']
    data['|E|/sec'] = data['|E|'] / (data['time/it'] / 1000)
    # data['EPS'] = (data['|E|'] / (data['rank_us'] / 1e6)) / 1e9
    data['EPS'] = (data['|E|'] / (data['time/it'] / 1e3)) / 1e9
    # data['PS'] = ((data['|V|'] + data['|E|']) / (data['rank_us'] / 1e6)) / 1e9
    data['PS'] = ((data['|V|'] + data['|E|']) / (data['time/it'] / 1e3)) / 1e9
    data['$\overline{D}$'] = data['|E|'] / data['|V|']

    return data

def merge_stats(fname, log):
    stats = pd.read_csv(fname, delim_whitespace=True, comment='#')
    stats = stats.drop(columns=['|V|', '|E|'])
    stats = stats.groupby(['DATASET'], as_index=False).mean()

    return pd.merge(log, stats, on='DATASET', how='left')