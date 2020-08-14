HGB
===
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

The Heterogenous Graph-processing Benchmark framework provides a toolset that eases loading, generating, and manipulating large-scale graphs in order to benchmark heterogeneous systems.

Features
--------

* Written in `C11`
* Benchmark kernels written in `OpenMP`, `OpenCL`, `CUDA`, `MPI`, `StarPU`
* Accelerator-ready graph data structures (`ELI`, `CSR`, `CSC`)
* Pluggable memory manager (pinned memory by default, out-of-core memory supported)
* Read/write common file formats (METIS, Matrix Market Exchange, compressed (un)directed adjacency lists)
* Large-scale synthetic graph generators (such as Erdos-Renyi, Barabasi, or Kronecker)
* Partitioning interface (block-based or edge-cut)
* Visualization tools for benchmark results and graph structure

Tools
-----

### graphdeg

Examine vertex degree.

```
Usage: graphdeg [OPTION...] [input_filename] [output_filename]

  -D, --disk[=directory]     Specify directory to store intermediates in
                             temporary files to reduce memory load. Working dir
                             by default.
  -h, --histogram[=bool]     Output vertex count per degree rather than degree
                             count per vertex (histogram).
  -i, --input=filename       Input filename.
  -I, -f, --input_format=string, --format=string
                             Force reader to assume this graph format (e.g.
                             `dimacs`,`csr`,`el`).
  -m, --method=['in','out','both']
                             Degree counting method.
  -o, --output=filename      Output filename.
  -s, --summary[=bool]       Print summary.
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
```

### graphgen

Generate a graph using one of the following generators: 'random', 'grg_random', 'barabasi', 'forest_fire', 'powerlaw', 'kronecker', 'regular', 'lattice', 'star', 'tree'.

```
Usage: graphgen [OPTION...] numvertices [output_filename]

  -a, --algo=alg             Generation algorithm.
  -D, --disk[=directory]     Specify directory to store intermediates in
                             temporary files to reduce memory load. Working dir
                             by default.
  -e, --nedge=num            Number of edges.
  -m, --medge=num            Number of edges modifier (multiplied by number of
                             vertices).
  -o, --output=filename      Output filename.
  -O, -f, --output_format=string, --format=string
                             Force writer to assume this graph format (e.g.
                             `dimacs`,`csr`,`el`).
  -r, --random=int           Random seed.
  -v, --nvert=num            Number of vertices.
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
```

### graphheat

Generate graph heat map.

```
Usage: graphheat [OPTION...] [input_filename] [output_filename]

  -b, --bins=integer         Number of row/col bins.
  -D, --disk[=directory]     Specify directory to store intermediates in
                             temporary files to reduce memory load. Working dir
                             by default.
  -i, --input=filename       Input filename.
  -I, -f, --input_format=string, --format=string
                             Force reader to assume this graph format (e.g.
                             `dimacs`,`csr`,`el`).
  -o, --output=filename      Output filename.
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
```

### graphpart

Distributes input graph over several partitions.

```
Usage: graphpart [OPTION...] num_parts output_filename [input_filename]

  -c, --no_crossgraph        Output graph containing the crossing edges between
                             partitions.
  -D, --disk[=directory]     Specify directory to store intermediates in
                             temporary files to reduce memory load. Working dir
                             by default.
  -f, --format=string        Force reader/writer to use this format (e.g.
                             `dimacs`,`csr`,`el`).
  -i, --input=filename       Input filename.
  -I, --input_format=string  Force reader to assume this graph format (e.g.
                             `dimacs`,`csr`,`el`).
  -m, --method=['block','random','file']
                             Graph partitioning strategy. Blocks by default.
  -n, --no_merge             Do not merge crossing edges with the same
                             destination.
  -o, --output=filename      Output filename base. Is appended suffix depending
                             on result file.
  -O, --output_format=string Force write to use this graph format (e.g.
                             `dimacs`,`csr`,`el`).
  -p, --input_parts=filename File which indicates partition per vector. Implies
                             method=file.
  -r, --random=int           Random seed.
  -s, --no_subgraph          Output graph files containing partition
                             subgraphs.
  -x, --no_index             Output index file containing partition and index
                             for each vertex.
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
```

### graphsan

Sanitizes graph structure and converts between graph formats.

```
Usage: graphsan [OPTION...] [input_filename] [output_filename]

  -a, --align=int            Align the number of edges for every vertex to a
                             multipe of this number.
  -c, --connected            Remove unconnected vertices.
  -d, --to_directed          For every edge (x:y), also add (y:x) to the graph.
                            
  -D, --disk[=directory]     Specify file template (ending with `XXXXXX`) to
                             store intermediates in temporary files to reduce
                             memory load. Working dir by default.
  -f, --format=string        Force reader/writer to use this format (e.g.
                             `dimacs`,`csr`,`el`).
  -i, --input=filename       Input filename.
  -I, --input_format=string  Force reader to assume this graph format (e.g.
                             `dimacs`,`csr`,`el`).
  -l, --noloops              Remove self loops.
  -m, --dimacs               Set settings according to the DIMACS specification
                             (produces valid input for e.g. Metis, Chaco,
                             KaHIP).
  -o, --output=filename      Output filename.
  -O, --output_format=string Force write to use this graph format (e.g.
                             `dimacs`,`csr`,`el`).
  -q, --unique               Remove duplicate edges.
  -s, --sort                 Sort vertices by out-degree.
  -t, --transpose            Reverse the direction of all edges.
  -u, --to_undirected        Only keep edges (x:y) where (x <= y).
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
```