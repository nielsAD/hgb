// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include "graph/csr.h"

#include <igraph.h>

#define igraph_eli_graph_t   eli_graph_v_t
#define igraph_eli_graph(ID) eli_graph_v ## ID

#define igraph_csr_graph_t   csr_graph_v_t
#define igraph_csr_graph(ID) csr_graph_v ## ID

#ifdef __cplusplus
extern "C" {
#endif

igraph_eli_graph_t *igraph_eli_graph(_from_igraph)(const igraph_t *graph, const graph_flags_enum_t flags);
igraph_csr_graph_t *igraph_csr_graph(_from_igraph)(const igraph_t *graph, const graph_flags_enum_t flags);

#ifdef __cplusplus
}
#endif