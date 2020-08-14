// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/pcsr.h"
#include "util/file.h"

#ifndef GRAPH_NAME
    #include "graph/template/bcsr.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pcsr_graph() {
    graph_size_t part;
    graph_size_t pcount;
    graph_size_t vcount;

    pcsr_local_graph_t *restrict local_graph;
    pcsr_cross_graph_t *restrict cross_graph;
} pcsr_graph_t;

pcsr_graph_t *pcsr_graph(_new_ex)(const graph_size_t part, const graph_size_t pcount, const graph_size_t vcount, pcsr_local_graph_t *restrict local, pcsr_cross_graph_t *restrict cross);
pcsr_graph_t *pcsr_graph(_new)(const graph_size_t part, const graph_size_t pcount, const graph_size_t vcount, const graph_size_t bcount, const graph_flags_enum_t flags);
pcsr_graph_t *pcsr_graph(_copy)(const pcsr_graph_t *base, graph_flags_enum_t flags);

pcsr_graph_t *pcsr_graph(_read_file_transposed)(const graph_size_t part, const graph_size_t pcount, const graph_size_t bcount, const char *const base, const bool transposed_local, const bool transposed_cross, const graph_flags_enum_t flags, const char *const force_ext);
pcsr_graph_t *pcsr_graph(_read_file)(const graph_size_t part, const graph_size_t pcount, const graph_size_t bcount, const char *const base, const graph_flags_enum_t flags, const char *const force_ext);

void pcsr_graph(_free)(pcsr_graph_t *graph);
bool pcsr_graph(_equals)(const pcsr_graph_t *restrict first, const pcsr_graph_t *restrict second) PURE_FUN;

void pcsr_graph(_filter_cross_graph)(pcsr_graph_t *restrict graph);

#ifdef __cplusplus
}
#endif