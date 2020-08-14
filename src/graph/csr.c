// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/csr.h"

static int _reorder_col_idx_compare(const void *a, const void *b)
{
    return *(graph_size_t*)a - *(graph_size_t*)b;
}

static int _reorder_by_degree_compare(const void *a, const void *b, void *c)
{
    const graph_size_t *ctx = (graph_size_t*)c;
    return ctx[*(graph_size_t*)b] - ctx[*(graph_size_t*)a];
}

static bool _idx_mapping_fun_v(const graph_size_t old_index, graph_size_t *restrict new_index, graph_size_t *restrict idx)
{
    *new_index = idx[old_index];
    return true;
}

static bool _idx_mapping_fun_e(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, graph_size_t *restrict idx)
{
    *new_src = idx[old_src];
    *new_dst = idx[old_dst];
    return true;
}

#define GRAPH_INCLUDE_FILE "graph/template/csr.c"
#include "graph/include_all_types.h"
#undef GRAPH_INCLUDE_FILE
