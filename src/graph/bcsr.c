// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/bcsr.h"

// Mapping functions for csr to bcsr
struct _bcsr_from_csr_mapping_args {
    graph_size_t bcount;
    graph_size_t row;
    graph_size_t col;
};

static bool _bcsr_from_csr_mapping_fun_v(const graph_size_t old_index, graph_size_t *restrict new_index, struct _bcsr_from_csr_mapping_args *restrict args)
{
    assert(new_index != NULL);
    assert(args != NULL);

    if (vertex_to_block_id(old_index, args->bcount) != args->row)
        return false;

    *new_index = vertex_to_block_index(old_index, args->bcount);
    return true;
}

static bool _bcsr_from_csr_mapping_fun_e(const graph_size_t old_src, const graph_size_t old_dst, graph_size_t *restrict new_src, graph_size_t *restrict new_dst, struct _bcsr_from_csr_mapping_args *restrict args)
{
    assert(new_src != NULL);
    assert(new_dst != NULL);
    assert(args != NULL);

    if ((vertex_to_block_id(old_src, args->bcount) != args->row) ||
        (vertex_to_block_id(old_dst, args->bcount) != args->col))
        return false;

    *new_src = vertex_to_block_index(old_src, args->bcount);
    *new_dst = vertex_to_block_index(old_dst, args->bcount);
    return true;
}

#define GRAPH_INCLUDE_FILE "graph/template/bcsr.c"
#include "graph/include_all_types.h"
#undef GRAPH_INCLUDE_FILE
