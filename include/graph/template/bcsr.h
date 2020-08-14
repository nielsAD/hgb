// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/bcsr.h"
#include "util/file.h"

#ifndef GRAPH_NAME
    #include "graph/template/csr.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcsr_graph() {
    graph_size_t bcount;
    graph_size_t vcount;
    graph_size_t ecount;

    graph_flags_enum_t flags;

    csr_graph_t *restrict *blocks;      //[bcount*bcount]
    csr_graph_t *restrict *blocks_diag; //[bcount]
} bcsr_graph_t;

bcsr_graph_t *bcsr_graph(_new_ex)(csr_graph_t *restrict *blocks, const graph_size_t bcount, const graph_flags_enum_t flags);
bcsr_graph_t *bcsr_graph(_new)(const graph_size_t bcount, const graph_flags_enum_t flags);
bcsr_graph_t *bcsr_graph(_new_random)(const graph_size_t bcount, graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags);

bcsr_graph_t *bcsr_graph(_read)(FILE *stream, const graph_flags_enum_t flags, const graph_size_t bcount);
bcsr_graph_t *bcsr_graph(_read_file)(const char *const filename, const graph_flags_enum_t flags, const graph_size_t bcount, const char *const force_ext);

bcsr_graph_t *bcsr_graph(_copy)(const bcsr_graph_t *base, graph_flags_enum_t flags);
bcsr_graph_t *bcsr_graph(_copy_from_csr)(csr_graph_t *base, graph_flags_enum_t flags, const graph_size_t bcount);

void bcsr_graph(_free)(bcsr_graph_t *graph);
void bcsr_graph(_clear)(bcsr_graph_t *graph);
bool bcsr_graph(_equals)(const bcsr_graph_t *restrict first, const bcsr_graph_t *restrict second) PURE_FUN;

void bcsr_graph(_update_block_pointers)(bcsr_graph_t *graph);
void bcsr_graph(_update_block_degrees)(bcsr_graph_t *graph);

      size_t bcsr_graph(_byte_size)(bcsr_graph_t *graph, bool allocated) PURE_FUN;
graph_size_t bcsr_graph(_align_edges)(bcsr_graph_t *graph, const graph_size_t alignment, const graph_size_t dst);
  void bcsr_graph(_set_size)(bcsr_graph_t *graph, graph_size_t vsize);
  void bcsr_graph(_grow)(bcsr_graph_t *graph, graph_size_t vgrow);
  void bcsr_graph(_shrink)(bcsr_graph_t *graph);
  void bcsr_graph(_transpose)(bcsr_graph_t *graph);

bool bcsr_graph(_toggle_flag)(bcsr_graph_t *graph, const graph_flags_enum_t flag, const bool enable);

bool bcsr_graph(_get_vertex_index)(const bcsr_graph_t *graph, const graph_size_t idx, graph_size_t *bid, graph_size_t *bidx);
bool bcsr_graph(_get_edge_index)(const bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst, graph_size_t *block, graph_size_t *idx);

void bcsr_graph(_insert_vertices)(bcsr_graph_t *graph, graph_size_t idx, const graph_size_t count);
void bcsr_graph(_insert_vertex)(bcsr_graph_t *graph, graph_size_t idx);
graph_size_t bcsr_graph(_add_vertices)(bcsr_graph_t *graph, const graph_size_t count);
graph_size_t bcsr_graph(_add_vertex)(bcsr_graph_t *graph);

void bcsr_graph(_add_edges)(bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst, const graph_size_t count);
void bcsr_graph(_add_edge)(bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst);
void bcsr_graph(_add_edgelist)(bcsr_graph_t *graph, const graph_size_t *src, const graph_size_t *dst, const graph_size_t count);

void bcsr_graph(_calc_vertex_degrees)(const csr_graph_t *graph, graph_size_t *deg);
void bcsr_graph(_calc_vertex_degrees_in)(const csr_graph_t *graph, graph_size_t *deg);
void bcsr_graph(_calc_vertex_degrees_out)(const csr_graph_t *graph, graph_size_t *deg);

graph_size_t bcsr_graph(_get_vertex_degree)(const bcsr_graph_t *graph, const graph_size_t idx) PURE_FUN;
graph_size_t bcsr_graph(_get_vertex_degree_in)(const bcsr_graph_t *graph, const graph_size_t idx) PURE_FUN;
graph_size_t bcsr_graph(_get_vertex_degree_out)(const bcsr_graph_t *graph, const graph_size_t idx) PURE_FUN;

#ifdef __cplusplus
}
#endif

static inline csr_graph_t *bcsr_graph(_get_block)(const bcsr_graph_t *graph, const graph_size_t idx)
{
    assert(idx < graph->bcount * graph->bcount);
    return graph->blocks[idx];
}

static inline void bcsr_graph(_clear_tags)(bcsr_graph_t *graph)
{
    assert(graph != NULL);

    bcsr_forall_blocks(block, graph)
    {
        csr_graph(_clear_tags)(graph->blocks[block]);
    }
}

#if defined(GRAPH_V_TYPE) || defined(GRAPH_E_TYPE)
    static inline void bcsr_graph(_clear_values)(bcsr_graph_t *graph)
    {
        assert(graph != NULL);

        bcsr_forall_blocks(block, graph)
        {
            csr_graph(_clear_values)(graph->blocks[block]);
        }
    }
#endif

static inline graph_size_t bcsr_graph(_remove_dup_edges)(bcsr_graph_t *graph, const graph_merge_tag_func_t merge_tag, void *merge_arg)
{
    assert(graph != NULL);

    graph_size_t res = 0;
    bcsr_forall_blocks(block, graph)
    {
        res += csr_graph(_remove_dup_edges)(graph->blocks[block], merge_tag, merge_arg);
    }
    return res;
}

static inline graph_size_t bcsr_graph(_remove_self_loops)(bcsr_graph_t *graph)
{
    assert(graph != NULL);

    graph_size_t res = 0;
    bcsr_forall_blocks(block, graph)
    {
        res += csr_graph(_remove_self_loops)(graph->blocks[block]);
    }
    return res;
}

static inline graph_tag_t bcsr_graph(_get_vertex_tag)(const bcsr_graph_t *graph, const graph_size_t idx)
{
    graph_size_t bid;
    graph_size_t bidx;
    UNUSED bool res = bcsr_graph(_get_vertex_index)(graph, idx, &bid, &bidx);
    assert(res && "invalid vertex index");

    return csr_graph(_get_vertex_tag)(graph->blocks_diag[bid], bidx);
}

static inline void bcsr_graph(_set_vertex_tag)(const bcsr_graph_t *graph, const graph_size_t idx, graph_tag_t tag)
{
    graph_size_t bid;
    graph_size_t bidx;
    UNUSED bool res = bcsr_graph(_get_vertex_index)(graph, idx, &bid, &bidx);
    assert(res && "invalid vertex index");

    csr_graph(_set_vertex_tag)(graph->blocks_diag[bid], bidx, tag);
}

static inline graph_tag_t bcsr_graph(_get_edge_tag)(const bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst)
{
    graph_size_t bid;
    graph_size_t bidx;
    UNUSED bool res = bcsr_graph(_get_edge_index)(graph, src, dst, &bid, &bidx);
    assert(res && "invalid edge index");

    return csr_graph(_get_edge_tag)(graph->blocks_diag[bid], bidx);
}

static inline void bcsr_graph(_set_edge_tag)(const bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst, graph_tag_t tag)
{
    graph_size_t bid;
    graph_size_t bidx;
    UNUSED bool res = bcsr_graph(_get_edge_index)(graph, src, dst, &bid, &bidx);
    assert(res && "invalid edge index");

    csr_graph(_set_edge_tag)(graph->blocks_diag[bid], bidx, tag);
}

#ifdef GRAPH_V_TYPE
    static inline GRAPH_V_TYPE bcsr_graph(_get_vertex_value)(const bcsr_graph_t *graph, const graph_size_t idx)
    {
        graph_size_t bid;
        graph_size_t bidx;
        UNUSED bool res = bcsr_graph(_get_vertex_index)(graph, idx, &bid, &bidx);
        assert(res && "invalid vertex index");

        return csr_graph(_get_vertex_value)(graph->blocks_diag[bid], bidx);
    }

    static inline void bcsr_graph(_set_vertex_value)(const bcsr_graph_t *graph, const graph_size_t idx, GRAPH_V_TYPE val)
    {
        graph_size_t bid;
        graph_size_t bidx;
        UNUSED bool res = bcsr_graph(_get_vertex_index)(graph, idx, &bid, &bidx);
        assert(res && "invalid vertex index");

        csr_graph(_set_vertex_value)(graph->blocks_diag[bid], bidx, val);
    }
#endif

#ifdef GRAPH_E_TYPE
    static inline GRAPH_E_TYPE bcsr_graph(_get_edge_value)(const bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst)
    {
        graph_size_t bid;
        graph_size_t bidx;
        UNUSED bool res = bcsr_graph(_get_edge_index)(graph, src, dst, &bid, &bidx);
        assert(res && "invalid edge index");

        return csr_graph(_get_edge_value)(graph->blocks_diag[bid], bidx);
    }

    static inline void bcsr_graph(_set_edge_value)(const bcsr_graph_t *graph, const graph_size_t src, const graph_size_t dst, GRAPH_E_TYPE val)
    {
        graph_size_t bid;
        graph_size_t bidx;
        UNUSED bool res = bcsr_graph(_get_edge_index)(graph, src, dst, &bid, &bidx);
        assert(res && "invalid edge index");

        csr_graph(_set_edge_value)(graph->blocks_diag[bid], bidx, val);
    }
#endif
