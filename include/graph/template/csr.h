// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/csr.h"
#include "util/file.h"

#ifndef GRAPH_NAME
    #include "graph/template/eli.h"
    #define GRAPH_V_TYPE float
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct csr_graph() {
    graph_size_t vcount;
    graph_size_t ecount;

    graph_size_t vsize;
    graph_size_t esize;

    graph_flags_enum_t flags;

    graph_size_t *restrict row_idx; //[vcount + 1]
    graph_size_t *restrict col_idx; //[ecount]

    graph_size_t *restrict deg_i; //[vcount]
    graph_size_t *restrict deg_o; //[vcount]

    graph_tag_t *restrict vtag; //[vcount]
    graph_tag_t *restrict etag; //[ecount]

#ifdef GRAPH_V_TYPE
    GRAPH_V_TYPE *restrict vval; //[vcount]
#endif

#ifdef GRAPH_E_TYPE
    GRAPH_E_TYPE *restrict eval; //[ecount]
#endif
} csr_graph_t;

csr_graph_t *csr_graph(_new_ex)(graph_size_t *restrict rid, graph_size_t *restrict cid, const graph_size_t vcount, const graph_flags_enum_t flags);
csr_graph_t *csr_graph(_new)(const graph_flags_enum_t flags);
csr_graph_t *csr_graph(_new_random)(const graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags);
csr_graph_t *csr_graph(_new_regular)(const graph_size_t vcount, graph_size_t deg, graph_size_t stride, const graph_flags_enum_t flags);
csr_graph_t *csr_graph(_new_kronecker)(const graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags);

csr_graph_t *csr_graph(_read)(FILE *stream, const graph_flags_enum_t flags);
csr_graph_t *csr_graph(_read_file)(const char *const filename, const graph_flags_enum_t flags, const char *const force_ext);

csr_graph_t *csr_graph(_copy)(const csr_graph_t *base, graph_flags_enum_t flags);
csr_graph_t *csr_graph(_mapped_copy_with_deg)(const csr_graph_t *base, const graph_size_t vcount, graph_size_t *local_deg_o, graph_flags_enum_t flags, const graph_map_vertex_func_t map_vertex, const graph_map_edge_func_t map_edge, void *map_arg);
csr_graph_t *csr_graph(_mapped_copy)(const csr_graph_t *base, const graph_flags_enum_t flags, const graph_map_vertex_func_t map_vertex, const graph_map_edge_func_t map_edge, void *map_arg);
csr_graph_t *csr_graph(_sorted_copy)(const csr_graph_t *base, const graph_flags_enum_t flags);

eli_graph_t *csr_graph(_get_eli_representation)(csr_graph_t *graph);
csr_graph_t *csr_graph(_convert_from_eli)(eli_graph_t *base, graph_flags_enum_t flags); // Invalidates base
csr_graph_t *csr_graph(_copy_from_eli)(eli_graph_t *base, graph_flags_enum_t flags);
       void  csr_graph(_free_eli_representation)(csr_graph_t *graph, eli_graph_t *eli);

void csr_graph(_free)(csr_graph_t *graph);
bool csr_graph(_equals)(const csr_graph_t *restrict first, const csr_graph_t *restrict second) PURE_FUN;

void csr_graph(_clear)(csr_graph_t *graph);
void csr_graph(_clear_tags)(csr_graph_t *graph);

void csr_graph(_write)(const csr_graph_t *graph, FILE *stream);
bool csr_graph(_write_file)(csr_graph_t *graph, const char *const filename, const char *const force_ext);

      size_t csr_graph(_byte_size)(csr_graph_t *graph, bool allocated) PURE_FUN;
graph_size_t csr_graph(_align_edges)(csr_graph_t *graph, const graph_size_t alignment, const graph_size_t dst);

void csr_graph(_set_size)(csr_graph_t *graph, graph_size_t vsize, graph_size_t esize);
void csr_graph(_grow)(csr_graph_t *graph, graph_size_t vgrow, graph_size_t egrow);
void csr_graph(_shrink)(csr_graph_t *graph);

bool csr_graph(_toggle_flag)(csr_graph_t *graph, const graph_flags_enum_t flag, const bool enable);

     void csr_graph(_transpose)(csr_graph_t *graph);
     void csr_graph(_to_directed)(csr_graph_t *graph);
graph_size_t csr_graph(_to_undirected)(csr_graph_t *graph);

graph_size_t csr_graph(_remove_dup_edges)(csr_graph_t *graph, const graph_merge_tag_func_t merge_tag, void *merge_arg);
graph_size_t csr_graph(_remove_self_loops)(csr_graph_t *graph);
graph_size_t csr_graph(_remove_unconnected)(csr_graph_t *graph);

bool csr_graph(_get_edge_index)(const csr_graph_t *graph, const graph_size_t src, const graph_size_t dst, graph_size_t *idx);

void csr_graph(_insert_vertices)(csr_graph_t *graph, graph_size_t idx, const graph_size_t count);
void csr_graph(_insert_vertex)(csr_graph_t *graph, graph_size_t idx);
graph_size_t csr_graph(_add_vertices)(csr_graph_t *graph, const graph_size_t count);
graph_size_t csr_graph(_add_vertex)(csr_graph_t *graph);

graph_size_t csr_graph(_add_edges)(csr_graph_t *graph, const graph_size_t src, const graph_size_t dst, const graph_size_t count);
graph_size_t csr_graph(_add_edge)(csr_graph_t *graph, const graph_size_t src, const graph_size_t dst);
        void csr_graph(_add_edgelist)(csr_graph_t *graph, const graph_size_t *src, const graph_size_t *dst, const graph_size_t count);

void csr_graph(_calc_vertex_degrees)(const csr_graph_t *graph, graph_size_t *deg);
void csr_graph(_calc_vertex_degrees_in)(const csr_graph_t *graph, graph_size_t *deg);
void csr_graph(_calc_vertex_degrees_out)(const csr_graph_t *graph, graph_size_t *deg);
void csr_graph(_recalc_managed_degrees)(const csr_graph_t *graph, const bool zero_mem);

      graph_size_t *csr_graph(_get_vertex_degrees)(csr_graph_t *graph);
const graph_size_t *csr_graph(_get_vertex_degrees_in)(csr_graph_t *graph);
const graph_size_t *csr_graph(_get_vertex_degrees_out)(csr_graph_t *graph);

graph_size_t csr_graph(_get_vertex_degree)(const csr_graph_t *graph, const graph_size_t idx) PURE_FUN;
graph_size_t csr_graph(_get_vertex_degree_in)(const csr_graph_t *graph, const graph_size_t idx) PURE_FUN;
graph_size_t csr_graph(_get_vertex_degree_out)(const csr_graph_t *graph, const graph_size_t idx) PURE_FUN;

double csr_graph(_avg_clustering_coefficient)(csr_graph_t *graph);
double csr_graph(_avg_neighbor_degree_in)(csr_graph_t *graph);
double csr_graph(_avg_neighbor_degree_out)(csr_graph_t *graph);
double csr_graph(_degree_assortativity)(csr_graph_t *graph);

#if defined(GRAPH_V_TYPE) || defined(GRAPH_E_TYPE)
    void csr_graph(_clear_values)(csr_graph_t *graph);
#endif

#ifdef __cplusplus
}
#endif

static inline graph_tag_t csr_graph(_get_vertex_tag)(const csr_graph_t *graph, const graph_size_t idx)
{
    assert(graph != NULL);
    assert(idx < graph->vcount);
    return (graph->vtag == NULL)
        ? 0
        : graph->vtag[idx];
}

static inline void csr_graph(_set_vertex_tag)(csr_graph_t *graph, const graph_size_t idx, graph_tag_t tag)
{
    assert(graph != NULL);
    assert(idx < graph->vcount);
    assert(graph->vtag != NULL);
    graph->vtag[idx] = tag;
}

static inline graph_tag_t csr_graph(_get_edge_tag)(const csr_graph_t *graph, const graph_size_t idx)
{
    assert(graph != NULL);
    assert(idx < graph->ecount);
    return (graph->etag == NULL)
        ? 0
        : graph->etag[idx];
}

static inline void csr_graph(_set_edge_tag)(csr_graph_t *graph, const graph_size_t idx, graph_tag_t tag)
{
    assert(graph != NULL);
    assert(idx < graph->ecount);
    assert(graph->etag != NULL);
    graph->etag[idx] = tag;
}

#ifdef GRAPH_V_TYPE
    static inline GRAPH_V_TYPE csr_graph(_get_vertex_value)(const csr_graph_t *graph, const graph_size_t idx)
    {
        assert(graph != NULL);
        assert(idx < graph->vcount);
        assert(graph->vval != NULL);
        return graph->vval[idx];
    }

    static inline void csr_graph(_set_vertex_value)(csr_graph_t *graph, const graph_size_t idx, GRAPH_V_TYPE val)
    {
        assert(graph != NULL);
        assert(idx < graph->vcount);
        assert(graph->vval != NULL);
        graph->vval[idx] = val;
    }
#endif

#ifdef GRAPH_E_TYPE
    static inline GRAPH_E_TYPE csr_graph(_get_edge_value)(const csr_graph_t *graph, const graph_size_t idx)
    {
        assert(graph != NULL);
        assert(idx < graph->ecount);
        assert(graph->eval != NULL);
        return graph->eval[idx];
    }

    static inline void csr_graph(_set_edge_value)(csr_graph_t *graph, const graph_size_t idx, GRAPH_E_TYPE val)
    {
        assert(graph != NULL);
        assert(idx < graph->ecount);
        assert(graph->eval != NULL);
        graph->eval[idx] = val;
    }
#endif
