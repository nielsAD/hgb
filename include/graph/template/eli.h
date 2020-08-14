// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/eli.h"
#include "util/file.h"

#ifndef GRAPH_NAME
    #error Undefined template macros, using placeholders for editor
    #define GRAPH_NAME(ID) __placeholder__ ## ID
    #define GRAPH_E_TYPE float
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eli_graph() {
    graph_size_t ecount;
    graph_size_t esize;

    graph_flags_enum_t flags;

    graph_size_t *restrict efr; //[ecount]
    graph_size_t *restrict eto; //[ecount]

    graph_tag_t *restrict etag; //[ecount]

#ifdef GRAPH_E_TYPE
    GRAPH_E_TYPE *restrict eval; //[ecount]
#endif
} eli_graph_t;

eli_graph_t *eli_graph(_new_ex)(graph_size_t *restrict efr, graph_size_t *restrict eto, const graph_size_t ecount, const graph_flags_enum_t flags);
eli_graph_t *eli_graph(_new)(const graph_flags_enum_t flags);
eli_graph_t *eli_graph(_new_random)(const graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags);
eli_graph_t *eli_graph(_new_regular)(const graph_size_t vcount, graph_size_t deg, graph_size_t stride, const graph_flags_enum_t flags);
eli_graph_t *eli_graph(_new_kronecker)(const graph_size_t vcount, graph_size_t ecount, const graph_flags_enum_t flags);

eli_graph_t *eli_graph(_deserialize)(FILE *stream);
eli_graph_t *eli_graph(_read)(FILE *stream, const graph_flags_enum_t flags);
eli_graph_t *eli_graph(_read_file)(const char *const filename, const graph_flags_enum_t flags, const char *const force_ext);

eli_graph_t *eli_graph(_copy)(const eli_graph_t *base, graph_flags_enum_t flags);
eli_graph_t *eli_graph(_mapped_copy)(const eli_graph_t *base, graph_flags_enum_t flags, const graph_map_edge_func_t map_edge, void *map_arg);

void eli_graph(_free)(eli_graph_t *graph);
void eli_graph(_clear)(eli_graph_t *graph);
void eli_graph(_clear_tags)(eli_graph_t *graph);
bool eli_graph(_equals)(const eli_graph_t *restrict first, const eli_graph_t *restrict second) PURE_FUN;

void eli_graph(_serialize)(const eli_graph_t *graph, FILE *stream);
void eli_graph(_write)(const eli_graph_t *graph, FILE *stream);
bool eli_graph(_write_file)(eli_graph_t *graph, const char *const filename, const char *const force_ext);

size_t eli_graph(_byte_size)(eli_graph_t *graph, bool allocated) PURE_FUN;
  void eli_graph(_set_size)(eli_graph_t *graph, graph_size_t esize);
  void eli_graph(_grow)(eli_graph_t *graph, graph_size_t egrow);
  void eli_graph(_shrink)(eli_graph_t *graph);

bool eli_graph(_toggle_flag)(eli_graph_t *graph, const graph_flags_enum_t flag, const bool enable);

        void eli_graph(_sort)(eli_graph_t *graph);
        void eli_graph(_transpose)(eli_graph_t *graph);
        void eli_graph(_to_directed)(eli_graph_t *graph);
graph_size_t eli_graph(_to_undirected)(eli_graph_t *graph);

graph_size_t eli_graph(_remove_dup_edges)(eli_graph_t *graph, const graph_merge_tag_func_t merge_tag, void *merge_arg);
graph_size_t eli_graph(_remove_self_loops)(eli_graph_t *graph);

void eli_graph(_get_vertex_range)(const eli_graph_t *graph, graph_size_t *min, graph_size_t *max);
void eli_graph(_shift_vertex_range)(eli_graph_t *graph, const int shift);
void eli_graph(_rebase_vertex_range)(eli_graph_t *graph, const graph_size_t idx_zero);
bool eli_graph(_get_edge_index)(const eli_graph_t *graph, const graph_size_t src, const graph_size_t dst, graph_size_t *idx);

graph_size_t eli_graph(_add_edges)(eli_graph_t *graph, const graph_size_t src, const graph_size_t dst, const graph_size_t count);
graph_size_t eli_graph(_add_edge)(eli_graph_t *graph, const graph_size_t src, const graph_size_t dst);
        void eli_graph(_add_edgelist)(eli_graph_t *graph, const graph_size_t *src, const graph_size_t *dst, const graph_size_t count);

#if defined(GRAPH_E_TYPE)
    void eli_graph(_clear_values)(eli_graph_t *graph);
#endif

#ifdef __cplusplus
}
#endif

static inline graph_tag_t eli_graph(_get_edge_tag)(const eli_graph_t *graph, const graph_size_t idx)
{
    assert(graph != NULL);
    assert(idx < graph->ecount);
    return (graph->etag == NULL)
        ? 0
        : graph->etag[idx];
}

static inline void eli_graph(_set_edge_tag)(eli_graph_t *graph, const graph_size_t idx, graph_tag_t tag)
{
    assert(graph != NULL);
    assert(idx < graph->ecount);
    assert(graph->etag != NULL);
    graph->etag[idx] = tag;
}

#ifdef GRAPH_E_TYPE
    static inline GRAPH_E_TYPE eli_graph(_get_edge_value)(const eli_graph_t *graph, const graph_size_t idx)
    {
        assert(graph != NULL);
        assert(idx < graph->ecount);
        assert(graph->eval != NULL);
        return graph->eval[idx];
    }

    static inline void eli_graph(_set_edge_value)(eli_graph_t *graph, const graph_size_t idx, GRAPH_E_TYPE val)
    {
        assert(graph != NULL);
        assert(idx < graph->ecount);
        assert(graph->eval != NULL);
        graph->eval[idx] = val;
    }
#endif
