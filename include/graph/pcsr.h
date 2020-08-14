// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "graph/bcsr.h"

#define PCSR_GRAPH_MAX_PCOUNT 128

#define PCSR_CROSS_FLAGS_DEFAULT ((graph_flags_enum_t)(E_GRAPH_FLAG_TAG_E | E_GRAPH_FLAG_DEG_IO))
#define PCSR_CROSS_FLAGS_INHERIT(F) ((graph_flags_enum_t)(((F) & E_GRAPH_FLAG_PIN) | PCSR_CROSS_FLAGS_DEFAULT))

#define pcsr_graph(ID) STATIC_CONCAT(p,csr_graph(ID))
#define pcsr_graph_t   pcsr_graph(_t)

#define pcsr_local_graph(ID) bcsr_graph(ID)
#define pcsr_local_graph_t   bcsr_graph_t

#define pcsr_cross_graph(ID)     csr_graph_v ## ID
#define pcsr_cross_graph_t       csr_graph_v_t
#define pcsr_cross_graph_eli(ID) eli_graph_v ## ID
#define pcsr_cross_graph_eli_t   eli_graph_v_t

#ifdef __cplusplus
                       static inline graph_size_t PCSR_PCOUNT(graph_size_t p) { return p; }
    template <class T> static inline graph_size_t PCSR_PCOUNT(T *graph) { return graph->pcount; }
                       static inline graph_size_t PCSR_VCOUNT(graph_size_t v) { return v; }
    template <class T> static inline graph_size_t PCSR_VCOUNT(T *graph) { return graph->vcount; }
#else
    #define PCSR_PCOUNT(V) STATIC_IF_INT((P),(P),((const pcsr_graph_v_t *const restrict)((size_t)(P)))->pcount)
    #define PCSR_VCOUNT(V) STATIC_IF_INT((V),(V),((const pcsr_graph_v_t *const restrict)((size_t)(V)))->vcount)
#endif

#define pcsr_forall_parts(PID,LEN) \
    for (graph_size_t PID = 0, _m = PCSR_PCOUNT(LEN); \
         PID < _m; \
         PID++ \
    )
#define pcsr_forall_vertices(VID,LEN) \
    for (graph_size_t VID = 0, _m = PCSR_VCOUNT(LEN); \
         VID < _m; \
         VID++ \
    )

#define pcsr_forall_parts_par(PID,LEN,OMP) \
    OMP_PRAGMA(omp parallel for OMP) \
    for (graph_size_t PID=0; \
         PID < PCSR_PCOUNT(LEN); \
         PID++ \
    )
#define pcsr_forall_vertices_par(VID,LEN,OMP) \
    OMP_PRAGMA(omp parallel for OMP) \
    for (graph_size_t VID=0; \
         VID < PCSR_VCOUNT(LEN); \
         VID++ \
    )

char *pcsr_graph_filename_index(const char *const base, const graph_size_t num_parts);
char *pcsr_graph_filename_crossgraph(const char *const base, const graph_size_t num_parts);
char *pcsr_graph_filename_partition(const char *const base, const graph_size_t num_parts, const graph_size_t part);

graph_size_t pcsr_global_graph_vcount(const graph_size_t num_parts, const char *const base);

graph_size_t pcsr_cross_graph_tag_encode_parts(const graph_size_t fr, const graph_size_t to) CONST_FUN;
        void pcsr_cross_graph_tag_decode_parts(const graph_size_t tag, graph_size_t *fr, graph_size_t *to);
graph_size_t pcsr_cross_graph_tag_transpose_parts(const graph_size_t tag) CONST_FUN;

pcsr_cross_graph_t *pcsr_cross_graph_from_eli(const pcsr_cross_graph_eli_t *restrict base, const graph_size_t part, const bool transposed);
pcsr_cross_graph_t *pcsr_cross_graph_read_file(const graph_size_t part, const graph_size_t num_parts, const bool transposed, const graph_flags_enum_t flags, const char *const base);
              void  pcsr_cross_graph_sort_eli(pcsr_cross_graph_eli_t *restrict eli);
              void  pcsr_cross_graph_transpose_eli(pcsr_cross_graph_eli_t *restrict eli);

#define GRAPH_INCLUDE_FILE "graph/template/pcsr.h"
#include "graph/include_all_types.h"
#undef GRAPH_INCLUDE_FILE
