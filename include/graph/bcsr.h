// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "graph/csr.h"

#define BCSR_GRAPH_MAX_BCOUNT  128
#define BCSR_GRAPH_VERTEX_PACK (128 / sizeof(graph_size_t)) // Size of an average cache line

#define BCSR_VALID_FLAGS_MASK CSR_VALID_FLAGS_MASK
#define BCSR_NONDIAG_VALID_FLAGS_MASK ((graph_flags_enum_t) ( \
    BCSR_VALID_FLAGS_MASK & \
    ~E_GRAPH_FLAG_TAG_V   & \
    ~E_GRAPH_FLAG_VAL_V   & \
    ~E_GRAPH_FLAG_DEG_I   & \
    ~E_GRAPH_FLAG_DEG_O     \
))

OMP_PRAGMA(omp declare simd uniform(bcount) linear(idx))
static inline graph_size_t vertex_to_block_id(const graph_size_t idx, const graph_size_t bcount)
{
    return (idx / BCSR_GRAPH_VERTEX_PACK) % bcount;
}

OMP_PRAGMA(omp declare simd uniform(bcount) linear(idx))
static inline graph_size_t vertex_to_block_index(const graph_size_t idx, const graph_size_t bcount)
{
    return ((idx / BCSR_GRAPH_VERTEX_PACK) / bcount) * BCSR_GRAPH_VERTEX_PACK + (idx % BCSR_GRAPH_VERTEX_PACK);
}

#define bcsr_graph(ID) STATIC_CONCAT(bcsr_,GRAPH_NAME(ID))
#define bcsr_graph_t bcsr_graph(_t)

#ifdef __cplusplus
                       static inline graph_size_t BCSR_BCOUNT(graph_size_t b) { return b; }
    template <class T> static inline graph_size_t BCSR_BCOUNT(T *graph) { return graph->bcount; }
                       static inline graph_size_t BCSR_VCOUNT(graph_size_t v) { return v; }
    template <class T> static inline graph_size_t BCSR_VCOUNT(T *graph) { return graph->vcount; }
                       static inline graph_size_t BCSR_ECOUNT(graph_size_t e) { return e; }
    template <class T> static inline graph_size_t BCSR_ECOUNT(T *graph) { return graph->ecount; }
#else
    #define BCSR_BCOUNT(B) STATIC_IF_INT((B),(B),((const bcsr_graph_v_t *const restrict)((size_t)(B)))->bcount)
    #define BCSR_VCOUNT(V) STATIC_IF_INT((V),(V),((const bcsr_graph_v_t *const restrict)((size_t)(V)))->vcount)
    #define BCSR_ECOUNT(E) STATIC_IF_INT((E),(E),((const bcsr_graph_v_t *const restrict)((size_t)(E)))->ecount)
#endif

#define bcsr_forall_blocks4(BID,ROW,COL,LEN) \
    for (graph_size_t BID = 0, ROW = 0, COL = 0, _c = BCSR_BCOUNT(LEN), _m = _c*_c; \
         BID < _m; \
         ROW=((COL+1==_c)?ROW+1:ROW), COL=((COL+1==_c)?0:COL+1), BID++ \
    )
#define bcsr_forall_blocks3(ROW,COL,LEN) bcsr_forall_blocks4(_b,ROW,COL,LEN)
#define bcsr_forall_blocks2(BID,LEN) \
    for (graph_size_t BID = 0, _c = BCSR_BCOUNT(LEN), _m = _c*_c; \
         BID < _m; \
         BID++ \
    )
#define bcsr_forall_blocks(...) VARARG(bcsr_forall_blocks,__VA_ARGS__)

#define bcsr_forall_diag_blocks3(BID,BIDX,LEN) \
    for (graph_size_t BID = 0, BIDX = 0, _m = BCSR_BCOUNT(LEN); \
         BID < _m; \
         BIDX += _m + 1, BID++ \
    )
#define bcsr_forall_diag_blocks2(BID,LEN) bcsr_forall_diag_blocks3(BID,_bidx,LEN)
#define bcsr_forall_diag_blocks(...) VARARG(bcsr_forall_diag_blocks,__VA_ARGS__)

#define bcsr_forall_nodes bcsr_forall_vertices
#define bcsr_forall_vertices5(VID,BID,BIDX,BLEN,VLEN) \
    for (graph_size_t VID = 0, BID, BIDX, _b = BCSR_BCOUNT(BLEN), _m = BCSR_VCOUNT(VLEN); \
         VID < _m && ((BID = vertex_to_block_id(VID,_b)) | (BIDX = vertex_to_block_index(VID,_b)) | 1); \
         VID++ \
    )
#define bcsr_forall_vertices4(VID,BID,BIDX,BCSR) bcsr_forall_vertices5(VID,BID,BIDX,BCSR,BCSR)
#define bcsr_forall_vertices3(BID,BIDX,BCSR) bcsr_forall_vertices5(_v,BID,BIDX,BCSR,BCSR)
#define bcsr_forall_vertices2(VID,LEN) \
    for (graph_size_t VID = 0, _m = BCSR_VCOUNT(LEN); \
         VID < _m; \
         VID++ \
    )
#define bcsr_forall_vertices(...) VARARG(bcsr_forall_vertices,__VA_ARGS__)

#define bcsr_forall_blocks_par4(ROW,COL,LEN,OMP) \
    OMP_PRAGMA(omp parallel for collapse(2) OMP) \
    for (graph_size_t ROW=0; ROW < BCSR_BCOUNT(LEN); ROW++) \
        for (graph_size_t COL=0; COL < BCSR_BCOUNT(LEN); COL++)
#define bcsr_forall_blocks_par3(BID,LEN,OMP) \
    OMP_PRAGMA(omp parallel for OMP) \
    for (graph_size_t BID=0; \
         BID < BCSR_BCOUNT(LEN)*BCSR_BCOUNT(LEN); \
         BID++ \
    )
#define bcsr_forall_blocks_par(...) VARARG(bcsr_forall_blocks_par,__VA_ARGS__)

#define bcsr_forall_diag_blocks_par(BID,LEN,OMP) \
    OMP_PRAGMA(omp parallel for OMP) \
    for (graph_size_t BID = 0; \
         BID < BCSR_BCOUNT(LEN); \
         BID++ \
    )

#define bcsr_forall_nodes_par bcsr_forall_vertices_par
#define bcsr_forall_vertices_par(VID,LEN,OMP) \
    OMP_PRAGMA(omp parallel for OMP) \
    for (graph_size_t VID = 0; \
         VID < BCSR_VCOUNT(LEN); \
         VID++ \
    )

#define GRAPH_INCLUDE_FILE "graph/template/bcsr.h"
#include "graph/include_all_types.h"
#undef GRAPH_INCLUDE_FILE
