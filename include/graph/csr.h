// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "graph/eli.h"

#define CSR_VALID_FLAGS_MASK GRAPH_VALID_FLAGS_MASK

#define csr_graph(ID) STATIC_CONCAT(csr_,GRAPH_NAME(ID))
#define csr_graph_t csr_graph(_t)

#ifdef __cplusplus
                       static inline graph_size_t CSR_VCOUNT(graph_size_t v) { return v; }
    template <class T> static inline graph_size_t CSR_VCOUNT(T *graph) { return graph->vcount; }
                       static inline graph_size_t CSR_ECOUNT(graph_size_t e) { return e; }
    template <class T> static inline graph_size_t CSR_ECOUNT(T *graph) { return graph->ecount; }
#else
    #define CSR_VCOUNT(V) STATIC_IF_INT((V),(V),((const csr_graph_v_t *const restrict)((size_t)(V)))->vcount)
    #define CSR_ECOUNT(E) STATIC_IF_INT((E),(E),((const csr_graph_v_t *const restrict)((size_t)(E)))->ecount)
#endif

#define csr_forall_nodes csr_forall_vertices
#define csr_forall_vertices(VID,LEN) \
    for (graph_size_t VID = 0, _m = CSR_VCOUNT(LEN); \
         VID < _m; \
         VID++ \
    )

#define csr_forall_out_edges5(EID,DST,SRC,RID,CID) \
    for (graph_size_t DST, _src=(SRC), *const restrict _rid = (RID), *const restrict _cid = (CID), EID = _rid[_src], _m = _rid[_src+1]; \
         EID < _m && ((DST = _cid[EID]) | 1); \
         EID++ \
    )
#define csr_forall_out_edges4(EID,DST,SRC,CSR) csr_forall_out_edges5(EID,DST,SRC,(CSR)->row_idx,(CSR)->col_idx)
#define csr_forall_out_edges3(DST,SRC,CSR)     csr_forall_out_edges4(_e,DST,SRC,CSR)
#define csr_forall_out_edges(...) VARARG(csr_forall_out_edges,__VA_ARGS__)

#define csr_forall_edges6(EID,SRC,DST,RID,CID,LEN) csr_forall_vertices(SRC,LEN) csr_forall_out_edges5(EID,DST,SRC,RID,CID)
#define csr_forall_edges5(SRC,DST,RID,CID,LEN)     csr_forall_vertices(SRC,LEN) csr_forall_out_edges5(_e,DST,SRC,RID,CID)
#define csr_forall_edges4(EID,SRC,DST,CSR)         csr_forall_vertices(SRC,CSR) csr_forall_out_edges4(EID,DST,SRC,CSR)
#define csr_forall_edges3(SRC,DST,CSR)             csr_forall_vertices(SRC,CSR) csr_forall_out_edges3(DST,SRC,CSR)
#define csr_forall_edges2(EID,LEN) \
    for (graph_size_t EID = 0, _m = CSR_ECOUNT(LEN); \
         EID < _m; \
         EID++ \
    )
#define csr_forall_edges(...) VARARG(csr_forall_edges,__VA_ARGS__)

#define csr_forall_nodes_par csr_forall_vertices_par
#define csr_forall_vertices_par(VID,LEN,OMP) \
    OMP_PRAGMA(omp parallel for OMP) \
    for (graph_size_t VID = 0; \
         VID < CSR_VCOUNT(LEN); \
         VID++ \
    )

#define csr_forall_edges_par7(EID,SRC,DST,RID,CID,LEN,OMP) csr_forall_vertices_par(SRC,LEN,OMP) csr_forall_out_edges5(EID,DST,SRC,RID,CID)
#define csr_forall_edges_par6(SRC,DST,RID,CID,LEN,OMP)     csr_forall_vertices_par(SRC,LEN,OMP) csr_forall_out_edges5(_e,DST,SRC,RID,CID)
#define csr_forall_edges_par5(EID,SRC,DST,CSR,OMP)         csr_forall_vertices_par(SRC,CSR,OMP) csr_forall_out_edges4(EID,DST,SRC,CSR)
#define csr_forall_edges_par4(SRC,DST,CSR,OMP)             csr_forall_vertices_par(SRC,CSR,OMP) csr_forall_out_edges3(DST,SRC,CSR)
#define csr_forall_edges_par3(EID,LEN,OMP) \
    OMP_PRAGMA(omp parallel for OMP) \
    for (graph_size_t EID = 0; \
         EID < CSR_ECOUNT(LEN); \
         EID++ \
    )
#define csr_forall_edges_par(...) VARARG(csr_forall_edges_par,__VA_ARGS__)

#define GRAPH_INCLUDE_FILE "graph/template/csr.h"
#include "graph/include_all_types.h"
#undef GRAPH_INCLUDE_FILE
