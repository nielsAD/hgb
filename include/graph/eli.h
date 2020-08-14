// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "graph/graph.h"
#include "util/string.h"
#include "util/openmp.h"

#define ELI_VALID_FLAGS_MASK ((graph_flags_enum_t) ( \
    GRAPH_VALID_FLAGS_MASK & \
    ~E_GRAPH_FLAG_TAG_V    & \
    ~E_GRAPH_FLAG_VAL_V    & \
    ~E_GRAPH_FLAG_DEG_I    & \
    ~E_GRAPH_FLAG_DEG_O      \
))

#define eli_graph(ID) STATIC_CONCAT(eli_,GRAPH_NAME(ID))
#define eli_graph_t eli_graph(_t)

#ifdef __cplusplus
                       static inline graph_size_t ELI_ECOUNT(graph_size_t e) { return e; }
    template <class T> static inline graph_size_t ELI_ECOUNT(T *graph) { return graph->ecount; }
#else
    #define ELI_ECOUNT(E) STATIC_IF_INT((E),(E),((const eli_graph_v_t *const restrict)((size_t)(E)))->ecount)
#endif

#define eli_forall_edges6(EID,SRC,DST,EFR,ETO,LEN) \
    for (graph_size_t EID = 0, SRC, DST, *const restrict _efr = (EFR), *const restrict _eto = (ETO), _m = ELI_ECOUNT(LEN); \
         EID < _m && ((SRC = _efr[EID]) | (DST = _eto[EID]) | 1); \
         EID++ \
    )
#define eli_forall_edges5(SRC,DST,EFR,ETO,LEN) eli_forall_edges6(_e,SRC,DST,EFR,ETO,LEN)
#define eli_forall_edges4(EID,SRC,DST,ELI)     eli_forall_edges6(EID,SRC,DST,(ELI)->efr,(ELI)->eto,ELI)
#define eli_forall_edges3(SRC,DST,ELI)         eli_forall_edges4(_e,SRC,DST,ELI)
#define eli_forall_edges2(EID,LEN) \
    for (graph_size_t EID = 0, _m = ELI_ECOUNT(LEN); \
         EID < _m; \
         EID++ \
    )
#define eli_forall_edges(...) VARARG(eli_forall_edges,__VA_ARGS__)

#define eli_forall_edges_par(EID,LEN,OMP) \
    OMP_PRAGMA(omp parallel for OMP) \
    for (graph_size_t EID = 0; \
         EID < ELI_ECOUNT(LEN); \
         EID++ \
    )

#define GRAPH_INCLUDE_FILE "graph/template/eli.h"
#include "graph/include_all_types.h"
#undef GRAPH_INCLUDE_FILE
