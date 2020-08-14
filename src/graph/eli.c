// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "graph/eli.h"

static int _reorder_idx_compare(const void *a, const void *b)
{
    return *(graph_size_t*)a - *(graph_size_t*)b;
}

#define GRAPH_INCLUDE_FILE "graph/template/eli.c"
#include "graph/include_all_types.h"
#undef GRAPH_INCLUDE_FILE
