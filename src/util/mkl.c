// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/mkl.h"
#include "graph/graph.h"

void mkl_initialize(void)
{
    mkl_set_interface_layer((sizeof(graph_size_t) == sizeof(int)) ? MKL_INTERFACE_LP64 : MKL_INTERFACE_ILP64);
    mkl_set_threading_layer(MKL_THREADING_GNU);
}

void mkl_finalize(void)
{
}
