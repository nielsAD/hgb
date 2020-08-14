// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/codelets.h"
#include "util/starpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void spu_pagerank_codelets_initialize(void);
void spu_pagerank_codelets_finalize(void);

extern struct starpu_codelet cl_pagerank_read_col;
extern struct starpu_codelet cl_pagerank_fill_arr;
extern struct starpu_codelet cl_pagerank_redux_zero_single;
extern struct starpu_codelet cl_pagerank_redux_zero;
extern struct starpu_codelet cl_pagerank_redux_add_single;
extern struct starpu_codelet cl_pagerank_redux_add;
extern struct starpu_codelet cl_pagerank_redux_sum;
extern struct starpu_codelet cl_pagerank_baserank;
extern struct starpu_codelet cl_pagerank_baserank_redux;
extern struct starpu_codelet cl_pagerank_update_rank_pull;
extern struct starpu_codelet cl_pagerank_update_rank_push;
extern struct starpu_codelet cl_pagerank_update_tmp;
extern struct starpu_codelet cl_pagerank_update_rank_tmp_pull;
extern struct starpu_codelet cl_pagerank_update_rank_tmp_push;
extern struct starpu_codelet cl_pagerank_redux_rank_tmp_pull;
extern struct starpu_codelet cl_pagerank_redux_rank_tmp_push;
extern struct starpu_codelet cl_pagerank_update_dest;
extern struct starpu_codelet cl_pagerank_calc_dest;
extern struct starpu_codelet cl_pagerank_calc_diff;

#ifdef __cplusplus
}
#endif