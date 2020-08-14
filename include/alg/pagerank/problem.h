// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "alg/pagerank/pagerank.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum pagerank_problem_data_global {
    E_PR_PROBLEM_GLOBAL_RNK = 0,
    E_PR_PROBLEM_GLOBAL_DIF,
    E_PR_PROBLEM_GLOBAL_SCR,
    E_PR_PROBLEM_GLOBAL_MAX
} pagerank_problem_data_global_enum_t;

typedef enum pagerank_problem_data_block {
    E_PR_PROBLEM_BLOCKS_TMP_VTX = 0, // [bcount]
    E_PR_PROBLEM_BLOCKS_TMP_RNK,     // [bcount]
    E_PR_PROBLEM_BLOCKS_TMP_DIF,     // [bcount]
    E_PR_PROBLEM_BLOCKS_TMP_DST,     // [bcount]
    E_PR_PROBLEM_BLOCKS_TMP_SCR,     // [bcount]

    E_PR_PROBLEM_BLOCKS_DEG,         // [bcount]
    E_PR_PROBLEM_BLOCKS_SRC,         // [bcount]
    E_PR_PROBLEM_BLOCKS_DST,         // [bcount]

    E_PR_PROBLEM_BLOCKS_RID,         // [bcount*bcount]
    E_PR_PROBLEM_BLOCKS_CID,         // [bcount*bcount]

    E_PR_PROBLEM_BLOCKS_MAX,
    E_PR_PROBLEM_BLOCKS_MAX_SHARED = E_PR_PROBLEM_BLOCKS_RID
} pagerank_problem_data_block_enum_t;

#ifdef __cplusplus
}
#endif