// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/spu_codelets.h"
#include "alg/pagerank/spu_wrappers.h"
#include "alg/pagerank/ocl_codelets.h"
#include "util/math.h"

#define SIZEBASE(NAME) spu_pagerank_ ## NAME ## _size
#define PERFMODEL(NAME) pm_pagerank_ ## NAME
#define CODELET(NAME)   cl_pagerank_ ## NAME

struct starpu_codelet CODELET(read_col);
struct starpu_codelet CODELET(fill_arr);
struct starpu_codelet CODELET(redux_zero_single);
struct starpu_codelet CODELET(redux_zero);
struct starpu_codelet CODELET(redux_add_single);
struct starpu_codelet CODELET(redux_add);
struct starpu_codelet CODELET(redux_sum);
struct starpu_codelet CODELET(baserank);
struct starpu_codelet CODELET(baserank_redux);
struct starpu_codelet CODELET(update_rank_pull);
struct starpu_codelet CODELET(update_rank_push);
struct starpu_codelet CODELET(update_tmp);
struct starpu_codelet CODELET(update_rank_tmp_pull);
struct starpu_codelet CODELET(update_rank_tmp_push);
struct starpu_codelet CODELET(redux_rank_tmp_pull);
struct starpu_codelet CODELET(redux_rank_tmp_push);
struct starpu_codelet CODELET(update_dest);
struct starpu_codelet CODELET(calc_dest);
struct starpu_codelet CODELET(calc_diff);

static size_t SIZEBASE(read_col)            (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[0]); }
static size_t SIZEBASE(fill_arr)            (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[0]); }
static size_t SIZEBASE(redux_zero_single)   (UNUSED struct starpu_task* task, UNUSED unsigned nimpl) { return 1; }
static size_t SIZEBASE(redux_zero)          (UNUSED struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[0]); }
static size_t SIZEBASE(redux_add_single)    (UNUSED struct starpu_task* task, UNUSED unsigned nimpl) { return 1; }
static size_t SIZEBASE(redux_add)           (UNUSED struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[0]); }
static size_t SIZEBASE(redux_sum)           (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[1]); }
static size_t SIZEBASE(baserank)            (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[0]); }
static size_t SIZEBASE(baserank_redux)      (UNUSED struct starpu_task* task, UNUSED unsigned nimpl) { return 1; }
static size_t SIZEBASE(update_rank_pull)    (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[3]) + starpu_vector_get_nx(task->handles[4]); }
static size_t SIZEBASE(update_rank_push)    (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[3]) + starpu_vector_get_nx(task->handles[4]); }
static size_t SIZEBASE(update_tmp)          (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[0]); }
static size_t SIZEBASE(update_rank_tmp_pull)(       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[2]) + starpu_vector_get_nx(task->handles[3]); }
static size_t SIZEBASE(update_rank_tmp_push)(       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[2]) + starpu_vector_get_nx(task->handles[3]); }
static size_t SIZEBASE(update_dest)         (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[0]); }
static size_t SIZEBASE(calc_dest)           (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[1]); }
static size_t SIZEBASE(calc_diff)           (       struct starpu_task* task, UNUSED unsigned nimpl) { return starpu_vector_get_nx(task->handles[0]); }

#define CPU_FUNC_NAME(A,I) (starpu_cpu_func_t) spu_pagerank_ ## A ## _cpu
#define CPU_FUNC_NAME_STR(A,I) (char*) ("spu_pagerank_" #A "_cpu")
#define ADD_CPU_FUNCS(NAME,...) \
    STATIC_ARRAY_INIT(CODELET(NAME).cpu_funcs,      STATIC_MAP(CPU_FUNC_NAME,    NAME,##__VA_ARGS__)) \
    STATIC_ARRAY_INIT(CODELET(NAME).cpu_funcs_name, STATIC_MAP(CPU_FUNC_NAME_STR,NAME,##__VA_ARGS__)) \

#define OCL_FUNC_NAME(A,I) (starpu_opencl_func_t) spu_pagerank_ ## A ## _ocl
#define OCL_FUNC_FLAG(A,I) STARPU_OPENCL_ASYNC
#define ADD_OCL_FUNCS(NAME,...) \
    STATIC_ARRAY_INIT(CODELET(NAME).opencl_funcs, STATIC_MAP(OCL_FUNC_NAME,NAME,##__VA_ARGS__)) \
    STATIC_ARRAY_INIT(CODELET(NAME).opencl_flags, STATIC_MAP(OCL_FUNC_FLAG,NAME,##__VA_ARGS__))

#define CUDA_FUNC_NAME(A,I) (starpu_cuda_func_t) spu_pagerank_ ## A ## _cuda
#define CUDA_FUNC_FLAG(A,I) STARPU_CUDA_ASYNC
#define ADD_CUDA_FUNCS(NAME,...) \
    STATIC_ARRAY_INIT(CODELET(NAME).cuda_funcs, STATIC_MAP(CUDA_FUNC_NAME,NAME,##__VA_ARGS__)) \
    STATIC_ARRAY_INIT(CODELET(NAME).cuda_flags, STATIC_MAP(CUDA_FUNC_FLAG,NAME,##__VA_ARGS__))

#define DEFINE_PERFMODEL(NAME,TYPE) \
    static struct starpu_perfmodel PERFMODEL(NAME); \
    PERFMODEL(NAME).type      = STARPU_ ## TYPE ## _BASED; \
    PERFMODEL(NAME).size_base = SIZEBASE(NAME); \
    PERFMODEL(NAME).symbol    = "spu_pagerank_" #NAME;

#define STARPU_BUFFER_MODE(M,I) (starpu_data_access_mode) (STARPU_ ## M)
#define DEFINE_CODELET(NAME,MODEL,...) \
    DEFINE_PERFMODEL(NAME,MODEL) \
    CODELET(NAME).name     = "spu_pagerank_" #NAME; \
    CODELET(NAME).nbuffers = VA_NARGS(__VA_ARGS__); \
    CODELET(NAME).model    = &PERFMODEL(NAME); \
    STATIC_ARRAY_INIT(CODELET(NAME).modes, STATIC_MAP(STARPU_BUFFER_MODE,__VA_ARGS__))

#define DEFINE_CODELET_WITH_ALL_FUNCS(NAME,MODEL,...) \
    DEFINE_CODELET(NAME,MODEL,__VA_ARGS__) \
    ADD_CPU_FUNCS(NAME) \
    ADD_OCL_FUNCS(NAME) \
    ADD_CUDA_FUNCS(NAME)

void spu_pagerank_codelets_initialize(void)
{
    ocl_pagerank_codelets_initialize();

    #define COMMUTE STARPU_COMMUTE

    DEFINE_CODELET_WITH_ALL_FUNCS(read_col,          REGRESSION, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(fill_arr,          REGRESSION, W)
    DEFINE_CODELET_WITH_ALL_FUNCS(redux_zero_single, HISTORY,    W)
    DEFINE_CODELET_WITH_ALL_FUNCS(redux_zero,        REGRESSION, W)
    DEFINE_CODELET_WITH_ALL_FUNCS(redux_add_single,  HISTORY,    RW, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(redux_add,         REGRESSION, RW, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(redux_sum,         REGRESSION, REDUX, R, SCRATCH)
    ADD_OCL_FUNCS( redux_sum, redux_parallel_sum)
    ADD_CUDA_FUNCS(redux_sum, redux_parallel_sum)

    DEFINE_CODELET_WITH_ALL_FUNCS(baserank,          REGRESSION, W, R, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(baserank_redux,    HISTORY,    RW)

    DEFINE_CODELET_WITH_ALL_FUNCS(update_rank_pull,       REGRESSION, RW|COMMUTE, R, R, R, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(update_rank_push,       REGRESSION, RW|COMMUTE, R, R, R, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(update_tmp,             REGRESSION, W, R, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(update_rank_tmp_pull,   REGRESSION, RW|COMMUTE, R, R, R, SCRATCH)
    ADD_OCL_FUNCS( update_rank_tmp_pull, update_rank_tmp_pull_warp)
    ADD_CUDA_FUNCS(update_rank_tmp_pull, update_rank_tmp_pull_warp, update_rank_tmp_pull_dyn)
    DEFINE_CODELET_WITH_ALL_FUNCS(update_rank_tmp_push,   REGRESSION, RW|COMMUTE, R, R, R, SCRATCH)
    ADD_OCL_FUNCS( update_rank_tmp_push, update_rank_tmp_push_warp)
    ADD_CUDA_FUNCS(update_rank_tmp_push, update_rank_tmp_push_warp, update_rank_tmp_push_dyn)

    DEFINE_CODELET_WITH_ALL_FUNCS(update_dest,       REGRESSION, RW, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(calc_dest,         REGRESSION, W, R, R)
    DEFINE_CODELET_WITH_ALL_FUNCS(calc_diff,         REGRESSION, W, R, R)

    CODELET(redux_rank_tmp_pull) = CODELET(update_rank_tmp_pull);
    CODELET(redux_rank_tmp_pull).modes[0] = STARPU_REDUX;
    CODELET(redux_rank_tmp_push) = CODELET(update_rank_tmp_push);
    CODELET(redux_rank_tmp_push).modes[0] = STARPU_REDUX;

    #undef COMMUTE
}

void spu_pagerank_codelets_finalize(void)
{
    ocl_pagerank_codelets_finalize();
}
