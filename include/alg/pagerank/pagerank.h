// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "graph/pcsr.h"
#include "util/math.h"
#include "util/mpi.h"
#include "util/time.h"

#define pr_eli_graph_t    eli_graph_v_t
#define pr_eli_graph(ID)  eli_graph_v ## ID

#define pr_csr_graph_t    csr_graph_v_t
#define pr_csr_graph(ID)  csr_graph_v ## ID
#define pr_csc_graph_t    csr_graph_v_t
#define pr_csc_graph(ID)  csr_graph_v ## ID

#define pr_bcsr_graph_t   bcsr_graph_v_t
#define pr_bcsr_graph(ID) bcsr_graph_v ## ID
#define pr_bcsc_graph_t   bcsr_graph_v_t
#define pr_bcsc_graph(ID) bcsr_graph_v ## ID

#define pr_pcsr_graph_t   pcsr_graph_v_t
#define pr_pcsr_graph(ID) pcsr_graph_v ## ID
#define pr_pcsc_graph_t   pcsr_graph_v_t
#define pr_pcsc_graph(ID) pcsr_graph_v ## ID

#define pr_pcsr_local_graph_t   pr_bcsr_graph_t
#define pr_pcsr_local_graph(ID) pr_bcsr_graph(ID)
#define pr_pcsc_local_graph_t   pr_bcsr_graph_t
#define pr_pcsc_local_graph(ID) pr_bcsr_graph(ID)

#ifdef __cplusplus
extern "C" {
#endif

typedef float pr_float;

static const pr_float PR_EPSILON = FLT_EPSILON;

#define PAGERANK_MEMORY_ALIGNMENT 64
#define PAGERANK_ASSUME_ALIGNED(P) (CAST_TO_TYPE_OF(P) ASSUME_ALIGNED(P, PAGERANK_MEMORY_ALIGNMENT))

void pagerank_initialize(void);
void pagerank_finalize(void);

typedef enum pagerank_stage {
    E_PR_STAGE_INIT = 0,
    E_PR_STAGE_BASERANK,
    E_PR_STAGE_UPDATE,
    E_PR_STAGE_DIFF,
    E_PR_STAGE_TRANSFER,
    E_PR_STAGE_MAX
} pagerank_stage_t;

typedef struct pagerank_options {
    pr_float damping;
    pr_float epsilon;
    pr_float *result;

    int devid;

    uint32_t local_iterations; // Local iterations before calculating new diff
    uint32_t min_iterations;
    uint32_t max_iterations;

    time_diff_t stage_time[E_PR_STAGE_MAX];
} pagerank_options_t;


// KERNEL BENCHMARKS
uint32_t bench_fill_cpu_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_asum_cpu_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_asum_cpu_lib(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_base_cpu_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_base_cpu_mapped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_cpu_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_cpu_mapped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_cpu_lib(const pr_csr_graph_t *graph, pagerank_options_t *options);

uint32_t bench_fill_omp_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_asum_omp_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_base_omp_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_base_omp_mapped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_omp_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_omp_mapped(const pr_csr_graph_t *graph, pagerank_options_t *options);

uint32_t bench_fill_ocl_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_asum_ocl_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_asum_ocl_parallel(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_base_ocl_mapped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_ocl_mapdef(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_ocl_mappar(const pr_csr_graph_t *graph, pagerank_options_t *options);

uint32_t bench_fill_cud_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_asum_cud_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_asum_cud_parallel(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_asum_cud_lib(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_base_cud_mapped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_cud_mapdef(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_cud_mappar(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_diff_cud_lib(const pr_csr_graph_t *graph, pagerank_options_t *options);

uint32_t bench_update_csr_cpu_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_cpu_default(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_cpu_stepped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_cpu_stepped(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_cpu_lib(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_cpu_lib(const pr_csc_graph_t *graph, pagerank_options_t *options);

uint32_t bench_update_csr_omp_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_omp_default(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_omp_stepped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_omp_stepped(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_omp_binsearch(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_omp_binsearch(const pr_csc_graph_t *graph, pagerank_options_t *options);

uint32_t bench_update_csr_ocl_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_ocl_default(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_ocl_stepped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_ocl_stepped(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_ocl_warp(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_ocl_warp(const pr_csc_graph_t *graph, pagerank_options_t *options);

uint32_t bench_update_csr_cud_default(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_cud_default(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_cud_stepped(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_cud_stepped(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_cud_warp(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_cud_warp(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_cud_dyn(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_cud_dyn(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csr_cud_lib(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t bench_update_csc_cud_lib(const pr_csc_graph_t *graph, pagerank_options_t *options);


// PAGERANK IMPLEMENTATIONS
uint32_t pagerank_csc_omp_binsearch(const pr_csc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_csr_omp_binsearch(const pr_csr_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_csc_mkl_lib(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_csr_mkl_lib(const pr_csc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_csc_cud_lib(const pr_csr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_csr_cud_lib(const pr_csc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_bcsc_ref_default(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_ref_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_ref_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_bcsr_ref_default(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_ref_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_ref_mapped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);


uint32_t pagerank_bcsc_omp_default(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_omp_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_omp_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_omp_redux(const pr_bcsc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_bcsr_omp_default(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_omp_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_omp_mapped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_omp_redux(const pr_bcsr_graph_t *graph, pagerank_options_t *options);


uint32_t pagerank_bcsc_ocl_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_ocl_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_ocl_stepped_warp(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_ocl_stepped_mix(const pr_bcsc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_bcsr_ocl_mapped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_ocl_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_ocl_stepped_warp(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_ocl_stepped_mix(const pr_bcsr_graph_t *graph, pagerank_options_t *options);


uint32_t pagerank_bcsc_cuda_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_cuda_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_cuda_stepped_warp(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_cuda_stepped_dyn(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_cuda_stepped_mix(const pr_bcsc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_bcsr_cuda_mapped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_cuda_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_cuda_stepped_warp(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_cuda_stepped_dyn(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_cuda_stepped_mix(const pr_bcsr_graph_t *graph, pagerank_options_t *options);


uint32_t pagerank_bcsc_spu_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_spu_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_spu_redux(const pr_bcsc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_bcsr_spu_mapped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_spu_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_spu_redux(const pr_bcsr_graph_t *graph, pagerank_options_t *options);


uint32_t pagerank_bcsc_mpi_default(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_mpi_stepped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_mpi_mapped(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsc_mpi_redux(const pr_bcsc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_bcsr_mpi_default(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_mpi_stepped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_mpi_mapped(const pr_bcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_bcsr_mpi_redux(const pr_bcsr_graph_t *graph, pagerank_options_t *options);


uint32_t pagerank_pcsc_mpi_default(MPI_Comm comm, const pr_pcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_pcsc_mpi_stepped(MPI_Comm comm, const pr_pcsc_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_pcsc_mpi_mapped(MPI_Comm comm, const pr_pcsc_graph_t *graph, pagerank_options_t *options);

uint32_t pagerank_pcsr_mpi_default(MPI_Comm comm, const pr_pcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_pcsr_mpi_stepped(MPI_Comm comm, const pr_pcsr_graph_t *graph, pagerank_options_t *options);
uint32_t pagerank_pcsr_mpi_mapped(MPI_Comm comm, const pr_pcsr_graph_t *graph, pagerank_options_t *options);


#define new_pagerank_solver_list(suf) \
    typedef struct suf ## _pagerank_solver \
    { \
        bool active; \
        bool check_result; \
        const char *const name; \
        const suf ## _pagerank_func_t func; \
    } suf ## _pagerank_solver_t; \
    extern suf ## _pagerank_solver_t suf ## _pagerank_solvers[]; \
    void print_ ## suf ## _pagerank_solvers(); \
    int find_ ## suf ## _pagerank_solver_by_func(const suf ## _pagerank_func_t func) PURE_FUN; \
    int find_ ## suf ## _pagerank_solver_by_name(const char *const name) PURE_FUN; \
    void toggle_ ## suf ## _pagerank_solver(const int index, bool active); \
    void toggle_ ## suf ## _pagerank_solvers(bool active); \
    void toggle_ ## suf ## _pagerank_solver_by_func(const suf ## _pagerank_func_t func, bool active); \
    void toggle_ ## suf ## _pagerank_solver_by_name(const char *const name, bool active); \
    void toggle_ ## suf ## _pagerank_solvers_by_pattern(const char *const pattern, bool match, bool active); \
    void reorder_active_ ## suf ## _pagerank_solvers(void); \
    size_t count_active_ ## suf ## _pagerank_solvers(void) PURE_FUN

typedef uint32_t (*eli_pagerank_func_t)(const pr_eli_graph_t *graph, pagerank_options_t *options);
typedef uint32_t (*csr_pagerank_func_t)(const pr_csr_graph_t *graph, pagerank_options_t *options);
typedef uint32_t (*csc_pagerank_func_t)(const pr_csc_graph_t *graph, pagerank_options_t *options);
typedef uint32_t (*bcsr_pagerank_func_t)(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
typedef uint32_t (*bcsc_pagerank_func_t)(const pr_bcsc_graph_t *graph, pagerank_options_t *options);
typedef uint32_t (*pcsr_pagerank_func_t)(MPI_Comm comm, const pr_pcsc_graph_t *graph, pagerank_options_t *options);
typedef uint32_t (*pcsc_pagerank_func_t)(MPI_Comm comm, const pr_pcsc_graph_t *graph, pagerank_options_t *options);

new_pagerank_solver_list(eli);
new_pagerank_solver_list(csr);
new_pagerank_solver_list(csc);
new_pagerank_solver_list(bcsr);
new_pagerank_solver_list(bcsc);
new_pagerank_solver_list(pcsr);
new_pagerank_solver_list(pcsc);

void print_pagerank_solvers();
void toggle_pagerank_solvers(bool active);
void toggle_pagerank_solver_by_name(const char *const name, bool active);
void toggle_pagerank_solvers_by_pattern(const char *const pattern, bool match, bool active);

void reorder_active_pagerank_solvers(void);
size_t count_active_pagerank_solvers(void) PURE_FUN;

#define pr_abs fabsf

#ifdef __cplusplus
}
#endif