// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/pagerank.h"
#include "alg/pagerank/ocl_codelets.h"
#include "alg/pagerank/spu_codelets.h"
#include "util/math.h"
#include "util/memory.h"
#include "util/string.h"
#include "util/starpu.h"

#include <fnmatch.h>

eli_pagerank_solver_t eli_pagerank_solvers[] = {};

csr_pagerank_solver_t csr_pagerank_solvers[] = {
    {false, false, "kern_fill_cpu_def_vec", bench_fill_cpu_default},
    {false, false, "kern_asum_cpu_def_vec", bench_asum_cpu_default},
    {false, false, "kern_asum_cpu_lib_vec", bench_asum_cpu_lib},
    {false, false, "kern_base_cpu_def_vec", bench_base_cpu_default},
    {false, false, "kern_base_cpu_map_vec", bench_base_cpu_mapped},
    {false, false, "kern_diff_cpu_def_vec", bench_diff_cpu_default},
    {false, false, "kern_diff_cpu_map_vec", bench_diff_cpu_mapped},
    {false, false, "kern_diff_cpu_lib_vec", bench_diff_cpu_lib},

    {false, false, "kern_fill_omp_def_vec", bench_fill_omp_default},
    {false, false, "kern_asum_omp_def_vec", bench_asum_omp_default},
    {false, false, "kern_base_omp_def_vec", bench_base_omp_default},
    {false, false, "kern_base_omp_map_vec", bench_base_omp_mapped},
    {false, false, "kern_diff_omp_def_vec", bench_diff_omp_default},
    {false, false, "kern_diff_omp_map_vec", bench_diff_omp_mapped},

    {false, false, "kern_fill_ocl_def_vec", bench_fill_ocl_default},
    {false, false, "kern_asum_ocl_def_vec", bench_asum_ocl_default},
    {false, false, "kern_asum_ocl_par_vec", bench_asum_ocl_parallel},
    {false, false, "kern_base_ocl_map_vec", bench_base_ocl_mapped},
    {false, false, "kern_diff_ocl_def_vec", bench_diff_ocl_mapdef},
    {false, false, "kern_diff_ocl_par_vec", bench_diff_ocl_mappar},

    {false, false, "kern_fill_cud_def_vec", bench_fill_cud_default},
    {false, false, "kern_asum_cud_def_vec", bench_asum_cud_default},
    {false, false, "kern_asum_cud_par_vec", bench_asum_cud_parallel},
    {false, false, "kern_asum_cud_lib_vec", bench_asum_cud_lib},
    {false, false, "kern_base_cud_map_vec", bench_base_cud_mapped},
    {false, false, "kern_diff_cud_def_vec", bench_diff_cud_mapdef},
    {false, false, "kern_diff_cud_par_vec", bench_diff_cud_mappar},
    {false, false, "kern_diff_cud_lib_vec", bench_diff_cud_lib},

    {false, false, "kern_rank_cpu_def_row", bench_update_csr_cpu_default},
    {false, false, "kern_rank_cpu_stp_row", bench_update_csr_cpu_stepped},
    {false, false, "kern_rank_cpu_lib_row", bench_update_csr_cpu_lib},

    {false, false, "kern_rank_omp_def_row", bench_update_csr_omp_default},
    {false, false, "kern_rank_omp_stp_row", bench_update_csr_omp_stepped},
    {false, false, "kern_rank_omp_bin_row", bench_update_csr_omp_binsearch},

    {false, false, "kern_rank_ocl_def_row", bench_update_csr_ocl_default},
    {false, false, "kern_rank_ocl_stp_row", bench_update_csr_ocl_stepped},
    {false, false, "kern_rank_ocl_wrp_row", bench_update_csr_ocl_warp},

    {false, false, "kern_rank_cud_def_row", bench_update_csr_cud_default},
    {false, false, "kern_rank_cud_stp_row", bench_update_csr_cud_stepped},
    {false, false, "kern_rank_cud_wrp_row", bench_update_csr_cud_warp},
    {false, false, "kern_rank_cud_dyn_row", bench_update_csr_cud_dyn},
    {false, false, "kern_rank_cud_lib_row", bench_update_csr_cud_lib},

    {true, true, "csr_omp_bin", pagerank_csr_omp_binsearch},
    {true, true, "csr_mkl_lib", pagerank_csr_mkl_lib},
    {true, true, "csr_cud_lib", pagerank_csr_cud_lib}
};

csc_pagerank_solver_t csc_pagerank_solvers[] = {
    {false, false, "kern_rank_cpu_def_col", bench_update_csc_cpu_default},
    {false, false, "kern_rank_cpu_stp_col", bench_update_csc_cpu_stepped},
    {false, false, "kern_rank_cpu_lib_col", bench_update_csc_cpu_lib},

    {false, false, "kern_rank_omp_def_col", bench_update_csc_omp_default},
    {false, false, "kern_rank_omp_stp_col", bench_update_csc_omp_stepped},
    {false, false, "kern_rank_omp_bin_col", bench_update_csc_omp_binsearch},

    {false, false, "kern_rank_ocl_def_col", bench_update_csc_ocl_default},
    {false, false, "kern_rank_ocl_stp_col", bench_update_csc_ocl_stepped},
    {false, false, "kern_rank_ocl_wrp_col", bench_update_csc_ocl_warp},

    {false, false, "kern_rank_cud_def_col", bench_update_csc_cud_default},
    {false, false, "kern_rank_cud_stp_col", bench_update_csc_cud_stepped},
    {false, false, "kern_rank_cud_wrp_col", bench_update_csc_cud_warp},
    {false, false, "kern_rank_cud_dyn_col", bench_update_csc_cud_dyn},
    {false, false, "kern_rank_cud_lib_col", bench_update_csc_cud_lib},

    {true, true, "csc_omp_bin", pagerank_csc_omp_binsearch},
    {true, true, "csc_mkl_lib", pagerank_csc_mkl_lib},
    {true, true, "csc_cud_lib", pagerank_csc_cud_lib},
};

bcsr_pagerank_solver_t bcsr_pagerank_solvers[] = {
    {true, true, "bcsr_ref_def", pagerank_bcsr_ref_default},
    {true, true, "bcsr_ref_stp", pagerank_bcsr_ref_stepped},
    {true, true, "bcsr_ref_map", pagerank_bcsr_ref_mapped},

    {true, true, "bcsr_omp_def", pagerank_bcsr_omp_default},
    {true, true, "bcsr_omp_stp", pagerank_bcsr_omp_stepped},
    {true, true, "bcsr_omp_map", pagerank_bcsr_omp_mapped},
    {true, true, "bcsr_omp_rdx", pagerank_bcsr_omp_redux},

    {true, true, "bcsr_ocl_map", pagerank_bcsr_ocl_mapped},
    {true, true, "bcsr_ocl_stp", pagerank_bcsr_ocl_stepped},
    {true, true, "bcsr_ocl_wrp", pagerank_bcsr_ocl_stepped_warp},
    {true, true, "bcsr_ocl_mix", pagerank_bcsr_ocl_stepped_mix},

    {true, true, "bcsr_cud_map", pagerank_bcsr_cuda_mapped},
    {true, true, "bcsr_cud_stp", pagerank_bcsr_cuda_stepped},
    {true, true, "bcsr_cud_wrp", pagerank_bcsr_cuda_stepped_warp},
    {true, true, "bcsr_cud_dyn", pagerank_bcsr_cuda_stepped_dyn},
    {true, true, "bcsr_cud_mix", pagerank_bcsr_cuda_stepped_mix},

    {true, true, "bcsr_spu_map", pagerank_bcsr_spu_mapped},
    {true, true, "bcsr_spu_stp", pagerank_bcsr_spu_stepped},
    {true, true, "bcsr_spu_rdx", pagerank_bcsr_spu_redux},

    {true, true, "bcsr_mpi_def", pagerank_bcsr_mpi_default},
    {true, true, "bcsr_mpi_stp", pagerank_bcsr_mpi_stepped},
    {true, true, "bcsr_mpi_map", pagerank_bcsr_mpi_mapped},
    {true, true, "bcsr_mpi_rdx", pagerank_bcsr_mpi_redux}
};

bcsc_pagerank_solver_t bcsc_pagerank_solvers[] = {
    {true, true, "bcsc_ref_def", pagerank_bcsc_ref_default},
    {true, true, "bcsc_ref_stp", pagerank_bcsc_ref_stepped},
    {true, true, "bcsc_ref_map", pagerank_bcsc_ref_mapped},

    {true, true, "bcsc_omp_def", pagerank_bcsc_omp_default},
    {true, true, "bcsc_omp_stp", pagerank_bcsc_omp_stepped},
    {true, true, "bcsc_omp_map", pagerank_bcsc_omp_mapped},
    {true, true, "bcsc_omp_rdx", pagerank_bcsc_omp_redux},

    {true, true, "bcsc_ocl_map", pagerank_bcsc_ocl_mapped},
    {true, true, "bcsc_ocl_stp", pagerank_bcsc_ocl_stepped},
    {true, true, "bcsc_ocl_wrp", pagerank_bcsc_ocl_stepped_warp},
    {true, true, "bcsc_ocl_mix", pagerank_bcsc_ocl_stepped_mix},

    {true, true, "bcsc_cud_map", pagerank_bcsc_cuda_mapped},
    {true, true, "bcsc_cud_stp", pagerank_bcsc_cuda_stepped},
    {true, true, "bcsc_cud_wrp", pagerank_bcsc_cuda_stepped_warp},
    {true, true, "bcsc_cud_dyn", pagerank_bcsc_cuda_stepped_dyn},
    {true, true, "bcsc_cud_mix", pagerank_bcsc_cuda_stepped_mix},

    {true, true, "bcsc_spu_map", pagerank_bcsc_spu_mapped},
    {true, true, "bcsc_spu_stp", pagerank_bcsc_spu_stepped},
    {true, true, "bcsc_spu_rdx", pagerank_bcsc_spu_redux},

    {true, true, "bcsc_mpi_def", pagerank_bcsc_mpi_default},
    {true, true, "bcsc_mpi_stp", pagerank_bcsc_mpi_stepped},
    {true, true, "bcsc_mpi_map", pagerank_bcsc_mpi_mapped},
    {true, true, "bcsc_mpi_rdx", pagerank_bcsc_mpi_redux}
};

pcsr_pagerank_solver_t pcsr_pagerank_solvers[] = {
    {true, true, "pcsr_mpi_def", pagerank_pcsr_mpi_default},
    {true, true, "pcsr_mpi_stp", pagerank_pcsr_mpi_stepped},
    {true, true, "pcsr_mpi_map", pagerank_pcsr_mpi_mapped}
};

pcsc_pagerank_solver_t pcsc_pagerank_solvers[] = {
    {true, true, "pcsc_mpi_def", pagerank_pcsc_mpi_default},
    {true, true, "pcsc_mpi_stp", pagerank_pcsc_mpi_stepped},
    {true, true, "pcsc_mpi_map", pagerank_pcsc_mpi_mapped}
};

void pagerank_initialize()
{
    if (memory_get_default_alignment() < PAGERANK_MEMORY_ALIGNMENT)
        memory_set_default_alignment(PAGERANK_MEMORY_ALIGNMENT);

    ocl_pagerank_codelets_initialize();
    spu_pagerank_codelets_initialize();

    #ifndef _OPENMP
        toggle_csr_pagerank_solver_by_func(bench_fill_omp_default, false);
        toggle_csr_pagerank_solver_by_func(bench_asum_omp_default, false);
        toggle_csr_pagerank_solver_by_func(bench_base_omp_default, false);
        toggle_csr_pagerank_solver_by_func(bench_base_omp_mapped, false);
        toggle_csr_pagerank_solver_by_func(bench_diff_omp_default, false);
        toggle_csr_pagerank_solver_by_func(bench_diff_omp_mapped, false);

        toggle_csr_pagerank_solver_by_func(bench_update_csr_omp_default, false);
        toggle_csr_pagerank_solver_by_func(bench_update_csr_omp_stepped, false);

        toggle_csc_pagerank_solver_by_func(bench_update_csc_omp_default, false);
        toggle_csc_pagerank_solver_by_func(bench_update_csc_omp_stepped, false);

        toggle_csr_pagerank_solver_by_func(pagerank_csr_omp_binsearch, false);
        toggle_csc_pagerank_solver_by_func(pagerank_csc_omp_binsearch, false);

        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_omp_default, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_omp_stepped, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_omp_mapped, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_omp_redux, false);

        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_omp_default, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_omp_stepped, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_omp_mapped, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_omp_redux, false);
    #endif

    if (starpu_opencl_worker_get_count() == 0)
    {
        toggle_csr_pagerank_solver_by_func(bench_fill_ocl_default, false);
        toggle_csr_pagerank_solver_by_func(bench_asum_ocl_default, false);
        toggle_csr_pagerank_solver_by_func(bench_asum_ocl_parallel, false);
        toggle_csr_pagerank_solver_by_func(bench_base_ocl_mapped, false);
        toggle_csr_pagerank_solver_by_func(bench_diff_ocl_mapdef, false);
        toggle_csr_pagerank_solver_by_func(bench_diff_ocl_mappar, false);

        toggle_csr_pagerank_solver_by_func(bench_update_csr_ocl_default, false);
        toggle_csr_pagerank_solver_by_func(bench_update_csr_ocl_stepped, false);
        toggle_csr_pagerank_solver_by_func(bench_update_csr_ocl_warp, false);

        toggle_csc_pagerank_solver_by_func(bench_update_csc_ocl_default, false);
        toggle_csc_pagerank_solver_by_func(bench_update_csc_ocl_stepped, false);
        toggle_csc_pagerank_solver_by_func(bench_update_csc_ocl_warp, false);

        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_ocl_mapped, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_ocl_stepped, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_ocl_stepped_warp, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_ocl_stepped_mix, false);

        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_ocl_mapped, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_ocl_stepped, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_ocl_stepped_warp, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_ocl_stepped_mix, false);
    }

    if (starpu_cuda_worker_get_count() == 0)
    {
        toggle_csr_pagerank_solver_by_func(bench_fill_cud_default, false);
        toggle_csr_pagerank_solver_by_func(bench_asum_cud_default, false);
        toggle_csr_pagerank_solver_by_func(bench_asum_cud_parallel, false);
        toggle_csr_pagerank_solver_by_func(bench_asum_cud_lib, false);
        toggle_csr_pagerank_solver_by_func(bench_base_cud_mapped, false);
        toggle_csr_pagerank_solver_by_func(bench_diff_cud_mapdef, false);
        toggle_csr_pagerank_solver_by_func(bench_diff_cud_mappar, false);
        toggle_csr_pagerank_solver_by_func(bench_diff_cud_lib, false);

        toggle_csr_pagerank_solver_by_func(bench_update_csr_cud_default, false);
        toggle_csr_pagerank_solver_by_func(bench_update_csr_cud_stepped, false);
        toggle_csr_pagerank_solver_by_func(bench_update_csr_cud_warp, false);
        toggle_csr_pagerank_solver_by_func(bench_update_csr_cud_dyn, false);
        toggle_csr_pagerank_solver_by_func(bench_update_csr_cud_lib, false);

        toggle_csr_pagerank_solver_by_func(pagerank_csr_cud_lib, false);

        toggle_csc_pagerank_solver_by_func(bench_update_csc_cud_default, false);
        toggle_csc_pagerank_solver_by_func(bench_update_csc_cud_stepped, false);
        toggle_csc_pagerank_solver_by_func(bench_update_csc_cud_warp, false);
        toggle_csc_pagerank_solver_by_func(bench_update_csc_cud_dyn, false);
        toggle_csc_pagerank_solver_by_func(bench_update_csc_cud_lib, false);

        toggle_csc_pagerank_solver_by_func(pagerank_csc_cud_lib, false);

        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_cuda_mapped, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_cuda_stepped, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_cuda_stepped_warp, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_cuda_stepped_dyn, false);
        toggle_bcsc_pagerank_solver_by_func(pagerank_bcsc_cuda_stepped_mix, false);

        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_cuda_mapped, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_cuda_stepped, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_cuda_stepped_warp, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_cuda_stepped_dyn, false);
        toggle_bcsr_pagerank_solver_by_func(pagerank_bcsr_cuda_stepped_mix, false);
    }
}

void pagerank_finalize()
{
    spu_pagerank_codelets_finalize();
    ocl_pagerank_codelets_finalize();
}

static int _reorder_active_pagerank_solvers_compare(const void *_a, const void *_b)
{
    return (*((bool*) _a) == *((bool*) _b))
        ? (int)((intptr_t)_a - (intptr_t)_b)       // Keep in order if "active" is the same
        : (int)*((bool*) _b) - (int)*((bool*) _a); // Order by "active"
}

#define implement_pagerank_solver_list(suf) \
    void print_ ## suf ## _pagerank_solvers() \
    { \
        printf("Available " #suf " pagerank solvers:\n"); \
        for (size_t i = 0; i < STATIC_ARR_SIZE(suf ## _pagerank_solvers); i++) \
        { \
            printf("   %02zu: %-7s (%s)\n", \
                i, \
                suf ## _pagerank_solvers[i].name, \
                suf ## _pagerank_solvers[i].active ? "enabled" : "disabled" \
            ); \
        } \
    } \
    int find_ ## suf ## _pagerank_solver_by_func(const suf ## _pagerank_func_t func) \
    { \
        for (size_t i = 0; i < STATIC_ARR_SIZE(suf ## _pagerank_solvers); i++) \
            if (func == suf ## _pagerank_solvers[i].func) \
                return i; \
        return -1; \
    } \
    int find_ ## suf ## _pagerank_solver_by_name(const char *const name) \
    { \
        for (size_t i = 0; i < STATIC_ARR_SIZE(suf ## _pagerank_solvers); i++) \
            if (strcasecmp(name, suf ## _pagerank_solvers[i].name) == 0) \
                return i; \
        return -1; \
    } \
    void toggle_ ## suf ## _pagerank_solver(const int index, bool active) \
    { \
        if (index >= 0 && index < (int) STATIC_ARR_SIZE(suf ## _pagerank_solvers)) \
            suf ## _pagerank_solvers[index].active = active; \
    } \
    void toggle_ ## suf ## _pagerank_solvers(bool active) \
    { \
        for (size_t i = 0; i < STATIC_ARR_SIZE(suf ## _pagerank_solvers); i++) \
            toggle_ ## suf ## _pagerank_solver(i, active); \
    } \
    void toggle_ ## suf ## _pagerank_solver_by_func(const suf ## _pagerank_func_t func, bool active) \
    { \
        toggle_ ## suf ## _pagerank_solver(find_ ## suf ## _pagerank_solver_by_func(func), active); \
    } \
    void toggle_ ## suf ## _pagerank_solver_by_name(const char *const name, bool active) \
    { \
        toggle_ ## suf ## _pagerank_solver(find_ ## suf ## _pagerank_solver_by_name(name), active); \
    } \
    void toggle_ ## suf ## _pagerank_solvers_by_pattern(const char *const pattern, bool match, bool active) \
    { \
        for (size_t i = 0; i < STATIC_ARR_SIZE(suf ## _pagerank_solvers); i++) \
            if ((fnmatch(pattern, suf ## _pagerank_solvers[i].name, 0) == 0) == match) \
                toggle_ ## suf ## _pagerank_solver(i, active); \
    } \
    size_t count_active_ ## suf ## _pagerank_solvers(void) \
    { \
        size_t count = STATIC_ARR_SIZE(suf ## _pagerank_solvers); \
        while (count > 0 && !suf ## _pagerank_solvers[count - 1].active) count--; \
        return count; \
    } \
    void reorder_active_ ## suf ## _pagerank_solvers(void) \
    { \
        qsort( \
            &suf ## _pagerank_solvers, \
            STATIC_ARR_SIZE(suf ## _pagerank_solvers), \
            sizeof(suf ## _pagerank_solver_t), \
            _reorder_active_pagerank_solvers_compare \
        ); \
    }

implement_pagerank_solver_list(eli);
implement_pagerank_solver_list(csr);
implement_pagerank_solver_list(csc);
implement_pagerank_solver_list(bcsr);
implement_pagerank_solver_list(bcsc);
implement_pagerank_solver_list(pcsr);
implement_pagerank_solver_list(pcsc);

void print_pagerank_solvers()
{
    print_eli_pagerank_solvers();
    print_csr_pagerank_solvers();
    print_csc_pagerank_solvers();
    print_bcsr_pagerank_solvers();
    print_bcsc_pagerank_solvers();
    print_pcsr_pagerank_solvers();
    print_pcsc_pagerank_solvers();
}

void toggle_pagerank_solvers(bool active)
{
    toggle_eli_pagerank_solvers(active);
    toggle_csr_pagerank_solvers(active);
    toggle_csc_pagerank_solvers(active);
    toggle_bcsr_pagerank_solvers(active);
    toggle_bcsc_pagerank_solvers(active);
    toggle_pcsr_pagerank_solvers(active);
    toggle_pcsc_pagerank_solvers(active);
}

void toggle_pagerank_solver_by_name(const char *const name, bool active)
{
    toggle_eli_pagerank_solver_by_name(name, active);
    toggle_csr_pagerank_solver_by_name(name, active);
    toggle_csc_pagerank_solver_by_name(name, active);
    toggle_bcsr_pagerank_solver_by_name(name, active);
    toggle_bcsc_pagerank_solver_by_name(name, active);
    toggle_pcsr_pagerank_solver_by_name(name, active);
    toggle_pcsc_pagerank_solver_by_name(name, active);
}

void toggle_pagerank_solvers_by_pattern(const char *const pattern, bool match, bool active)
{
    toggle_eli_pagerank_solvers_by_pattern(pattern, match, active);
    toggle_csr_pagerank_solvers_by_pattern(pattern, match, active);
    toggle_csc_pagerank_solvers_by_pattern(pattern, match, active);
    toggle_bcsr_pagerank_solvers_by_pattern(pattern, match, active);
    toggle_bcsc_pagerank_solvers_by_pattern(pattern, match, active);
    toggle_pcsr_pagerank_solvers_by_pattern(pattern, match, active);
    toggle_pcsc_pagerank_solvers_by_pattern(pattern, match, active);
}

void reorder_active_pagerank_solvers(void)
{
    reorder_active_eli_pagerank_solvers();
    reorder_active_csr_pagerank_solvers();
    reorder_active_csc_pagerank_solvers();
    reorder_active_bcsr_pagerank_solvers();
    reorder_active_bcsc_pagerank_solvers();
    reorder_active_pcsr_pagerank_solvers();
    reorder_active_pcsc_pagerank_solvers();
}

size_t count_active_pagerank_solvers(void)
{
    return count_active_eli_pagerank_solvers()
        +  count_active_csr_pagerank_solvers()
        +  count_active_csc_pagerank_solvers()
        + count_active_bcsr_pagerank_solvers()
        + count_active_bcsc_pagerank_solvers()
        + count_active_pcsr_pagerank_solvers()
        + count_active_pcsc_pagerank_solvers();
}
