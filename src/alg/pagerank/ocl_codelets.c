// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/ocl_codelets.h"
#include "util/math.h"
#include "util/string.h"

struct starpu_opencl_program ocl_pagerank_program;

ocl_pagerank_kernel_t ocl_pagerank_kernels[STARPU_MAXOPENCLDEVS][E_PR_PROBLEM_KERNEL_MAX];

static void ocl_pagerank_kernel_initialize(const int devid, const ocl_pagerank_kernel_index_enum_t kernel)
{
    ocl_pagerank_kernel_t *k = &ocl_pagerank_kernels[devid][kernel];

    if (ocl_pagerank_kernel_names[kernel] == NULL)
    {
        starpu_opencl_get_queue(devid, &k->queue);
        return;
    }

    OPENCL_ASSERT(starpu_opencl_load_kernel(
        &k->kernel,
        &k->queue,
        &ocl_pagerank_program,
        ocl_pagerank_kernel_names[kernel],
        devid
    ));

    cl_device_id device_id;
    starpu_opencl_get_device(devid, &device_id);

    OPENCL_ASSERT(clGetKernelWorkGroupInfo(
        k->kernel,
        device_id,
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(k->work_group_size),
        &k->work_group_size,
        NULL
    ));

    OPENCL_ASSERT(clGetKernelWorkGroupInfo(
        k->kernel,
        device_id,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(k->preferred_work_group_size_multiple),
        &k->preferred_work_group_size_multiple,
        NULL
    ));
}

static void ocl_pagerank_kernel_finalize(const int devid, const ocl_pagerank_kernel_index_enum_t kernel)
{
    if (ocl_pagerank_kernel_names[kernel] != NULL)
        starpu_opencl_release_kernel(
            ocl_pagerank_kernels[devid][kernel].kernel
        );
}

static int ocl_codelets_ref_count = 0;

void ocl_pagerank_codelets_initialize(void)
{
    if (++ocl_codelets_ref_count != 1)
        return;

    static const char *const graph_size_str =
        (sizeof(graph_size_t) == sizeof(cl_ulong))  ? "ulong"  :
        (sizeof(graph_size_t) == sizeof(cl_int))    ? "uint"   :
        (sizeof(graph_size_t) == sizeof(cl_ushort)) ? "ushort" :
        (sizeof(graph_size_t) == sizeof(cl_uchar))  ? "uchar"  : NULL;
    assert(graph_size_str && "unknown graph_size_t size");

    static const char *const float_str =
        (sizeof(pr_float) == sizeof(cl_double)) ? "double" :
        (sizeof(pr_float) == sizeof(cl_float))  ? "float"  :
        (sizeof(pr_float) == sizeof(cl_half))   ? "half"   : NULL;
    assert(float_str && "unknown pr_float size");

    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%s -Dgraph_size_t=%s -Dpr_float=%s", STATIC_STR(OCL_FLAGS), graph_size_str, float_str);

    const int ret = starpu_opencl_load_opencl_from_file(
        "include/alg/pagerank/ocl_codelets.cl",
        &ocl_pagerank_program,
        buffer
    );

    STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_ocl_from_file");

    const size_t worker_count = starpu_opencl_worker_get_count();

    for (size_t d = 0; d < worker_count; d++)
        for (size_t k = 0; k < E_PR_PROBLEM_KERNEL_MAX; k++)
            ocl_pagerank_kernel_initialize(d, (ocl_pagerank_kernel_index_enum_t) k);

    ocl_codelets_ref_count++;
}

void ocl_pagerank_codelets_finalize(void)
{
    if (--ocl_codelets_ref_count > 0)
        return;

    const size_t worker_count = starpu_opencl_worker_get_count();

    for (size_t d = 0; d < worker_count; d++)
        for (size_t k = 0; k < E_PR_PROBLEM_KERNEL_MAX; k++)
            ocl_pagerank_kernel_finalize(d, (ocl_pagerank_kernel_index_enum_t) k);

    const int ret = starpu_opencl_unload_opencl(&ocl_pagerank_program);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_ocl");
}

void ocl_pagerank_determine_work_groups(const ocl_pagerank_kernel_t *kernel, size_t *global_work_items, size_t *local_work_items)
{
    assert(ocl_codelets_ref_count > 0);

    if (global_work_items != NULL)
        *global_work_items = ROUND_TO_MULT(*global_work_items, kernel->preferred_work_group_size_multiple);

    if (local_work_items != NULL)
    {
        *local_work_items = ROUND_TO_MULT(*local_work_items, kernel->preferred_work_group_size_multiple);

        if (global_work_items != NULL && *local_work_items > *global_work_items)
            *local_work_items = *global_work_items;
        if (*local_work_items > kernel->work_group_size && kernel->work_group_size > kernel->preferred_work_group_size_multiple)
            *local_work_items = (kernel->work_group_size / kernel->preferred_work_group_size_multiple) * kernel->preferred_work_group_size_multiple;

        if (global_work_items != NULL)
            *global_work_items = ROUND_TO_MULT(*global_work_items, *local_work_items);
    }
}

static inline cl_event ocl_pagerank_submit(const ocl_pagerank_kernel_t *kernel, size_t global_work_items, size_t local_work_items, bool determine_wg, cl_uint num_wait, const cl_event *wait_list)
{
    size_t *wg_global = &global_work_items;
    size_t *wg_local  = (local_work_items  == 0) ? NULL : &local_work_items;

    if (determine_wg)
        ocl_pagerank_determine_work_groups(
            kernel,
            wg_global,
            wg_local
        );

    cl_event event;

    OPENCL_ASSERT(clEnqueueNDRangeKernel(
        kernel->queue,
        kernel->kernel,
        1, NULL,
        wg_global, wg_local,
        num_wait, (num_wait == 0) ? NULL : wait_list,
        &event
    ));

    return event;
}

cl_event ocl_pagerank_read_col(const int devid, pr_float *dst, const cl_mem src, const graph_size_t dst_offset, const graph_size_t dst_cols, const graph_size_t src_cols, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_COPY_COL];

    cl_event event;
    const size_t buffer_origin[] = {0, 0, 0};
    const size_t host_origin[]   = {sizeof(*dst) * dst_offset, 0, 0};
    const size_t region[]        = {sizeof(*dst) * src_cols, DIVIDE_BY_INC(size, src_cols), 1};

    const cl_int err = clEnqueueReadBufferRect(
        kernel->queue,
        src,
        CL_FALSE,
        buffer_origin,
        host_origin,
        region,
        sizeof(*dst) * src_cols, 0,
        sizeof(*dst) * dst_cols, 0,
        dst,
        num_wait, (num_wait == 0) ? NULL : wait_list,
        &event
    );
    OPENCL_ASSERT(err);

    return event;
}

cl_event ocl_pagerank_fill_arr(const int devid, const cl_mem arr, const pr_float val, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_FILL_ARR];

    cl_event event;
    const cl_int err = clEnqueueFillBuffer(
        kernel->queue,
        arr,
        &val, sizeof(val),
        0, size * sizeof(val),
        num_wait, (num_wait == 0) ? NULL : wait_list,
        &event
    );
    OPENCL_ASSERT(err);

    return event;
}

cl_event ocl_pagerank_add_arr(const int devid, const cl_mem a, const cl_mem b, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_ADD_ARR];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(a),    &a);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(b),    &b);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(size), &size);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, size, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_sum_arr(const int devid, const cl_mem res, const cl_mem arr, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_SUM_ARR];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(res),  &res);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(arr),  &arr);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(size), &size);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, 1, 1, false, num_wait, wait_list);
}

cl_event ocl_pagerank_sum_arr_offset(const int devid, const cl_mem res, const cl_mem arr, const graph_size_t offset, const graph_size_t size, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_SUM_ARR_OFFSET];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(res),    &res);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(arr),    &arr);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(offset), &offset);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(size),   &size);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, 1, 1, false, num_wait, wait_list);
}

cl_event ocl_pagerank_parallel_sum_arr(const int devid, const cl_mem res, const cl_mem arr, const cl_mem scr, const graph_size_t arr_size, const graph_size_t scr_size, cl_uint num_wait, const cl_event *wait_list)
{
    if (arr_size < 2*scr_size)
        return ocl_pagerank_sum_arr(devid, res, arr, arr_size, num_wait, wait_list);

    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_SUM_ARR_PARALLEL];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(scr),      &scr);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(arr),      &arr);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(arr_size), &arr_size);
    OPENCL_ASSERT(err);

    cl_event event  = ocl_pagerank_submit(kernel, 128 * scr_size, 128, false, num_wait, wait_list);
    cl_event result = ocl_pagerank_sum_arr(devid, res, scr, scr_size, 1, &event);

    clReleaseEvent(event);
    return result;
}

cl_event ocl_pagerank_parallel_sum_arr_offset(const int devid, const cl_mem res, const cl_mem arr, const cl_mem scr, const graph_size_t offset, const graph_size_t arr_size, const graph_size_t scr_size, cl_uint num_wait, const cl_event *wait_list)
{
    if (arr_size < 2*scr_size)
        return ocl_pagerank_sum_arr_offset(devid, res, arr, offset, arr_size, num_wait, wait_list);

    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_SUM_ARR_PARALLEL];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(scr),      &scr);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(arr),      &arr);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(arr_size), &arr_size);
    OPENCL_ASSERT(err);

    cl_event event  = ocl_pagerank_submit(kernel, 128 * scr_size, 128, false, num_wait, wait_list);
    cl_event result = ocl_pagerank_sum_arr_offset(devid, res, scr, offset, scr_size, 1, &event);

    clReleaseEvent(event);
    return result;
}

cl_event ocl_pagerank_baserank(const int devid, const cl_mem tmp, const cl_mem src, const cl_mem deg, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_BASERANK];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(tmp),    &tmp);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(src),    &src);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(deg),    &deg);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(vcount), &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_baserank_redux(const int devid, const cl_mem tmp, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_BASERANK_REDUX];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(tmp),     &tmp);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(damping), &damping);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(vcount),  &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, 1, 1, false, num_wait, wait_list);
}

cl_event ocl_pagerank_update_rank_pull(const int devid, const cl_mem dst, const cl_mem src, const cl_mem rid, const cl_mem cid, const cl_mem deg, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_RANK_PULL];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),    &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(src),    &src);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(rid),    &rid);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(cid),    &cid);
    err |= clSetKernelArg(kernel->kernel, 4, sizeof(deg),    &deg);
    err |= clSetKernelArg(kernel->kernel, 5, sizeof(vcount), &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_update_rank_push(const int devid, const cl_mem dst, const cl_mem src, const cl_mem rid, const cl_mem cid, const cl_mem deg, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_RANK_PUSH];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),    &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(src),    &src);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(rid),    &rid);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(cid),    &cid);
    err |= clSetKernelArg(kernel->kernel, 4, sizeof(deg),    &deg);
    err |= clSetKernelArg(kernel->kernel, 5, sizeof(vcount), &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_update_tmp(const int devid, const cl_mem tmp, const cl_mem src, const cl_mem deg, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_TMP];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(tmp),    &tmp);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(src),    &src);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(deg),    &deg);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(vcount), &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_update_rank_tmp_pull(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem rid, const cl_mem cid, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_RANK_TMP_PULL];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),    &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(tmp),    &tmp);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(rid),    &rid);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(cid),    &cid);
    err |= clSetKernelArg(kernel->kernel, 4, sizeof(vcount), &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_update_rank_tmp_push(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem rid, const cl_mem cid, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_RANK_TMP_PUSH];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),    &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(tmp),    &tmp);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(rid),    &rid);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(cid),    &cid);
    err |= clSetKernelArg(kernel->kernel, 4, sizeof(vcount), &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_update_rank_tmp_pull_warp(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem rid, const cl_mem cid, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_RANK_TMP_PULL_WARP];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),    &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(tmp),    &tmp);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(rid),    &rid);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(cid),    &cid);
    err |= clSetKernelArg(kernel->kernel, 4, sizeof(vcount), &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel,  32 * STARPU_MIN(vcount, 1024), 32, false, num_wait, wait_list);
}

cl_event ocl_pagerank_update_rank_tmp_push_warp(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem rid, const cl_mem cid, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_RANK_TMP_PUSH_WARP];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),    &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(tmp),    &tmp);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(rid),    &rid);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(cid),    &cid);
    err |= clSetKernelArg(kernel->kernel, 4, sizeof(vcount), &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel,  32 * STARPU_MIN(vcount, 1024), 32, false, num_wait, wait_list);
}

cl_event ocl_pagerank_update_dest(const int devid, const cl_mem dst, const cl_mem base_rank, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_DEST];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),       &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(base_rank), &base_rank);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(damping),   &damping);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(vcount),    &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_update_dest_raw(const int devid, const cl_mem dst, const pr_float base_rank, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_UPDATE_DEST_RAW];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),       &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(base_rank), &base_rank);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(damping),   &damping);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(vcount),    &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_calc_dest(const int devid, const cl_mem dst, const cl_mem tmp, const cl_mem base_rank, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_CALC_DEST];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),       &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(tmp),       &tmp);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(base_rank), &base_rank);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(damping),   &damping);
    err |= clSetKernelArg(kernel->kernel, 4, sizeof(vcount),    &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_calc_dest_raw(const int devid, const cl_mem dst, const cl_mem tmp, const pr_float base_rank, const pr_float damping, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_CALC_DEST_RAW];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(dst),       &dst);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(tmp),       &tmp);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(base_rank), &base_rank);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(damping),   &damping);
    err |= clSetKernelArg(kernel->kernel, 4, sizeof(vcount),    &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}

cl_event ocl_pagerank_calc_diff(const int devid, const cl_mem tmp, const cl_mem src, const cl_mem dst, const graph_size_t vcount, cl_uint num_wait, const cl_event *wait_list)
{
    assert(ocl_codelets_ref_count > 0);
    const ocl_pagerank_kernel_t *kernel = &ocl_pagerank_kernels[devid][E_PR_PROBLEM_KERNEL_CALC_DIFF];

    cl_int err;
    err  = clSetKernelArg(kernel->kernel, 0, sizeof(tmp),     &tmp);
    err |= clSetKernelArg(kernel->kernel, 1, sizeof(src),     &src);
    err |= clSetKernelArg(kernel->kernel, 2, sizeof(dst),     &dst);
    err |= clSetKernelArg(kernel->kernel, 3, sizeof(vcount),  &vcount);
    OPENCL_ASSERT(err);

    return ocl_pagerank_submit(kernel, vcount, PAGERANK_BLOCK_SIZE, true, num_wait, wait_list);
}
