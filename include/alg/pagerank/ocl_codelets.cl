#if !defined(graph_size_t) || !defined(pr_float)
    #error Undefined kernel macros, using placeholders for editor

    #define graph_size_t uint
    #define pr_float     float
#endif

// http://suhorukov.blogspot.nl/2011/12/opencl-11-atomic-operations-on-floating.html
inline void atomicAdd(volatile global float *addr, const float val)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;

    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = expected.f32 + val;
        current.u32  = atomic_cmpxchg((volatile global unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// Rather use clEnqueueFillBuffer when available
kernel void pagerank_fill_arr_single(
          global pr_float *arr,
    const        pr_float  val,
    const        graph_size_t size)
{
    const graph_size_t idx = get_global_id(0);
    if (idx < size)
        arr[idx] = val;
}

kernel void pagerank_redux_add_single(
          global pr_float *restrict a,
    const global pr_float *restrict b,
    const        graph_size_t size)
{
    const graph_size_t idx = get_global_id(0);
    if (idx < size)
        a[idx] += b[idx];
}

kernel void pagerank_redux_sum(
          global pr_float *res,
    const global pr_float *arr,
    const        graph_size_t size)
{
    pr_float tmp_res = 0.0;
    for (graph_size_t idx = 0; idx < size; idx++)
        tmp_res += arr[idx];
    *res += tmp_res;
}

kernel void pagerank_redux_sum_offset(
          global pr_float *res,
    const global pr_float *arr,
    const        graph_size_t offset,
    const        graph_size_t size)
{
    pr_float tmp_res = 0.0;
    for (graph_size_t idx = 0; idx < size; idx++)
        tmp_res += arr[idx];
    res[offset] = tmp_res;
}
//https://github.com/sschaetz/nvidia-opencl-examples/tree/master/OpenCL/src/oclReduction
kernel void pagerank_redux_sum_parallel(
          global pr_float *res,
    const global pr_float *arr,
    const        graph_size_t size)
{
    #define lsi 128
    local volatile pr_float scr[lsi];

    const graph_size_t lid = get_local_id(0);
    //const graph_size_t lsi = get_local_size(0);
    const graph_size_t gid = get_group_id(0);
    const graph_size_t gsi = lsi*get_num_groups(0);

    pr_float sum = 0.0;

    for (graph_size_t idx = gid*lsi + lid; idx < size; idx += gsi)
        sum += arr[idx];

    scr[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(graph_size_t rdx = lsi >> 1; rdx > 0; rdx >>= 1)
    {
        if (lid < rdx)
            scr[lid] += scr[lid + rdx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        res[gid] = scr[0];

    #undef lsi
}

kernel void pagerank_baserank_single(
          global pr_float     *restrict res,
    const global pr_float     *restrict src,
    const global graph_size_t *restrict deg,
    const        graph_size_t  vcount)
{
    const graph_size_t node = get_global_id(0);
    if (node < vcount)
    {
        //ISO C -> comparison yields 1 if equal
        res[node] = (deg[node] == 0) * src[node];
    }
}

kernel void pagerank_baserank_redux(
          global pr_float *redux,
    const        pr_float  damping,
    const        graph_size_t vcount)
{
    pr_float r = *redux;

    r /= vcount;
    r *= damping;
    r += (1.0 - damping) / vcount;

    *redux = r;
}

kernel void pagerank_update_rank_pull_single(
          global pr_float     *restrict dst,
    const global pr_float     *restrict src,
    const global graph_size_t *restrict rid,
    const global graph_size_t *restrict cid,
    const global graph_size_t *restrict deg,
    const        graph_size_t  vcount)
{
    const graph_size_t node = get_global_id(0);

    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        pr_float new_rank = 0.0;

        for (graph_size_t edge = efr; edge < eto; edge++)
            new_rank += src[cid[edge]] / deg[cid[edge]];

        dst[node] += new_rank;
    }
}

kernel void pagerank_update_rank_push_single(
          global pr_float     *restrict dst,
    const global pr_float     *restrict src,
    const global graph_size_t *restrict rid,
    const global graph_size_t *restrict cid,
    const global graph_size_t *restrict deg,
    const        graph_size_t  vcount)
{
    const graph_size_t node = get_global_id(0);

    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        const pr_float send_rank = src[node] / deg[node];

        for (graph_size_t edge = efr; edge < eto; edge++)
            atomicAdd(&dst[cid[edge]], send_rank);
    }
}

kernel void pagerank_update_tmp_single(
          global pr_float     *restrict tmp,
    const global pr_float     *restrict src,
    const global graph_size_t *restrict deg,
    const        graph_size_t  vcount)
{
    const graph_size_t node = get_global_id(0);
    if (node < vcount)
        tmp[node] = src[node] / deg[node];
}

kernel void pagerank_update_rank_tmp_pull_single(
          global pr_float     *restrict dst,
    const global pr_float     *restrict tmp,
    const global graph_size_t *restrict rid,
    const global graph_size_t *restrict cid,
    const        graph_size_t  vcount)
{
    const graph_size_t node = get_global_id(0);
    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        pr_float new_rank = 0.0;

        for (graph_size_t edge = efr; edge < eto; edge++)
            new_rank += tmp[cid[edge]];

        dst[node] += new_rank;
    }
}

kernel void pagerank_update_rank_tmp_push_single(
          global pr_float     *restrict dst,
    const global pr_float     *restrict tmp,
    const global graph_size_t *restrict rid,
    const global graph_size_t *restrict cid,
    const        graph_size_t  vcount)
{
    const graph_size_t node = get_global_id(0);
    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        for (graph_size_t edge = efr; edge < eto; edge++)
            atomicAdd(&dst[cid[edge]], tmp[node]);
    }
}

//https://github.com/jpola/cl_sparse/blob/master/cl_sparse/csr_kernels
kernel void pagerank_update_rank_tmp_pull_warp(
          global pr_float     *restrict dst,
    const global pr_float     *restrict tmp,
    const global graph_size_t *restrict rid,
    const global graph_size_t *restrict cid,
    const        graph_size_t  vcount)
{
    #define lsi 32
    local volatile pr_float scr[lsi];

    const graph_size_t lid = get_local_id(0);
    //const graph_size_t lsi = get_local_size(0);
    const graph_size_t gid = get_group_id(0);
    const graph_size_t gsi = get_num_groups(0);

    for (graph_size_t node = gid; node < vcount; node += gsi)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        pr_float new_rank = 0.0;

        for (graph_size_t edge = efr + lid; edge < eto; edge += lsi)
            new_rank += tmp[cid[edge]];

        scr[lid] = new_rank;
        barrier(CLK_LOCAL_MEM_FENCE);

        for(graph_size_t rdx = lsi >> 1; rdx > 0; rdx >>= 1)
        {
            if (lid < rdx)
                scr[lid] += scr[lid + rdx];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0)
            dst[node] += scr[0];
    }
    #undef lsi
}

kernel void pagerank_update_rank_tmp_push_warp(
          global pr_float     *restrict dst,
    const global pr_float     *restrict tmp,
    const global graph_size_t *restrict rid,
    const global graph_size_t *restrict cid,
    const        graph_size_t  vcount)
{
    #define lsi 32

    const graph_size_t lid = get_local_id(0);
    //const graph_size_t lsi = get_local_size(0);
    const graph_size_t gid = get_group_id(0);
    const graph_size_t gsi = get_num_groups(0);

    for (graph_size_t node = gid; node < vcount; node += gsi)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        for (graph_size_t edge = efr + lid; edge < eto; edge += lsi)
            atomicAdd(&dst[cid[edge]], tmp[node]);
    }
    #undef lsi
}

kernel void pagerank_update_dest_single(
          global pr_float *restrict dst,
    const global pr_float *restrict brp,
    const        pr_float  damping,
    const        graph_size_t vcount)
{
    const graph_size_t node = get_global_id(0);

    if (node < vcount)
        dst[node] = (damping * dst[node]) + *brp;
}

kernel void pagerank_update_dest_raw_single(
          global pr_float *restrict dst,
    const        pr_float brp,
    const        pr_float damping,
    const        graph_size_t vcount)
{
    const graph_size_t node = get_global_id(0);

    if (node < vcount)
        dst[node] = (damping * dst[node]) + brp;
}

kernel void pagerank_calc_dest_single(
          global pr_float *restrict dst,
    const global pr_float *restrict tmp,
    const global pr_float *restrict brp,
    const        pr_float     damping,
    const        graph_size_t vcount)
{
    const graph_size_t node = get_global_id(0);

    if (node < vcount)
        dst[node] = (damping * tmp[node]) + *brp;
}

kernel void pagerank_calc_dest_raw_single(
          global pr_float *restrict dst,
    const global pr_float *restrict tmp,
    const        pr_float brp,
    const        pr_float damping,
    const        graph_size_t vcount)
{
    const graph_size_t node = get_global_id(0);

    if (node < vcount)
        dst[node] = (damping * tmp[node]) + brp;
}

kernel void pagerank_calc_diff_single(
          global pr_float *restrict dif,
    const global pr_float *restrict src,
    const global pr_float *restrict dst,
    const        graph_size_t vcount)
{
    const graph_size_t node = get_global_id(0);
    if (node < vcount)
        dif[node] = fabs(src[node] - dst[node]);
}
