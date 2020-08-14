// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/starpu.h"
#include "util/opencl.h"
#include "util/memory.h"

void starpu_inititialize_conf(int argc, char *argv[])
{
    static struct starpu_conf conf;
    STARPU_CHECK_RETURN_VALUE(starpu_conf_init(&conf), "starpu_conf_init");
    STARPU_CHECK_RETURN_VALUE(starpu_initialize(&conf, &argc, &argv), "starpu_init");

    starpu_malloc_set_align(memory_get_default_alignment());
    if (starpu_cuda_worker_get_count() > 0)
        memory_set_default_pinned_manager(E_MM_CUDA);
    else if (starpu_opencl_worker_get_count() > 0)
    {
        int devid = starpu_worker_get_devid(starpu_worker_get_by_type(STARPU_OPENCL_WORKER, 0));
        assert(devid >= 0);

        cl_context context;
        cl_command_queue queue;
        starpu_opencl_get_context(devid, &context);
        starpu_opencl_get_queue(devid, &queue);

        memory_set_opencl_context(context, queue);
        //memory_set_default_pinned_manager(E_MM_OPENCL); this might allocate precious device memory as well

        for (unsigned int i = 0; i < starpu_opencl_worker_get_count(); i++)
        {
            cl_command_queue queue;
            starpu_opencl_get_queue((int)i, &queue);

            OPENCL_ASSERT(clSetCommandQueueProperty(queue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE, true, NULL));
        }
    }
    else
        memory_set_default_pinned_manager(E_MM_STARPU);

    starpu_pause();
}

void starpu_finalize()
{
    memory_set_default_pinned_manager(E_MM_DEFAULT);
    starpu_resume();
    starpu_shutdown();
}
