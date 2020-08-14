// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#ifndef OCL_FLAGS
	#define OCL_FLAGS
#endif

#include "util/starpu.h"
#define OPENCL_CHECK_ERROR STARPU_OPENCL_CHECK_ERROR

#ifdef NDEBUG
	#define OPENCL_ASSERT(x) { if(x){} }
#else
	#define OPENCL_ASSERT OPENCL_CHECK_ERROR
#endif

// Deprecated
extern cl_int clSetCommandQueueProperty(cl_command_queue, cl_command_queue_properties, cl_bool, cl_command_queue_properties*);
