// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef __cplusplus
extern "C" {
#endif

unsigned long long int line_count(FILE *stream);
const char *file_extension(const char *const filename);

#ifdef __cplusplus
}
#endif

static inline bool file_allocate(int fno, size_t size)
{
    return fallocate(fno, 0, 0, size) == 0
        || pwrite(fno, "\0", 1, size - 1) != -1
        || posix_fallocate(fno, 0, size) == 0;
}
