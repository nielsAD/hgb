// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"

#include <mkl.h>

#ifdef __cplusplus
extern "C" {
#endif

void mkl_initialize(void);
void mkl_finalize(void);

#ifdef __cplusplus
}
#endif