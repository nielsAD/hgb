// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include <string.h>
#include <strings.h>

#ifdef __cplusplus
extern "C" {
#endif

bool strtob(const char *const a, const bool def) PURE_FUN;
unsigned long long int sstrtoull(const char *const a);
unsigned long long int strtoullr(const char *const a, const unsigned long long int min, const unsigned long long int max);

char *strrepl(const char *str, const char *pat, const char *repl);

#ifdef __cplusplus
}
#endif

#define STATIC_STR_(s) #s
#define STATIC_STR(s) STATIC_STR_(s)

#define STATIC_CONCAT_(a,b)     a ## b
#define STATIC_CONCAT2(a,b)     STATIC_CONCAT_(a,b)
#define STATIC_CONCAT3(a,b,c)   STATIC_CONCAT2(STATIC_CONCAT2(a,b),c)
#define STATIC_CONCAT4(a,b,c,d) STATIC_CONCAT2(STATIC_CONCAT3(a,b,c),d)
#define STATIC_CONCAT(...)      VARARG(STATIC_CONCAT,__VA_ARGS__)
