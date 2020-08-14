// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
    #define restrict __restrict__
#else
    #include <stdbool.h>
#endif

#ifdef __GNUC__
    #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)

    #define PURE_FUN  __attribute__((pure))
    #define CONST_FUN __attribute__((const))
    #define UNUSED    __attribute__((unused))

    #define fallthrough __attribute__ ((fallthrough))

    #define ASSUME_ALIGNED(p,a) __builtin_assume_aligned(p,a)

    #define DECLARE_TYPE_OF(x)     __typeof__(x)
    #define CAST_TO_TYPE_OF(x)    (__typeof__(x))
    #define CAST_TO_POINTER_OF(x) (__typeof__(x)*)
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)

    #define PURE_FUN
    #define CONST_FUN
    #define UNUSED
    #define fallthrough

    #define ASSUME_ALIGNED(p,a) (p)

    #define DECLARE_TYPE_OF(x) auto
    #define CAST_TO_TYPE_OF(x)
    #define CAST_TO_POINTER_OF(x) (void*)
#endif

/*
  Overloaded macro functions based on the number of arguments
  http://stackoverflow.com/questions/5365440/variadic-macro-trick
*/
#define VA_NARGS_IMPL(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,N,...) N
#define VA_NARGS(...) VA_NARGS_IMPL(X,##__VA_ARGS__,9,8,7,6,5,4,3,2,1,0)

#define VARARG_IMPL2(base,count,...) base##count(__VA_ARGS__)
#define VARARG_IMPL(base,count,...)  VARARG_IMPL2(base,count,__VA_ARGS__)
#define VARARG(base,...)             VARARG_IMPL(base,VA_NARGS(__VA_ARGS__),__VA_ARGS__)

#define STATIC_IF_INT(IF,THEN,ELSE) _Generic((IF), \
    char: THEN,           unsigned char: THEN, \
    short: THEN,          unsigned short: THEN, \
    int: THEN,            unsigned int: THEN, \
    long int: THEN,       unsigned long int: THEN, \
    long long int: THEN,  unsigned long long int: THEN, \
    default: ELSE)

#define STATIC_IF_FLOAT(IF,THEN,ELSE) _Generic((IF), \
    float: THEN, \
    double: THEN, \
    long double: THEN, \
    default: ELSE)
