// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include <math.h>
#include <float.h>

#define MIN(a,b) ({ \
    DECLARE_TYPE_OF(a) _a = (a); \
    DECLARE_TYPE_OF(b) _b = (b); \
    _a < _b ? _a : _b; \
})

#define MAX(a,b) ({ \
    DECLARE_TYPE_OF(a) _a = (a); \
    DECLARE_TYPE_OF(b) _b = (b); \
    _a > _b ? _a : _b; \
})

#define SWAP_VALUES(a, b) { \
    DECLARE_TYPE_OF(a) tmp = (a); \
    (a) = (b); \
    (b) = tmp; \
}

#define DIVIDE_BY_INC(num, div) ({ \
    DECLARE_TYPE_OF(num) _num = (num); \
    DECLARE_TYPE_OF(div) _div = (div); \
    _div == 0 ? 0 : (_num + _div - 1) / _div; \
})

#define ROUND_TO_MULT(num, mul) ({ \
    DECLARE_TYPE_OF(mul) _mul = (mul); \
    DIVIDE_BY_INC(num, _mul) * _mul; \
})

#define BINARY_SEARCH5(key, arr, low, high, res) ({ \
    bool _ret = false; \
    size_t _lo = (low); \
    size_t _hi = (high) + 1; \
    while (_hi > _lo) \
    { \
        const size_t _idx = _lo + ((_hi - _lo - 1) >> 1); \
        const DECLARE_TYPE_OF(*arr) _val = arr[_idx]; \
        if(_val == key) \
        { \
            res = _idx; \
            _ret = true; \
            break; \
        } \
        else if (_val < key) \
            _lo = _idx + 1; \
        else \
            _hi = _idx; \
    } \
    if (!_ret) res = _hi; \
    _ret; \
})
#define BINARY_SEARCH4(key, arr, low, high) ({ \
    size_t _res; \
    BINARY_SEARCH5(key, arr, low, high, _res); \
})
#define BINARY_SEARCH(...) VARARG(BINARY_SEARCH,__VA_ARGS__)

#define STATIC_MAP1(m)
#define STATIC_MAP2(m,a)               m(a,0)
#define STATIC_MAP3(m,a,b)             m(a,0),m(b,1)
#define STATIC_MAP4(m,a,b,c)           m(a,0),m(b,1),m(c,2)
#define STATIC_MAP5(m,a,b,c,d)         m(a,0),m(b,1),m(c,2),m(d,3)
#define STATIC_MAP6(m,a,b,c,d,e)       m(a,0),m(b,1),m(c,2),m(d,3),m(e,4)
#define STATIC_MAP7(m,a,b,c,d,e,f)     m(a,0),m(b,1),m(c,2),m(d,3),m(e,4),m(f,5)
#define STATIC_MAP8(m,a,b,c,d,e,f,g)   m(a,0),m(b,1),m(c,2),m(d,3),m(e,4),m(f,5),m(g,6)
#define STATIC_MAP9(m,a,b,c,d,e,f,g,h) m(a,0),m(b,1),m(c,2),m(d,3),m(e,4),m(f,5),m(g,6),m(h,7)
#define STATIC_MAP(m,...)              VARARG(STATIC_MAP,m,__VA_ARGS__)

#define __STATIC_ARRAY_INIT__(arg,idx) __arr[idx] = arg
#define STATIC_ARRAY_INIT(var,...) \
    { \
        DECLARE_TYPE_OF(&var[0]) __arr = &var[0]; \
        STATIC_MAP(__STATIC_ARRAY_INIT__,__VA_ARGS__); \
    }

#define REPEAT0(m)
#define REPEAT1(m) m(0)
#define REPEAT2(m) m(0) m(1)
#define REPEAT3(m) m(0) m(1) m(2)
#define REPEAT4(m) m(0) m(1) m(2) m(3)
#define REPEAT5(m) m(0) m(1) m(2) m(3) m(4)
#define REPEAT6(m) m(0) m(1) m(2) m(3) m(4) m(5)
#define REPEAT7(m) m(0) m(1) m(2) m(3) m(4) m(5) m(6)
#define REPEAT8(m) m(0) m(1) m(2) m(3) m(4) m(5) m(6) m(7)
#define REPEAT9(m) m(0) m(1) m(2) m(3) m(4) m(5) m(6) m(7) m(8)
#define REPEAT(m,count) REPEAT##count(m)

#define FIRST_ARG(a,...)             a

#define LAST_ARG1(a)                 a
#define LAST_ARG2(a,b)               b
#define LAST_ARG3(a,b,c)             c
#define LAST_ARG4(a,b,c,d)           d
#define LAST_ARG5(a,b,c,d,e)         e
#define LAST_ARG6(a,b,c,d,e,f)       f
#define LAST_ARG7(a,b,c,d,e,f,g)     g
#define LAST_ARG8(a,b,c,d,e,f,g,h)   h
#define LAST_ARG9(a,b,c,d,e,f,g,h,i) i
#define LAST_ARG(...)                VARARG(LAST_ARG,__VA_ARGS__)