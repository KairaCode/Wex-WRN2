/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_ENGINE_COMPAT_H
#define WEX_FUZZ_ENGINE_COMPAT_H
/* STAND-IN, for wex4/fuzz/fuzz_store_wex.c only: c/expert_store_wex.c includes the engine's compat.h, which is
 * not in this repository. The store takes one thing from it: Sleep() outside Windows, for the second read of
 * WEX_READ_RETRY=1 (off by default). */
#ifndef _WIN32
#include <time.h>
static inline void Sleep(unsigned int milliseconds) {
    struct timespec t;
    t.tv_sec = (time_t)(milliseconds / 1000u);
    t.tv_nsec = (long)(milliseconds % 1000u) * 1000000L;
    nanosleep(&t, NULL);
}
#endif
#endif /* WEX_FUZZ_ENGINE_COMPAT_H */
