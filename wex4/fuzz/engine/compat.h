/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_ENGINE_COMPAT_H
#define WEX_FUZZ_ENGINE_COMPAT_H
/* STAND-IN, for wex4/fuzz/fuzz_store_wex.c only: c/expert_store_wex.c includes the engine's compat.h, which is
 * not in this repository. The store uses nothing of it: on Windows it calls the Win32 API itself, elsewhere the
 * POSIX functions of the headers it includes. So this file declares nothing.
 *
 * Checked against colibri dev 98327913f980514970173b6df7bf67210ad0dc16, c/compat.h. Until 2026-11-15 this
 * file defined Sleep() outside Windows, for the pause of WEX_READ_RETRY=1; colibri's compat.h defines no
 * Sleep() on any platform (on Windows it comes from windows.h), so the store did not build against it on Linux
 * or macOS. The store now has its own pause (wex_retry_pause in c/expert_store_wex.c). */
#endif /* WEX_FUZZ_ENGINE_COMPAT_H */
