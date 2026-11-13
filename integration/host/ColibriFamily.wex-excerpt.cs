// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
// REFERENCE EXCERPT, not compilable on its own.
// Source: KAIRA src/host/ColibriFamily.cs at commit 88ddb07 (2026-11-07), as committed.
// Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
// KAIRA host: the V4.1 env allow-list (item H1: no WEX name in it).

// ==== ColibriFamily.cs:22-40  V41EnvNames ====
    private static readonly string[] V4ModelTypes = { "deepseek_v4" };

    // Engine-side variables the DeepSeek V4.1 engine reads (deepseek_v41.c getenv list, plus the Vulkan switch
    // and the OpenMP names read by the launcher). Everything else in a V4.1 profile env is V4-only (WEX, V4_*,
    // DSV4_*), or set by KAIRA or the launcher and therefore not taken from the profile:
    // CTX (from --ctx), KV_SLOTS (set by the gateway) and COLI_MODEL_MIRROR (from mirror_dir).
    private static readonly string[] V41EnvNames =
    {
        "COLI_VULKAN", "COLI_MODEL_DIRS", "COLI_DISK_WEIGHTS", "COLI_USAGE", "COLI_KV_PREFIX",
        "COLI_PREFIX_LOG", "COLI_NO_OMP_TUNE", "OMP_NUM_THREADS",
        "COLI_CUDA", "COLI_CUDA_DENSE", "COLI_CUDA_DENSE_GB",
        // names the V4.1 engine reads and that the upstream tree added (verified against src/colibri/modern:
        // deepseek_v41.c getenv list — GGUF_CHECK validates the checkpoint loader, GQ_EXACT selects the exact
        // grouped-query path, SERVE switches the serve entry point). Without them a profile setting them is
        // silently dropped by IsV41Env.
        "GGUF_CHECK", "GQ_EXACT", "SERVE",
        "CLUSTER_WORKERS", "CLUSTER_WORKER_CACHE", "CLUSTER_WORKER_PORT", "EXPERT_WORKER",
        "SNAP", "SNAP_MIRROR", "SEED", "DUMP",
    };

