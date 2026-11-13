// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
// REFERENCE EXCERPT, not compilable on its own.
// Source: KAIRA src/host/Config.cs at commit 88ddb07 (2026-11-07), as committed.
// Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
// One value is not verbatim: the path of COLI_SETUP_HOME is shortened to <HOME>\...\colibri-state.
// KAIRA host (C#, outside colibri): profile defaults. COLI_WEX_PATH here is STALE (deleted V4-0731 container).

// ==== Config.cs:193-215  ColibriConfig.MirrorDir and DefaultEnv (COLI_EXPERT_STORE=wex, COLI_WEX_PATH, COLI_DISK_WEIGHTS) ====
    [JsonPropertyName("model_dir")] public string? ModelDir { get; set; } = @"C:\models\DeepSeek-V4-Flash-0731";
    /// <summary>Second copy of the checkpoint on another SSD (COLI_MODEL_MIRROR), or null.</summary>
    [JsonPropertyName("mirror_dir")] public string? MirrorDir { get; set; } = @"U:\models\DeepSeek-V4-Flash-0731";
    /// <summary>Colibri RAM budget in decimal GB (legacy JSON key ram_gib); 0 = available memory minus desktop headroom.</summary>
    [JsonPropertyName("ram_gib")] public int RamGiB { get; set; }
    [JsonPropertyName("ctx")] public int Ctx { get; set; } = 8192;
    [JsonPropertyName("extra_args")] public List<string>? ExtraArgs { get; set; }
    /// <summary>Folders put in front of PATH (the CUDA runtime DLLs of Colibri's backend).</summary>
    [JsonPropertyName("path_prepend")] public List<string>? PathPrepend { get; set; } = new()
    {
        @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64",
    };
    /// <summary>Environment for the engine (Colibri's COLI_* / DSV4_* / V4_* switches).</summary>
    [JsonPropertyName("env")] public Dictionary<string, string>? Env { get; set; } = DefaultEnv();

    /// <summary>The DeepSeek V4 Flash setup measured on this PC (patched engine, colibri-v4-patch.diff).</summary>
    public static Dictionary<string, string> DefaultEnv() => new()
    {
        ["COLI_EXPERT_STORE"] = "wex",
        ["COLI_WEX_PATH"] = @"U:\models\DeepSeek-V4-Flash-0731\v4_flash_experts.wex",
        ["COLI_SETUP_HOME"] = @"<HOME>\...\colibri-state",
        ["COLI_DISK_WEIGHTS"] = "58,42", // C: T700 / U: mirror, by measured bandwidth
        ["COLI_V4_DIRECT"] = "1",

