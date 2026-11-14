// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
// REFERENCE EXCERPT, not compilable on its own.
// Source: the maintainer's application tree, src/host/Engine.cs as of 2026-11-07, as committed.
// Line numbers are of that version; the live tree moves them. Code between ranges is omitted.
// Application host: how the profile env and mirror_dir reach the engine process.

// ==== Engine.cs:58-78  ApplyProfileEnv: V4.1 on deepseek_v41.exe gets only IsV41Env names (drops every WEX variable) ====

    /// <summary>
    /// Copies the profile env into the engine env (a copy of KAIRA's own). A V4.1 checkpoint on deepseek_v41.exe gets
    /// only the variables that engine reads (the preflight reports the rest); on deepseek_v4.exe (COLI_FAMILY_ENGINE) it
    /// gets them all. Returns true for deepseek_v4.exe.
    /// </summary>
    internal static bool ApplyProfileEnv(IDictionary<string, string?> environment, ColibriConfig c, bool deepSeekV41)
    {
        bool v41OnV4 = deepSeekV41 && ColibriFamilies.RunsOnV4Engine(c);
        if (c.Env != null)
            foreach (var kv in c.Env)
            {
                if (string.IsNullOrEmpty(kv.Key)) continue;
                if (deepSeekV41 && !v41OnV4 && !ColibriFamilies.IsV41Env(kv.Key)) continue;
                environment[kv.Key] = kv.Value;
            }
        // coli picks the engine from this variable: it must be the one the host checked and set up, not one inherited
        // from KAIRA's environment that the profile overrode (with "" or deepseek_v41)
        if (v41OnV4) environment[ColibriFamilies.FamilyEngineEnv] = "deepseek_v4";
        else if (deepSeekV41) environment.Remove(ColibriFamilies.FamilyEngineEnv);
        return v41OnV4;

// ==== Engine.cs:510-520  mirror_dir must hold config.json, else no mirror ====
        }
        if (string.IsNullOrWhiteSpace(model) || !File.Exists(Path.Combine(model, "config.json")))
        {
            SetError(gen, $"Modello non trovato: {model}. Controlla \"colibri.model_dir\" in kaira.json.");
            return;
        }
        string? mirror = string.IsNullOrWhiteSpace(c.MirrorDir) ? null : c.MirrorDir;
        if (mirror != null && !File.Exists(Path.Combine(mirror, "config.json")))
        {
            Log($"[KAIRA] Copia di specchio non trovata ({mirror}): leggo solo da {model}.");
            mirror = null;

// ==== Engine.cs:650-657  COLI_MODEL_MIRROR from mirror_dir; without a mirror COLI_DISK_WEIGHTS is removed ====
            Log($"[KAIRA] Cache VRAM GLM: riserva transitoria {vramReserve.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)} GB; Gemma {helperState}; riserva aggiuntiva {pendingGemma} MiB.");
        }
        if (mirror != null) psi.Environment["COLI_MODEL_MIRROR"] = mirror;
        else
        {
            psi.Environment.Remove("COLI_MODEL_MIRROR");
            psi.Environment.Remove("COLI_DISK_WEIGHTS"); // the split only means something with a mirror
        }

