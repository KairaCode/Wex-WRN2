// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
// REFERENCE EXCERPT, not compilable on its own.
// Source: KAIRA src/host/DeepSeekV41Preflight.cs at commit 88ddb07 (2026-11-07), as committed.
// Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
// KAIRA host: preflight warning on COLI_EXPERT_STORE/COLI_WEX_PATH for V4.1 (STALE since the V4.1 WEX1/WNR2 exist).

// ==== DeepSeekV41Preflight.cs:242-247  WEX warning ====
        }
        // deepseek_v4.exe does read the expert container, and no V4.1 one exists yet (decision D-3): a V4 .wex is the wrong model.
        string[] wex = onV4 ? new[] { "COLI_EXPERT_STORE", "COLI_WEX_PATH" }.Where(k => c.Env?.TryGetValue(k, out string? v) == true && !string.IsNullOrWhiteSpace(v)).ToArray() : Array.Empty<string>();
        if (wex.Length > 0)
            Add(PreflightLevel.Warn, "ambiente", $"{string.Join(" e ", wex)} nel profilo: deepseek_v4.exe leggerebbe gli esperti da un contenitore WEX, ma per DeepSeek V4.1 non ne esiste ancora uno (quello V4 è di un altro modello). Toglile dal profilo per leggere gli esperti dagli shard. Un profilo senza \"env\" proprio eredita quelle V4.");
        return items;

