export const meta = {
  name: 'wnr-kit-team7-portability',
  description: 'WNR kit, team 7 (portable): WNR on NVFP4 checkpoints, WNR3 on other MXFP4 MoE models, a lossless random-access Engram format and a disk-placement plan for the KAIRA machine, then a critic',
  whenToUse: 'On the machine that received the WNR test kit (ideally the KAIRA machine, for the real drive table), after setup\\fetch_data.py. Pass args.root and args.data; set args.downloads = false to forbid all network access.',
  phases: [
    { title: 'Research', detail: 'engram-format -> disk-plan chain, nvfp4, other-models' },
    { title: 'Critic', detail: 'cross-check numbers and claims, write the team summary' },
  ],
}

// args = { root: '<kit root>', data: '<data dir>' (default <root>\data), out: '<results dir>' (default <root>\results\wf-team7),
//          python: 'python', downloads: true (false = no network at all; members work from the copied stats/results),
//          download_gb: { nvfp4: 1.5, 'other-models': 1.5, 'engram-format': 1.0 } (per-member caps, GB),
//          members: ['engram-format', 'disk-plan', 'nvfp4', 'other-models'] (subset; default all),
//          machine: 'free text about the PC (optional)', kaira: '<KAIRA source dir, read-only, optional>',
//          drives: 'optional free text: drive letters / roles on this PC (e.g. U: = expert NVMe)' }
const A = (args && typeof args === 'object') ? args : {}
if (!A.root || typeof A.root !== 'string') throw new Error('args.root (absolute path of the kit root) is required, e.g. {"root": "D:\\\\wnr-testkit", "data": "D:\\\\wnr-testkit\\\\data"}')
const SEP = (A.root.indexOf('\\') >= 0 || /^[A-Za-z]:/.test(A.root)) ? '\\' : '/'
const trim = p => p.replace(/[\\/]+$/, '')
const ROOT = trim(A.root)
const J = (...xs) => xs.join(SEP)
const DATA = trim(A.data || J(ROOT, 'data'))
const OUT = trim(A.out || J(ROOT, 'results', 'wf-team7'))
const PY = A.python || 'python'
const NET = A.downloads !== false
const GB = Object.assign({ nvfp4: 1.5, 'other-models': 1.5, 'engram-format': 1.0 }, (A.download_gb && typeof A.download_gb === 'object') ? A.download_gb : {})
const ONLY = Array.isArray(A.members) ? A.members : null
const want = k => !ONLY || ONLY.indexOf(k) >= 0
const SAMPLE = J(DATA, 'v41-sample')
const EXTRA = J(DATA, 'extra')
const DL = J(DATA, 'downloads')
const PART = J(ROOT, 'partial')
const SH = J(PART, '_shared')
const ENVSET = `$env:WNR_SAMPLE='${SAMPLE}'; $env:WNR_EXTRA='${EXTRA}'; $env:WNR_T4MAN='${J(SH, 'team4-data-manifests')}'`
const dlRule = m => NET
  ? `Downloads ONLY via public HTTP from Hugging Face (pinned revisions, HTTP Range reads of tensor bytes; no credentials, no gated repos, no accepting licenses, nothing from a repo is executed) into ${J(DL, m)}\\, at most ${GB[m] !== undefined ? GB[m] : 0} GB in total for this member (count every byte; log each file and its size to ${J(DL, m)}\\downloads.log). Re-fetching data the stopped run had downloaded on the original PC counts against the same cap.`
  : `NO network access in this run (args.downloads is false): work only from the copied stats/results and the local sample; say which conclusions would need data you could not fetch.`

const CTX = `
CONTEXT (portable WNR test kit, running on the machine that received it; expected: Windows, RTX 5090, a stronger CPU than the original i7-7700K PC. ${A.machine ? 'User note about this PC: ' + A.machine + '.' : ''}${A.drives ? ' Drives: ' + A.drives + '.' : ''} First run "${PY} -I ${J(ROOT, 'setup', 'check_env.py')} --root ${ROOT} --data ${DATA}" (read-only) and record OS, CPU, RAM, GPU, compilers and free disk; cite it when you compare with the original PC. On Linux read every '\\' in the paths below as '/' and use export VAR=... instead of $env:VAR=...)
- Project: KAIRA's WNR2 = lossless static-context rANS coding of MXFP4 MoE expert records (DeepSeek V4.1 REAP-192, 7680 records x 18,800,640 B raw). Read first:
  * ${J(ROOT, 'docs', 'WNR-RICERCA-2026-11-07.md')} (report, Italian) and ${J(ROOT, 'docs', 'measured-v41.md')}
  * ${J(ROOT, 'docs', 'WNR3-DRAFT-v2.1.md')} (S:<line>; v1 profile CEIL|LENC|HASH; ceil-form f = halve 'doubled' blocks, a byte-exact bijection gated per matrix on 'no stored block with max magnitude 5'; anchor g0)
  * ${J(ROOT, 'docs', 'WNR2-REFERENCE.md')} (REF:<line>; drive table and free-space figures; checkpoint layout; Engram)${A.kaira ? `\n  * KAIRA sources (READ-ONLY, never modify, never build): ${A.kaira}` : '\n  * KAIRA sources are not part of the kit; mark anything that needs them as OPEN-K.'}
  * ${J(PART, 'STATUS.md')}: what each stopped member had finished, where to resume, environment variables the copied code reads.
- Prior results to build on (copies in ${SH}): team4-ceil-origin (the doubled-block artifact appears in QAT-native MXFP4 checkpoints (DeepSeek V4.x, gpt-oss) and not in PTQ ones; fetch_upstream.py / fetch_other.py HTTP Range fetchers, mxfp.py, other_*.json results); team4-data-manifests (manifests with sha256 of team 4's extra/origin/other slices, REAP keepset); team3-other-tensors (scripts, specs, results: Engram tables 202.76 GB = 55% of the 365.8 GB checkpoint, rows ~264 B measured at ~0.79 compressibility, 48 rows/token read with addresses known from token ids; FP8 dense 7.40 GB; MTP experts; tensors_subset.json = the safetensors header entries of every tensor in specs\\*.json, usable as TENSORS_JSON for scripts\\fetch_tensors.py); team4-capacity-update (container sizes WNR2 129.905 GB, WNR3 127.169, WNR3t 126.850, WNR3z 123.214 GB; tok/s model capacity_update.py). The wnr2 tools (exact 14-bit quantizer mxanalyze.py quantize, mxceil.py ceil transform, mxcount.py load_expert) are copied in ${J(PART, 'team5', 'headroom', 'scripts')}.
- Data (args.data = ${DATA}): ${SAMPLE} (160 V4.1 experts + manifest.json), ${EXTRA} (70 routed + 6 MTP, optional). Downloads of this workflow go to ${DL}\\<member>\\.
- The copied code reads environment variables (PowerShell state does not persist between commands, so prefix every command): ${ENVSET}; nvfp4 fetch_nv.py also reads WNR_T4_ORIGIN_UP (folder with team 4's UP_L*_U*.bin upstream experts, re-fetchable with team4-ceil-origin\\fetch_upstream.py).
- Old paths: the docs cite the original PC's absolute paths (C:\\..., D:\\..., AppData scratchpads) as provenance. They do not exist here: look them up in ${J(ROOT, 'docs', 'README.md')} (old path -> kit location); an old path not in that table is provenance only and not available.
- CONCURRENCY on this PC: this workflow is the only research team running: no other kit workflow, no setup\\run_all.ps1 and no plain runner at the same time (no member of this team needs one). Up to 3 members run at once: use at most max(1, logical_cores / 3) worker processes each and at most 4 parallel HTTP connections each. GPU: not needed by this team (do not use it).
RULES: never modify ${J(ROOT, 'docs')}, ${J(ROOT, 'tests')}, ${PART} or ${DATA}\\v41-sample / extra: copy what you need into ${OUT}\\<your-member-name>\\ (create it) and work only there (downloads in ${DL}\\<member>\\). Never touch, move or delete files outside these folders, even when a plan says files should be moved (only list them). No KAIRA changes, no posting, no git push, no system-wide installs. Tag claims as measured (file), source (URL / path:line) or inferred. Before returning, write your returned object as JSON to ${OUT}\\<your-member-name>\\member_result.json. Return raw data.`

const SCHEMA = {
  type: 'object',
  properties: {
    member: { type: 'string' },
    verdict: { type: 'string' },
    environment: { type: 'string' },
    downloads_bytes: { type: 'number', description: 'bytes downloaded by this member in this run (0 if none)' },
    findings: { type: 'array', items: { type: 'object', properties: {
      claim: { type: 'string' }, evidence: { type: 'string' },
      kind: { type: 'string', enum: ['measured', 'read-in-code', 'source', 'inferred'] },
      impact: { type: 'string' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] } },
      required: ['claim', 'evidence', 'kind', 'impact', 'confidence'] } },
    numbers: { type: 'array', items: { type: 'object', properties: { what: { type: 'string' }, value: { type: 'string' }, conditions: { type: 'string' }, original_pc_value: { type: 'string' } }, required: ['what', 'value', 'conditions'] } },
    spec_impact: { type: 'array', items: { type: 'string' } },
    files_written: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
  required: ['member', 'verdict', 'environment', 'downloads_bytes', 'findings', 'numbers', 'spec_impact', 'files_written', 'open_questions'],
}

const ENGRAM = `Member "engram-format". Disk space is binding (no V4.1 expert container fits the 122.6 GB free on U: of the reference machine), and the Engram tables are 202.76 GB. Design and measure a LOSSLESS, row-random-access storage format for them. Resume from ${J(PART, 'team7', 'engram-format')} (copy to ${OUT}\\engram-format\\): scripts\\fetch_segments.py (Engram segment layout = primes of inference/engram.py EngramLayout + per-segment row samples by HTTP Range from deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e10...; usage OUT_DIR [CHUNKS_PER_SEG] [ROWS_PER_CHUNK]), scripts\\prep.py (merges team 3's Engram samples and the segment samples into rows.npz; team 3's samples are re-fetchable with ${J(SH, 'team3-other-tensors', 'scripts', 'fetch_tensors.py')} TENSORS_JSON=${J(SH, 'team3-other-tensors', 'tensors_subset.json')} SPEC=${J(SH, 'team3-other-tensors', 'specs', 'engram.json')}), scripts\\models.py (held-out ideal code lengths of static context models), scripts\\erow.c + build.bat (per-row static rANS codec DLL: count/encode/verify/bench; MSVC; gcc -O2 -shared also works if you adapt), scripts\\codec_run.py (train, encode held-out, verify round trip, time decode). Results on the original PC: results\\codec.json: 237,568 sample rows, model 9,894 B, held-out mean 209.10 B/row (std 2.6, max 234, 0 rows needing the 264-B fallback, 0 mismatching rows) vs 264 B raw; results\\models.json: candidate model sizes. ${dlRule('engram-format')} Steps: (1) re-fetch the rows (same seeds/specs; check sha256 where the manifests have them), rebuild rows.npz and erow.dll, reproduce codec.json; (2) compare the candidates: (a) fixed-size slots per row after per-row entropy coding (e.g. 224 B; fallback rate), (b) per-row rANS with a shared static model and a row-offset index, (c) groups of N rows coded together with an index, (d) byte-plane / exponent-only coding of the FP8 values plus raw mantissas; for each: bytes per row, total GB saved on 202.76 GB, index size, worst-case row size and fallback, decode cost per token on THIS CPU (48 rows per token; time it, single thread and threaded), and disk reads per token (must not increase; the reference/team 3 note that a scale-next-to-row layout could halve reads); (3) recommend one format with its expected disk saving and what KAIRA's Engram loader would have to change (cite REF and the copied inference code lines if you fetched them).`

const PLAN = eng => `Member "disk-plan". Produce placement plans for the full V4.1 deployment on the KAIRA machine's drives that FIT, using the reference's drive table and free-space figures (cite REF lines). If this PC is the KAIRA machine (check: drive letters/sizes from Get-PSDrive -PSProvider FileSystem and Get-Volume, read-only, vs REF), ALSO use the live free space and say how it differs from REF; otherwise state that only REF numbers were used. Pieces: expert container (WEX1 raw 144.39 GB; WNR2 129.905; WNR3 127.169; WNR3t 126.850; WNR3z 123.214 GB from ${J(SH, 'team4-capacity-update')}), Engram tables (raw 202.76 GB or compressed per engram-format below), dense/attention FP8 (7.40 GB), embeddings/head, MTP experts, and any mirror/backup copies the reference requires (W3 same-drive copy rules, .verified markers). For each plan: which drive holds what, free space left, the expected tok/s from team 4's capacity-update model (copy capacity_update.py; same-drive vs split rules; the model's G2+W2 assumption) and from the CPU-only path if relevant (use ${J(ROOT, 'results', 'wf-team6')}\\cpu-path-value\\member_result.json if team 6's workflow has been run). Show which plans need the user to delete or move existing files (list them from REF and, on the KAIRA machine, from a read-only directory size listing; NEVER touch any file). Rank the plans.

ENGRAM-FORMAT RESULT:
${eng ? JSON.stringify(eng) : `engram-format did not run in this workflow run: use ${OUT}\\engram-format\\member_result.json if it exists, else the original-PC codec result (209.10 B/row of 264, i.e. ~0.792) from ${J(PART, 'team7', 'engram-format', 'results', 'codec.json')}, and say so.`}`

const NVFP4 = `Member "nvfp4". Future checkpoints may ship NVFP4 instead of MXFP4 (E2M1 codes, one FP8 E4M3 scale per 16 values, plus a per-tensor FP32 scale). Resume from ${J(PART, 'team7', 'nvfp4')} (copy to ${OUT}\\nvfp4\\): hf_search.py + hf_search.json (public NVFP4 MoE repos found), fetch_meta.py (index/config), fetch_nv.py (family -> pinned repo/revision and (layer, expert) picks: ds41_1017 = nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 paired with the MXFP4 sample, qwen36_nv, qwen36_rh, glm53_nv, nemo3s_nv; usage FAMILY INDEX_JSON OUT_DIR MAX_MB), nvcount.py (WNR-style context counts adapted to 16-blocks), nvanalyze.py, nvscales.py (E4M3 scale entropy), nvpair_ds.py (NVFP4 vs MXFP4 of the same V4.1 experts), nvstruct.py / nvinspect.py. Results on the original PC: res_ds41_1017.json, res_qwen36_nv.json, res_qwen36_rh.json (weights payload bit/w per context set, held-out record estimate, block-max16 histograms, doubled16, zero sign, scale bits); the count files (220 MB) were NOT copied; nemo3s_nv was counted but never analysed; glm53_nv was never fetched. ${dlRule('nvfp4')} Steps: (1) check the three copied results for consistency (recompute the held-out record estimate from the numbers in each file); (2) fetch + count + analyse the missing families (nemo3s_nv, glm53_nv if its repo is public and ungated at the pinned revision; otherwise say so) and, if budget remains, re-fetch ds41_1017 to rerun nvpair_ds.py; (3) measure: code entropy with WNR-style contexts on 16-blocks (srel, pm, mx), E4M3 scale entropy (order-0 and with neighbour contexts), block-max distribution (is the block max pinned to the top code like MXFP4's floor rule? any double-rounding / re-export artifact similar to ceil-form?), sign-of-zero statistics, and the held-out ratio a WNR-NV variant would reach (record bytes / raw); (4) sketch the WNR-NV differences from WNR3 (contexts, scale coding, block size 16, per-tensor scale) and its GPU decode implications. Cite model cards and the NVFP4 format docs.`

const OTHER = `Member "other-models". Which other MXFP4 MoE models could KAIRA run with a WNR3 container, and how well? Resume from ${J(PART, 'team7', 'other-models')} (copy to ${OUT}\\other-models\\): fetch7.py (usage FAMILY INDEX_JSON OUT_DIR MAX_MB [--plan]), stats7.py (per-expert statistics + context counts per matrix: usage DATA_DIR OUT_DIR [LABEL]), ana7.py (held-out ratios and transfer from the stats: usage STATS_DIR OUT_JSON), stats\\<checkpoint>\\*.npz (COPIED: amd_gptoss120, amd_r1, ds_v41flash, ds_v4flash, ds_v4pro, glm53x, gptoss120, gptoss20, kimi26, minimax3, r1x, rh_glm53, v41extra complete; v41sample incomplete: 75 of 160 experts), stats.log, fetch.log; team 4's earlier per-checkpoint analysis in ${J(SH, 'team4-ceil-origin')} (other_*.json, fetch_other.py) and manifests in ${J(SH, 'team4-data-manifests', 'other')}. ana7.py was never run. ${dlRule('other-models')} Steps: (1) finish stats7.py for v41sample on ${SAMPLE} (and v41extra check on ${EXTRA} if present) and run ana7.py on all stats; (2) for each model: expert geometry and record size, QAT-native vs PTQ (doubled-block share, block-max histogram, the ceil gate outcome per matrix), held-out ratio with a model trained on that checkpoint for WNR2-form, ceil-form + g0 (choose the anchor per the spec's rule and check the anchor0 consistency of S section 5.3), and canonical zero; and the transfer ratio when using the V4.1 model unchanged (does a 'universal QAT MXFP4 model' work, or is one WNM per checkpoint required?); (3) only if a conclusion needs more experts, fetch more slices within the cap (start with what team 4 indexed: amd Kimi-K2.6-MXFP4, amd MiniMax-M3-MXFP4, openai gpt-oss-120b, RedHatAI GLM-5.3-MXFP4, amd DeepSeek-R1-0528-MXFP4). Summarise as a table of expected ratios and the per-checkpoint training needed.`

phase('Research')
log(`kit root ${ROOT}; data ${DATA}; results ${OUT}; members: ${ONLY ? ONLY.join(', ') : 'all'}; downloads ${NET ? JSON.stringify(GB) + ' GB caps' : 'OFF'}`)
const chain = (async () => {
  let eng = null, plan = null
  if (want('engram-format')) eng = await agent(`${CTX}\n\nTASK:\n${ENGRAM}`, { label: 'engram-format', phase: 'Research', schema: SCHEMA, effort: 'high' })
  if (want('disk-plan')) plan = await agent(`${CTX}\n\nTASK:\n${PLAN(eng)}`, { label: 'disk-plan', phase: 'Research', schema: SCHEMA })
  return [eng, plan]
})()
const others = []
if (want('nvfp4')) others.push(() => agent(`${CTX}\n\nTASK:\n${NVFP4}`, { label: 'nvfp4', phase: 'Research', schema: SCHEMA }))
if (want('other-models')) others.push(() => agent(`${CTX}\n\nTASK:\n${OTHER}`, { label: 'other-models', phase: 'Research', schema: SCHEMA }))
const [chainRes, otherRes] = await Promise.all([chain, parallel(others)])
const all = [...chainRes, ...otherRes].filter(Boolean)
log(`Team 7 members returned: ${all.map(r => r.member).join(', ') || 'none'}`)

phase('Critic')
let critic = null
if (all.length) {
  critic = await agent(`${CTX}\n\nTASK:\nMember "critic". Adversarially review team 7's results below: open the cited files and recheck every number (ratios, GB totals, per-token decode times, free-space arithmetic against REF lines and, if measured, the live drive figures), check that downloads stayed within each member's cap (${NET ? JSON.stringify(GB) + ' GB' : 'no downloads allowed'}; read ${DL}\\<member>\\downloads.log) and were public and ungated at pinned revisions, check held-out protocols for leakage, check that no plan touched files, and list contradictions, unsupported claims and missing checks, most serious first. Finally write ${OUT}\\team7_summary.json = {"members": <the RESULTS array below verbatim>, "critic": <your returned object>} and ${OUT}\\team7_summary.md (English, one page: verdicts, key numbers vs original PC, open items).\n\nRESULTS:\n${JSON.stringify(all)}`,
    { label: 'critic', phase: 'Critic', schema: SCHEMA, effort: 'high' })
} else log('no member result: critic skipped')
return { members: all, critic }
