export const meta = {
  name: 'wnr-kit-team5-conformance',
  description: 'WNR kit, team 5 (portable): regenerate and re-check the spec-conformant WNR3 v1 encoder, write an independent C decoder (T13), finish the GPU v1 profile on the RTX 5090 and the ratio-headroom study, then a critic',
  whenToUse: 'On the machine that received the WNR test kit, after setup\\fetch_data.py has placed the 160-expert sample in args.data. Pass args.root (kit root) and args.data.',
  phases: [
    { title: 'Research', detail: 'conf-encoder -> c-decoder chain, gpu-v1 (5090), headroom' },
    { title: 'Critic', detail: 'cross-check every number and claim, write the team summary' },
  ],
}

// ---------------------------------------------------------------- arguments
// args = { root: '<kit root>', data: '<data dir>' (default <root>\data), out: '<results dir>' (default <root>\results\wf-team5),
//          python: 'python' (interpreter with numpy; CuPy for gpu-v1), gpu: true (false skips gpu-v1),
//          members: ['conf-encoder', 'c-decoder', 'gpu-v1', 'headroom'] (subset to run; default all),
//          machine: 'free text about the PC (optional)', kaira: '<KAIRA source dir, read-only, optional>' }
const A = (args && typeof args === 'object') ? args : {}
if (!A.root || typeof A.root !== 'string') throw new Error('args.root (absolute path of the kit root) is required, e.g. {"root": "D:\\\\wnr-testkit", "data": "D:\\\\wnr-testkit\\\\data"}')
const SEP = (A.root.indexOf('\\') >= 0 || /^[A-Za-z]:/.test(A.root)) ? '\\' : '/'
const trim = p => p.replace(/[\\/]+$/, '')
const ROOT = trim(A.root)
const J = (...xs) => xs.join(SEP)
const DATA = trim(A.data || J(ROOT, 'data'))
const OUT = trim(A.out || J(ROOT, 'results', 'wf-team5'))
const PY = A.python || 'python'
const GPU = A.gpu !== false
const ONLY = Array.isArray(A.members) ? A.members : null
const want = k => !ONLY || ONLY.indexOf(k) >= 0
const SAMPLE = J(DATA, 'v41-sample')
const EXTRA = J(DATA, 'extra')
const PART = J(ROOT, 'partial')
const SH = J(PART, '_shared')
const ENVSET = `$env:WNR_SAMPLE='${SAMPLE}'; $env:WNR_EXTRA='${EXTRA}'; $env:WNR_CEILREF_MODEL='${J(SH, 'ceil-ref', 'models', 'even.npz')}'`

const CTX = `
CONTEXT (portable WNR test kit, running on the machine that received it; expected: Windows, NVIDIA RTX 5090 = sm_120 / compute_120, a stronger CPU than the original i7-7700K + RTX 3070 PC. ${A.machine ? 'User note about this PC: ' + A.machine + '.' : ''} First run "${PY} -I ${J(ROOT, 'setup', 'check_env.py')} --root ${ROOT} --data ${DATA}" (read-only) and record OS, CPU, cores, AVX2/AVX-512, RAM, GPU, driver, CUDA/NVRTC/CuPy versions and the C compilers it finds; cite that output whenever you compare with the original PC. On Linux read every '\\' in the paths below as '/' and use export VAR=... instead of $env:VAR=...)
- Project: KAIRA's WNR2 format = lossless static-context rANS coding of MXFP4 MoE expert records (DeepSeek V4.1 REAP-192, 7680 records of 18,800,640 B raw; rANS L = 2^23, 14-bit CDFs, byte renorm, streams of 2048 weights / 512 scales, 567 contexts (srel7 x pm9 x mx9) x 16 symbols). WNR3 = the next format, specified in the draft below.
- Kit root: ${ROOT}. Read first:
  * ${J(ROOT, 'docs', 'WNR3-DRAFT-v2.1.md')} (DRAFT 0.2.1, cite as S:<line>; v1 default profile CEIL|LENC|HASH, anchor rule g0 = constant 120; tests T1..T14 in section 11; section 0.4 lists rules never run in a codec)
  * ${J(ROOT, 'docs', 'WNR-RICERCA-2026-11-07.md')} (consolidated report, Italian) and ${J(ROOT, 'docs', 'measured-v41.md')} (raw measurements)
  * ${J(ROOT, 'docs', 'WNR2-REFERENCE.md')} (internal reference; cite as REF:<line>)${A.kaira ? `\n  * KAIRA sources (READ-ONLY, never modify, never build): ${A.kaira}` : '\n  * KAIRA sources are not part of the kit; mark anything that needs them as OPEN-K.'}
  * ${J(PART, 'STATUS.md')}: what each stopped member had finished on the original PC, which scripts to resume from, environment variables the copied code reads.
- Data (args.data = ${DATA}): ${SAMPLE} = 160 V4.1 experts L{LL}_E{EEE}.bin (tensors w1.scale w2.scale w3.scale w1.weight w2.weight w3.weight back to back) + manifest.json (shapes, sha256); w1/w3 weight I8 [2304,2560] packed E2M1 (low nibble = even column), scale F8_E8M0 [2304,160]; w2 weight [5120,1152], scale [5120,72]. ${EXTRA} = 70 extra routed + 6 MTP experts (optional, from fetch_data.py --extra). If ${SAMPLE} is missing or "${PY} -I ${J(ROOT, 'setup', 'fetch_data.py')} --root ${ROOT} --data ${DATA} --verify-only --quick" fails, STOP and report it (do not download the sample yourself).
- Kit code: plain runners in ${J(ROOT, 'tests', 'cpu')}, ${J(ROOT, 'tests', 'gpu')}, ${J(ROOT, 'tests', 'audit')} (each has ref\\ or reference\\ files with the original PC's values); stopped-team work in ${J(PART, 'team5')}\\<member>\\; shared copies in ${SH}: ceil-ref (team 4 real-record coder wnr_common.py with features f ceil-form, g/g0 anchors, a, b, c canonical zero, d coded lengths, e 23-bit hash in x0, t tail seeding; models\\even.npz trained on the 80 even-indexed experts; results\\heldout_summary*.json and heldout_g0.jsonl = reference values), team3-capacity-value, team4-capacity-update, team4-ceil-origin, team3-other-tensors (scripts/specs/results only), team4-data-manifests. Held-out protocol everywhere: models from the 80 even-indexed manifest experts, test on the 40 odd-indexed experts team 3 used (one per layer).
- The copied code reads these environment variables (PowerShell state does not persist between your commands, so prefix every command): ${ENVSET}; GPU code under partial\\ reads WNR_NVRTC_ARCH as a bare arch such as 'sm_120' (no '-arch=' prefix; default: detected from the device, sm_120 on the 5090) and WNR_NVSMI (default nvidia-smi on PATH). The kit's own copy tests\\gpu\\src_v1\\v1bench.py reads WNRKIT_ARCH (set by tests\\gpu\\run_gpu_v1_profile.py) or WNR_NVRTC_ARCH, accepts 'sm_120' or '-arch=sm_120', defaults to the device arch, and needs WNRKIT_SAMPLE = the sample dir when run directly; prefer running it through run_gpu_v1_profile.py.
- Old paths: the docs cite the original PC's absolute paths (C:\\..., D:\\..., AppData scratchpads) as provenance. They do not exist here: look them up in ${J(ROOT, 'docs', 'README.md')} (old path -> kit location); an old path not in that table is provenance only and not available.
- Established facts (original PC): WNR2 held-out real record 0.89943 of raw; f+g0 0.88152 (-1.791%); f+g0+d+e 0.88021 (-1.922%); f+g0+c+d+e 0.85282; exact rANS flush ~3.5 B/stream; the structural check (final state == L, exact length) silently accepts ~0.26-0.63% of random single-bit flips (each = one weight's sign flipped); x0 = L + h23 catches them; on ceil-form, side-bit/K/pm2/position contexts give ~0. GPU (3070): ~300 cycles/symbol/lane; REF: 0.2 ms/record kernel-only on the 5090.
- Already covered by the kit's PLAIN runners (do not redo them here; cite their summary.json if you need the numbers): WNR2 encoder-only anchor change (tests\\cpu\\wnr2_g0.py), corruption audit of R0/HASH/TAIL on ceil-form records (tests\\audit\\run_audit.py), training-set learning curve and MTP storage (tests\\cpu\\train_curve_mtp.py), real-record held-out table (tests\\cpu\\heldout_configs.py), GPU bench and canonical-zero bit test (tests\\gpu\\run_gpu_bench.py and the zero-bit-test runner there). Results of runners already run on this PC are in ${J(ROOT, 'results')}\\<test-name>\\summary.json.
- CONCURRENCY on this PC: this workflow is the only research team running: no other kit workflow and no setup\\run_all.ps1 at the same time. Members may run the specific kit runners their own task names (e.g. gpu-v1: run_gpu_v1_profile.py and run_gpu_bench.py --encode-only), always with --out inside their own folder ${OUT}\\<member>\\; do not start any other plain runner. Up to 3 members of this workflow run at once: use at most max(1, logical_cores / 4) worker processes for heavy CPU jobs; for any CPU timing record the concurrent load (Get-Counter '\\Processor(_Total)\\% Processor Time') and report best-of-N with the load, flagging numbers taken under load as lower bounds. The GPU (RTX 5090) belongs to member gpu-v1 only.
RULES: never modify ${J(ROOT, 'docs')}, ${J(ROOT, 'tests')}, ${PART} or ${DATA}\\v41-sample / extra: copy what you need into your own folder ${OUT}\\<your-member-name>\\ (create it) and work only there (large intermediates included). Run Python as "${PY} -I <script>" with scripts outside data folders. No downloads (all data is local), no system-wide installs, no KAIRA changes, no network posting, no git push, no credentials. Tag every claim as measured (with result file), read-in-code/source (path:line or URL) or inferred. Before returning, write your returned object as JSON to ${OUT}\\<your-member-name>\\member_result.json. Return raw data.`

const SCHEMA = {
  type: 'object',
  properties: {
    member: { type: 'string' },
    verdict: { type: 'string', description: 'One dense paragraph: what was done, what holds, what broke, key numbers' },
    environment: { type: 'string', description: 'CPU / GPU / driver / CUDA / compiler actually used (from check_env.py)' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          evidence: { type: 'string' },
          kind: { type: 'string', enum: ['measured', 'read-in-code', 'source', 'inferred'] },
          impact: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['claim', 'evidence', 'kind', 'impact', 'confidence'],
      },
    },
    numbers: {
      type: 'array',
      items: {
        type: 'object',
        properties: { what: { type: 'string' }, value: { type: 'string' }, conditions: { type: 'string' }, original_pc_value: { type: 'string' } },
        required: ['what', 'value', 'conditions'],
      },
    },
    spec_impact: { type: 'array', items: { type: 'string' }, description: 'concrete changes the WNR3 spec 0.2.1 needs because of your results (section + change), or empty' },
    files_written: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
  required: ['member', 'verdict', 'environment', 'findings', 'numbers', 'spec_impact', 'files_written', 'open_questions'],
}

const CONF = `Member "conf-encoder". The spec-conformant WNR3 v1 encoder (test T9 and the encoder half of T13) was FINISHED on the original PC; its code is in ${J(PART, 'team5', 'conf-encoder')} (wnr3c.py codec, train.py WNM3 blob, encode.py records + index, corrupt.py, synth.py, vectors.py H32 vectors, signflip_all.py) but the 660 MB of records were not copied. Copy the folder to ${OUT}\\conf-encoder\\ and REGENERATE everything there from the local sample (read the Usage line of each script):
 (1) train.py SAMPLE_DIR OUT_BLOB OUT_REPORT_JSON [CEILREF_MODELS_NPZ] with the ceil-ref model ${J(SH, 'ceil-ref', 'models', 'even.npz')} as the cross-check; compare train_report.json with the copied one (bit-identical model expected; same model_id).
 (2) encode.py SAMPLE_DIR BLOB OUT_DIR CR_HELDOUT_JSONL --override-anchor-stop with CR_HELDOUT_JSONL = ${J(SH, 'ceil-ref', 'results', 'heldout_g0.jsonl')}; the anchor pre-pass stops on L39_E156 w2 (densest pair 121 != anchor0 120, 1 of 40) exactly as on the original PC: keep the override and report it. Reference (original PC, encode.log): 40 records, ratio_of_raw 0.8803159, cr ratio 0.8802124, delta +0.0103 pp (CI95 +1331..+2560 B), T9 pass, all_byte_exact true, flags_seen [28], ceil_masks_seen [7], max_over_mean 1.0112. Every record must decode byte-exact against the ORIGINAL .bin bytes (sha256 from manifest.json) and the record sha256 values must match ${J(PART, 'team5', 'conf-encoder', 'records', 'index.json')} (same code + same data => identical bytes; any difference is a finding).
 (3) corrupt.py (cases A sign flip, B LENC byte, C header byte, D ceil_mask with recomputed header_check: all must give status -1 with the listed reasons), synth.py (ceil_mask 0/1/2/4 synthetic records), vectors.py (H32 test vectors vs the spec's), signflip_all.py (structure-accepted single-bit flips are all rejected by HASH).
 (4) Then close what the original run did not: re-read every spec section the encoder implements and list each ambiguity / contradiction / under-specified rule with S:<line> and the reading chosen (look in build_report.jsonl and the wnr3c.py comments for the ones already found); check container/entry flags and the WND3 descriptor against the spec; time encode per record on this CPU.
Write the records, model blob, index.json (expert, layer, source sha256, expected decoded sha256, record sha256) and the corrupted/synthetic sets under ${OUT}\\conf-encoder\\records\\ and return their paths: an independent decoder will be built from the spec and these files only.`

const CDEC = enc => `Member "c-decoder". Write an INDEPENDENT WNR3 v1 decoder in C (C11, single .c file plus a small header), using ONLY the spec text ${J(ROOT, 'docs', 'WNR3-DRAFT-v2.1.md')} and the encoder's record index and records. You must NOT open, read or copy any Python or C codec code (no wnr3c.py or anything else under ${J(PART, 'team5', 'conf-encoder')} except records/ and index files, no ${J(SH, 'ceil-ref')}, no tests\\cpu or tests\\audit sources, no team6 cpu-simd wnrdec.c, no GPU kernels): the point is test T13, conformance by an independent implementation. Use a C compiler found by check_env.py (MSVC cl from a VS developer prompt, gcc/clang from MSYS2/mingw/LLVM; install nothing system-wide; if none exists, write the decoder in pure Python from the spec and say so). Decode every record the encoder wrote (WNM3 blob parse included, record header and header check, the ceil inverse and ceil_mask rule, LENC offsets, anchor0 check, HASH with H32 identity layout), compare with the ORIGINAL source .bin bytes of ${SAMPLE} (byte-exact, every stream), check that the corrupted records give the expected status and that the synthetic ceil_mask records decode. Measure single-thread CPU decode time per record with and without the ceil inverse and with/without HASH (-O2 and -O3 -march=native / cl /O2 /arch:AVX2 or AVX512), and multi-thread if trivial. Record every place where the spec was insufficient or ambiguous for writing the decoder (S:<line>) and every disagreement with the encoder's bytes. Put the decoder source, build script and results under ${OUT}\\c-decoder\\.

ENCODER RESULT:
${enc ? JSON.stringify(enc) : `conf-encoder did not run in this workflow run. Use ${OUT}\\conf-encoder\\records\\index.json if it exists; otherwise RUN (do not read) the scripts in ${J(PART, 'team5', 'conf-encoder')} exactly as their docstring usage lines say to regenerate the records into ${OUT}\\c-decoder\\records\\ (train.py, then encode.py with ${J(SH, 'ceil-ref', 'results', 'heldout_g0.jsonl')} and --override-anchor-stop, then corrupt.py); reading only the "Usage:" line printed by running a script with no arguments is allowed.`}`

const GPUV1 = `Member "gpu-v1" (the ONLY member allowed to use the GPU, RTX 5090 sm_120). The WNR3 v1 GPU profile CEIL|LENC|HASH (kernel wdec3<NT,HASH,LENC,CEIL,DIAG>, in-kernel H32 and warp-scan LENC offsets) was finished and measured on the RTX 3070: code in ${J(PART, 'team5', 'gpu-v1')}\\src (v1enc.py encoder, v1bench.py bench with stages h32,scale,weight,neg,corrupt,lenc,timing,pipe; v1summ.py), 3070 results in ...\\out (bench_v1_table.txt, bench_v1_summary.json, corrupt.json, lenc_corrupt.json). 3070 reference (median paired ratios): HASH cost +0.7..+1.8%, LENC cost -0.8..0.0%, HASH+LENC +0.1..+1.1%, v1 vs CF5 +0.6..+3.4%, all pipe checks 0 bad streams. The kit may already contain a plain runner for it (look in ${J(ROOT, 'tests', 'gpu')} for gpu_v1_profile* and its README; the kit's copy of the sources is tests\\gpu\\src_v1): if it exists, run it first (quick, then full) with "--out ${OUT}\\gpu-v1\\v1profile --t4enc ${OUT}\\gpu-v1\\t4enc" and build on its summary.json; otherwise copy partial\\team5\\gpu-v1 to ${OUT}\\gpu-v1\\ and run it there. v1bench.py also needs team 4's encoded A259/CF baseline streams (T4_ENC_DIR): produce them on the CPU with the kit's GPU bench encoder: "${PY} -I ${J(ROOT, 'tests', 'gpu', 'run_gpu_bench.py')} --root ${ROOT} --data ${DATA} --out ${OUT}\\gpu-v1\\bench --encode-only --enc ${OUT}\\gpu-v1\\t4enc" (check its --help first; same 8 experts PICK = [17, 0, 40, 80, 120, 159, 60, 100]; it verifies the output against team 4's sha256). On the 5090: (1) byte-exact against the ORIGINAL safetensors bytes for every stream and launch config, and the negative controls / corruption checks of v1bench (sign flip, last byte, LENC byte) must be reported by the GPU check; (2) paired timing (>= 21 rounds, the method of team 2/4) of G5, G5H, G5L, G5HL vs CF5 and A259 for 1 matrix, 1 record, 8 records, 8 streams, over a launch-config grid re-chosen for 170 SMs (NT 32..256, records per launch up to 64; report achieved warps/SM); (3) registers, shared memory, occupancy per kernel on sm_120 (cuFuncGetAttribute); (4) state what changed vs the 3070 and whether HASH/LENC costs stay below 2%, and the GiB/s per record vs REF's 0.2 ms/record and 102.8 GiB/s (8 concurrent records). Never run two GPU benchmarks at once; record nvidia-smi clocks/power before and after.`

const HEAD = `Member "headroom". Question: after ceil-form, how much lossless ratio headroom is left, so we know whether to stop context-model research? The stopped run's code is in ${J(PART, 'team5', 'headroom')}\\scripts (hfeat.py causal feature extraction on ceil-form anchor 120, hctx.py shared helpers + exact 14-bit quantizer, hgreedy2.py greedy forward selection under a row budget, hadapt.py adaptive upper bound, hmlp.py residual-MLP static upper bound, hscale.py scale-stream contexts, hpath.py exact held-out evaluation with per-expert costs, plus copies of mxcount.py / mxanalyze.py / mxceil.py); results in ...\\out. DONE on the original PC: greedy static search with budgets 0 and 3072 rows: base (srel7, pm, mx) 3.693190 bit/w held-out, best set + rowp4 3.691817 bit/w (-0.0014 bit/w = -0.032% of raw), nothing else helped. NOT DONE: (1) adaptive ceiling (hadapt.py: static / own / Dirichlet-adaptive / back-off chains on full test matrices), (2) hmlp.py rich static bound, (3) hscale.py scale streams on ceil-form (order-0 vs left/upper/row contexts), (4) hpath.py exact evaluation of the chosen sets with the reverse fold and bootstrap CI, (5) a check that table budgets (<= 96 KB per matrix table of 16-entry u16 rows, <= 4096 rows) were respected and that no feature leaks across streams (decoder-known only). Copy the scripts to ${OUT}\\headroom\\, regenerate the stride-16 feature files with hfeat.py (2.9 GB, into ${OUT}\\headroom\\feat\\s16; set $env:WNR_HEADROOM_FEAT to it), re-run hgreedy2.py once to reproduce the reference number, then run (1)-(5) on the held-out split (train even-indexed, test odd-indexed). Give a verdict: residual headroom in % of raw for static models within budget, the adaptive ceiling, and whether to stop context-model research.`

phase('Research')
log(`kit root ${ROOT}; data ${DATA}; results ${OUT}; members: ${ONLY ? ONLY.join(', ') : 'all'}; gpu ${GPU}`)
log('Covered by plain runners, not agents here: wnr2-g0 -> tests\\cpu\\wnr2_g0.py, audit-ceil -> tests\\audit\\run_audit.py, mtp-training -> tests\\cpu\\train_curve_mtp.py')
const chain = (async () => {
  let enc = null, dec = null
  if (want('conf-encoder')) enc = await agent(`${CTX}\n\nTASK:\n${CONF}`, { label: 'conf-encoder', phase: 'Research', schema: SCHEMA, effort: 'high' })
  if (want('c-decoder')) dec = await agent(`${CTX}\n\nTASK:\n${CDEC(enc)}`, { label: 'c-decoder', phase: 'Research', schema: SCHEMA, effort: 'high' })
  return [enc, dec]
})()
const others = []
if (GPU && want('gpu-v1')) others.push(() => agent(`${CTX}\n\nTASK:\n${GPUV1}`, { label: 'gpu-v1', phase: 'Research', schema: SCHEMA, effort: 'high' }))
else log('gpu-v1 skipped (args.gpu === false or not in args.members)')
if (want('headroom')) others.push(() => agent(`${CTX}\n\nTASK:\n${HEAD}`, { label: 'headroom', phase: 'Research', schema: SCHEMA }))
const [chainRes, otherRes] = await Promise.all([chain, parallel(others)])
const all = [...chainRes, ...otherRes].filter(Boolean)
log(`Members returned: ${all.map(r => r.member).join(', ') || 'none'}`)

phase('Critic')
let critic = null
if (all.length) {
  critic = await agent(`${CTX}\n\nTASK:\nMember "critic". Adversarially review team 5's results below. (1) Open the files each member cites and check every number (recompute ratios = bytes/18,800,640, bit/w = weight payload*8/35,389,440, CIs where data allows) and compare with the original-PC reference values quoted in the tasks and in ${J(PART, 'STATUS.md')}. (2) Check that the c-decoder really is independent (it must not have read the Python codec code: inspect its sources, notes and command history in its folder) and that byte-exact claims were checked against the original source bytes of ${SAMPLE}, not against re-encoded data. (3) Check the GPU numbers: arch actually sm_120, paired method, clocks recorded, corruption checks really executed on the GPU. (4) Check the headroom study for train/test leakage, decoder-causality of every feature and table budgets. (5) List contradictions between members, unsupported claims and what is still missing, most serious first. Finally write ${OUT}\\team5_summary.json = {"members": <the RESULTS array below verbatim>, "critic": <your returned object>} and ${OUT}\\team5_summary.md (English, one page: verdicts, key numbers vs original PC, open items).\n\nRESULTS:\n${JSON.stringify(all)}`,
    { label: 'critic', phase: 'Critic', schema: SCHEMA, effort: 'high' })
} else log('no member result: critic skipped')
return { members: all, critic, covered_by_runners: { 'wnr2-g0': 'tests/cpu/wnr2_g0.py', 'audit-ceil': 'tests/audit/run_audit.py', 'mtp-training': 'tests/cpu/train_curve_mtp.py' } }
