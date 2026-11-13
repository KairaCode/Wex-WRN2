export const meta = {
  name: 'wnr-kit-team6-cpu-path',
  description: 'WNR kit, team 6 (portable): fast AVX2/AVX-512 CPU decoder and what WNR is worth on the CPU-only V4.1 path of this machine, the value-exact compacted sign-plane layout, then a critic',
  whenToUse: 'On the machine that received the WNR test kit (ideally the KAIRA machine itself, whose CPU and drives are the ones that matter), after setup\\fetch_data.py. Pass args.root and args.data.',
  phases: [
    { title: 'Research', detail: 'cpu-simd -> cpu-path-value chain, signplane-z' },
    { title: 'Critic', detail: 'cross-check numbers and claims, write the team summary' },
  ],
}

// args = { root: '<kit root>', data: '<data dir>' (default <root>\data), out: '<results dir>' (default <root>\results\wf-team6),
//          python: 'python', gpu: true (false: signplane-z gives CPU numbers + inferred GPU estimate only),
//          members: ['cpu-simd', 'cpu-path-value', 'signplane-z'] (subset; default all),
//          machine: 'free text about the PC (optional)', kaira: '<KAIRA source dir, read-only, optional>',
//          drives: 'optional free text: which drive letters hold the KAIRA containers (e.g. U: = expert NVMe, E: = ...)' }
const A = (args && typeof args === 'object') ? args : {}
if (!A.root || typeof A.root !== 'string') throw new Error('args.root (absolute path of the kit root) is required, e.g. {"root": "D:\\\\wnr-testkit", "data": "D:\\\\wnr-testkit\\\\data"}')
const SEP = (A.root.indexOf('\\') >= 0 || /^[A-Za-z]:/.test(A.root)) ? '\\' : '/'
const trim = p => p.replace(/[\\/]+$/, '')
const ROOT = trim(A.root)
const J = (...xs) => xs.join(SEP)
const DATA = trim(A.data || J(ROOT, 'data'))
const OUT = trim(A.out || J(ROOT, 'results', 'wf-team6'))
const PY = A.python || 'python'
const GPU = A.gpu !== false
const ONLY = Array.isArray(A.members) ? A.members : null
const want = k => !ONLY || ONLY.indexOf(k) >= 0
const SAMPLE = J(DATA, 'v41-sample')
const PART = J(ROOT, 'partial')
const SH = J(PART, '_shared')
const ENVSET = `$env:WNR_SAMPLE='${SAMPLE}'; $env:WNR_CEILREF_MODEL='${J(SH, 'ceil-ref', 'models', 'even.npz')}'`

const CTX = `
CONTEXT (portable WNR test kit, running on the machine that received it; expected: Windows, NVIDIA RTX 5090 = sm_120, a stronger CPU than the original i7-7700K (4C/8T, AVX2, no AVX-512) + RTX 3070 PC. ${A.machine ? 'User note about this PC: ' + A.machine + '.' : ''}${A.drives ? ' Drives: ' + A.drives + '.' : ''} First run "${PY} -I ${J(ROOT, 'setup', 'check_env.py')} --root ${ROOT} --data ${DATA}" (read-only) and record OS, CPU model, cores/threads, AVX2/AVX-512, RAM, GPU, C compilers found; cite it whenever you compare with the original PC. On Linux read every '\\' in the paths below as '/' and use export VAR=... instead of $env:VAR=...)
- Project: KAIRA's WNR2 = lossless static-context rANS coding of MXFP4 MoE expert records (DeepSeek V4.1 REAP-192, 7680 records x 18,800,640 B raw; rANS L = 2^23, 14-bit CDFs, byte renorm, streams of 2048 weights / 512 scales, 567 contexts (srel7 x pm9 x mx9) x 16 symbols). Read first:
  * ${J(ROOT, 'docs', 'WNR-RICERCA-2026-11-07.md')} (report, Italian) and ${J(ROOT, 'docs', 'measured-v41.md')}
  * ${J(ROOT, 'docs', 'WNR3-DRAFT-v2.1.md')} (WNR3 spec, S:<line>; v1 profile CEIL|LENC|HASH, ceil-form f, anchor g0 = 120)
  * ${J(ROOT, 'docs', 'WNR2-REFERENCE.md')} (REF:<line>; e.g. REF:523 says V4.1 experts run CPU-only today; CPU decode in the reference ~0.3 s/expert; drive table with GB/s per drive; M0 measured 1.11-1.14 tok/s WEX1 CPU-only on U:)${A.kaira ? `\n  * KAIRA sources (READ-ONLY, never modify, never build): ${A.kaira}` : '\n  * KAIRA sources are not part of the kit; mark anything that needs them as OPEN-K.'}
  * ${J(PART, 'STATUS.md')}: what each stopped member had finished, where to resume, environment variables the copied code reads.
- Data (args.data = ${DATA}): ${SAMPLE} (160 experts, manifest.json with shapes and sha256; tensors w1.scale w2.scale w3.scale w1.weight w2.weight w3.weight back to back; weight I8 packed E2M1, low nibble = even column; scale F8_E8M0). If it is missing or "${PY} -I ${J(ROOT, 'setup', 'fetch_data.py')} --root ${ROOT} --data ${DATA} --verify-only --quick" fails, STOP and report it.
- Kit code: stopped-team work in ${J(PART, 'team6')}\\<member>\\; shared copies in ${SH}: ceil-ref (team 4 real-record coder wnr_common.py: features f ceil-form, g0, c canonical zero, d coded lengths, e hash, t tail; models\\even.npz trained on the 80 even-indexed experts; results\\heldout_g0.jsonl / heldout_summary_g0.json = reference values; held-out protocol: train on the 80 even-indexed, test on the 40 odd-indexed experts), team3-capacity-value (capacity_value.py), team4-capacity-update (capacity_update.py, results JSON, container sizes). GPU plain runners and their 3070 reference values in ${J(ROOT, 'tests', 'gpu')} (README there; encoder for team 2/4's A259/D/CF/CFD streams).
- The copied code reads environment variables (PowerShell state does not persist between your commands, so prefix every command): ${ENVSET}; GPU code under partial\\: WNR_NVRTC_ARCH as a bare arch such as 'sm_120' (no '-arch=' prefix; default detected from the device: sm_120 on the 5090), WNR_NVSMI (default nvidia-smi); cpu-simd build.ps1: WNR_GCC_BIN (folder of gcc.exe if not on PATH).
- Old paths: the docs cite the original PC's absolute paths (C:\\..., D:\\..., AppData scratchpads) as provenance. They do not exist here: look them up in ${J(ROOT, 'docs', 'README.md')} (old path -> kit location); an old path not in that table is provenance only and not available.
- Established facts (original PC): WNR2 held-out real record 0.89943 of raw; f+g0+d+e 0.88021; f+g0+c+d+e 0.85282; sign bypass costs 1e-6 bit/w but a FIXED sign plane pays the signs of zeros, cancelling canonical zero's gain; GPU decode ~0.2 ms/record on the 5090 (REF), ~300 cycles/symbol/lane on the 3070 prototype.
- Covered by the kit's PLAIN runner, not an agent here: stream chunk-size sweep (tests\\cpu\\chunk_sweep.py; cite its summary.json in ${J(ROOT, 'results')} if it was run).
- CONCURRENCY on this PC: this workflow is the only research team running: no other kit workflow and no setup\\run_all.ps1 at the same time. Members may run the specific kit runners their own task names (e.g. signplane-z: run_gpu_bench.py --encode-only), always with --out inside their own folder ${OUT}\\<member>\\; do not start any other plain runner. cpu-simd's timings need a quiet CPU: signplane-z must keep its heavy CPU jobs to at most max(1, logical_cores / 4) processes and pause new heavy jobs while ${OUT}\\cpu-simd\\TIMING_IN_PROGRESS exists (cpu-simd creates that file before a timing batch and deletes it after). For every CPU timing record the concurrent CPU load (Get-Counter '\\Processor(_Total)\\% Processor Time') and report best-of-N with the load; numbers taken under load are lower bounds and must be labelled. The GPU (RTX 5090) is used only by signplane-z${GPU ? '' : ' (and not at all in this run: args.gpu is false)'}; check nvidia-smi --query-compute-apps=pid,process_name --format=csv first and label timings taken with another compute process as contaminated.
RULES: never modify ${J(ROOT, 'docs')}, ${J(ROOT, 'tests')}, ${PART} or ${DATA}\\v41-sample: copy what you need into ${OUT}\\<your-member-name>\\ (create it) and work only there. Run Python as "${PY} -I <script>", scripts outside data folders. No downloads, no system-wide installs, no KAIRA changes, no posting, no git push, no credentials. Tag claims as measured (result file), read-in-code/source (path:line or URL) or inferred. Before returning, write your returned object as JSON to ${OUT}\\<your-member-name>\\member_result.json. Return raw data.`

const SCHEMA = {
  type: 'object',
  properties: {
    member: { type: 'string' },
    verdict: { type: 'string' },
    environment: { type: 'string', description: 'CPU / GPU / compiler / flags actually used' },
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
  required: ['member', 'verdict', 'environment', 'findings', 'numbers', 'spec_impact', 'files_written', 'open_questions'],
}

const SIMD = `Member "cpu-simd". Build a FAST CPU decoder in C for WNR2 and for the WNR3 v1 profile (ceil inverse + HASH; LENC optional), to answer whether WNR can be decoded on the CPU-only V4.1 path fast enough ON THIS MACHINE's CPU. Resume from ${J(PART, 'team6', 'cpu-simd')} (copy it to ${OUT}\\cpu-simd\\ first): src\\wnrdec.c (interleaved/AVX2 rANS decoder with LUT variants NBSH 5/6/7, OpenMP, modes h32 / decode / bench, Windows API timers; byte-exact compare against the source .bin), src\\wdec_v8.h + wdec_il.h, src\\gen_records.py (WNR2 and f+g0+e records with the COPIED ceil-ref coder src\\wnr_common.py and model even.npz; usage: SAMPLE_DIR OUT_DIR IDX,...), src\\mklist.py (writes list_*.txt; wnrdec reads them with fscanf %s, so the sample path must not contain spaces: mklist uses a relative path and stops otherwise), src\\build.ps1 (gcc; header of wnrdec.c also gives the MSVC line cl /O2 /arch:AVX2 /openmp). Status on the original PC: records for the 8 held-out experts idx 1,23,41,63,81,103,121,143 generated and Python-verified byte-exact (ratios: WNR2 ~0.896, fge ~0.880), decoder built, but NO timing was recorded. Steps: (1) regenerate the 16 records with gen_records.py into ${OUT}\\cpu-simd\\rec (check sizes/sha against rec\\records.json of the copy), mklist.py, build (gcc -O3 -march=native, and MSVC /arch:AVX2 and /arch:AVX512 if available); (2) verify byte-exact output against the source .bin bytes for every stream of every record, and the h32 mode against the spec vector; (3) measure decoded GB/s and ms/record for: scalar baseline, AVX2 single thread, 1/2/4/8/16/... threads up to the logical core count, with and without the ceil inverse and the hash check; if the CPU has AVX-512, add a 16-lane variant if it is a contained change (record design and gain); cite Giesen's interleaved/SIMD rANS work and published AVX2 rANS decoders for design choices; (4) compare with the NVMe read time of a record on each drive of the reference's drive table (REF lines, GB/s) and, if possible, a READ-ONLY sequential-read measurement of a large existing file on the drives of this PC, to state whether CPU decode would be the bottleneck. Create ${OUT}\\cpu-simd\\TIMING_IN_PROGRESS before each timing batch and delete it after.`

const VAL = simd => `Member "cpu-path-value". Today V4.1 experts run CPU-only on the KAIRA machine (REF:523; M0 measured 1.11-1.14 tok/s with WEX1 on U:; read the M0 lines in the reference). Build a small, explicit model of tok/s on that CPU-only path for WEX1, WNR2, WNR3 (f+g0+d+e, ratio 0.88021) and WNR3z (0.85282), using: misses per token and record sizes from the reference, the drive bandwidth/latency table, the RAM cache size (read the actual RAM of this PC from check_env.py; the reference machine has 32 GB), and the CPU decode throughput measured by cpu-simd below on THIS CPU (if this PC is not the KAIRA machine, scale with an explicit, stated assumption and also give results as a function of decode GB/s). Include whether decode can overlap NVMe reads (threads), the extra RAM-cache capacity from compression (records fit = cache bytes / record bytes), and the break-even decode speed above which WNR beats WEX1. You may adapt team 3's capacity model (copy ${J(SH, 'team3-capacity-value', 'capacity_value.py')}) or team 4's update (${J(SH, 'team4-capacity-update', 'capacity_update.py')}; it reads ${J(SH, 'ceil-ref', 'results')} and the team-3 results JSON next to it) or write a simpler one, but state every parameter and its source (REF line, measured file, or assumption). If this is the KAIRA machine you may read drive letters, sizes and free space (Get-PSDrive / Get-Volume, read-only) and compare them with REF. Output: tok/s medians and ranges for each format and drive (U:, E: of the reference, plus this PC's drives if different), the break-even decode GB/s, and a recommendation for whether a CPU WNR decoder is worth building before G2.

CPU-SIMD RESULT:
${simd ? JSON.stringify(simd) : `cpu-simd did not run in this workflow run: use ${OUT}\\cpu-simd\\member_result.json if it exists, otherwise give results as a function of decode GB/s and say so.`}`

const SIGN = `Member "signplane-z". Value-exact layout that combines canonical zero with sign bypass: code only the 3-bit magnitude with rANS (8-symbol CDF per context, ceil-form data with g0), and store sign bits ONLY for nonzero magnitudes in a compacted raw sign plane (Z1: plane at the stream tail read backwards; Z2: Z1 + the last <= 7 signs carried in x0 above the 23-bit hash). Resume from ${J(PART, 'team6', 'signplane-z')} (copy to ${OUT}\\signplane-z\\ first): src\\signplane.py (layout on a verbatim copy of ceil-ref wnr_common.py), train_z.py (out\\zmodel.npz), run_z.py (usage: SAMPLE_DIR ZMODEL_NPZ CEILREF_MODELS_NPZ OUT_JSONL EXPERT_INDICES CONFIGS [SAVE_DIR]; configs z1de, z2de, z1e, ref_fgcde, ref_fgde), summarize_z.py (paired bootstrap vs team 4's g0 records: TEAM4_G0_JSONL = ${J(SH, 'ceil-ref', 'results', 'heldout_g0.jsonl')}), integrity_z.py (single-bit corruption trials incl. sign-plane flips), make_jobs.py + c\\zdec.c (CPU decoder for CF / D / Z1 / Z2 weight streams, paired timing), gpu\\wnr2dec_z.cu + gpu\\zbench.py (Z kernel wdecz vs team 4's D, CFD5, CF5, A259 on team 4's encoded data). Status on the original PC: model trained; out\\heldout_z.jsonl has z1de and z2de for 18 of the 40 held-out experts (all value-exact); integrity, C timing and GPU timing not run. Steps: (1) run_z.py for all 40 held-out experts (z1de, z2de, plus ref_fgcde and ref_fgde as reproduction check; reuse the 18 finished rows only if your re-run of 2 of them is bit-identical), summarize_z.py: real record bytes vs f+g0+c+d+e (0.85282) and vs f+g0+d+e (0.88021) with bootstrap CIs; (2) integrity_z.py: acceptance of single-bit flips in rANS bytes vs sign plane, with and without HASH, and how the HASH covers the raw signs; (3) build zdec.c (gcc -O3 -march=native or MSVC) and time CF / D / Z1 / Z2 paired on one expert (respect cpu-simd's TIMING_IN_PROGRESS marker); (4) ${GPU ? `GPU: produce team 4's encoded streams (A259, D, CF, CFD, ...) on the CPU with "${PY} -I ${J(ROOT, 'tests', 'gpu', 'run_gpu_bench.py')} --root ${ROOT} --data ${DATA} --out ${OUT}\\signplane-z\\bench --encode-only --enc ${OUT}\\signplane-z\\t4enc" (check its --help first; same 8 experts), then run zbench.py on the RTX 5090 (arch auto-detected; >= 21 paired rounds; check byte-exact output: Z output = canonical raw, -0 -> +0, scales after the deferred patch must equal the ORIGINAL scale bytes) and report Z1 vs D, CFD5, CF5, A259` : 'no GPU in this run: give an inferred GPU estimate from team 2/4 D/A259/CFD5 ratios in tests\\gpu\\ref'}. Recommend whether Z1/Z2 should replace c (canonical zero in the 16-symbol alphabet) in WNR3 and what the spec would need.`

phase('Research')
log(`kit root ${ROOT}; data ${DATA}; results ${OUT}; members: ${ONLY ? ONLY.join(', ') : 'all'}; gpu ${GPU}`)
log('Covered by a plain runner, not an agent here: chunk-size -> tests\\cpu\\chunk_sweep.py')
const chain = (async () => {
  let simd = null, val = null
  if (want('cpu-simd')) simd = await agent(`${CTX}\n\nTASK:\n${SIMD}`, { label: 'cpu-simd', phase: 'Research', schema: SCHEMA, effort: 'high' })
  if (want('cpu-path-value')) val = await agent(`${CTX}\n\nTASK:\n${VAL(simd)}`, { label: 'cpu-path-value', phase: 'Research', schema: SCHEMA })
  return [simd, val]
})()
const others = []
if (want('signplane-z')) others.push(() => agent(`${CTX}\n\nTASK:\n${SIGN}`, { label: 'signplane-z', phase: 'Research', schema: SCHEMA }))
const [chainRes, otherRes] = await Promise.all([chain, parallel(others)])
const all = [...chainRes, ...otherRes].filter(Boolean)
log(`Team 6 members returned: ${all.map(r => r.member).join(', ') || 'none'}`)

phase('Critic')
let critic = null
if (all.length) {
  critic = await agent(`${CTX}\n\nTASK:\nMember "critic". Adversarially review team 6's results below: open the cited files and recheck every number (ratios = bytes/18,800,640; GB/s computations; thread scaling vs core count of this CPU), check that byte-exact / value-exact claims were checked against the ORIGINAL source bytes of ${SAMPLE}, check that CPU timings report their load and are not presented as clean when contaminated (and that the TIMING_IN_PROGRESS protocol was followed), check the CPU-path model's parameters against the reference lines and this PC's measured hardware, check GPU timings for arch sm_120 and the paired method, and list contradictions, unsupported claims and missing checks, most serious first. Finally write ${OUT}\\team6_summary.json = {"members": <the RESULTS array below verbatim>, "critic": <your returned object>} and ${OUT}\\team6_summary.md (English, one page: verdicts, key numbers vs original PC, open items).\n\nRESULTS:\n${JSON.stringify(all)}`,
    { label: 'critic', phase: 'Critic', schema: SCHEMA, effort: 'high' })
} else log('no member result: critic skipped')
return { members: all, critic, covered_by_runners: { 'chunk-size': 'tests/cpu/chunk_sweep.py' } }
