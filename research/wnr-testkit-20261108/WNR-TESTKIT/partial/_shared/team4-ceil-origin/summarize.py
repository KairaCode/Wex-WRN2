"""Collect all result JSONs of team4/ceil-origin into summary.json (raw numbers, one row per checkpoint).

Usage: python -I summarize.py RESULT_DIR
i.i.d. expectation of even blocks = sum over matrices of (#max-7 blocks x P(no code 1/3 among 31 other symbols)),
with P from that matrix's own magnitude shares in its max-7 blocks (one max symbol per block removed).
"""
import json, os, sys

FILES = [('deepseek-ai/DeepSeek-V4.1-Flash-1017 (upstream, 4 kept + 8 pruned + 1 MTP)', 'upstream_v41_1017.json'),
         ('deepseek-ai/DeepSeek-V4.1-Flash', 'other_ds_v41flash.json'),
         ('deepseek-ai/DeepSeek-V4-Flash', 'other_ds_v4flash.json'),
         ('deepseek-ai/DeepSeek-V4-Pro', 'other_ds_v4pro.json'),
         ('openai/gpt-oss-20b', 'other_gptoss20.json'),
         ('openai/gpt-oss-120b', 'other_gptoss120.json'),
         ('amd/gpt-oss-120b-w-mxfp4-a-fp8 (Quark PTQ of gpt-oss-120b)', 'other_amd_gptoss120.json'),
         ('amd/DeepSeek-R1-0528-MXFP4 (Quark PTQ from FP8)', 'other_amd_r1.json'),
         ('RedHatAI/GLM-5.3-MXFP4 (llm-compressor PTQ)', 'other_rh_glm53.json')]


def main():
    d = sys.argv[1]
    rows = []
    for name, f in FILES:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        T = j['totals']
        exp_even = 0.0
        per_exp = []
        for e in j['experts']:
            for m in e['matrices'].values():
                if m['p_iid_even_given_max7'] == m['p_iid_even_given_max7']:
                    exp_even += m['bmax_hist'][7] * m['p_iid_even_given_max7']
            per_exp.append(round(e['even_pct'], 3))
        rows.append({
            'checkpoint': name, 'file': f, 'items': j.get('n_experts', j.get('n_items')), 'weights': j['weights'],
            'blocks': T['blocks'], 'bmax_share_pct_code6_code7': [round(T['bmax_share_pct'][6], 3), round(T['bmax_share_pct'][7], 3)],
            'blocks_max_below_6': T['max_lt6'], 'even_pct': round(T['even_pct'], 4),
            'even_pct_iid_expected': round(100 * exp_even / T['blocks'], 5),
            'even_pct_per_item_min_max': [min(per_exp), max(per_exp)],
            'max6_without_1_3_blocks': T['max6_no13'], 'roundtrip_failures': T['roundtrip_fail'], 'scale_overflow_255': T['scale_ovf'],
            'zero_pos_pct': round(T['zero_pos_pct'], 3), 'zero_neg_pct': round(T['zero_neg_pct'], 3),
            'w_bpw_orig_in_ho': [round(j['w_orig']['in_bpw'], 5), round(j['w_orig']['ho_bpw'], 5)],
            'w_bpw_ceil_in_ho': [round(j['w_ceil']['in_bpw'], 5), round(j['w_ceil']['ho_bpw'], 5)],
            'w_ceil_delta_pct_ho': round(j['w_ceil']['delta_pct_vs_orig_ho'], 4),
            'w_ceil_delta_pct_in': round(100 * (j['w_ceil']['in_bpw'] - j['w_orig']['in_bpw']) / j['w_orig']['in_bpw'], 4),
            'w_ceil_z_delta_pct_ho': round(j['w_ceil_z']['delta_pct_vs_orig_ho'], 4),
            'w_orig_z_delta_pct_ho': round(j['w_orig_z']['delta_pct_vs_orig_ho'], 4),
            's_bps_orig_in_ho': [round(j['s_orig']['in_bps'], 4), round(j['s_orig']['ho_bps'], 4)],
            's_bps_ceil_in_ho': [round(j['s_ceil']['in_bps'], 4), round(j['s_ceil']['ho_bps'], 4)],
            'record_delta_pct_of_raw_ho': round(j['ceil_delta_pct_of_raw'], 4),
            'ratio_est_orig_to_ceil_ho': [round(j['orig_ratio_est'], 5), round(j['ceil_ratio_est'], 5)],
        })
    json.dump(rows, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
    for r in rows:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
