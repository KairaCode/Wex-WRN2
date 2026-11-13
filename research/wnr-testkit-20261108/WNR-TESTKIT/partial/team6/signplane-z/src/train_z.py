"""Train the Z (magnitude-only) model from team 4 ceil-ref's training counts (80 even-indexed experts).
Usage: python -I train_z.py SAMPLE_DIR COUNTS_DIR CEILREF_MODELS_NPZ OUT_NPZ OUT_JSON"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signplane as Z

sample_dir, counts_dir, ref_models, out_npz, out_json = sys.argv[1:6]
info = Z.train(sample_dir, counts_dir, out_npz, ref_models)
t = Z.load_tables(out_npz)
info['model_id'] = t.model_id
info['model_blob_bytes'] = len(t.blob)
info['etab_bits_per_block'] = (t.etab / 256).round(2).tolist()
json.dump(info, open(out_json, 'w'), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != 'etab_bits_per_block'}, indent=1))
