import sys, json, cProfile, pstats, os
sys.path.insert(0, r'<RESEARCH-PC>\team2\audit')
import audit_count as a
man = json.load(open(r'<RESEARCH-PC>\v41-sample\manifest.json'))
p = os.path.join(r'<RESEARCH-PC>\team2\audit\counts', 'L00_E097.npz')
if os.path.exists(p):
    os.remove(p)
job = (r'<RESEARCH-PC>\v41-sample', r'<RESEARCH-PC>\team2\audit\counts', man['experts'][0])
cProfile.run('a.do_expert(job)', r'<RESEARCH-PC>\team2\audit\prof.out')
pstats.Stats(r'<RESEARCH-PC>\team2\audit\prof.out').sort_stats('tottime').print_stats(10)
