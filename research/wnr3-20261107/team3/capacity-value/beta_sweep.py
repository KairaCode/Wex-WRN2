"""Shape check: WNR2-vs-WEX1 gain on U: as beta and qmix vary (others at center)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import capacity_value as cvm

R = cvm.format_ratios()
rng = np.random.default_rng(99)
U = cvm.make_uniforms(rng, 1, 1500, 40, 6)
for drive in ('U', 'E'):
    for qmix in (0.0, 0.5, 1.0):
        row = []
        for b in (0.42, 0.5, 0.6, 0.7, 0.85, 1.0):
            P = cvm.center_params(beta=b, qmix=qmix)
            w = cvm.simulate(P, cvm.V41, 1.0, 'none', None, drive, True, U)
            n = cvm.simulate(P, cvm.V41, R['WNR2'], 'none', None, drive, True, U)
            z = cvm.simulate(P, cvm.V41, R['WNR3z'], 'none', None, drive, True, U)
            row.append(f'b={b}: WEX1 {1/w["t"][0]:.2f} t/s rho {w["rho"][0]:.2f} WNR2 {cvm.pct(w["t"], n["t"])[0]:+.1f}% '
                       f'3z/2 {cvm.pct(n["t"], z["t"])[0]:+.1f}%')
        print(drive, f'qmix={qmix}', ' | '.join(row))
