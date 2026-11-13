"""Monte Carlo check of the block model in parametric.py (synthetic data only).
1) OCP quantiser on Gaussian blocks: block max magnitude must be 6 or 7.
2) Rejection-sample |u| ~ halfnormal(a) | (|u|<8, max>=4), quantise, compare the empirical first-position magnitude
   distribution and the mean block log-likelihood with the closed form."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parametric import fam_tables, loglik, LAM

rng = np.random.default_rng(1)
GRID = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])


def q_mag(u):
    """RNE onto the E2M1 magnitude grid with saturation; returns magnitude index 0..7"""
    a = np.abs(u)
    t = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5])
    return np.searchsorted(t, a, side='left')  # ties are measure-zero


# 1) real quantiser
w = rng.normal(size=(200000, 32))
amax = np.abs(w).max(1)
X = np.exp2(np.floor(np.log2(amax)) - 2)
m = q_mag(w / X[:, None])
print('block max magnitude histogram (expect only 6,7):', np.bincount(m.max(1), minlength=8))

# 2) closed form vs simulation at a fixed lam
for lam in (0.5, 0.75, 1.5):
    a = 2.0 ** lam
    blocks = []
    need = 60000
    while sum(len(b) for b in blocks) < need:
        u = np.abs(rng.normal(scale=a / np.sqrt(2), size=(200000, 32)))  # gg(beta=2, a) has sigma = a/sqrt(2)
        ok = (u.max(1) < 8) & (u.max(1) >= 4)
        blocks.append(u[ok])
    u = np.concatenate(blocks)[:need]
    mm = q_mag(u)
    emp = np.bincount(mm[:, 0], minlength=8) / len(mm)
    logp, logr, lognorm = fam_tables('gg', 2.0, np.array([lam]))
    p = np.exp(logp[0]); q = p.copy(); q[6] = np.exp(logr[0]) * p[6]; q[7] = 0
    Q = q.sum()
    model = (p - q * Q ** 31) / (1 - Q ** 32)
    N = np.stack([(mm == k).sum(1) for k in range(8)], 1).astype(float)
    ll_model = loglik(N, (logp, logr, lognorm))[:, 0].mean()
    # empirical check of normalisation: exact log-prob of the magnitudes should match model to MC error; compare with
    # a nearby wrong lam (must be worse)
    lw = fam_tables('gg', 2.0, np.array([lam + 0.25]))
    ll_wrong = loglik(N, lw)[:, 0].mean()
    lw2 = fam_tables('gg', 2.0, np.array([lam - 0.25]))
    ll_wrong = (ll_wrong, loglik(N, lw2)[:, 0].mean())
    print(f'lam {lam}: first-pos emp {np.round(emp, 4)}\n          model    {np.round(model, 4)}  sum {model.sum():.6f}'
          f'\n          mean loglik true {ll_model:.4f} nats, lam+-0.25 {ll_wrong[0]:.4f} {ll_wrong[1]:.4f}')
