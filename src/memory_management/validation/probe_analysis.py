"""Paired statistics for V-B; adaptive greedy sets are evaluated on held-out seeds only."""
from __future__ import annotations

import numpy as np
from scipy import stats


def standard_error(values, axis=-1):
    values = np.asarray(values, dtype=float)
    return np.std(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def split_reliability(gains):
    """Slot ranking across even/odd seeds and Spearman--Brown correction."""
    gains = np.asarray(gains, dtype=float)
    a, b = gains[:, ::2].mean(axis=1), gains[:, 1::2].mean(axis=1)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"spearman": None, "spearman_brown": None, "top1_agreement": None, "regret": None}
    r = float(stats.spearmanr(a, b).statistic)
    corrected = 2*r/(1+r) if r > -1 else None
    ia, ib = int(np.argmax(a)), int(np.argmax(b))
    regret = 0.5*((b.max()-b[ia])+(a.max()-a[ib]))
    return {"spearman": r, "spearman_brown": corrected, "top1_agreement": float(ia==ib), "regret": float(regret)}


def crossfit_best(gains):
    """Choose on one half and score on the other; SE never uses the selecting half's value."""
    gains = np.asarray(gains, dtype=float)
    a, b = gains[:, ::2], gains[:, 1::2]
    ia, ib = int(np.argmax(a.mean(axis=1))), int(np.argmax(b.mean(axis=1)))
    held_a, held_b = a[ib], b[ia]
    score = 0.5*(held_a.mean()+held_b.mean())
    # Conservative two-fold uncertainty (fold selections are statistically dependent).
    se = 0.5*(float(standard_error(held_a))+float(standard_error(held_b)))
    return {"gain": float(score), "heldout_se": se, "selected_on_even": ia, "selected_on_odd": ib}


def positive_pvalues(gains):
    """One-sided paired t test of per-seed gains; handle deterministic identity explicitly."""
    values = np.asarray(gains, float)
    mean, se = values.mean(axis=-1), standard_error(values)
    t = np.divide(mean, se, out=np.zeros_like(mean), where=se>0)
    p = stats.t.sf(t, values.shape[-1]-1)
    return np.where(se>0, p, np.where(mean>0, 0., 1.))


def bh_significant(pvalues, q=0.1):
    p = np.asarray(pvalues, dtype=float)
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good], kind="stable")]
    passed = p[order] <= q*np.arange(1,len(order)+1)/max(len(order),1)
    result = np.zeros(len(p),dtype=bool)
    if passed.any():
        result[order[:np.flatnonzero(passed)[-1]+1]] = True
    return result


def clustered_ci(values, episodes, *, draws=2000, seed=20261022):
    """Resample episodes, keeping all decision boundaries of each selected episode."""
    values, episodes = np.asarray(values,float), np.asarray(episodes)
    valid = np.isfinite(values)
    values, episodes = values[valid], episodes[valid]
    ids = np.unique(episodes)
    if len(ids) < 2:
        return {"mean": float(values.mean()) if len(values) else None, "ci95": None, "episodes": len(ids)}
    sums = np.array([values[episodes==i].sum() for i in ids])
    counts = np.array([(episodes==i).sum() for i in ids])
    rng = np.random.default_rng(seed)
    index = rng.integers(0,len(ids),(draws,len(ids)))
    sampled = sums[index].sum(axis=1)/counts[index].sum(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(sampled,[.025,.975]).tolist(), "episodes":len(ids)}


def additivity_fit(expected, actual):
    x,y = np.asarray(expected,float), np.asarray(actual,float)
    if len(x)<3 or np.ptp(x)==0:
        return {"slope": None,"r2":None,"n":len(x)}
    fit = stats.linregress(x,y)
    return {"slope":float(fit.slope),"r2":float(fit.rvalue**2),"n":len(x)}
