"""Predefined layout-cluster bootstrap, one-sided randomization and four-test Holm."""
import hashlib
import numpy as np

BOOTSTRAP_DRAWS = 10000
PERMUTATION_DRAWS = 100000
BASE_SEED = 20261103


def rng_for(name, stream):
    seed = int.from_bytes(hashlib.sha256(f'{BASE_SEED}|{name}|{stream}'.encode()).digest()[:8], 'big')
    return np.random.default_rng(seed), seed


def contrast(a, b, name, primary=False, bootstrap_draws=BOOTSTRAP_DRAWS, permutation_draws=PERMUTATION_DRAWS):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[0] > 0
    assert np.isfinite(a).all() and np.isfinite(b).all()
    delta = (a-b).mean(axis=1)
    rng, bootstrap_seed = rng_for(name, 'bootstrap')
    selected = rng.integers(0, len(delta), size=(bootstrap_draws, len(delta)))
    draws = delta[selected].mean(axis=1)
    result = {'contrast': name, 'layouts': len(delta), 'seeds_per_layout': a.shape[1],
        'difference': float(delta.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
        'bootstrap_draws': bootstrap_draws, 'bootstrap_seed': bootstrap_seed,
        'bootstrap_values_sha256': hashlib.sha256(draws.astype('<f8').tobytes()).hexdigest()}
    if primary:
        assert np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()
        integer_delta = (a-b).sum(axis=1).astype(np.int64)
        observed = int(integer_delta.sum())
        rng, permutation_seed = rng_for(name, 'randomization')
        signs = rng.integers(0, 2, size=(permutation_draws, len(delta)), dtype=np.int8)*2-1
        permuted = signs@integer_delta
        exceed = int(np.count_nonzero(permuted >= observed))
        result.update(p_one_sided=(exceed+1)/(permutation_draws+1), permutation_draws=permutation_draws,
            permutation_seed=permutation_seed, permutation_exceedances_including_ties=exceed,
            observed_integer_sum=observed)
    return result


def holm(pvalues):
    keys = sorted(pvalues, key=lambda k: (pvalues[k], k))
    adjusted = {}
    running = 0.
    for i, key in enumerate(keys):
        p = float(pvalues[key])
        assert 0 <= p <= 1
        running = max(running, (len(keys)-i)*p)
        adjusted[key] = min(1., running)
    return {key: adjusted[key] for key in pvalues}


def ratio_contrast(na, da, nb, db, name):
    """Condition-specific conditional rates, with denominators resampled too."""
    na, da, nb, db = [np.asarray(v, dtype=float) for v in (na, da, nb, db)]
    assert na.shape == da.shape == nb.shape == db.shape and na.ndim == 2
    rng, seed = rng_for(name, 'bootstrap')
    selected = rng.integers(0, len(na), size=(BOOTSTRAP_DRAWS, len(na)))
    xa, ya, xb, yb = [v[selected].sum(axis=(1, 2)) for v in (na, da, nb, db)]
    valid = (ya > 0) & (yb > 0)
    draws = xa[valid]/ya[valid]-xb[valid]/yb[valid]
    return {'contrast': name, 'difference': float(na.sum()/da.sum()-nb.sum()/db.sum()) if da.sum() and db.sum() else None,
        'ci95': np.quantile(draws, [.025, .975]).tolist() if len(draws) else None,
        'bootstrap_draws': BOOTSTRAP_DRAWS, 'undefined_bootstrap_draws': int((~valid).sum()),
        'bootstrap_seed': seed, 'numerators': [int(na.sum()), int(nb.sum())],
        'denominators': [int(da.sum()), int(db.sum())]}
