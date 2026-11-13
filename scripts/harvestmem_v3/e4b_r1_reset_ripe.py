#!/usr/bin/env python3
"""Check recovered seed metadata against live CPU-simulation environment resets."""
import json
from pathlib import Path
from eval_episode import DEV_SEED_BASE, make_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/harvestmem_v3_E4b_r1'


def main():
    expected = json.loads((OUT / 'seeded_ripe.json').read_text())['episodes']
    env, _ = make_env('occluded_ripe')
    rows = []
    try:
        for episode in range(250, 260):
            seed = DEV_SEED_BASE + 16 * episode
            env.reset(seed=seed)
            actual = [bool(f['ripe']) for f in env.unwrapped.original_geometry]
            assert actual == expected[str(episode)]['ripe'], (episode, actual, expected[str(episode)])
            rows.append({'episode': episode, 'seed': seed, 'ripe': actual})
            print(f'ep{episode}: verified', flush=True)
    finally:
        env.close()
    (OUT / 'reset_ripe_verification.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
