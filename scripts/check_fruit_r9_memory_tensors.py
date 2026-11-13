#!/usr/bin/env python3
"""R9 tensor gates on one recorded D5 episode. Read-only inputs; no rollout or training."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from v_a_common import ECM, POLICY, OPENPI_DATA_HOME, read_json, read_jsonl, sha256, timestamp
from fruit_r9_common import OUT, proof, d5_cases, recorded_distances
sys.path[:0] = [str(ECM), str(ECM / 'src'), str(POLICY / 'src'), str(POLICY / 'packages/openpi-client/src')]
os.environ.setdefault('OPENPI_DATA_HOME', str(OPENPI_DATA_HOME))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def main():
    path = OUT / 'memory_tensor_checks.json'
    if path.exists():
        saved = read_json(path)
        if not saved['passed']:raise RuntimeError('Failed tensor check retained; no automatic retry')
        print(saved);return
    result = {'revision': 9, 'passed': False, 'created_utc': timestamp()}
    try:
        import jax.numpy as jnp
        from mme_vla_suite.policies.policy_config import create_trained_policy
        from mme_vla_suite.training import config
        from mme_vla_suite.training import dataset as official_dataset
        from mme_vla_suite.shared.mem_buffer import MemoryBuffer
        from mme_vla_suite.shared.posemb_3d import PosEmb3D
        from memory_management import event_keyframes as ek
        from memory_management.policy_adapter import pack_frame_memory, install_temporal_index
        from memory_management.fruit_v2_distance import from_snapshot, RecordedDistance
        from fruit_picking.task_v2_audit_r9 import MEMORIES
        from finetune_event_keyframes import install_sampler

        source = Path(d5_cases()[0]['source']);distances = recorded_distances(source)
        times = np.linspace(0, len(distances) - 1, 10, dtype=int).tolist()
        spec = importlib.util.spec_from_file_location('_r9_tensor_original', OUT / 'baseline_sources/src/memory_management/event_keyframes.py')
        legacy = importlib.util.module_from_spec(spec);sys.modules[spec.name] = legacy;spec.loader.exec_module(legacy)
        required = set(range(16))
        for t in times:
            for cfg in [ek.KeyframeConfig(events=4), *MEMORIES.values()]:
                required.update(ek.select(distances, t, cfg)[0])
        data = {};cursor = 0
        for shard in read_jsonl(source / 'raw_shards.jsonl'):
            stored = Path(shard['path'])
            if not stored.exists():stored = source / 'raw' / stored.name
            assert sha256(stored) == shard['sha256']
            with np.load(stored) as archive:
                for k in range(len(archive['front'])):
                    if cursor + k in required:
                        data[cursor + k] = (archive['front'][k].copy(), archive['state'][k].copy())
                cursor += len(archive['front'])
            if cursor > max(required):break
        assert set(data) == required
        geometry = {'tcp_position':[0.,0.,0.], 'fruit_positions':[[.01,0.,0.],[.1,0.,0.],[0.,0.,0.]],
                    'detached':[False,False,False], 'present':[True,True,False],
                    'ripe':[False,True,True], 'fruit_distance':.1}
        assert from_snapshot(geometry) == .01
        geometry['ripe'] = [True,False,False];geometry['fruit_distance'] = 999.
        assert from_snapshot(geometry) == .01
        recorded = RecordedDistance();recorded.reset({'directory':str(source)})
        first = read_jsonl(source/'raw_frames.jsonl')[0]
        assert recorded.distance(SimpleNamespace(frame_id=0,image=data[0][0])) == from_snapshot(first['gt'])
        result['any_hanging_inference_factory'] = {'passed':True,'maturity_and_legacy_distance_ignored':True,
                                                  'inactive_reserve_excluded':True,'recorded_frame_hash_checked':True}
        checkpoint = ECM / 'runs/fruit_k0v2_lowlevel_rescue_20261102/ckpt/mme_vla_suite/k0/11999'
        policy = create_trained_policy(config.get_config('mme_vla_suite'), checkpoint, seed=7)
        bank = MemoryBuffer(prepare_buffer=False)
        bank.vision_enc = policy._vision_encode;bank.pos_emb_dict = policy.mem_buffer.pos_emb_dict
        ids = sorted(required)
        for start in range(0, len(ids), 16):
            batch = ids[start:start + 16]
            bank.add_buffer(np.stack([data[t][0] for t in batch])[:, None],
                            np.stack([data[t][1] for t in batch]), batch)
        names = ('image', 'position', 'state', 'mask')
        absolute_checks = []
        original_pack = bank._prepare_frame_sampling
        for t in times:
            for events in (0, 4, 8):
                old_frames = legacy.select(distances, t, legacy.KeyframeConfig(events=events))[0]
                new_frames = ek.select(distances, t, ek.KeyframeConfig(events=events))[0]
                before = original_pack(bank._history_feats, old_frames, 512, 16)
                after = pack_frame_memory(bank, bank._history_feats, new_frames, 512, 16)
                assert all(np.array_equal(a, b) and a.dtype == b.dtype and a.tobytes()==b.tobytes()
                           for a, b in zip(before, after))
                absolute_checks.append({'step': t, 'events': events,
                    'elementwise_equal': dict.fromkeys(names, True)})
        # Real frozen-encoder features reconstructed once from recorded RGB/state;
        # both paths receive exactly those features. No D5 action or environment is rerun.
        result['absolute_default'] = {'passed': True, 'source': str(source), 'checks': absolute_checks,
            'scope': 'Before/after packing on identical frozen-encoder features of recorded D5 input; historical files unchanged.'}
        fixture = OUT / 'tensor_unit_fixture';(fixture / 'features/episode_0').mkdir(parents=True, exist_ok=True)
        np.save(fixture / 'features/episode_0/distance.npy', distances)
        original_sampler = official_dataset.RoboMMEDataset.prepare_frame_sampling
        policy.mem_buffer = bank;install_temporal_index(policy, 'order')
        order_checks = []
        try:
            for condition, cfg in MEMORIES.items():
                install_sampler(cfg, fixture, temporal_index='order')
                fake = SimpleNamespace(mem_buffer=MemoryBuffer(prepare_buffer=False),
                    history_config=SimpleNamespace(budget=512, token_per_image=16, num_views=1),
                    _gather_history_feat=lambda frames, episode: {t: bank._history_feats[t] for t in frames})
                for t in times:
                    frames = ek.select(distances, t, cfg)[0]
                    training = official_dataset.RoboMMEDataset.prepare_frame_sampling(fake, 0, t)
                    serving = bank._prepare_frame_sampling(bank._history_feats, frames, 512, 16)
                    assert all(np.array_equal(a, b) and a.dtype == b.dtype and a.tobytes()==b.tobytes()
                               for a, b in zip(training, serving))
                    for rank, frame in enumerate(frames):
                        encoded = serving[1][rank * 16:(rank + 1) * 16]
                        reference = bank.pos_emb_dict['4x4'][rank].reshape(16, 768)
                        source_position = bank._history_feats[frame]['pos_emb_4x4'].reshape(16, 768)
                        assert np.array_equal(encoded[:, :256], reference[:, :256])
                        assert np.array_equal(encoded[:, 256:], source_position[:, 256:])
                    ordinary = original_pack(bank._history_feats, frames, 512, 16)
                    assert all(np.array_equal(serving[i], ordinary[i]) for i in (0, 2, 3))
                    order_checks.append({'condition': condition, 'step': t,
                        'train_inference_elementwise_equal': dict.fromkeys(names, True),
                        'temporal_rank_equal': True, 'spatial_unchanged': True})
        finally:
            official_dataset.RoboMMEDataset.prepare_frame_sampling = original_sampler
        result['order_train_inference'] = {'passed': True, 'times': times, 'checks': order_checks}
        encoder = PosEmb3D(768)
        boundary = np.asarray(encoder(jnp.array([2047, 2048, 2800, 4095, 4096]), 4))
        assert boundary.shape == (5, 16, 768) and np.isfinite(boundary).all()
        table = bank.pos_emb_dict['4x4']
        assert table[2800:2801].shape == (1, 16, 768)
        assert table[4096:4097].shape == (0, 16, 768)
        result['feature_length_boundary'] = {'passed': True, 'initial_temporal_cache': 2048,
            'PosEmb3D_automatically_expands': True, 'buffer_max_steps_default': 4096,
            'R9_budget': 2800, 'R9_safe_under_default_table': True,
            'beyond_max_steps': 'Empty position slice; explicitly allocate max_steps above the largest frame index for longer future data. Default unchanged.'}
        result.update(passed=True, selected_raw_frames=len(required), checkpoint=str(checkpoint),
                      default_selection_source_sha256=sha256(source / 'event_signal.jsonl'),
                      shared_packer_sha256=sha256(ECM / 'src/memory_management/policy_adapter.py'),
                      train_sampler_sha256=sha256(ECM / 'scripts/finetune_event_keyframes.py'))
        proof('memory_tensor_checks.json', result)
        print({'passed': True, 'absolute_tensor_cases': 30, 'order_parity_cases': 30,
               'recorded_episode_times': 10, 'new_rollouts': 0, 'training': False}, flush=True)
    except Exception as exc:
        result['error'] = str(exc);proof('memory_tensor_checks.json', result);raise


if __name__ == '__main__':main()
