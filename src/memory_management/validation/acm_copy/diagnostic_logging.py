"""External, reversible instrumentation of the fixed official policy and runner.

No checker/evaluate calls, RNG draws, transforms, or model changes occur here.
Ground truth is written by the simulator; D0 never reads it in the policy.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = "acm-m3-v1"
STATIC = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask")


def digest(array):
    a = np.asarray(array)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def append(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def simple(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): simple(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [simple(v) for v in value]
    return value


class PolicyLogger:
    """Wrap an instance, preserving original methods for uninstrumented replay."""
    def __init__(self, policy, context_path=None, context=None):
        self.policy = policy
        self.context_path = Path(context_path) if context_path else None
        self.context = context
        self.original = {name: getattr(policy, name) for name in ("reset", "add_buffer", "_prepare_history", "infer")}
        self.active = False
        self.selected = []
        self.decision = 0
        self.frames = {}
        self.attach()

    def attach(self):
        if self.active:
            return
        p = self.policy
        p.reset, p.add_buffer = self.reset, self.add_buffer
        p._prepare_history, p.infer = self.prepare_history, self.infer
        self.active = True
        self.attach_buffer()

    def detach(self):
        for name, method in self.original.items():
            setattr(self.policy, name, method)
        if self.policy.mem_buffer is not None:
            self.policy.mem_buffer.get_frame_sampling_indices = self.selector_original
        self.active = False

    def attach_buffer(self):
        self.selector_original = self.policy.mem_buffer.get_frame_sampling_indices
        self.policy.mem_buffer.get_frame_sampling_indices = self.select

    def emit(self, name, row):
        if self.context:
            append(Path(self.context["directory"]) / name, {"schema": SCHEMA,
                   **{k: self.context[k] for k in ("run_id", "task", "episode_id", "variant") if k in self.context},
                   **row})

    def reset(self):
        result = self.original["reset"]()
        self.attach_buffer()
        self.selected, self.frames, self.decision = [], {}, 0
        if self.context_path:
            self.context = json.loads(self.context_path.read_text()) if self.context_path.exists() else None
        self.emit("policy_lifecycle.jsonl", {"kind": "reset", "step_idx": self.policy.step_idx,
                  "exec_start_idx": self.policy.exec_start_idx, "model_seed": self.policy._seed})
        return result

    def add_buffer(self, obs):
        first = self.policy.step_idx + 1
        result = self.original["add_buffer"](obs)
        images, states = obs["images"], obs["state"]
        n = len(images)
        rows = []
        for k in range(n):
            idx = first+k
            row = {"frame_id": idx, "front_sha256": digest(images[k, 0]),
                   "state_sha256": digest(states[k]), "control_step": idx-self.policy.exec_start_idx,
                   "is_demo": idx < self.policy.exec_start_idx}
            self.frames[idx] = row
            rows.append(row)
        self.emit("policy_ingest.jsonl", {"decision_idx": self.decision, "first_frame_id": first,
                  "last_frame_id": self.policy.step_idx, "samples": n, "image_shape": list(images.shape),
                  "state_shape": list(states.shape), "exec_start_idx": self.policy.exec_start_idx,
                  "frames": rows})
        return result

    def select(self, step_idx, token_budget, token_per_image):
        # Call the official selector exactly once. D0 returns its exact result.
        indices = self.selector_original(step_idx, token_budget, token_per_image)
        variant = self.context.get("variant", "D0") if self.context else "D0"
        if variant != "D0":
            from diagnostic_selection import choose
            indices = choose(variant, indices, self.policy.mem_buffer, step_idx, self.context)
        self.selected = list(indices)
        if self.selected != sorted(set(self.selected)) or len(indices)*token_per_image > token_budget:
            raise ValueError("Invalid selection or token budget")
        if any(i < 0 or i > step_idx for i in indices):
            raise ValueError("Noncausal selection")
        return indices

    def prepare_history(self, obs):
        result = self.original["_prepare_history"](obs)
        arrays = {k: np.asarray(result[k]) for k in STATIC}
        mask = arrays["static_mask"]
        n = len(self.selected)*16
        assert [arrays[k].shape for k in STATIC] == [(512, 2048), (512, 768), (512, 8), (512,)]
        assert mask.dtype == np.bool_ and int(mask.sum()) == n
        assert mask[:n].all() and not mask[n:].any()
        # Verify ORIGINAL source-time positional features; no re-timestamping.
        for slot, idx in enumerate(self.selected):
            expected = self.policy.mem_buffer._history_feats[idx]["pos_emb_4x4"].reshape(16, 768)
            assert np.array_equal(arrays["static_pos_emb"][slot*16:(slot+1)*16], expected)
        row = {"decision_idx": self.decision, "available_max_frame_id": self.policy.step_idx,
               "control_step": self.policy.step_idx-self.policy.exec_start_idx,
               "exec_start_idx": self.policy.exec_start_idx, "history_count": len(self.policy.mem_buffer._history_feats),
               "selected_frame_ids": self.selected, "selected_control_steps": [i-self.policy.exec_start_idx for i in self.selected],
               "selected_demo_flags": [i < self.policy.exec_start_idx for i in self.selected],
               "source_position_frame_ids": self.selected, "token_budget": 512, "tokens_per_frame": 16,
               "num_views": 1, "valid_tokens": n, "padding_tokens": 512-n, "mask": mask.tolist(),
               "tensor_sha256": {k: digest(v) for k, v in arrays.items()},
               "tensor_dtype": {k: str(v.dtype) for k, v in arrays.items()},
               "selected_front_sha256": [self.frames[i]["front_sha256"] for i in self.selected]}
        self.emit("policy_decisions.jsonl", row)
        return result

    def infer(self, obs):
        # Save exact observations on client; server independently hashes the received input.
        input_hashes = {k: digest(v) for k, v in obs.items() if isinstance(v, np.ndarray)}
        result = self.original["infer"](obs)
        if self.context:
            path = Path(self.context["directory"]) / "actions"
            path.mkdir(exist_ok=True)
            np.savez(path / f"decision_{self.decision:04d}.npz",
                     **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        self.emit("policy_actions.jsonl", {"decision_idx": self.decision, "input_sha256": input_hashes,
                  "prompt": obs.get("prompt"), "actions_sha256": digest(result["actions"]),
                  "action_shape": list(result["actions"].shape), "execution_horizon": 16,
                  "infer_time_ms": result["infer_time_ms"]})
        self.decision += 1
        return result


def snapshot(env, info):
    """Read already-computed GT. Never invoke a predicate/checker/evaluate."""
    gt = {"info": simple({k: v for k, v in info.items() if k in
          ("simple_subgoal_online", "grounded_subgoal_online", "status", "success", "fail")})}
    names = ("elapsed_steps", "current_task_index", "current_task_name", "demonstration_record_traj", "after_demo",
             "red_cubes_in_bin", "blue_cubes_in_bin", "green_cubes_in_bin", "binfill_language_sequence",
             "red_cubes_target_number", "blue_cubes_target_number", "green_cubes_target_number",
             "target_cube_names", "target_cube_colors", "target_cube_pickup_counts", "target_cube_pickup_active",
             "color_names", "swap_times", "pick_times", "highlight_starts", "match")
    for name in names:
        if hasattr(env, name):
            gt[name] = simple(getattr(env, name))
    tasks = getattr(env, "task_list", [])
    index = gt.get("current_task_index", -1)
    if isinstance(index, (int, float)) and 0 <= index < len(tasks):
        gt["task_phase"] = tasks[int(index)].get("name")
    for name in ("selected_buttons", "achieved_list", "selected_bins", "all_cubes", "target_cubes", "spawned_bins"):
        if hasattr(env, name):
            gt[name] = [{"name": getattr(a, "name", str(i)), "position": simple(a.pose.p)}
                        for i, a in enumerate(getattr(env, name))]
    if hasattr(env, "swap_schedule"):
        gt["swap_schedule"] = [[getattr(a, "name", None), getattr(b, "name", None), s, e]
                                for a, b, s, e in env.swap_schedule]
    gt["tcp_position"] = simple(env.agent.tcp.pose.p)
    return gt


class RawRecorder:
    def __init__(self, directory, task):
        self.directory, self.task = Path(directory), task
        (self.directory / "raw").mkdir(exist_ok=True)
        (self.directory / "observations").mkdir(exist_ok=True)
        self.pending, self.batch = [], []
        self.rows = []
        self.exec_start = None
        self.control_step = 0
        self.shard = 0

    def install(self, runner):
        self.runner = runner
        wrapped = runner.env
        while not hasattr(type(wrapped), "_augment_obs_and_info"):
            wrapped = wrapped.env
        original_augment = wrapped._augment_obs_and_info

        def augment(obs, info, action):
            out, details = original_augment(obs, info, action)
            gt = snapshot(runner.env.unwrapped, details)
            # The original wrapper already produced this segmentation mapping.
            # Pixel presence supports visibility, but is not a sufficiency judgment.
            segmentation = obs.get("sensor_data", {}).get("base_camera", {}).get("segmentation")
            if segmentation is not None:
                if hasattr(segmentation, "detach"):
                    segmentation = segmentation.detach().cpu().numpy()
                ids, counts = np.unique(np.asarray(segmentation), return_counts=True)
                count_map = dict(zip(ids.tolist(), counts.tolist()))
                mapping = getattr(wrapped, "segmentation_id_map", {})
                pixels = {}
                for sid, actor in mapping.items():
                    name = getattr(actor, "name", str(sid))
                    pixels[name] = pixels.get(name, 0) + count_map.get(int(sid), 0)
                gt["front_visible_pixels"] = pixels
            cache = getattr(runner.env.unwrapped, "_highlight_cache", {})
            gt["highlighted_targets"] = [getattr(a, "name", str(i))
                  for i, a in enumerate(getattr(runner.env.unwrapped, "target_cubes", []))
                  if cache.get(id(a), {}).get("is_highlighted", False)]
            self.pending.append({"front_sha256": digest(out["front_rgb_list"]),
                                 "gt": gt,
                                 "timestamp_ns": time.time_ns()})
            return out, details

        wrapped._augment_obs_and_info = augment
        original_init, original_step = runner.get_init_obs, runner.step

        def init():
            result = original_init()
            self.exec_start = len(result["images"])-1
            cursor = 0
            for i, (front, wrist, state) in enumerate(zip(result["images"], result["wrist_images"], result["states"])):
                h = digest(front)
                while cursor < len(self.pending) and self.pending[cursor]["front_sha256"] != h:
                    cursor += 1
                if cursor == len(self.pending):
                    raise ValueError("Cannot align demonstration image to original wrapper snapshot")
                self.add(front, wrist, state, self.pending[cursor], i, i-self.exec_start, i < self.exec_start)
                cursor += 1
            self.pending.clear()
            self.flush()
            return result

        def step(action):
            self.pending.clear()
            result = original_step(action)
            self.control_step += 1
            (front, wrist, state), stop, outcome = result
            if front is not None:
                h = digest(front)
                matches = [p for p in self.pending if p["front_sha256"] == h]
                if not matches:
                    raise ValueError("Cannot align execution image to wrapper snapshot")
                self.add(front, wrist, state, matches[-1], len(self.rows), self.control_step, False)
            append(self.directory / "control_actions.jsonl", {"control_step": self.control_step,
                   "action": simple(action), "stop": bool(stop), "outcome": outcome,
                   "raw_frame_id": len(self.rows)-1 if front is not None else None})
            return result

        runner.get_init_obs, runner.step = init, step

    def add(self, front, wrist, state, details, idx, control, demo):
        row = {"schema": SCHEMA, "frame_id": idx, "control_step": control, "is_demo": demo,
               "exec_start_idx": self.exec_start, **details, "wrist_sha256": digest(wrist),
               "state_sha256": digest(state), "shape": list(front.shape),
               "shard": f"raw/shard_{self.shard:04d}.npz", "shard_offset": len(self.batch)}
        self.rows.append(row)
        self.batch.append((front.copy(), wrist.copy(), state.copy()))
        append(self.directory / "raw_frames.jsonl", row)
        if len(self.batch) >= 32:
            self.flush()

    def flush(self):
        if not self.batch:
            return
        path = self.directory / "raw" / f"shard_{self.shard:04d}.npz"
        front, wrist, state = zip(*self.batch)
        np.savez_compressed(path, front=np.stack(front), wrist=np.stack(wrist), state=np.stack(state))
        append(self.directory / "raw_shards.jsonl", {"path": str(path), "size": path.stat().st_size,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frames": len(self.batch)})
        self.shard += 1
        self.batch.clear()

    def install_client(self, official_eval):
        original = official_eval._websocket_client_policy.MMEVLAWebsocketClientPolicy
        recorder = self

        class LoggingClient(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.decision_idx = 0
                self.next_frame = 0

            def add_buffer(self, obs):
                count = len(obs["images"])
                for j in range(count):
                    row = recorder.rows[self.next_frame+j]
                    assert digest(obs["images"][j, 0]) == row["front_sha256"]
                    assert digest(obs["state"][j]) == row["state_sha256"]
                recorder.flush()
                response = super().add_buffer(obs)
                append(recorder.directory / "client_ingest.jsonl", {"decision_idx": self.decision_idx,
                       "first_frame_id": self.next_frame, "last_frame_id": self.next_frame+count-1,
                       "exec_start_idx": obs.get("exec_start_idx", 0), "samples": count})
                self.next_frame += count
                return response

            def infer(self, obs):
                np.savez_compressed(recorder.directory / "observations" / f"decision_{self.decision_idx:04d}.npz",
                    **{k: v for k, v in obs.items() if isinstance(v, np.ndarray)}, prompt=np.asarray(obs["prompt"]))
                # Oracle metadata remains local; the WebSocket input is unchanged.
                response = super().infer(obs)
                append(recorder.directory / "client_decisions.jsonl", {"decision_idx": self.decision_idx,
                       "available_max_frame_id": self.next_frame-1,
                       "input_sha256": {k: digest(v) for k, v in obs.items() if isinstance(v, np.ndarray)},
                       "actions_sha256": digest(response["actions"])})
                self.decision_idx += 1
                return response

        official_eval._websocket_client_policy.MMEVLAWebsocketClientPolicy = LoggingClient
