#!/usr/bin/env python3
"""HarvestMem v3 closed-loop evaluator for SG, K0, and NM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

import imageio.v2 as imageio
import numpy as np

ECM = Path("/media/fulei/fuleivla/ecm_vla")
POLICY = Path("/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning")
sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"),
                str(ECM), str(ECM / "src"), str(ECM / "scripts"), str(ECM / "scripts/harvestmem_v3")]

DEV_SEED_BASE = 4200000
TRAIN_SEED_BASE = 4100000
GUARD = json.loads((ECM / "results/fruit_k0v2_gate_20261102/guard_v2.json").read_text())

from oracle import HarvestMemOracle  # noqa: E402
from subgoal import compose_prompt, project_world_to_pixel, quantize_unique  # noqa: E402

ARM_MODEL = {"A": "sg", "AR": "sg", "D": "k0", "DR": "k0", "E": "nm", "ER": "nm",
             "sg": "sg", "k0": "k0", "nm": "nm",
             "DP1R": "k0", "DPaR": "k0", "EP1R": "nm", "EPaR": "nm",
             "DP": "k0", "DPa": "k0", "EP": "nm", "EPa": "nm"}


def make_env(task: str):
    import gymnasium as gym
    import fruit_picking.task_v2_env  # noqa: F401

    budget = 2800 if task == "count" else 1300
    env = gym.make(
        "FruitPickTaskV2-v0", task=task, task_version="v2", task_revision=9,
        robot_uids="rm75b_eg2", no_lamp_wait=True,
        obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
        render_mode="rgb_array", sim_backend="cpu", max_episode_steps=budget,
    )
    return env, budget


def _snapshot(u, obs: dict[str, object], timestep: int) -> dict:
    """Keep the GT row fields aligned with fruit_r9_data.H5Recorder.record."""
    from memory_management.fruit_v2_distance import any_hanging

    q = u.agent.robot.get_qpos()[0].cpu().numpy()
    tcp = u.agent.tcp_pose
    position = tcp.p[0].cpu().numpy()
    quaternion = tcp.q[0].cpu().numpy()
    positions = u.fruit_positions()
    inside = u.in_basket()
    hanging = u.hanging()
    seg = obs["sensor_data"]["front_camera"]["segmentation"][0, ..., 0].cpu().numpy()
    fruits = []
    for i in range(len(positions)):
        fruits.append({
            "pos": positions[i].tolist(),
            "attached": bool(u.attached[i]),
            "present": bool(u.present[i]),
            "wind_detached": bool(u.wind_detached[i]),
            "in_basket": bool(inside[i]),
            "front_pixels": int(np.sum(seg == int(u.fruits[i].per_scene_id[0]))),
        })
    return {
        "timestep": int(timestep),
        "phase": None,
        "subgoal": None,
        "target": None,
        "tcp_pos": position.tolist(),
        "tcp_quat": quaternion.tolist(),
        "gripper_qpos": q[7:9].tolist(),
        "fruits": fruits,
        "n_in_basket": int(inside.sum()),
        "lamp_green": bool(u.lamp_green),
        "fruit_distance": float(any_hanging(position, positions, hanging)),
        "distance": float(any_hanging(position, positions, hanging)),
        "distance_signal": "any_hanging",
    }


def _front_wrist_state(u, obs):
    sd = obs["sensor_data"]
    q = u.agent.robot.get_qpos()[0].cpu().numpy()
    n = len(u.agent.arm_joint_names)
    front = sd["front_camera"]["rgb"][0].cpu().numpy().astype(np.uint8)
    wrist = sd["hand_camera"]["rgb"][0].cpu().numpy().astype(np.uint8)
    state = np.concatenate([q[:n], q[n:n + 1]]).astype(np.float32)
    return front, wrist, state


def _slot_pixels(u, obs):
    params = obs["sensor_param"]["front_camera"]
    intrinsic = params["intrinsic_cv"][0].cpu().numpy()
    extrinsic = params["extrinsic_cv"][0].cpu().numpy()
    projected = {
        slot: project_world_to_pixel(intrinsic, extrinsic, geometry["pos"])
        for slot, geometry in enumerate(u.original_geometry)
    }
    return quantize_unique(projected)


def _instruction_selfcheck():
    import gymnasium as gym
    import fruit_picking.task_v2_env  # noqa: F401
    env = gym.make(
        "FruitPickTaskV2-v0", task="count", task_version="v2", task_revision=9,
        robot_uids="rm75b_eg2", no_lamp_wait=True,
        obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
        render_mode="rgb_array", sim_backend="cpu", max_episode_steps=2810,
    )
    try:
        env.reset(seed=TRAIN_SEED_BASE)
        instruction = env.unwrapped.layout.instruction.lower()
    finally:
        env.close()
    with (ECM / "data/harvestmem_v3_dataset_20261110/data/0.pkl").open("rb") as handle:
        dataset_prompt = pickle.load(handle)["prompt"]
    if instruction != dataset_prompt:
        raise RuntimeError(f"train prompt mismatch: {instruction!r} != {dataset_prompt!r}")
    return {"train_clean_count_ep0_seed": TRAIN_SEED_BASE,
            "train_prompt_exact_match": True, "train_prompt": dataset_prompt}


class Episode:
    def __init__(self, task: str, episode: int, arm: str, out: Path):
        self.task = task
        self.episode = episode
        self.arm = arm
        self.out = out
        self.env, self.budget = make_env(task)
        self.seed = DEV_SEED_BASE + 16 * episode
        self.u = None
        self.obs = None
        self.oracle = None
        self.latest_oracle = None
        self.trace_file = None
        self.actions_file = None
        self.timeline_file = None
        self.video = None
        self.last_info = {}
        self.last_outcome = "ongoing"
        self.self_checks = {}
        self.stop_trigger_step = None
        self.stop_trigger_n = None
        self.stop_arrive_step = None
        self.stop_end_step = None
        self.stop_primitive_timeout = False
        self.planner_reach = "off"
        self.planner_calls = 0
        self.planner_failures = []
        self.planner_replans = 0
        self.planner_steps = 0
        self.planner_would_clip_steps = 0
        self.planner_projection_steps = 0
        self.planner_images = []
        self.planner_states = []
        self.handoffs = []
        self.planner_done_first = False
        self.planner_pending = False
        self.capture_handoff_frames = False
        self.handoff_frame_schedule = {}

    def _write_trace(self, snap, oracle_output):
        snap = dict(snap)
        snap["oracle_subgoal"] = oracle_output.subgoal
        snap["oracle_phase"] = oracle_output.phase
        snap["oracle_target_slot"] = oracle_output.target_slot
        snap["oracle_events"] = list(oracle_output.events)
        self.trace_file.write(json.dumps(snap) + "\n")
        if (oracle_output.events or self.latest_oracle is None
                or oracle_output.subgoal != self.latest_oracle.subgoal):
            self.timeline_file.write(json.dumps({
                "timestep": snap["timestep"], "subgoal": oracle_output.subgoal,
                "phase": oracle_output.phase, "target_slot": oracle_output.target_slot,
                "events": list(oracle_output.events),
            }) + "\n")
        self.latest_oracle = oracle_output

    def reset(self):
        self.obs, _ = self.env.reset(seed=self.seed)
        self.u = self.env.unwrapped
        self.instruction = self.u.layout.instruction
        self.slot_xy = _slot_pixels(self.u, self.obs)
        ripe = [bool(item["ripe"]) for item in self.u.original_geometry]
        slot_pos = [list(item["pos"]) for item in self.u.original_geometry]
        self.oracle = HarvestMemOracle(self.task, int(self.u.layout.n_pick), self.u.slot_of,
                                       self.slot_xy, ripe, slot_pos)
        self.front, self.wrist, self.state = _front_wrist_state(self.u, self.obs)
        self.out.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(self.out / "first_front.png", self.front)
        imageio.imwrite(self.out / "first_wrist.png", self.wrist)
        imageio.imwrite(self.out / "first_front_wrist.png",
                        np.concatenate([self.front, self.wrist], axis=1))
        self.video = imageio.get_writer(self.out / "front_camera.mp4", fps=30, codec="libx264", quality=6)
        self.video.append_data(self.front)
        self.trace_file = (self.out / "trace.jsonl").open("w")
        self.actions_file = (self.out / "actions.jsonl").open("w")
        self.timeline_file = (self.out / "subgoal_timeline.jsonl").open("w")
        initial = _snapshot(self.u, self.obs, 0)
        initial["instruction"] = self.instruction
        self._write_trace(initial, self.oracle.step(initial))
        dev_path = ECM / "results/harvestmem_v3_S1/replays/dev_expert" / self.task / f"ep{self.episode}_attempt0/result.json"
        if dev_path.exists():
            dev = json.loads(dev_path.read_text())
            expected_raw = dev.get("harvestmem_v3_slot_pixels")
            has_layout_pixels = isinstance(expected_raw, dict) and bool(expected_raw)
            expected_pixels = {int(k): list(v) for k, v in (expected_raw or {}).items()}
            if has_layout_pixels and int(self.u.layout.n_pick) != int(dev["N"]):
                raise RuntimeError(f"N mismatch: live={self.u.layout.n_pick} dev={dev['N']}")
            if has_layout_pixels and {k: list(v) for k, v in self.slot_xy.items()} != expected_pixels:
                raise RuntimeError(f"slot pixel mismatch: live={self.slot_xy} dev={expected_pixels}")
            self.self_checks["dev_layout_exact"] = has_layout_pixels
        else:
            self.self_checks["dev_layout_exact"] = None
        self.self_checks.update({
            "env_id": self.env.spec.id,
            "robot": "rm75b_eg2",
            "task_revision": int(self.u.task_revision),
            "no_lamp_wait": bool(self.u.no_lamp_wait),
            "seed": self.seed,
            "expected_seed": DEV_SEED_BASE + 16 * self.episode,
            "count_prompt_has_no_work_light": "work light" not in self.instruction.lower()
                if self.task == "count" else True,
            "slot_xy": {str(k): list(v) for k, v in self.slot_xy.items()},
        })
        if self.self_checks["env_id"] != "FruitPickTaskV2-v0" or self.self_checks["robot"] != "rm75b_eg2":
            raise RuntimeError("environment or robot self-check failed")
        if not self.self_checks["no_lamp_wait"] or self.self_checks["task_revision"] != 9:
            raise RuntimeError("task variant self-check failed")
        return self.front, self.wrist, self.state

    def step(self, action, guard=True):
        action = np.asarray(action, dtype=np.float64)
        if guard:
            from memory_management.fruit_rescue_platform import guard_v2
            q = self.u.agent.robot.get_qpos()[0].cpu().numpy()
            action, clipping = guard_v2(action, q, GUARD)
        else:
            clipping = {}
        self.obs, _, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.front, self.wrist, self.state = _front_wrist_state(self.u, self.obs)
        for handoff_index, offset in self.handoff_frame_schedule.pop(int(self.u.control_steps), []):
            imageio.imwrite(self.out / f"handoff_{handoff_index:03d}_plus_{offset:02d}.png", self.front)
        self.video.append_data(self.front)
        snap = _snapshot(self.u, self.obs, int(self.u.control_steps))
        oracle_output = self.oracle.step(snap)
        self._write_trace(snap, oracle_output)
        success = bool(info["success"][0])
        fail = bool(info["fail"][0])
        timeout = bool(truncated) or self.u.control_steps >= self.budget
        self.last_outcome = "success" if success else "fail" if fail else "timeout" if timeout else "ongoing"
        self.last_info = info
        return self.front, self.wrist, self.state, bool(terminated) or timeout or success or fail, self.last_outcome, clipping

    def close(self):
        if self.video is not None:
            self.video.close()
        for handle in (self.trace_file, self.actions_file, self.timeline_file):
            if handle is not None:
                handle.close()
        self.env.close()

    def write_action(self, action, source, *, step=None, **extra):
        self.actions_file.write(json.dumps({
            "step": int(self.u.control_steps if step is None else step), "source": source,
            "action": np.asarray(action).round(6).tolist(), **extra,
        }) + "\n")

    def run_stop_primitive(self):
        rest = np.asarray(self.u.agent.keyframes["rest"].qpos[:7], dtype=np.float64)
        current = np.asarray(self.u.agent.robot.get_qpos()[0].cpu().numpy()[:7], dtype=np.float64)
        delta = rest - current
        steps = max(1, int(np.ceil(np.max(np.abs(delta)) / 0.01)))
        for index in range(steps):
            alpha = (index + 1) / steps
            action = np.r_[current + alpha * delta, 1.0]
            self.write_action(action, "stop_primitive", stop_phase="return_to_rest")
            _, _, _, done, _, _ = self.step(action, guard=False)
            if done:
                self.stop_end_step = int(self.u.control_steps)
                return
        self.stop_arrive_step = int(self.u.control_steps)
        for _ in range(300):
            action = np.r_[rest, 1.0]
            self.write_action(action, "stop_primitive", stop_phase="hold_rest")
            _, _, _, done, _, _ = self.step(action, guard=False)
            if done:
                self.stop_end_step = int(self.u.control_steps)
                return
        self.stop_primitive_timeout = True
        self.stop_end_step = int(self.u.control_steps)

    def should_stop(self, debug_step=None, enforce_stop=False):
        if self.task != "count":
            return False
        if debug_step is not None and self.u.control_steps >= debug_step:
            return True
        return bool(enforce_stop and self.latest_oracle.phase == "return"
                    and int(self.u.in_basket().sum()) >= int(self.u.layout.n_pick))

    def planner_step(self, action, labels):
        from memory_management.fruit_rescue_platform import guard_v2
        q = self.u.agent.robot.get_qpos()[0].cpu().numpy()
        _, clipping = guard_v2(np.asarray(action, dtype=np.float64), q, GUARD)
        would_clip = bool(clipping["guard_clipped"])
        self.planner_would_clip_steps += int(would_clip)
        self.planner_projection_steps += int(clipping["guard_projection"])
        self.planner_steps += 1
        self.write_action(action, "planner", planner_phase=labels.get("phase"),
                          planner_target=labels.get("target"), would_clip=would_clip,
                          guard_projection=bool(clipping["guard_projection"]))
        result = self.step(action, guard=False)
        self.planner_images.append(self.front.copy())
        self.planner_states.append(self.state.copy())
        if self.should_stop(None, self.enforce_stop):
            self.stop_trigger_step = int(self.u.control_steps)
            self.stop_trigger_n = int(self.u.in_basket().sum())
            self.run_stop_primitive()
        return result

    def planner_reach_to_target(self, target_slot):
        from fruit_picking.solver import FruitPickSolver, PlanningFailure, PRE_GRASP, horizontal_grasp_pose, OPEN
        from fruit_picking.canonical_solver import configure
        actor = self.u.hanging_slot[target_slot] if target_slot < len(self.u.hanging_slot) else None
        if actor is None:
            raise PlanningFailure(f"no live actor for slot {target_slot}")
        if bool(self.u._grasped()):
            self.timeline_file.write(json.dumps({"timestep": int(self.u.control_steps), "events": ["planner_skipped_grasped"]}) + "\n")
            return False
        solver = FruitPickSolver(self.env, self.planner_step, ik_mode="canonical_v31", ik_seed=0, ik_candidates=64)
        configure(solver, {"settle_steps": 2})
        fruit = self.u.fruit_positions()[actor]
        approach = np.array([fruit[0], fruit[1], 0.0], dtype=np.float64)
        approach /= np.linalg.norm(approach[:2])
        pre = horizontal_grasp_pose(fruit - PRE_GRASP * approach, approach)
        solver.refresh_obstacles(skip=(actor,))
        solver.gripper = OPEN
        labels = {"subgoal": f"planner reach slot {target_slot}", "target": actor, "phase": "reach"}
        solver.move(pre, labels)
        return True

    def maybe_planner(self, args):
        if args.planner_reach == "off" or self.task != "count" and args.planner_reach == "first":
            return
        if not self.planner_pending:
            return
        if args.planner_reach == "first" and self.planner_done_first:
            self.planner_pending = False
            return
        target = self.latest_oracle.target_slot
        if target is None or self.latest_oracle.phase != "pick":
            self.planner_pending = False
            return
        max_attempts = 3 if args.planner_reach == "all" else 1
        for attempt in range(max_attempts):
            self.planner_calls += 1
            start_step = int(self.u.control_steps)
            try:
                reached = self.planner_reach_to_target(target)
                if not reached:
                    self.planner_pending = False
                    return
                self.handoffs.append({"start_step": start_step,
                    "timestep": int(self.u.control_steps), "target_slot": target,
                    "end_target_slot": self.latest_oracle.target_slot,
                    "tcp_pos": self._current_tcp()})
                if self.capture_handoff_frames:
                    idx = len(self.handoffs) - 1
                    imageio.imwrite(self.out / f"handoff_{idx:03d}_plus_00.png", self.front)
                    for offset in (8, 16):
                        self.handoff_frame_schedule.setdefault(int(self.u.control_steps) + offset, []).append((idx, offset))
                self.planner_pending = False
                self.planner_done_first = True
                if args.planner_reach == "all" and self.latest_oracle.phase == "pick" and self.latest_oracle.target_slot != target:
                    self.planner_replans += 1
                    target = self.latest_oracle.target_slot
                    self.planner_pending = True
                    continue
                return
            except Exception as exc:
                self.planner_failures.append({"timestep": int(self.u.control_steps), "target_slot": target, "error": repr(exc)})
                self.planner_pending = False
                return

    def _current_tcp(self):
        return self.u.agent.tcp_pose.p[0].cpu().numpy().tolist()


def run_episode(args) -> dict:
    started = time.monotonic()
    episode = Episode(args.task, args.episode, args.arm, args.out)
    result = {"task": args.task, "episode": args.episode, "arm": args.arm,
              "outcome": "error", "error": "", "self_checks": {}}
    try:
        episode.enforce_stop = bool(args.enforce_stop)
        episode.planner_reach = args.planner_reach
        episode.capture_handoff_frames = bool(args.capture_handoff_frames)
        result["train_prompt_self_check"] = _instruction_selfcheck()
        front, wrist, state = episode.reset()
        result.update(seed=episode.seed, N=int(episode.u.layout.n_pick),
                      instruction=episode.instruction, checkpoint=args.checkpoint,
                      budget=episode.budget, slot_xy=episode.self_checks["slot_xy"],
                      slot_initial_ripe=[bool(item["ripe"]) for item in episode.u.original_geometry])
        result["self_checks"] = episode.self_checks
        result["self_checks"]["checkpoint_path_ok"] = "harvestmem_v3_S2" in args.checkpoint
        if not result["self_checks"]["checkpoint_path_ok"]:
            raise RuntimeError("checkpoint path self-check failed")

        from openpi_client import websocket_client_policy as wcp
        from utils import pack_buffer
        client = wcp.MMEVLAWebsocketClientPolicy("127.0.0.1", args.port)
        if args.context:
            args.context.parent.mkdir(parents=True, exist_ok=True)
            args.context.write_text(json.dumps({
                "memory": args.memory,
                "trace": str(args.out / "trace.jsonl"),
            }))
        reset = client.reset()
        if not reset.get("reset_finished"):
            raise RuntimeError(f"policy reset failed: {reset}")

        images, states = [front], [state]
        fixed = np.r_[state[:7], 1.0].astype(np.float32)
        for _ in range(30):
            front, wrist, state, stop, outcome, _ = episode.step(fixed, guard=False)
            images.append(front); states.append(state)
            if episode.latest_oracle and "target_selected" in episode.latest_oracle.events:
                episode.planner_pending = True
            if episode.should_stop(args.debug_stop_at_step, args.enforce_stop):
                episode.stop_trigger_step = int(episode.u.control_steps)
                episode.stop_trigger_n = int(episode.u.in_basket().sum())
                episode.run_stop_primitive()
                break
            if stop:
                break
        episode.maybe_planner(args)
        if episode.stop_trigger_step is not None:
            pending_images, pending_states = [], []
        else:
            pending_images, pending_states = images, states
        exec_start = len(images) - 1
        if episode.planner_images:
            pending_images.extend(episode.planner_images)
            pending_states.extend(episode.planner_states)
            episode.planner_images.clear(); episode.planner_states.clear()
            front, wrist, state = episode.front, episode.wrist, episode.state
        while (episode.last_outcome == "ongoing" and episode.u.control_steps < (args.max_steps or episode.budget)
               and episode.stop_trigger_step is None):
            episode.actions_file.flush(); episode.trace_file.flush(); episode.timeline_file.flush()
            client.add_buffer(pack_buffer(pending_images, pending_states, exec_start))
            prompt = episode.instruction.lower()
            if ARM_MODEL[args.arm] == "sg":
                prompt = compose_prompt(episode.instruction, episode.latest_oracle.subgoal)
            prediction = client.infer({"observation/image": front, "observation/wrist_image": wrist,
                                       "observation/state": state, "prompt": prompt})
            chunk = np.asarray(prediction["actions"], dtype=np.float32)[:16]
            new_images, new_states = [], []
            for action in chunk:
                action_step = int(episode.u.control_steps)
                action_labels = {"prompt": prompt, "oracle_subgoal": episode.latest_oracle.subgoal,
                                 "oracle_phase": episode.latest_oracle.phase,
                                 "oracle_target_slot": episode.latest_oracle.target_slot}
                front, wrist, state, stop, outcome, clipping = episode.step(action, guard=True)
                episode.write_action(action, "policy", step=action_step, **action_labels,
                                     guard_clipped=bool(clipping["guard_clipped"]),
                                     guard_projection=bool(clipping["guard_projection"]))
                new_images.append(front); new_states.append(state)
                if episode.latest_oracle and "target_selected" in episode.latest_oracle.events:
                    episode.planner_pending = True
                    if args.planner_reach == "all":
                        break
                if episode.should_stop(args.debug_stop_at_step, args.enforce_stop):
                    episode.stop_trigger_step = int(episode.u.control_steps)
                    episode.stop_trigger_n = int(episode.u.in_basket().sum())
                    episode.run_stop_primitive()
                    break
                if stop or (args.max_steps and episode.u.control_steps >= args.max_steps):
                    break
            pending_images, pending_states = new_images, new_states
            exec_start = 0
            episode.maybe_planner(args)
            if episode.planner_images:
                pending_images.extend(episode.planner_images)
                pending_states.extend(episode.planner_states)
                episode.planner_images.clear(); episode.planner_states.clear()
                front, wrist, state = episode.front, episode.wrist, episode.state
            if episode.stop_trigger_step is not None:
                break
        result.update(outcome=episode.last_outcome, steps=int(episode.u.control_steps),
                      n_in_basket=int(episode.last_info.get("n_in_basket", [episode.oracle.n_pick])[0])
                      if episode.last_info else int(episode.oracle.n_pick),
                      reason=episode.last_info.get("v2_reason") if episode.last_info else "smoke_timeout",
                      enforce_stop=bool(args.enforce_stop),
                      debug_stop_at_step=args.debug_stop_at_step,
                      stop_trigger_step=episode.stop_trigger_step,
                      stop_trigger_n=episode.stop_trigger_n,
                      stop_arrive_step=episode.stop_arrive_step,
                      stop_end_step=episode.stop_end_step,
                      stop_primitive_timeout=episode.stop_primitive_timeout,
                      planner_reach=args.planner_reach,
                      planner_calls=episode.planner_calls,
                      planner_failures=episode.planner_failures,
                      planner_replans=episode.planner_replans,
                      planner_steps=episode.planner_steps,
                      planner_would_clip_steps=episode.planner_would_clip_steps,
                      planner_projection_steps=episode.planner_projection_steps,
                      handoffs=episode.handoffs)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        result["wall_s"] = round(time.monotonic() - started, 1)
        result["self_checks"] = {**result.get("self_checks", {}), **episode.self_checks}
        try:
            episode.close()
        except Exception:
            result["error"] += "\nclose:\n" + traceback.format_exc()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "error"}, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task", choices=("count", "occluded_ripe"), required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_MODEL), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memory", choices=("uniform", "none"), default="uniform")
    parser.add_argument("--context", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--enforce-stop", action="store_true")
    parser.add_argument("--debug-stop-at-step", type=int)
    parser.add_argument("--planner-reach", choices=("off", "first", "all"), default="off")
    parser.add_argument("--capture-handoff-frames", action="store_true")
    result = run_episode(parser.parse_args())
    return int(bool(result.get("error")))


if __name__ == "__main__":
    raise SystemExit(main())
