"""Actual Double-DQN learning with crossed, chronological behavior tapes."""
from dataclasses import dataclass
from collections import deque
from functools import lru_cache
import hashlib
import json
from dataclasses import asdict
import time

import numpy as np

from .env import World
from .learner import DQN
from .tabular import TabularQ


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


class Replay:
    def __init__(self, capacity, next_uid=0):
        self.capacity = capacity
        self.rows = np.zeros((capacity, 5), dtype=np.float64)
        self.uids = np.full(capacity, -1, dtype=np.int64)
        self.next_uid = next_uid
        self.size = 0

    def append(self, row):
        slot = self.next_uid % self.capacity
        self.rows[slot] = row
        self.uids[slot] = self.next_uid
        self.next_uid += 1
        self.size = min(self.capacity, self.size + 1)

    def clone(self):
        result = Replay(self.capacity, self.next_uid)
        result.rows = self.rows.copy()
        result.uids = self.uids.copy()
        result.size = self.size
        return result

    def recent(self, keep):
        result = Replay(keep, max(0, self.next_uid - min(keep, self.size)))
        for uid in range(result.next_uid, self.next_uid):
            result.append(self.rows[uid % self.capacity])
        return result

    def sample_uids(self, rng, count):
        return rng.integers(self.next_uid - self.size, self.next_uid, size=count)

    def batch(self, uids, embeddings):
        slots = uids % self.capacity
        if not np.array_equal(self.uids[slots], uids):
            raise ValueError("replay UID is missing or overwritten")
        rows = self.rows[slots]
        return (embeddings[rows[:, 0].astype(np.int64)], rows[:, 1].astype(np.int64),
                rows[:, 2], embeddings[rows[:, 3].astype(np.int64)], rows[:, 4].astype(bool))


@dataclass
class Tape:
    rows: np.ndarray
    sampled_uids: tuple
    sync: tuple
    actor_fingerprints: tuple
    scores: list
    final_fingerprint: str
    evaluation_steps: int
    work: dict

    def digest(self):
        digest = hashlib.sha256(self.rows.tobytes())
        for uids, sync in zip(self.sampled_uids, self.sync):
            digest.update(np.asarray(uids, dtype=np.int64).tobytes())
            digest.update(bytes([sync]))
        return digest.hexdigest()


@lru_cache(maxsize=128)
def references(world, context):
    # Each dynamic program queries two actions in nine nonterminal states.
    return float(np.max(world.oracle(context)[0])), world.random_return(context)


def work_counts(model, initial, evaluation_forward):
    return {"network_forward_examples": model.forward_examples - initial[0],
            "frozen_network_forward_examples": model.frozen_forward_examples - initial[1],
            "evaluation_network_forward_examples": evaluation_forward}


def evaluate(model, world, context, step):
    q = model.q_values(world.embeddings)
    state, total, discount, calls = 0, 0.0, 1.0, 0
    while state != 9:
        action = int(np.argmax(q[state]))
        state, reward, done = world.step(state, action, context)
        total += discount * reward
        discount *= world.gamma
        calls += 1
        if done:
            break
    optimal, random_value = references(world, context)
    score = (total - random_value) / (optimal - random_value)
    return {"step": step, "return": total, "normalized_return": score,
            "clairvoyant_optimal_return": optimal, "uniform_return": random_value}, calls


def auc(scores):
    horizon = scores[-1]["step"]
    return sum((right["step"] - left["step"]) * (left["normalized_return"] + right["normalized_return"]) / 2
               for left, right in zip(scores, scores[1:])) / horizon


def age(world, init_seed, config, model=None):
    if model is None:
        model = DQN(input_dim=config["observation_dim"], hidden=config["hidden"], seed=init_seed,
                    learning_rate=config["learning_rate"], gamma=world.gamma)
    replay = Replay(config["replay_capacity"])
    action_rng = np.random.default_rng(seed_for(init_seed, "age-actions"))
    replay_rng = np.random.default_rng(seed_for(init_seed, "age-replay"))
    state = 0
    context_index = 0
    context = world.training_context(0)
    steps, updates, episode_return = 0, 0, 0.0
    recent_returns = deque(maxlen=100)
    while True:
        explore = action_rng.random() < config["epsilon"]
        random_action = int(action_rng.integers(2))
        action = random_action if explore else int(np.argmax(model.q_values(world.embeddings[state])[0]))
        next_state, reward, done = world.step(state, action, context)
        replay.append((state, action, reward, next_state, done))
        steps += 1
        episode_return += reward
        if steps % config["update_every"] == 0 and replay.size >= config["batch_size"]:
            uids = replay.sample_uids(replay_rng, config["batch_size"])
            updates += 1
            model.update(*replay.batch(uids, world.embeddings), sync_target=updates % config["target_every"] == 0)
        state = next_state
        if done:
            recent_returns.append(episode_return)
            episode_return = 0.0
            state = 0
            if steps >= config["age_steps"]:
                break
            new_context_index = steps // config["block_steps"]
            if new_context_index != context_index:
                context_index = new_context_index
                context = world.training_context(context_index)
    return model, replay, context, {"training_interactions": steps, "optimizer_updates": updates,
                                   "last_context_index": context_index,
                                   "recent_undiscounted_episode_return": float(np.mean(recent_returns)),
                                   "work": work_counts(model, (0, 0), 0)}


def generate(model, replay, world, context, config, rng_seed, initial_updates, *, uniform=False, epsilon_restart=False):
    model, replay = model.clone(), replay.clone()
    initial_work = (model.forward_examples, model.frozen_forward_examples)
    action_rng = np.random.default_rng(seed_for(rng_seed, "actions"))
    replay_rng = np.random.default_rng(seed_for(rng_seed, "replay"))
    rows, samples, syncs, fingerprints = [], [], [], []
    first, eval_calls = evaluate(model, world, context, 0)
    evaluation_forward = model.forward_examples - initial_work[0]
    scores = [first]
    state, updates = 0, initial_updates
    for index in range(1, config["adapt_steps"] + 1):
        epsilon = 1.0 if uniform else (0.5 if epsilon_restart and index <= config["restart_steps"] else config["epsilon"])
        explore = action_rng.random() < epsilon
        random_action = int(action_rng.integers(2))
        action = random_action if explore else int(np.argmax(model.q_values(world.embeddings[state])[0]))
        next_state, reward, done = world.step(state, action, context)
        row = (state, action, reward, next_state, done)
        rows.append(row)
        replay.append(row)
        uids = np.empty(0, dtype=np.int64)
        sync = False
        if replay.next_uid % config["update_every"] == 0 and replay.size >= config["batch_size"]:
            uids = replay.sample_uids(replay_rng, config["batch_size"])
            updates += 1
            sync = updates % config["target_every"] == 0
            model.update(*replay.batch(uids, world.embeddings), sync_target=sync)
            fingerprints.append(model.fingerprint())
        samples.append(uids)
        syncs.append(sync)
        state = 0 if done else next_state
        if index % config["evaluate_every"] == 0 or index == config["adapt_steps"]:
            before_eval = model.forward_examples
            score, calls = evaluate(model, world, context, index)
            scores.append(score)
            eval_calls += calls
            evaluation_forward += model.forward_examples - before_eval
    return Tape(np.asarray(rows, dtype=np.float64), tuple(samples), tuple(syncs), tuple(fingerprints),
                scores, model.fingerprint(), eval_calls, work_counts(model, initial_work, evaluation_forward))


def consume(model, replay, tape, world, context, config, *, diagonal=False):
    model, replay = model.clone(), replay.clone()
    initial_work = (model.forward_examples, model.frozen_forward_examples)
    score, eval_calls = evaluate(model, world, context, 0)
    if diagonal and score != tape.scores[0]:
        raise AssertionError("initial diagonal evaluation diverged")
    evaluation_forward = model.forward_examples - initial_work[0]
    scores = [score]
    update_index = 0
    for index, (row, uids, sync) in enumerate(zip(tape.rows, tape.sampled_uids, tape.sync), 1):
        replay.append(row)
        if len(uids):
            model.update(*replay.batch(uids, world.embeddings), sync_target=sync)
            if diagonal and model.fingerprint() != tape.actor_fingerprints[update_index]:
                raise AssertionError(f"diagonal learner diverged at update {update_index}")
            update_index += 1
        if index % config["evaluate_every"] == 0 or index == len(tape.rows):
            before_eval = model.forward_examples
            score, calls = evaluate(model, world, context, index)
            scores.append(score)
            eval_calls += calls
            evaluation_forward += model.forward_examples - before_eval
            if diagonal and score != tape.scores[len(scores) - 1]:
                raise AssertionError("diagonal evaluation diverged")
    if diagonal and model.fingerprint() != tape.final_fingerprint:
        raise AssertionError("final diagonal state diverged")
    return {"scores": scores, "auc": auc(scores), "fingerprint": model.fingerprint(),
            "optimizer_updates": update_index, "evaluation_interactions": eval_calls,
            "diagonal_verified": diagonal, "work": work_counts(model, initial_work, evaluation_forward)}


def transformed(aged, intervention_seed, kind):
    model = aged.clone()
    if kind == "injection":
        model.inject(intervention_seed)
    elif kind == "head_reset":
        model.reset_head(intervention_seed)
    elif kind == "optimizer_reset":
        for value in model.m + model.v:
            value.fill(0.0)
        model.steps[:] = [0] * len(model.steps)
    elif kind != "identity":
        raise ValueError("unknown intervention")
    return model


def save_checkpoint(model, replay, output, name):
    arrays = {f"{group}_{index}": value for group in ("params", "target_params", "m", "v")
              for index, value in enumerate(getattr(model, group))}
    arrays.update(steps=np.asarray(model.steps, dtype=np.int64), replay_rows=replay.rows,
                  replay_uids=replay.uids, replay_state=np.asarray([replay.capacity, replay.next_uid, replay.size]))
    np.savez_compressed(output / name, **arrays)


def load_checkpoint(path, config, expected_fingerprint):
    """Restore an uninjected aged learner and replay; verify learning identity."""
    model = DQN(config["observation_dim"], config["hidden"], 0, config["learning_rate"], config["gamma"])
    with np.load(path, allow_pickle=False) as arrays:
        for group in ("params", "target_params", "m", "v"):
            for index, destination in enumerate(getattr(model, group)):
                source = arrays[f"{group}_{index}"]
                if source.shape != destination.shape or source.dtype != np.float64 or not np.isfinite(source).all():
                    raise ValueError("checkpoint parameters do not match configuration")
                np.copyto(destination, source)
        model.steps = [int(step) for step in arrays["steps"]]
        capacity, next_uid, size = map(int, arrays["replay_state"])
        if capacity != config["replay_capacity"] or not 0 <= size <= min(capacity, next_uid):
            raise ValueError("invalid replay checkpoint")
        replay = Replay(capacity, next_uid)
        np.copyto(replay.rows, arrays["replay_rows"])
        np.copyto(replay.uids, arrays["replay_uids"])
        replay.size = size
        expected_uids = np.arange(next_uid - size, next_uid)
        if not np.array_equal(replay.uids[expected_uids % capacity], expected_uids):
            raise ValueError("checkpoint replay UIDs are inconsistent")
    if model.fingerprint() != expected_fingerprint:
        raise ValueError("checkpoint learning fingerprint mismatch")
    return model, replay


def run_pair(structural_seed, init_seed, config, output):
    started = time.perf_counter()
    world = World(structural_seed, observation_dim=config["observation_dim"], gamma=config["gamma"])
    aged, replay, anchor, age_counts = age(world, init_seed, config)
    save_checkpoint(aged, replay, output, f"checkpoint-{structural_seed}-{init_seed}.npz")
    tabular, tabular_replay, tabular_anchor, tabular_age = age(
        world, init_seed, config, TabularQ(world.embeddings, gamma=world.gamma))
    if tabular_anchor != anchor:
        raise ValueError("aged comparators ended in different contexts; use complete aging blocks")
    intervention_seed = seed_for(init_seed, "intervention")
    injected = transformed(aged, intervention_seed, config["intervention"])
    if config["intervention"] in {"injection", "identity"}:
        if not np.array_equal(aged.q_values(world.embeddings), injected.q_values(world.embeddings)):
            raise AssertionError("primary intervention changed initial predictions")
    fresh = DQN(input_dim=config["observation_dim"], hidden=config["hidden"], seed=seed_for(init_seed, "fresh"),
                learning_rate=config["learning_rate"], gamma=config["gamma"])
    result = {"structural_seed": structural_seed, "learner_seed": init_seed,
              "age": age_counts, "tabular_age": tabular_age, "anchor_context": asdict(anchor),
              "aged_fingerprint": aged.fingerprint(), "switches": {}}
    for kind in config["switches"]:
        context = world.post_context(anchor, kind)
        rng_seed = seed_for(structural_seed, init_seed, kind, "post")
        tapes = {"A": generate(aged, replay, world, context, config, rng_seed, age_counts["optimizer_updates"]),
                 "I": generate(injected, replay, world, context, config, rng_seed, age_counts["optimizer_updates"])}
        cells = {}
        for source, tape in tapes.items():
            for label, learner in (("A", aged), ("I", injected), ("F", fresh)):
                cells[label + source] = consume(learner, replay, tape, world, context, config, diagonal=label == source)
        aa, ia, ai, ii = (cells[name]["auc"] for name in ("AA", "IA", "AI", "II"))
        learner_effect = ((ia - aa) + (ii - ai)) / 2
        data_effect = ((ai - aa) + (ii - ia)) / 2
        if not np.isclose(ii - aa, learner_effect + data_effect, atol=1e-12, rtol=0):
            raise AssertionError("factorial decomposition failed")
        controls = {}
        control_tapes = {}
        for label, learner, buffer, restart in (
            ("epsilon_restart", aged, replay, True),
            ("recent_replay", aged, replay.recent(config["recent_keep"]), False),
        ):
            tape = generate(learner, buffer, world, context, config, rng_seed, age_counts["optimizer_updates"], epsilon_restart=restart)
            control_tapes[label] = tape
            controls[label] = {"auc": auc(tape.scores), "scores": tape.scores, "tape_digest": tape.digest(),
                               "evaluation_interactions": tape.evaluation_steps,
                               "optimizer_updates": len(tape.actor_fingerprints), "work": tape.work}
        # Independently aged comparator: same budget and observation information,
        # favorable tabular representation, own closed-loop experience.
        tabular_context = world.post_context(tabular_anchor, kind)
        tabular_tape = generate(tabular, tabular_replay.recent(config["recent_keep"]), world,
                                tabular_context, config, rng_seed, tabular_age["optimizer_updates"])
        control_tapes["tabular_recent_replay"] = tabular_tape
        controls["tabular_recent_replay"] = {
            "auc": auc(tabular_tape.scores), "scores": tabular_tape.scores,
            "tape_digest": tabular_tape.digest(), "evaluation_interactions": tabular_tape.evaluation_steps,
            "q_minibatch_updates": len(tabular_tape.actor_fingerprints), "work": tabular_tape.work,
            "anchor_context": asdict(tabular_anchor)}
        empty = Replay(config["replay_capacity"])
        rich = generate(aged, empty, world, context, config, rng_seed, age_counts["optimizer_updates"], uniform=True)
        calibration = {}
        for label, learner in (("A", aged), ("I", injected),
                               ("O", transformed(aged, intervention_seed, "optimizer_reset"))):
            calibration[label] = consume(learner, empty, rich, world, context, config, diagonal=label == "A")
        state_coverage = {source: np.bincount(tape.rows[:, 0].astype(int), minlength=10).tolist() for source, tape in tapes.items()}
        result["switches"][kind] = {
            "cells": cells, "total_effect": ii - aa, "learner_state_effect": learner_effect,
            "generated_data_effect": data_effect, "interaction": ii - ia - ai + aa,
            "controls": controls, "clean_tape_calibration": calibration,
            "state_visit_counts": state_coverage, "tape_digests": {name: tape.digest() for name, tape in tapes.items()},
            "rich_tape_digest": rich.digest(),
            "training_simulator_interactions": 6 * config["adapt_steps"],
            "actor_work": {name: tape.work for name, tape in {**tapes, "rich": rich}.items()},
            "reference_transition_queries": 36 * (1 + (tabular_context != context)),
            "actor_evaluation_interactions": sum(tape.evaluation_steps for tape in tapes.values()) + rich.evaluation_steps,
            "actor_optimizer_updates": sum(len(tape.actor_fingerprints) for tape in tapes.values()) + len(rich.actor_fingerprints),
        }
        arrays = {}
        for name, tape in {**tapes, **control_tapes, "rich": rich}.items():
            arrays[name + "_transitions"] = tape.rows
            # A separate offset array retains updates at their chronological arrival time.
            arrays[name + "_update_steps"] = np.asarray([i for i, u in enumerate(tape.sampled_uids, 1) if len(u)], dtype=np.int64)
            arrays[name + "_sample_uids"] = np.asarray([u for u in tape.sampled_uids if len(u)], dtype=np.int64)
            arrays[name + "_target_sync_steps"] = np.asarray([i for i, sync in enumerate(tape.sync, 1) if sync], dtype=np.int64)
            arrays[name + "_actor_fingerprints"] = np.asarray(tape.actor_fingerprints, dtype="U64")
        np.savez_compressed(output / f"tapes-{structural_seed}-{init_seed}-{kind}.npz", **arrays)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result
