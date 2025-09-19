import sys
import pathlib
import numpy as np
import os
import json
from datetime import datetime

# Ensure package root (JaxMARL) is on sys.path when running this file directly
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from llm_command_helper import (
    llm_decide_actions,
    _lazy_init,
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

_lazy_init()

from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX


# Import JAX only after vLLM is initialized (mirrors MAPPO intent)
import jax
import jax.numpy as jnp

# ------------- Simple runtime settings -------------
# Number of episodes to run for evaluation
EPISODES = 10
# Query the LLM every MACRO_STEPS env steps (1 = every step)
MACRO_STEPS = 1
# Per-step console logging
VERBOSE_STEPS = False
# Write final summary file in addition to per-episode JSONL
WRITE_SUMMARY = True
# Micro-control: LLM outputs per-ally discrete actions directly

# Output files (JSONL per-episode + summary JSON)
OUTPUT_DIR = os.path.join(str(pathlib.Path(__file__).resolve().parents[2]), "JaxMARL", "outputs")
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
EVAL_PREFIX = f"smax_llm_raw_eval_{_ts}"
JSONL_PATH = os.path.join(OUTPUT_DIR, f"{EVAL_PREFIX}.jsonl")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, f"{EVAL_PREFIX}_summary.json")

def build_env():
    scenario = map_name_to_scenario("2s3z")
    env = HeuristicEnemySMAX(
        scenario=scenario,
        see_enemy_actions=True,
        walls_cause_death=True,
        attack_mode="closest",
    )
    return env


def _movement_vectors(num_movement_actions: int):
    vecs = []
    for action in range(max(0, num_movement_actions - 1)):
        v = jnp.array([
            ((-1) ** (action // 2)) * (1.0 / jnp.sqrt(2.0)),
            ((-1) ** (action // 2 + action % 2)) * (1.0 / jnp.sqrt(2.0)),
        ])
        rotation = jnp.array(
            [
                [1.0 / jnp.sqrt(2.0), -1.0 / jnp.sqrt(2.0)],
                [1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)],
            ]
        )
        v = rotation @ v
        vecs.append(np.array(v, dtype=np.float32))
    return np.stack(vecs, axis=0) if vecs else np.zeros((0, 2), dtype=np.float32)


def _nearest_movement_action(towards_vec: np.ndarray, move_vecs: np.ndarray) -> int:
    if move_vecs.shape[0] == 0:
        return 0
    tv = towards_vec
    n = np.linalg.norm(tv) + 1e-8
    tv = tv / n
    dots = move_vecs @ tv
    return int(np.argmax(dots))


def postprocess_llm_actions(env: HeuristicEnemySMAX, state, actions: dict) -> dict:
    s = state.state
    num_allies = env.num_allies
    num_enemies = env.num_enemies
    nmove = int(env._env.num_movement_actions)
    stop_idx = nmove - 1
    move_vecs = _movement_vectors(nmove)

    for i in range(num_allies):
        key = f"ally_{i}"
        a = int(actions.get(key, stop_idx))
        if a < nmove:
            continue
        enemy_local = a - nmove
        if not (0 <= enemy_local < num_enemies):
            actions[key] = np.int32(stop_idx)
            continue
        ally_idx = i
        enemy_idx = num_allies + enemy_local
        if (not bool(s.unit_alive[ally_idx])) or (not bool(s.unit_alive[enemy_idx])):
            actions[key] = np.int32(stop_idx)
            continue
        ally_type = int(s.unit_types[ally_idx])
        attack_range = float(env._env.unit_type_attack_ranges[ally_type])
        dv = np.array(s.unit_positions[enemy_idx] - s.unit_positions[ally_idx])
        dist = float(np.linalg.norm(dv))
        if dist < attack_range:
            actions[key] = np.int32(a)
        else:
            move_a = _nearest_movement_action(dv, move_vecs)
            actions[key] = np.int32(move_a)
    return actions

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = build_env()

    key = jax.random.PRNGKey(0)

    results = []
    wins = 0
    losses = 0
    draws = 0

    with open(JSONL_PATH, "w") as f:
        for ep in range(EPISODES):
            # Reset per episode
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key)
            episode_reward = 0.0
            step_i = 0

            # Action management (macro-step refresh)
            current_actions = None

            while True:
                # Refresh LLM actions every MACRO_STEPS
                if (step_i % MACRO_STEPS == 0) or (current_actions is None):
                    world_state = env.get_world_state(state)
                    world_state_np = np.asarray(world_state).reshape(-1)
                    # Build per-ally available-action masks for fairness
                    avail = env.get_avail_actions(state)
                    avail_masks = [list(map(int, avail[f"ally_{i}"].tolist())) for i in range(env.num_allies)]
                    current_actions = llm_decide_actions(
                        world_state_np,
                        num_allies=env.num_allies,
                        num_enemies=env.num_enemies,
                        num_movement_actions=env._env.num_movement_actions,
                        avail_masks=avail_masks,
                    )

                # Step env with micro actions (apply out-of-range→move fallback)
                actions = postprocess_llm_actions(env, state, dict(current_actions))
                key, step_key = jax.random.split(key)
                obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)

                step_rew = float(sum(float(rewards[a]) for a in rewards))
                episode_reward += step_rew
                step_i += 1

                if VERBOSE_STEPS:
                    print(
                        f"ep={ep+1}/{EPISODES} step={step_i} micro_actions={list(actions.values())} step_reward={step_rew:.3f} total={episode_reward:.3f}"
                    )

                if bool(dones.get("__all__", False)) or step_i >= 200:
                    # Determine outcome
                    s = state.state
                    num_allies = env.num_allies
                    allies_alive = bool(jnp.any(s.unit_alive[:num_allies]))
                    enemies_alive = bool(jnp.any(s.unit_alive[num_allies:]))
                    if allies_alive and not enemies_alive:
                        outcome = "win"
                        wins += 1
                    elif (not allies_alive) and enemies_alive:
                        outcome = "loss"
                        losses += 1
                    else:
                        outcome = "draw"
                        draws += 1

                    # Write episode record
                    rec = {
                        "episode": ep + 1,
                        "steps": step_i,
                        "return": episode_reward,
                        "outcome": outcome,
                        "macro_steps": MACRO_STEPS,
                        "timestamp": datetime.now().isoformat(),
                    }
                    f.write(json.dumps(rec) + "\n")
                    if not VERBOSE_STEPS:
                        print(
                            f"episode={ep+1} outcome={outcome} steps={step_i} return={episode_reward:.3f}"
                        )
                    results.append(rec)
                    break

    total = len(results)
    if WRITE_SUMMARY:
        win_rate = wins / total if total > 0 else 0.0
        summary = {
            "episodes": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "macro_steps": MACRO_STEPS,
            "jsonl": JSONL_PATH,
        }
        with open(SUMMARY_PATH, "w") as sf:
            json.dump(summary, sf, indent=2)
        print(
            f"Evaluation done: episodes={total} win_rate={win_rate:.3f} (W/L/D={wins}/{losses}/{draws}).\n"
            f"Saved: {JSONL_PATH}, {SUMMARY_PATH}"
        )
    else:
        print(f"Evaluation done: episodes={total}. Per-episode records saved: {JSONL_PATH}")


if __name__ == "__main__":
    # Keep argv handling simple / consistent with other scripts
    sys.exit(main())


