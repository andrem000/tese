"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import functools
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import os
import json
import pickle
from datetime import datetime
import pathlib

# --- LLM Command Helper ---
from llm_command_helper import (
    llm_decide_cmd,
    llm_decide_cmd_batch,
    CMD_DIM,
    _lazy_init,
    llm_record_feedback,
)
from jax.experimental import io_callback

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"   # cap JAX to ~0.6 GiB on 6 GiB GPU

# Initialize vLLM before JAX to avoid CUDA driver conflicts
_lazy_init()

from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

class SMAXWorldStateWrapper(JaxMARLWrapper):
    """
    Provides a `"world_state"` observation for the centralised critic.
    world state observation of dimension: (num_agents, world_state_size)    
    """
    
    def __init__(self,
                 env: HeuristicEnemySMAX,
                 obs_with_agent_id=True,):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id
        
        if not self.obs_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id
            
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state_fn(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        #return all_obs
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state
        
    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        #all_obs = jnp.array([obs[agent] for agent in self._env.agents])
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)
        
    def world_state_size(self):
   
        return self._world_state_size 

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        #print('ins', ins)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Command-gated policy head: the LLM selects which head is active via one-hot cmd
        # The last CMD_DIM features of obs are the appended one-hot command
        cmd_one_hot = obs[..., -CMD_DIM:]

        x_head = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        x_head = nn.relu(x_head)
        # Produce per-command logits and select via the one-hot command
        logits_all = nn.Dense(
            CMD_DIM * int(self.action_dim), kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x_head)
        logits_all = logits_all.reshape(logits_all.shape[:-1] + (CMD_DIM, int(self.action_dim)))
        action_logits = (logits_all * cmd_one_hot[..., None]).sum(axis=-2)

        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    cmd: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    #print('batchify', x.shape)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    # Macro-step frequency for LLM command refresh
    MACRO_STEPS = int(config.get("MACRO_STEPS", 4))
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    # Respect a pre-set NUM_UPDATES (e.g., when resuming), otherwise compute default
    default_total_updates = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES"] = int(config.get("NUM_UPDATES", default_total_updates))
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)
    # Expose movement action count for shaping/penalties
    config["NUM_MOVEMENT_ACTIONS"] = int(env._env.num_movement_actions)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # LLM batching flag and world-state compression parameters
        use_batch_llm = config.get("LLM", {}).get("BATCH_LLM", True)
        # Checkpoint/resume controls
        CKPT_EVERY_UPDATES = int(config.get("CKPT_EVERY_UPDATES", 10))
        START_UPDATE = int(config.get("START_UPDATE", 0))

        # Macro guidance flags (disabled by default)
        MACRO_DEV_COEF = float(config.get("MACRO_DEVIATION_COEF", 0.0))
        FOCUS_SHAPING_SCALE = float(config.get("FOCUS_SHAPING_SCALE", 0.0))
        LLM_FEEDBACK = bool(config.get("LLM_FEEDBACK", False))
        NUM_MOVES = int(config.get("NUM_MOVEMENT_ACTIONS", env._env.num_movement_actions))
        # Map command id → focused enemy index, else -1
        focus_base_cmd = 2
        focus_map_list = [-1 for _ in range(CMD_DIM)]
        for j in range(min(env.num_enemies, max(0, CMD_DIM - focus_base_cmd))):
            focus_map_list[focus_base_cmd + j] = j
        FOCUS_MAP = jnp.asarray(focus_map_list, dtype=jnp.int32)

        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        obs_dim_aug = env.observation_space(env.agents[0]).shape[0] + CMD_DIM
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], obs_dim_aug)),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size() + CMD_DIM,)),  
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
        
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # If resuming, replace params and optimizer state from config-injected trees
        if bool(config.get("RESUME", False)):
            init_actor_params = config.get("INIT_ACTOR_PARAMS", None)
            init_critic_params = config.get("INIT_CRITIC_PARAMS", None)
            if init_actor_params is not None:
                actor_train_state = actor_train_state.replace(params=init_actor_params)
            if init_critic_params is not None:
                critic_train_state = critic_train_state.replace(params=init_critic_params)
            # Optional: restore optimizer state if provided
            init_actor_opt = config.get("INIT_ACTOR_OPT_STATE", None)
            init_critic_opt = config.get("INIT_CRITIC_OPT_STATE", None)
            if init_actor_opt is not None:
                actor_train_state = actor_train_state.replace(opt_state=init_actor_opt)
            if init_critic_opt is not None:
                critic_train_state = critic_train_state.replace(opt_state=init_critic_opt)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            WORLD_STATE_FLAT_SIZE = env.world_state_size() * env.num_agents  # constant

            def _env_step(runner_state, step_i):
                train_states, env_state, last_obs, last_done, hstates, rng, prev_cmd_ids = runner_state

                # SELECT ACTION (include LLM command)
                rng, _rng = jax.random.split(rng)

                # --- Compute avail_actions and base obs first (needed for broadcasting) --
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch_base = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                # ---------------------------------------------------------------------
                # Query LLM every macro step via io_callback and persist otherwise
                step_idx = update_steps * config["NUM_STEPS"] + step_i
                needs_new_cmd = (step_idx % MACRO_STEPS) == 0

                if use_batch_llm:
                    # --- Batched variant: one prompt per env ----------------
                    def _get_cmds(ws_batch):
                        return io_callback(
                            llm_decide_cmd_batch,
                            jax.ShapeDtypeStruct((config["NUM_ENVS"],), jnp.int32),
                            ws_batch,
                        )

                    ws_batch = last_obs["world_state"].reshape((config["NUM_ENVS"], WORLD_STATE_FLAT_SIZE))

                    def _get(_in):
                        prev, ws = _in
                        return _get_cmds(ws)
                    def _keep(_in):
                        prev, ws = _in
                        return prev
                    cmd_ids = jax.lax.cond(
                        needs_new_cmd,
                        _get,
                        _keep,
                        operand=(prev_cmd_ids, ws_batch),
                    )

                    # one-hot encode and repeat for each agent
                    cmd_one_hot = jax.nn.one_hot(cmd_ids, CMD_DIM)  # (NUM_ENVS, CMD_DIM)
                    cmd_mat = jnp.repeat(cmd_one_hot, env.num_agents, axis=0)  # (NUM_ACTORS, CMD_DIM)
                    cmd_switch_env = (cmd_ids != prev_cmd_ids).astype(jnp.int32) * needs_new_cmd.astype(jnp.int32)
                    llm_calls_env = jnp.full((config["NUM_ENVS"],), needs_new_cmd.astype(jnp.int32), dtype=jnp.int32)
                else:
                    # --- Single-prompt (legacy) variant ---------------------
                    def _get_cmd(flat_ws):
                        return io_callback(
                            llm_decide_cmd,
                            jax.ShapeDtypeStruct((), jnp.int32),
                            flat_ws,
                        )

                    flat_world_state = last_obs["world_state"].reshape(-1)
                    prev_cmd_id = prev_cmd_ids
                    def _get(ws_and_prev):
                        ws, prev = ws_and_prev
                        return _get_cmd(ws)
                    def _keep(ws_and_prev):
                        ws, prev = ws_and_prev
                        return prev
                    cmd_id = jax.lax.cond(
                        needs_new_cmd,
                        _get,
                        _keep,
                        operand=(flat_world_state, prev_cmd_id),
                    )
                    cmd_one_hot = jax.nn.one_hot(cmd_id, CMD_DIM)
                    cmd_mat = jnp.repeat(cmd_one_hot[None, :], obs_batch_base.shape[0], axis=0)
                    cmd_switch_env = jnp.full((config["NUM_ENVS"],), ((cmd_id != prev_cmd_id) & needs_new_cmd).astype(jnp.int32), dtype=jnp.int32)
                    llm_calls_env = jnp.full((config["NUM_ENVS"],), needs_new_cmd.astype(jnp.int32), dtype=jnp.int32)
                    
                    # Optional feedback to LLM when focus attack is infeasible
                    if LLM_FEEDBACK:
                        is_focus_cmd = (cmd_id >= jnp.int32(focus_base_cmd)) & (cmd_id < jnp.int32(focus_base_cmd + env.num_enemies))
                        focus_k = cmd_id - jnp.int32(focus_base_cmd)
                        pref_attack_idx = jnp.where(is_focus_cmd, NUM_MOVES + focus_k, jnp.int32(-1))
                        safe_idx = jnp.clip(pref_attack_idx, 0, avail_actions.shape[1] - 1)
                        pref_avail_col = avail_actions[:, safe_idx]
                        feasible_any = (jnp.sum(pref_avail_col) > 0) & is_focus_cmd
                        def _send_feedback_infeasible(k):
                            def _cb(k_host):
                                try:
                                    llm_record_feedback("Command focus_enemy_" + str(int(k_host)) + " infeasible (out of range/dead). Agents will reposition/kite.")
                                except Exception:
                                    pass
                                return ()
                            return jax.experimental.io_callback(_cb, None, k)
                        def _no_send(k):
                            return ()
                        _ = jax.lax.cond(needs_new_cmd & (~feasible_any), _send_feedback_infeasible, _no_send, operand=focus_k)

                        # Feedback on changed target between consecutive macros
                        prev_is_focus = (prev_cmd_id >= jnp.int32(focus_base_cmd)) & (prev_cmd_id < jnp.int32(focus_base_cmd + env.num_enemies))
                        changed_focus = needs_new_cmd & prev_is_focus & is_focus_cmd & (cmd_id != prev_cmd_id)
                        prev_k = prev_cmd_id - jnp.int32(focus_base_cmd)
                        def _send_feedback_changed(args):
                            k_prev, k_new = args
                            def _cb(p_host, n_host):
                                try:
                                    llm_record_feedback("Switching focus from enemy_" + str(int(p_host)) + " to enemy_" + str(int(n_host)) + ". Agents adapting.")
                                except Exception:
                                    pass
                                return ()
                            return jax.experimental.io_callback(_cb, None, k_prev, k_new)
                        def _no_send_changed(args):
                            return ()
                        _ = jax.lax.cond(changed_focus, _send_feedback_changed, _no_send_changed, operand=(prev_k, focus_k))
 
                obs_batch = jnp.concatenate([obs_batch_base, cmd_mat], axis=1)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                #print('env step ac in', ac_in)
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                # Keep a consistent shape: (NUM_ENVS,) for each agent. Avoid squeezing
                # so that vmap over envs still sees a rank-1 array even when NUM_ENVS=1.
                env_act = {k: v.reshape((config["NUM_ENVS"],)) for k, v in env_act.items()}

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0,1)  
                world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                # critic conditions on command
                world_state_aug = jnp.concatenate([world_state, cmd_mat], axis=1)

                cr_in = (
                    world_state_aug[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                # Optional: shaping reward to encourage focus fire when preferred attack is available
                cmd_arg = jnp.argmax(cmd_mat, axis=1)
                focus_k = jnp.take(FOCUS_MAP, cmd_arg)
                is_focus = focus_k >= jnp.int32(0)
                pref_attack_idx = NUM_MOVES + focus_k
                pref_attack_idx = jnp.where(is_focus, pref_attack_idx, -jnp.ones_like(pref_attack_idx))
                gather_idx = jnp.clip(pref_attack_idx[:, None], 0, avail_actions.shape[1] - 1)
                pref_avail = jnp.where(is_focus, jnp.take_along_axis(avail_actions, gather_idx, axis=1).squeeze(-1), 0)
                sampled_action = action.squeeze()
                is_match = (sampled_action == pref_attack_idx) & (pref_avail == 1)
                is_deviate = (sampled_action != pref_attack_idx) & (pref_avail == 1) & is_focus
                shaping = FOCUS_SHAPING_SCALE * (is_match.astype(jnp.float32) - is_deviate.astype(jnp.float32))

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                info["cmd_switch"] = jnp.repeat(cmd_switch_env, env.num_agents, axis=0)
                info["llm_calls"] = jnp.repeat(llm_calls_env, env.num_agents, axis=0)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                # Apply shaping if enabled
                rew_batch = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                if FOCUS_SHAPING_SCALE != 0.0:
                    rew_batch = rew_batch + shaping
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    rew_batch,
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    cmd_mat,
                    info,
                    avail_actions,
                )
                next_cmd_ids = cmd_ids if use_batch_llm else cmd_id
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng, next_cmd_ids)
                return runner_state, transition

            initial_hstates = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"]), config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng, last_cmd_ids = runner_state
            
            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            if use_batch_llm:
                last_cmd_one_hot = jax.nn.one_hot(last_cmd_ids, CMD_DIM)
                last_cmd_mat = jnp.repeat(last_cmd_one_hot, env.num_agents, axis=0)
            else:
                last_cmd_one_hot = jax.nn.one_hot(last_cmd_ids, CMD_DIM)
                last_cmd_mat = jnp.repeat(last_cmd_one_hot[None, :], config["NUM_ACTORS"], axis=0)
            last_world_state_aug = jnp.concatenate([last_world_state, last_cmd_mat], axis=1)

            cr_in = (
                last_world_state_aug[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy

                        # Macro deviation penalty (encourage following focus attack when available)
                        if MACRO_DEV_COEF != 0.0:
                            # Prepare tensors; allow any leading dims and gather along the last axis
                            cmd_mb = traj_batch.cmd
                            avail_mb = traj_batch.avail_actions
                            act_mb = traj_batch.action

                            # Ensure last axis semantics: cmd one-hot over commands; avail over actions
                            action_dim = avail_mb.shape[-1]

                            # Compute preferred attack index with same leading shape as cmd_mb[..., 0]
                            cmd_arg = jnp.argmax(cmd_mb, axis=-1)
                            focus_k = jnp.take(FOCUS_MAP, cmd_arg)
                            is_focus = focus_k >= jnp.int32(0)
                            pref_attack_idx = NUM_MOVES + focus_k
                            pref_attack_idx = jnp.where(is_focus, pref_attack_idx, -jnp.ones_like(pref_attack_idx))

                            # Flatten leading dims dynamically; avoid Python int conversions
                            avail_flat = jnp.reshape(avail_mb, (-1, action_dim))
                            act_flat = jnp.reshape(act_mb, (-1,))
                            pref_flat = jnp.reshape(pref_attack_idx, (-1,))
                            focus_flat = jnp.reshape(is_focus, (-1,))

                            # Safe gather index: for non-focus, set to zero to avoid negative index
                            safe_idx = jnp.where(
                                focus_flat,
                                jnp.clip(pref_flat, 0, action_dim - 1),
                                jnp.zeros_like(pref_flat),
                            )
                            pref_avail_flat = jnp.take_along_axis(avail_flat, safe_idx[:, None], axis=1).squeeze(-1)

                            deviated_flat = (act_flat != pref_flat) & (pref_avail_flat == 1) & focus_flat
                            actor_loss = actor_loss + MACRO_DEV_COEF * jnp.mean(deviated_flat.astype(jnp.float32))
                        
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK with command conditioning
                        world_state_aug = jnp.concatenate([traj_batch.world_state, traj_batch.cmd], axis=-1)
                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (world_state_aug,  traj_batch.done)) 
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree_util.tree_map(lambda x: jnp.reshape(
                    x, (1, config["NUM_ACTORS"], -1)
                ), init_hstates)
                
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                #train_states = (actor_train_state, critic_train_state)
                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree_util.tree_map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            
            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            rng = update_state[-1]

            def callback_log(metric, global_update_host):
                # Extract key metrics for logging
                returns = metric["returned_episode_returns"][:, :, 0][
                    metric["returned_episode"][:, :, 0]
                ].mean()
                win_rate = metric["returned_won_episode"][:, :, 0][
                    metric["returned_episode"][:, :, 0]
                ].mean()
                # LLM metrics (averaged across time/envs)
                if "cmd_switch" in metric:
                    cmd_switch_rate = metric["cmd_switch"][:, :, 0].mean()
                else:
                    cmd_switch_rate = 0.0
                if "llm_calls" in metric:
                    llm_calls_rate = metric["llm_calls"][:, :, 0].mean()
                else:
                    llm_calls_rate = 0.0
                # Use global update step including any START_UPDATE offset
                global_update = int(global_update_host)
                env_step = global_update * config["NUM_ENVS"] * config["NUM_STEPS"]
                
                # Create log entry
                log_entry = {
                    "returns": float(returns),
                    "win_rate": float(win_rate),
                    "cmd_switch_rate": float(cmd_switch_rate),
                    "llm_calls_rate": float(llm_calls_rate),
                    "env_step": int(env_step),
                    "update_steps": int(global_update),
                    "timestamp": datetime.now().isoformat(),
                    **{k: float(v) for k, v in metric["loss"].items()}
                }
                
                # Save metrics to file (no model saving or printing here)
                metrics_file = os.path.join(config["OUTPUT_DIR"], "training_metrics.jsonl")
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            def callback_ckpt(actor_params_host, critic_params_host, actor_opt_state_host, critic_opt_state_host, global_update_host):
                global_update = int(global_update_host)
                env_step = global_update * config["NUM_ENVS"] * config["NUM_STEPS"]
                # Convert pytrees to numpy for safe pickling
                to_numpy = lambda t: jax.tree_util.tree_map(lambda x: np.asarray(x), t)
                actor_params_np = to_numpy(actor_params_host)
                critic_params_np = to_numpy(critic_params_host)
                actor_opt_np = to_numpy(actor_opt_state_host)
                critic_opt_np = to_numpy(critic_opt_state_host)
                # Drop heavy INIT_* entries from config before saving
                config_for_ckpt = {k: v for k, v in config.items() if not str(k).startswith("INIT_")}
                ckpt = {
                    "actor_params": actor_params_np,
                    "critic_params": critic_params_np,
                    "actor_opt_state": actor_opt_np,
                    "critic_opt_state": critic_opt_np,
                    "config": config_for_ckpt,
                    "env_step": int(env_step),
                    "update_steps": int(global_update),
                    "timestamp": datetime.now().isoformat(),
                }
                ckpt_name = f"ckpt_{config['MAP_NAME']}_step_{int(env_step)}.pkl"
                ckpt_path = os.path.join(config["OUTPUT_DIR"], ckpt_name)
                with open(ckpt_path, "wb") as f:
                    pickle.dump(ckpt, f)
            
            metric["update_steps"] = update_steps
            # Compute global update index and log metrics
            global_update = START_UPDATE + update_steps
            jax.experimental.io_callback(
                callback_log,
                None,
                metric,
                global_update,
            )
            # Periodically checkpoint: only transfer params when needed
            if CKPT_EVERY_UPDATES > 0:
                need_ckpt = jnp.equal(jnp.mod(global_update, CKPT_EVERY_UPDATES), 0)
                def _do_ckpt(args):
                    ts = args
                    jax.experimental.io_callback(
                        callback_ckpt,
                        None,
                        ts[0].params,
                        ts[1].params,
                        ts[0].opt_state,
                        ts[1].opt_state,
                        global_update,
                    )
                    return ()
                def _no_ckpt(args):
                    return ()
                _ = jax.lax.cond(need_ckpt, _do_ckpt, _no_ckpt, operand=train_states)
            update_steps = update_steps + 1
            # preserve last_cmd_ids in outer runner_state carry
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng, last_cmd_ids)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        init_cmd_ids = jnp.zeros((config["NUM_ENVS"]), dtype=jnp.int32) if use_batch_llm else jnp.int32(0)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
            init_cmd_ids,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

@hydra.main(version_base=None, config_path="config", config_name="mappo_homogenous_rnn_smax_llm")
def main(config):

    config = OmegaConf.to_container(config)

    # Ensure output directory exists (absolute path provided in YAML)
    import pathlib, os
    pathlib.Path(config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # Optionally resume from checkpoint
    resume_path = config.get("RESUME_FROM", None)
    start_update = 0
    if resume_path and os.path.isfile(resume_path):
        try:
            with open(resume_path, "rb") as f:
                ckpt = pickle.load(f)
            # Backward compatibility: handle both full and param-only checkpoints
            resume_only_actor = bool(config.get("RESUME_ONLY_ACTOR", False))
            config["INIT_ACTOR_PARAMS"] = ckpt.get("actor_params", None)
            config["INIT_ACTOR_OPT_STATE"] = ckpt.get("actor_opt_state", None)
            if not resume_only_actor:
                config["INIT_CRITIC_PARAMS"] = ckpt.get("critic_params", None)
                config["INIT_CRITIC_OPT_STATE"] = ckpt.get("critic_opt_state", None)
            start_update = int(ckpt.get("update_steps", 0))
            config["RESUME"] = True
            config["START_UPDATE"] = start_update
            # Reduce planned updates so total remains consistent
            total_updates = int(config["NUM_UPDATES"]) if "NUM_UPDATES" in config else (
                config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
            )
            remaining = max(0, total_updates - start_update)
            config["NUM_UPDATES"] = remaining
            print(f"Resuming from {resume_path}: start_update={start_update}, remaining_updates={remaining}")
        except Exception as e:
            print(f"Failed to load checkpoint from {resume_path}: {e}. Starting fresh.")
            config.pop("RESUME", None)
            for k in ["INIT_ACTOR_PARAMS", "INIT_CRITIC_PARAMS", "INIT_ACTOR_OPT_STATE", "INIT_CRITIC_OPT_STATE", "START_UPDATE"]:
                if k in config:
                    config.pop(k)

    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config)) 
        out = train_jit(rng)

        # Post-process: Save model weights and print progress outside JIT
        runner_state = out["runner_state"]
        
        # runner_state: (train_states_tuple, final_update_steps)
        # runner_state[0] is a tuple of arrays, where each array contains TrainState objects for each step
        # runner_state[1] is the final update_steps (scalar)
        train_states_arrays = runner_state[0]  # tuple of (actor_array, critic_array)
        final_update_steps = runner_state[1]  # final update_steps (scalar)
        
        # Get the final actor and critic TrainStates from train_states_arrays[0]
        final_actor_train_state = train_states_arrays[0][0]
        final_critic_train_state = train_states_arrays[0][1]
        
        actor_params = final_actor_train_state.params
        critic_params = final_critic_train_state.params
        
        # Include START_UPDATE offset if any when naming final checkpoint
        start_update = int(config.get("START_UPDATE", 0))
        env_step = int((start_update + final_update_steps) * config["NUM_ENVS"] * config["NUM_STEPS"])
        model_filename = f"model_{config['MAP_NAME']}_step_{env_step}.pkl"
        model_filepath = os.path.join(config["OUTPUT_DIR"], model_filename)
        
        # Numpy-convert and prune INIT_* from config in final save as well
        actor_params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), actor_params)
        critic_params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), critic_params)
        config_for_save = {k: v for k, v in config.items() if not str(k).startswith("INIT_")}
        with open(model_filepath, "wb") as f:
            pickle.dump({
                "actor_params": actor_params_np,
                "critic_params": critic_params_np,
                "config": config_for_save,
                "env_step": env_step,
                "update_steps": int(start_update + final_update_steps)
            }, f)
        print(f"Training finished. Final model saved at step {env_step}.")

if __name__=="__main__":
    main()