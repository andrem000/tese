import json
import re
import os
from typing import List

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Global singleton model / tokenizer (initialised lazily on first call)
# ---------------------------------------------------------------------------
_TOKENIZER = None  # type: AutoTokenizer | None
_LLM = None       # type: LLM | None

# Sensible defaults; can be overridden via Hydra (see below)
STATE_SIZE = 1800
K_CONTEXT = 2

# Attempt to load overrides directly from the fully-resolved Hydra config
# (Hydra writes the absolute path to the final YAML in $HYDRA_FULL_CONFIG).
try:
    from omegaconf import OmegaConf  # slowish but only runs at import

    _cfg_path = os.getenv("HYDRA_FULL_CONFIG")
    if _cfg_path and os.path.isfile(_cfg_path):
        _full_cfg = OmegaConf.load(_cfg_path)
        _llm_cfg = _full_cfg.get("LLM") if _full_cfg is not None else None
        if _llm_cfg:
            STATE_SIZE = int(_llm_cfg.get("STATE_SIZE", STATE_SIZE))
            K_CONTEXT = int(_llm_cfg.get("K_CONTEXT", K_CONTEXT))
except Exception as _e:  # pragma: no cover – never crash on config issues
    print(f"[llm_command_helper] Could not read Hydra config → {_e}. Using defaults.")


# We rely on the fact that the assistant reply is at most 3 tokens and the
# system prompt is short.  We keep dropping the oldest user+assistant pair
# whenever adding the next pair would exceed the decoder’s `max_model_len`.
_CHAT_HISTORY: List[dict] = []  # rolling list length = 2*K_CONTEXT

# Hard-coded command list for the SMAX project (extend as needed)
COMMAND_LIST = [
    "0 = AttackClosest",
    "1 = AttackLowestHP",
    "2 = FocusEnemy0",
    "3 = FocusEnemy1",
    "4 = FocusEnemy2",
    "5 = FocusEnemy3",
    "6 = FocusEnemy4",
    "7 = Retreat",
    "8 = Regroup",
]
CMD_DIM = len(COMMAND_LIST)

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are the high-level commander for SMAX agents.\n"
        "Respond **only** with an integer (0-{max_id}) that selects a command.\n".format(max_id=CMD_DIM - 1)
        + "Valid commands:\n" + "\n".join(COMMAND_LIST)
    ),
}

# ---------------------------------------------------------------------------
# Helper to shrink the world-state representation passed to the LLM
# ---------------------------------------------------------------------------

def _compress_world_state(world_state: np.ndarray, *, quantize: bool = False) -> str:
    """Convert *world_state* to a compact space-separated string.

    Parameters
    ----------
    world_state : np.ndarray
        Vector to compress (any shape is flattened).
    quantize : bool, optional
        If ``True`` multiply by 10, round and cast to int so each number is an
        integer (e.g. 12.3 → 123). This removes the decimal point and saves
        ~30-40 % tokens. Default ``False`` keeps one decimal place.
    """

    if quantize:
        ws = np.round(world_state.astype(np.float32) * 10).astype(np.int32).flatten()
        str_vals = (str(int(v)) for v in ws)
    else:
        ws = np.round(world_state.astype(np.float32), 1).flatten()
        str_vals = (f"{v:.1f}" for v in ws)

    return " ".join(str_vals)


def _lazy_init():
    """Load tokenizer & model the first time we’re called."""
    global _TOKENIZER, _LLM
    if _TOKENIZER is None:
        # Use a single source for both tokenizer and LLM model name
        model_name = os.getenv("VLLM_MODEL", "Qwen/Qwen2-1.5B-Instruct")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _LLM = LLM(
            model=model_name,
            max_model_len=4000,
            gpu_memory_utilization=0.95,
            download_dir="./"
        )


def llm_decide_cmd(world_state: np.ndarray) -> np.int32:
    """Return an int cmd_id selected by the language model.

    Parameters
    ----------
    world_state : np.ndarray
        Flattened world_state vector from SMAXWorldStateWrapper.

    Returns
    -------
    np.int32
        Integer in [0, CMD_DIM-1]. Falls back to 0 on any error.
    """
    _lazy_init()
    #dump_world_state(world_state, "outputs/world_state_raw.json")

    # Build user JSON with 1-decimal, space-separated flat string
    world_state_str = _compress_world_state(world_state, quantize=True)
    user_msg = {
        "role": "user",
        "content": json.dumps({"world_state": world_state_str}),
    }

    messages = [SYSTEM_MESSAGE] + _CHAT_HISTORY + [user_msg]
    prompt = _TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        out = _LLM.generate(
            [prompt],
            sampling_params=SamplingParams(max_tokens=3, temperature=0.0, top_p=1.0),
        )
        reply = out[0].outputs[0].text.strip()
    except Exception as e:
        print(f"[llm_decide_cmd] vLLM error → {e}. Falling back to 0.")
        reply = "0"

    # Update chat history (truncate to avoid huge context)
    _CHAT_HISTORY.append(user_msg)
    _CHAT_HISTORY.append({"role": "assistant", "content": reply})

    # --- Trim history dynamically to fit within model context --------------
    def _approx_tokens(s: str) -> int:
        return len(s)

    def _current_tokens() -> int:
        return sum(_approx_tokens(m["content"]) for m in _CHAT_HISTORY)

    # If keeping the new round would overflow, drop oldest rounds (each = 2 msgs)
    while _current_tokens() + _approx_tokens(world_state_str) + 3 > STATE_SIZE * K_CONTEXT and len(_CHAT_HISTORY) >= 2:
        _CHAT_HISTORY.pop(0)  # drop oldest user
        _CHAT_HISTORY.pop(0)  # drop corresponding assistant

    # Parse integer safely
    try:
        cmd_id = int(reply)
    except ValueError:
        cmd_id = 0
    if not 0 <= cmd_id < CMD_DIM:
        cmd_id = 0

    return np.int32(cmd_id)

def llm_decide_cmd_batch(world_states: np.ndarray) -> np.ndarray:
    """Return an array of cmd_id for each *world_state* in the batch.

    Parameters
    ----------
    world_states : np.ndarray
        2-D array with shape (batch, flat_dim).  Each row is flattened
        world-state that should be fed to the LLM.

    Returns
    -------
    np.ndarray
        1-D array (int32) of length *batch* with values in ``[0, CMD_DIM-1]``.
        Any parsing/LLM error falls back to all-zero commands.
    """
    _lazy_init()

    # Ensure we are dealing with 2-D shape (B, D)
    if world_states.ndim == 1:
        world_states = world_states[None, :]

    batch_size = world_states.shape[0]

    # Build prompts for the whole batch
    prompts = []
    for ws in world_states:
        ws_str = _compress_world_state(ws, quantize=True)
        user_msg = {
            "role": "user",
            "content": json.dumps({"world_state": ws_str}),
        }
        # For batching we *do not* attach the rolling chat history to avoid
        # quadratic context growth across N prompts.
        prompt = _TOKENIZER.apply_chat_template(
            [SYSTEM_MESSAGE, user_msg], tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Run vLLM once for the whole batch
    try:
        outs = _LLM.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=3, temperature=0.0, top_p=1.0),
        )
        replies = [o.outputs[0].text.strip() for o in outs]
    except Exception as e:
        print(f"[llm_decide_cmd_batch] vLLM error → {e}. Falling back to zeros.")
        replies = ["0"] * batch_size

    # Parse replies safely
    cmd_ids = []
    for reply in replies:
        try:
            cmd_id = int(reply)
        except ValueError:
            cmd_id = 0
        if not 0 <= cmd_id < CMD_DIM:
            cmd_id = 0
        cmd_ids.append(cmd_id)

    return np.asarray(cmd_ids, dtype=np.int32)
 
# ---------------------------------------------------------------------------
# Optional: record feedback to inform future commands (single-prompt mode)
# ---------------------------------------------------------------------------

def llm_record_feedback(feedback: str) -> None:
    """Append a brief feedback message to the rolling chat history.

    Only affects single-prompt mode where _CHAT_HISTORY is included.
    """
    try:
        _CHAT_HISTORY.append({"role": "user", "content": str(feedback)[:512]})
        # Trim history if needed
        def _approx_tokens(s: str) -> int:
            return len(s)
        while sum(_approx_tokens(m["content"]) for m in _CHAT_HISTORY) > STATE_SIZE * K_CONTEXT and len(_CHAT_HISTORY) >= 2:
            _CHAT_HISTORY.pop(0)
            _CHAT_HISTORY.pop(0)
    except Exception as _e:
        print(f"[llm_record_feedback] Could not record feedback → {_e}")

# ---------------------------------------------------------------------------
# Optional: record feedback to inform future commands (single-prompt mode)
# ---------------------------------------------------------------------------

def llm_record_feedback(feedback: str) -> None:
    """Append a brief feedback message to the rolling chat history.

    This is only used by single-prompt mode where _CHAT_HISTORY is attached
    to the prompt; batched mode ignores history by design.
    """
    try:
        _CHAT_HISTORY.append({"role": "user", "content": str(feedback)[:512]})
        # Trim if necessary (reuse same policy as in llm_decide_cmd)
        def _approx_tokens(s: str) -> int:
            return len(s)
        while sum(_approx_tokens(m["content"]) for m in _CHAT_HISTORY) > STATE_SIZE * K_CONTEXT and len(_CHAT_HISTORY) >= 2:
            _CHAT_HISTORY.pop(0)
            _CHAT_HISTORY.pop(0)
    except Exception as _e:
        print(f"[llm_record_feedback] Could not record feedback → {_e}")

# ---------------------------------------------------------------------------
# Utility: save current world_state to disk for offline compression testing
# ---------------------------------------------------------------------------

def dump_world_state(world_state: np.ndarray, path: str = "world_state_raw.json", decimals: int = 3) -> None:
    """Write *world_state* to *path* as JSON list.

    Parameters
    ----------
    world_state : np.ndarray
        Flattened vector (or any array-like) to persist.
    path : str, optional
        Output filename. Defaults to ``"world_state_raw.json"`` in cwd.
    decimals : int, optional
        Number of decimals to round before saving to reduce file size.
    """
    arr = np.round(world_state.astype(np.float32), decimals).tolist()
    with open(path, "w") as f:
        json.dump(arr, f)
    print(f"Saved world_state of length {len(arr)} to {path}") 


# ---------------------------------------------------------------------------
# Micro-control helper: return per-ally discrete actions
# ---------------------------------------------------------------------------

def llm_decide_actions(
    world_state: np.ndarray,
    num_allies: int,
    num_enemies: int,
    num_movement_actions: int = 5,
    avail_masks: list | None = None,
    debug: bool = False,
) -> dict:
    """Return a dict of per-ally discrete actions selected by the LLM.

    The action index convention matches SMAX discrete actions:
      - 0..(num_movement_actions-2): movement directions
      - (num_movement_actions-1): stop/hold position
      - num_movement_actions..num_movement_actions+num_enemies-1: attack enemy j
        where j = action - num_movement_actions

    Returns a dict: {"ally_i": np.int32(action_idx)}.
    Falls back to all-stop on any parsing error.
    """
    _lazy_init()

    world_state_str = _compress_world_state(world_state, quantize=True)

    system_msg = {
        "role": "system",
        "content": (
            "You control SMAX allies at the micro level.\n"
            f"There are {num_allies} allies and {num_enemies} enemies.\n"
            f"Each ally's discrete action is an integer in [0, {num_movement_actions + num_enemies - 1}].\n"
            f"Movement: 0..{num_movement_actions-2}; stop: {num_movement_actions-1};"
            f" attack enemy j via {num_movement_actions}+j.\n"
            "You must choose allowed actions only (mask[ally_i][k] == 1).\n"
            "If an attack is out-of-range (mask=0), choose a movement action.\n"
            "Return your answer wrapped in <json>...</json> containing ONLY a JSON object with keys ally_0..ally_{N}.\n"
            "Example: <json>{\"ally_0\": 4, \"ally_1\": 7, \"ally_2\": 6}</json>\n"
            .replace("{N}", str(int(num_allies) - 1))
        ),
    }
    user_msg = {
        "role": "user",
        "content": json.dumps({
            "world_state": world_state_str,
            "num_allies": int(num_allies),
            "num_enemies": int(num_enemies),
            "num_movement_actions": int(num_movement_actions),
            "avail_masks": avail_masks,
        }),
    }

    prompt = _TOKENIZER.apply_chat_template(
        [system_msg, user_msg], tokenize=False, add_generation_prompt=True
    )

    parse_mode = "none"
    try:
        out = _LLM.generate(
            [prompt],
            sampling_params=SamplingParams(
                max_tokens=max(64, num_allies * 8),
                temperature=0.0,
                top_p=1.0,
                stop=["</json>"]
            ),
        )
        reply = out[0].outputs[0].text.strip()
        # First, try strict JSON
        try:
            parsed = json.loads(reply)
            parse_mode = "json"
        except Exception:
            # Try to extract JSON-like substring and coerce
            # Prefer a tagged JSON region
            tag_start = reply.find("<json>")
            tag_end = reply.rfind("</json>")
            if tag_start != -1 and tag_end != -1 and tag_end > tag_start:
                candidate = reply[tag_start + len("<json>"):tag_end]
                parse_mode = "tagged"
            else:
                brace_start = reply.find("{")
                brace_end = reply.rfind("}")
                candidate = reply[brace_start:brace_end + 1] if (brace_start != -1 and brace_end != -1 and brace_end > brace_start) else ""
                if candidate:
                    parse_mode = "braces"
            parsed = {}
            if candidate:
                # Remove code fences/backticks
                candidate = candidate.replace("```json", "").replace("```", "").strip()
                # Replace single quotes with double quotes to help JSON parsing
                candidate_clean = re.sub(r"'", '"', candidate)
                try:
                    parsed = json.loads(candidate_clean)
                except Exception:
                    parsed = {}
            if not parsed:
                # Regex fallback: extract ally_i:number pairs anywhere in the text
                pairs = re.findall(r"ally_(\d+)\s*[:=]\s*(-?\d+)", reply)
                parsed = {f"ally_{k}": int(v) for k, v in pairs}
                if parsed:
                    parse_mode = "pairs"
            if not parsed:
                # As a last resort, take the first num_allies integers found in order
                nums = re.findall(r"-?\d+", reply)
                if len(nums) >= int(num_allies):
                    parsed = {f"ally_{i}": int(nums[i]) for i in range(int(num_allies))}
                    parse_mode = "numbers"
                else:
                    raise ValueError("Could not parse any ally actions from reply")
    except Exception as e:
        print(f"[llm_decide_actions] vLLM/parse error → {e}. Falling back to all-stop.")
        parsed = {}

    # Validate and sanitize
    actions: dict = {}
    stop_idx = int(num_movement_actions - 1)
    min_idx = 0
    max_idx = int(num_movement_actions + num_enemies - 1)
    if debug:
        # Truncate long replies for readability
        _preview = (reply[:240] + "…") if isinstance(locals().get("reply"), str) and len(locals().get("reply")) > 240 else locals().get("reply")
        print(f"[llm_decide_actions][debug] parse_mode={parse_mode} reply_preview={_preview}")
    for i in range(int(num_allies)):
        key = f"ally_{i}"
        val = parsed.get(key, stop_idx)
        original_val = val
        try:
            val = int(val)
        except Exception:
            val = stop_idx
        range_ok = (min_idx <= val <= max_idx)
        if not range_ok:
            val = stop_idx
        mask_ok = True
        chosen_via = "parsed"
        if avail_masks is not None and i < len(avail_masks) and isinstance(avail_masks[i], (list, tuple)) and len(avail_masks[i]) > val:
            try:
                mask_ok = bool(int(avail_masks[i][val]) == 1)
            except Exception:
                mask_ok = True
        # Enforce availability mask with a deterministic policy, but preserve
        # attack intent: if the chosen action is an attack (>= num_movement_actions)
        # and it is masked (out-of-range now), keep it so postprocess can convert
        # it into a movement towards the target. Only override masked movement.
        if avail_masks is not None and i < len(avail_masks) and isinstance(avail_masks[i], (list, tuple)):
            mask = list(avail_masks[i])
            if val >= 0 and val < len(mask) and not (mask[val] == 1):
                is_attack_intent = val >= int(num_movement_actions)
                if is_attack_intent:
                    # Keep masked attacks (even if target later turns out dead);
                    # we do not enforce the mask on the LLM output here.
                    chosen_via = "kept_attack_mask0"
                else:
                    # Movement masked: choose a safe fallback
                    # Prefer any legal movement direction first
                    picked = None
                    for mv in range(0, max(0, int(num_movement_actions) - 1)):
                        if mv < len(mask) and mask[mv] == 1:
                            picked = mv
                            break
                    if picked is not None:
                        val = int(picked)
                        chosen_via = "fallback_move"
                    elif stop_idx < len(mask) and mask[stop_idx] == 1:
                        val = stop_idx
                        chosen_via = "fallback_stop"
                    else:
                        # Pick first any allowed action
                        for k, ok in enumerate(mask):
                            if ok == 1:
                                val = int(k)
                                chosen_via = "fallback_any"
                                break
        if debug:
            print(
                f"[llm_decide_actions][debug] {key}: parsed={original_val} → chosen={val} (range_ok={range_ok}, mask_ok={mask_ok}, via={chosen_via})"
            )
        actions[key] = np.int32(val)

    return actions