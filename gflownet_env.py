import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Tuple

class EnvState(NamedTuple):
    seq: jnp.ndarray        # (L, 5) float32 one-hot representation
    mutated: jnp.ndarray    # (L,) bool mask of already-mutated positions
    step_num: jnp.int32     # Scalar tracking the number of edits

class GFlowNetEnv:
    """
    Pure JAX Markov Decision Process (MDP) for GFlowNet DNA mutation.
    """
    def __init__(self, seq_len: int, max_edits: int = 100):
        self.seq_len = seq_len
        self.max_edits = max_edits
        self.vocab_size = 5
        self.terminate_action = self.seq_len * self.vocab_size

    def reset(self, wt_seq: jnp.ndarray) -> EnvState:
        """Initializes the environment state from a wild-type sequence."""
        return EnvState(
            seq=wt_seq,
            mutated=jnp.zeros((self.seq_len,), dtype=jnp.bool_),
            step_num=jnp.int32(0)
        )

    def get_valid_actions(self, state: EnvState) -> jnp.ndarray:
        """
        Returns boolean mask of valid actions, shape (seq_len * 5 + 1,)
        Action A at position P is valid if:
          1) Position P has not been mutated yet.
          2) The target base does not match the current existing base at P.
        """
        valid_positions = ~state.mutated  # (L,)
        valid_expanded = jnp.repeat(valid_positions[:, None], self.vocab_size, axis=1) # (L, 5)

        # Do not allow mutating to the current existing base (e.g. A -> A)
        is_novel_base = ~(state.seq.astype(jnp.bool_))

        valid_2d = valid_expanded & is_novel_base
        valid_flat = valid_2d.flatten()

        # The terminate action is always valid conceptually
        terminate_valid = jnp.array([True], dtype=jnp.bool_)

        return jnp.concatenate([valid_flat, terminate_valid])

    def step(self, state: EnvState, action: jnp.int32) -> Tuple[EnvState, jnp.bool_]:
        """
        Executes an action purely in JAX. Returns (next_state, is_terminal).
        """
        is_terminate_action = (action == self.terminate_action)
        is_max_edits = (state.step_num >= self.max_edits - 1)
        is_terminal = is_terminate_action | is_max_edits

        pos = action // self.vocab_size
        base = action % self.vocab_size

        new_base_onehot = jax.nn.one_hot(base, self.vocab_size, dtype=jnp.float32)

        # Standard JAX at[].set() for out-of-place static slice updates
        updated_seq = state.seq.at[pos].set(new_base_onehot)
        updated_mutated = state.mutated.at[pos].set(True)

        next_seq = jnp.where(is_terminate_action, state.seq, updated_seq)
        next_mutated = jnp.where(is_terminate_action, state.mutated, updated_mutated)

        next_state = EnvState(
            seq=next_seq,
            mutated=next_mutated,
            step_num=state.step_num + 1
        )

        return next_state, is_terminal


class GeneratorPolicy(nn.Module):
    """
    Forward Policy P_F with 1D-Convolutional spatial stem and factored action head.

    Replaces the catastrophic mean-pool design with aggressive-stride convolutions
    that preserve local positional windows while remaining under 5M parameters.

    The action head is FACTORED to avoid a single massive Dense(128→500001):
      - Position scores: Conv output per-position → (B, L)
      - Base scores: Dense(128→5) — which base to mutate to
      - Full logits = position_scores[:, None] + base_scores[None, :] → (B, L*5)
      - Terminate logit: Dense(128→1) appended

    Input channels: 6 = 5 (one-hot DNA) + 1 (mutated mask)
    Architecture:
        (B, L, 6) → Conv1D(16, k=10, s=10) → (B, L/10, 16) → ReLU
                   → Conv1D(32, k=10, s=10) → (B, L/100, 32) → ReLU
                   → Conv1D(1, k=1, s=1)    → (B, L/100, 1)  — position scores at coarse resolution
                   → Upsample back to L via repeat

        Global path: mean-pool conv features → Dense(128) → Dense(5) for base scores
                                              → Dense(1) for terminate logit
    """
    seq_len: int
    vocab_size: int = 5
    input_channels: int = 6  # 5 one-hot + 1 mutated mask

    @nn.compact
    def __call__(self, state_input: jnp.ndarray, target_metadata: jnp.ndarray):
        """
        state_input:      (B, L, 6) — one-hot DNA concat with mutated mask
        target_metadata:  (B, M)
        """
        B = state_input.shape[0]
        L = self.seq_len

        # --- Convolutional spatial stem ---
        # Layer 1: 100k×6 → 10k×16  (stride 10 captures 10-bp local windows)
        x = nn.Conv(features=16, kernel_size=(10,), strides=(10,), name="conv_stem_1")(state_input)
        x = jax.nn.relu(x)

        # Layer 2: 10k×16 → 1k×32  (stride 10 captures 100-bp local windows)
        x = nn.Conv(features=32, kernel_size=(10,), strides=(10,), name="conv_stem_2")(x)
        x = jax.nn.relu(x)
        # x is now (B, L/100, 32) — e.g. (B, 1000, 32)

        # --- Factored Action Head ---

        # POSITION BRANCH: per-position scoring at coarse resolution
        # (B, 1000, 32) → (B, 1000, 1) → squeeze → (B, 1000)
        pos_coarse = nn.Conv(features=1, kernel_size=(1,), strides=(1,), name="pos_score_conv")(x)
        pos_coarse = pos_coarse.squeeze(-1)  # (B, L/100)

        # Upsample coarse position scores back to full resolution via repeat
        # Each coarse bin covers 100 positions
        stride_factor = 100
        pos_scores = jnp.repeat(pos_coarse, stride_factor, axis=-1)  # (B, L)
        # Trim or pad to exact seq_len (handles non-divisible lengths)
        pos_scores = pos_scores[:, :L]

        # GLOBAL BRANCH: pool spatial features for base scores + terminate
        global_features = jnp.mean(x, axis=1)  # (B, 32)
        global_features = jnp.concatenate([global_features, target_metadata], axis=-1)  # (B, 32+M)

        g = nn.Dense(128, name="global_dense_1")(global_features)
        g = jax.nn.relu(g)
        g = nn.Dense(64, name="global_dense_2")(g)
        g = jax.nn.relu(g)

        # Base preference scores: which of 5 bases to mutate to
        base_scores = nn.Dense(self.vocab_size, name="base_scores")(g)  # (B, 5)

        # Terminate logit
        terminate_logit = nn.Dense(1, name="terminate_logit")(g)  # (B, 1)

        # COMBINE: factored logits = pos_scores[:, :, None] + base_scores[:, None, :]
        # → (B, L, 5) → flatten → (B, L*5) → concat terminate → (B, L*5+1)
        combined = pos_scores[:, :, None] + base_scores[:, None, :]  # (B, L, 5)
        mutation_logits = combined.reshape((B, L * self.vocab_size))  # (B, L*5)

        logits = jnp.concatenate([mutation_logits, terminate_logit], axis=-1)  # (B, L*5+1)
        return logits
