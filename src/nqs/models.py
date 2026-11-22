"""Neural Quantum State (NQS) wavefunction architectures: Causal GPT, ARNNDense, and RBM."""

from __future__ import annotations

from typing import Any

from nqs.config import ExperimentConfig


def build_variational_state(cfg: ExperimentConfig, hilbert: Any) -> tuple[Any, Any, Any]:
    """Build (vstate, optimizer, preconditioner) for NetKet VMC."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import netket as nk

    class CausalGPTWaveFunction(nk.models.AbstractARNN):
        """Autoregressive Causal Transformer (Mini-GPT) Neural Quantum State.

        Factorizes the many-body wavefunction via the chain rule:
            psi(s_1, ..., s_N) = prod_{i=1}^N psi(s_i | s_1, ..., s_{i-1})

        Because each site's amplitude/phase depends strictly on preceding spins (< i),
        NetKet's ARDirectSampler draws 100% uncorrelated exact samples in O(N) forward passes
        on GPU without Markov Chain Monte Carlo burn-in or autocorrelation!
        """

        d_model: int = 32
        n_heads: int = 4
        n_layers: int = 2
        mlp_ratio: int = 2
        use_phase: bool = False
        machine_pow: int = 2

        @nn.compact
        def conditionals_log_psi(self, inputs: jax.Array) -> jax.Array:
            # Handle both 1D (N,) and batched 2D (B, N) inputs safely
            squeeze_batch = inputs.ndim == 1
            x_in = inputs[None, :] if squeeze_batch else inputs
            x_in = self.reorder(x_in)

            batch_size, n_sites = x_in.shape
            local_size = self.hilbert.local_size

            # Strict right-shift by 1 site:
            # Site 0 sees only learned <BOS>; site i sees [<BOS>, s_0, ..., s_{i-1}]
            bos = self.param(
                "bos_embed",
                nn.initializers.normal(stddev=0.02),
                (1, 1, self.d_model),
            )
            bos_batch = jnp.broadcast_to(bos, (batch_size, 1, self.d_model))

            if n_sites > 1:
                prev_spins = x_in[:, :-1, None].astype(jnp.float32)
                spin_tokens = nn.Dense(self.d_model, name="spin_proj")(prev_spins)
                h = jnp.concatenate([bos_batch, spin_tokens], axis=1)
            else:
                h = bos_batch

            pos_emb = self.param(
                "pos_embed",
                nn.initializers.normal(stddev=0.02),
                (1, n_sites, self.d_model),
            )
            h = h + pos_emb

            # Strict lower-triangular causal attention mask: shape (1, 1, N, N)
            causal_mask = jnp.tril(jnp.ones((1, 1, n_sites, n_sites), dtype=bool))

            for layer_idx in range(self.n_layers):
                # Pre-LayerNorm Causal Multi-Head Self-Attention
                h_norm = nn.LayerNorm(name=f"ln1_{layer_idx}")(h)
                attn_out = nn.MultiHeadDotProductAttention(
                    num_heads=self.n_heads,
                    qkv_features=self.d_model,
                    deterministic=True,
                    name=f"attn_{layer_idx}",
                )(h_norm, mask=causal_mask)
                h = h + attn_out

                # Pre-LayerNorm Feed-Forward Network (GELU MLP)
                h_norm2 = nn.LayerNorm(name=f"ln2_{layer_idx}")(h)
                mlp_hidden = nn.Dense(self.d_model * self.mlp_ratio, name=f"mlp_fc1_{layer_idx}")(h_norm2)
                mlp_hidden = nn.gelu(mlp_hidden)
                mlp_out = nn.Dense(self.d_model, name=f"mlp_fc2_{layer_idx}")(mlp_hidden)
                h = h + mlp_out

            h = nn.LayerNorm(name="ln_final")(h)

            # Normalized conditional log-modulus so sum_{s_i} |psi(s_i | s_<i)|^2 == 1
            amp_logits = nn.Dense(local_size, name="amp_head")(h)
            log_amp = jax.nn.log_softmax(amp_logits, axis=-1) / self.machine_pow

            if self.use_phase:
                phase_raw = nn.Dense(local_size, name="phase_head")(h)
                phase = jnp.pi * jnp.tanh(phase_raw)
                log_psi = log_amp.astype(jnp.complex64) + 1j * phase.astype(jnp.complex64)
            else:
                log_psi = log_amp

            # Restore original spatial ordering if custom graph ordering is used
            log_psi = self.inverse_reorder(log_psi, axis=1)
            return log_psi[0] if squeeze_batch else log_psi

    m_cfg = cfg.model
    t_cfg = cfg.training

    if m_cfg.ansatz == "gpt":
        model = CausalGPTWaveFunction(
            hilbert=hilbert,
            d_model=m_cfg.d_model,
            n_heads=m_cfg.n_heads,
            n_layers=m_cfg.n_layers,
            mlp_ratio=m_cfg.mlp_ratio,
            use_phase=m_cfg.use_phase,
        )
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=t_cfg.n_samples,
            seed=t_cfg.seed,
        )
        holomorphic = False
    elif m_cfg.ansatz == "arnn":
        param_dtype = complex if m_cfg.use_phase else float
        model = nk.models.ARNNDense(
            hilbert=hilbert,
            layers=m_cfg.arnn_layers,
            features=m_cfg.arnn_features,
            param_dtype=param_dtype,
        )
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=t_cfg.n_samples,
            seed=t_cfg.seed,
        )
        holomorphic = False
    elif m_cfg.ansatz == "rbm":
        model = nk.models.RBM(alpha=m_cfg.rbm_alpha, param_dtype=complex)
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=16)
        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=t_cfg.n_samples,
            n_discard_per_chain=t_cfg.n_discard_per_chain,
            seed=t_cfg.seed,
        )
        holomorphic = True
    else:
        raise ValueError(f"Unsupported ansatz: {m_cfg.ansatz}")

    if t_cfg.optimizer == "adam":
        optimizer = nk.optimizer.Adam(learning_rate=t_cfg.learning_rate)
    else:
        optimizer = nk.optimizer.Sgd(learning_rate=t_cfg.learning_rate)

    if t_cfg.use_sr:
        if holomorphic:
            preconditioner = nk.optimizer.SR(diag_shift=t_cfg.diag_shift, holomorphic=True)
        else:
            preconditioner = nk.optimizer.SR(
                qgt=nk.optimizer.qgt.QGTJacobianPyTree,
                diag_shift=t_cfg.diag_shift,
                holomorphic=False,
            )
    else:
        preconditioner = nk.optimizer.identity_preconditioner

    return vstate, optimizer, preconditioner
