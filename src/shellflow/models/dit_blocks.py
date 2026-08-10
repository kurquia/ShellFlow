"""Transformer building blocks for the Riemannian DiTs model.

SinusoidalPositionEmbedding, RMSNorm, SwiGLU, MultiHeadAttention (with
varlen support), METCrossAttention, and the adaLN-Zero DiT block
(Peebles & Xie, 2023).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embedding of the continuous flow-matching time t.

    The caller scales t ∈ [0, 1] (typically by 1000) so the sinusoid
    frequencies cover a useful dynamic range.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (batch_size,) tensor of (scaled) flow times
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out)
        self.w2 = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w2(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional attention mask.

    Uses PyTorch's F.scaled_dot_product_attention for better performance
    and Flash Attention support when available.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        multiplicative: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        self.multiplicative = multiplicative

        # QKV projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Multiplicative attention: also compute q ⊙ (attn @ v) for pairwise products
        if multiplicative:
            self.out_proj = nn.Linear(2 * d_model, d_model, bias=False)
            # Gate initialized at 0 → sigmoid(0)=0.5, but the branch output
            # starts small because weights are random. Stable ramp-up.
            self.mult_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # QK-Norm: stabilize attention logits, enable sharper patterns
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        pair_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len) - 1 for valid, 0 for padding
            return_attn_weights: if True, return attention weights averaged over heads
            pair_bias: (batch_size, num_heads, seq_len, seq_len) - added to attn logits

        Returns:
            If return_attn_weights=False: (batch_size, seq_len, d_model)
            If return_attn_weights=True: tuple of output and attn_weights (batch, seq, seq)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK-Norm: normalize before attention scores
        # Save raw q for multiplicative branch (preserves magnitude information)
        q_raw = q if self.multiplicative else None
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Prepare attention mask — combine padding mask with pair bias into float mask
        if pair_bias is not None:
            # pair_bias: (B, H, N, N) float — added to logits before softmax
            attn_mask = pair_bias
            if attention_mask is not None:
                pad = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
                attn_mask = attn_mask.masked_fill(~pad, float("-inf"))
        elif attention_mask is not None:
            attn_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        else:
            attn_mask = None

        if return_attn_weights:
            # Manual attention computation to get weights
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn_weights = (
                        attn_weights + attn_mask
                    )  # float mask (pair_bias + padding)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights_out = attn_weights.mean(
                dim=1
            )  # average over heads → (B, N, N)
            # No dropout: `if self.training` ternary breaks torch.compile
            attn_v = torch.matmul(attn_weights, v)  # (B, H, N, D_head)
            additive_out = attn_v.transpose(1, 2).reshape(
                batch_size, seq_len, self.d_model
            )
            if self.multiplicative:
                # Use raw (un-normalized) q to preserve magnitude for pairwise products
                mult_out = (
                    (q_raw * attn_v)
                    .transpose(1, 2)
                    .reshape(batch_size, seq_len, self.d_model)
                )
                mult_out = torch.sigmoid(self.mult_gate) * mult_out
                out = self.out_proj(torch.cat([additive_out, mult_out], dim=-1))
            else:
                out = self.out_proj(additive_out)
            return out, attn_weights_out
        else:
            # Use PyTorch's optimized scaled dot-product attention
            # dropout_p=0.0 unconditionally: the `self.training` ternary causes
            # torch.compile graph breaks and NaN in validation.  Attention dropout
            # is negligible for this model size; removing it is the simplest fix.
            attn_v = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )  # (batch, heads, seq, head_dim)
            additive_out = attn_v.transpose(1, 2).reshape(
                batch_size, seq_len, self.d_model
            )
            if self.multiplicative:
                # Use raw (un-normalized) q to preserve magnitude for pairwise products
                mult_out = (
                    (q_raw * attn_v)
                    .transpose(1, 2)
                    .reshape(batch_size, seq_len, self.d_model)
                )
                mult_out = torch.sigmoid(self.mult_gate) * mult_out
                out = self.out_proj(torch.cat([additive_out, mult_out], dim=-1))
            else:
                out = self.out_proj(additive_out)
            return out

    def forward_geometric(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Geometric (varlen) forward — no padding, block-diagonal attention.

        Args:
            x:          (T_total, d_model) flat concatenated particle features.
            cu_seqlens: (B+1,) int32 cumulative sequence lengths.
            max_seqlen: maximum sequence length in the batch.

        Returns:
            (T_total, d_model) attention output.
        """
        T = x.shape[0]

        # QKV projection: (T, 3*D) -> (T, 3, H, D_head)
        qkv = self.qkv(x).reshape(T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Save raw q for multiplicative branch before normalization
        q_raw = q.clone() if self.multiplicative else None
        q = self.q_norm(q)
        k = self.k_norm(k)

        # varlen_attn requires bf16 or fp16
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        attn_v = varlen_attn(
            q,
            k,
            v,
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
        )  # (T, H, D_head)

        if orig_dtype == torch.float32:
            attn_v = attn_v.to(orig_dtype)

        additive_out = attn_v.reshape(T, self.d_model)

        if self.multiplicative:
            mult_out = (q_raw * attn_v).reshape(T, self.d_model)
            mult_out = torch.sigmoid(self.mult_gate) * mult_out
            out = self.out_proj(torch.cat([additive_out, mult_out], dim=-1))
        else:
            out = self.out_proj(additive_out)
        return out


class METCrossAttention(nn.Module):
    """Multi-head cross-attention from particles (Q) to MET context tokens (KV).

    Optional tanh gate (disabled by default). When gated=True, output is
    multiplied by tanh(learnable_scalar) initialized at 0, ramping up during
    training. When gated=False, output passes through directly.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.1, gated: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.gated = gated

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # QK-norm (matching self-attention convention)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.dropout_p = dropout
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, met_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (B, N, D)  particle representations (queries)
            met_context: (B, M, D)  MET context tokens (keys/values)
        Returns:
            (B, N, D)  cross-attention output
        """
        B, N, _ = x.shape
        M = met_context.shape[1]

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(met_context)
            .view(B, M, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(met_context)
            .view(B, M, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,  # no dropout on cross-attn to few MET tokens (torch.compile safe)
            is_causal=False,
        )  # (B, H, N, head_dim)

        out = attn_out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.out_proj(out)
        if self.gated:
            out = torch.tanh(self.gate) * out
        return out

    def forward_geometric(
        self,
        x: torch.Tensor,
        met_context: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Geometric (varlen) cross-attention from particles to MET.

        Args:
            x:            (T_total, d_model) flat particle features (queries).
            met_context:  (B, M, d_model) MET context tokens (keys/values).
            cu_seqlens_q: (B+1,) int32 cumulative particle counts.
            max_seqlen_q: max particles in any event.

        Returns:
            (T_total, d_model) cross-attention output.
        """
        T = x.shape[0]
        B, M, _ = met_context.shape

        q = self.q_proj(x).reshape(T, self.num_heads, self.head_dim)
        k = self.k_proj(met_context).reshape(B * M, self.num_heads, self.head_dim)
        v = self.v_proj(met_context).reshape(B * M, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV cu_seqlens: uniform, every event has exactly M MET tokens
        cu_seqlens_k = torch.arange(
            0, (B + 1) * M, M, dtype=torch.int32, device=x.device
        )

        # varlen_attn requires bf16 or fp16
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        attn_out = varlen_attn(
            q,
            k,
            v,
            cu_seq_q=cu_seqlens_q,
            cu_seq_k=cu_seqlens_k,
            max_q=max_seqlen_q,
            max_k=M,
        )  # (T, H, D_head)

        if orig_dtype == torch.float32:
            attn_out = attn_out.to(orig_dtype)

        out = attn_out.reshape(T, self.d_model)
        out = self.out_proj(out)
        if self.gated:
            out = torch.tanh(self.gate) * out
        return out


class AdaLNZeroDiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (Peebles & Xie, 2023).

    Each sub-layer (self-attn, cross-attn, FFN) gets per-layer time-dependent
    scale (γ), shift (β), and output gate (α).  All α gates are zero-initialized
    so every block starts as identity → prevents rank collapse in deep residual
    streams.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        multiplicative_attention: bool = False,
        met_cross_attention: bool = False,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model, num_heads, dropout, multiplicative=multiplicative_attention
        )

        # Optional MET cross-attention
        self.met_cross_attention = met_cross_attention
        if met_cross_attention:
            self.cross_norm = RMSNorm(d_model)
            self.cross_attn = METCrossAttention(
                d_model, num_heads, dropout, gated=False
            )

        # FFN
        self.norm2 = RMSNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            SwiGLU(d_model, mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

        # adaLN-Zero: project time_emb → modulation vectors
        # With cross-attn: 8 vectors (γ1,β1,α1, γ_c,α_c, γ2,β2,α2)
        # Without:         6 vectors (γ1,β1,α1, γ2,β2,α2)
        n_adaln = 8 if met_cross_attention else 6
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, n_adaln * d_model),
        )
        # Zero-init: all gates start at 0 → identity blocks at init
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        met_context: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Compute all modulation vectors
        # - If cond is None: time-only, per-event → chunks unsqueezed to (B, 1, D) for broadcast
        # - If cond is (B, N, D): per-particle base = time_emb.unsqueeze(1) + cond
        if cond is not None:
            base = time_emb.unsqueeze(1) + cond  # (B, N, D)
            adaln_out = self.adaln_mlp(base)  # (B, N, n*D)
            if self.met_cross_attention:
                gamma1, beta1, alpha1, gamma_c, alpha_c, gamma2, beta2, alpha2 = (
                    adaln_out.chunk(8, dim=-1)
                )
            else:
                gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_out.chunk(
                    6, dim=-1
                )
        else:
            adaln_out = self.adaln_mlp(time_emb)  # (B, n*D)
            if self.met_cross_attention:
                gamma1, beta1, alpha1, gamma_c, alpha_c, gamma2, beta2, alpha2 = (
                    v.unsqueeze(1) for v in adaln_out.chunk(8, dim=-1)
                )
            else:
                gamma1, beta1, alpha1, gamma2, beta2, alpha2 = (
                    v.unsqueeze(1) for v in adaln_out.chunk(6, dim=-1)
                )

        # Modulated self-attention
        x_norm = self.norm1(x) * (1 + gamma1) + beta1
        if return_attn_weights:
            attn_out, attn_weights = self.attn(
                x_norm,
                attention_mask=attention_mask,
                return_attn_weights=True,
            )
        else:
            attn_out = self.attn(x_norm, attention_mask=attention_mask)
        x = x + alpha1 * attn_out

        # Gated MET cross-attention
        if self.met_cross_attention and met_context is not None:
            x_cn = self.cross_norm(x) * (1 + gamma_c)
            x = x + alpha_c * self.cross_attn(x_cn, met_context)

        # Modulated FFN
        x = x + alpha2 * self.mlp(self.norm2(x) * (1 + gamma2) + beta2)

        if return_attn_weights:
            return x, attn_weights
        return x

    def forward_geometric(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        batch: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        met_context: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Geometric (varlen) forward — no padding.

        Args:
            x:           (T_total, d_model) flat particle features.
            time_emb:    (B, d_model) per-event time embedding.
            batch:       (T_total,) int64 event index per particle.
            cu_seqlens:  (B+1,) int32 cumulative particle counts.
            max_seqlen:  max particles in any event.
            met_context: (B, M, d_model) MET tokens, or None.
            cond:        (T_total, d_model) per-particle adaLN cond, or None.

        Returns:
            (T_total, d_model) block output.
        """
        if cond is not None:
            # Per-particle: expand time to flat, add cond, then adaLN — all chunks (T, D)
            base_flat = time_emb[batch] + cond  # (T_total, D)
            adaln_out = self.adaln_mlp(base_flat)  # (T_total, n*D)
            if self.met_cross_attention:
                g1, b1, a1, g_c, a_c, g2, b2, a2 = adaln_out.chunk(8, dim=-1)
            else:
                g1, b1, a1, g2, b2, a2 = adaln_out.chunk(6, dim=-1)

            # Modulated self-attention (chunks already per-particle; no [batch] gather)
            x_norm = self.norm1(x) * (1 + g1) + b1
            attn_out = self.attn.forward_geometric(x_norm, cu_seqlens, max_seqlen)
            x = x + a1 * attn_out

            # Gated MET cross-attention
            if self.met_cross_attention and met_context is not None:
                x_cn = self.cross_norm(x) * (1 + g_c)
                x = x + a_c * self.cross_attn.forward_geometric(
                    x_cn, met_context, cu_seqlens, max_seqlen
                )

            # Modulated FFN
            x = x + a2 * self.mlp(self.norm2(x) * (1 + g2) + b2)
            return x

        # Legacy per-event path (time only)
        adaln_out = self.adaln_mlp(time_emb)  # (B, n*D)
        if self.met_cross_attention:
            g1, b1, a1, g_c, a_c, g2, b2, a2 = adaln_out.chunk(8, dim=-1)
        else:
            g1, b1, a1, g2, b2, a2 = adaln_out.chunk(6, dim=-1)

        # Expand per-event → per-particle via batch indexing
        # Modulated self-attention
        x_norm = self.norm1(x) * (1 + g1[batch]) + b1[batch]
        attn_out = self.attn.forward_geometric(x_norm, cu_seqlens, max_seqlen)
        x = x + a1[batch] * attn_out

        # Gated MET cross-attention
        if self.met_cross_attention and met_context is not None:
            x_cn = self.cross_norm(x) * (1 + g_c[batch])
            x = x + a_c[batch] * self.cross_attn.forward_geometric(
                x_cn, met_context, cu_seqlens, max_seqlen
            )

        # Modulated FFN
        x = x + a2[batch] * self.mlp(self.norm2(x) * (1 + g2[batch]) + b2[batch])
        return x

    def forward_flat(
        self,
        x: torch.Tensor,
        time_emb_flat: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        met_context_flat: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fully flat forward — no [batch] gathers, fully compilable.

        All inputs are pre-expanded to per-particle (T, D). The only
        event-aware operation is varlen_attn via cu_seqlens.
        Only real particles should be passed — no padding tokens.

        Args:
            x:               (T, d_model) particle features (real only)
            time_emb_flat:   (T, d_model) time embedding (pre-expanded)
            cu_seqlens:      (B+1,) int32 cumulative counts
            max_seqlen:      max particles in any event
            met_context_flat: (B, M, d_model) MET context, or None
            cond:            (T, d_model) per-particle adaLN cond, or None

        Returns:
            (T, d_model) block output
        """
        # AdaLN modulation on (T, D) — optionally fused with per-particle cond
        base = time_emb_flat + cond if cond is not None else time_emb_flat
        adaln_out = self.adaln_mlp(base)  # (T, n*D)
        if self.met_cross_attention:
            g1, b1, a1, g_c, a_c, g2, b2, a2 = adaln_out.chunk(8, dim=-1)
        else:
            g1, b1, a1, g2, b2, a2 = adaln_out.chunk(6, dim=-1)

        # Modulated self-attention
        x_norm = self.norm1(x) * (1 + g1) + b1
        attn_out = self.attn.forward_geometric(x_norm, cu_seqlens, max_seqlen)
        x = x + a1 * attn_out

        # Gated MET cross-attention
        if self.met_cross_attention and met_context_flat is not None:
            x_cn = self.cross_norm(x) * (1 + g_c)
            cross_out = self.cross_attn.forward_geometric(
                x_cn, met_context_flat, cu_seqlens, max_seqlen
            )
            x = x + a_c * cross_out

        # Modulated FFN
        x = x + a2 * self.mlp(self.norm2(x) * (1 + g2) + b2)
        return x

    def _njt_pre_attn(self, x_njt, g1, b1):
        """NJT norm + modulation before attention (compilable)."""
        return self.norm1(x_njt) * (1 + g1) + b1

    def _njt_post_attn(self, x_njt, a1, attn_njt, g2, b2, a2):
        """NJT residual + FFN after attention (compilable)."""
        x_njt = x_njt + a1 * attn_njt
        x_njt = x_njt + a2 * self.mlp(self.norm2(x_njt) * (1 + g2) + b2)
        return x_njt

    def _njt_pre_cross_attn(self, x_njt, g_c):
        """NJT norm before cross-attention (compilable)."""
        return self.cross_norm(x_njt) * (1 + g_c)

    def forward_njt(
        self,
        x_njt: torch.Tensor,
        time_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        met_context: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """NJT forward — compilable NJT ops + varlen attention.

        The NJT-pure parts (norm, modulation, MLP, residuals) are split
        into helper methods that torch.compile can fuse. The attention
        calls use flat tensors via varlen_attn.

        When cond is provided (flat (T, D)), adaLN modulation is per-particle
        rather than per-event. All modulation tensors become NJT for elementwise
        ops against x_njt (no compiled helpers on the per-particle path — they
        assume regular broadcast tensors).
        """
        offsets = x_njt.offsets()

        if cond is not None:
            # Per-particle modulation path — compute NJT chunks, bypass compiled helpers
            cond_njt = torch.nested.nested_tensor_from_jagged(cond, offsets=offsets)
            base_njt = time_emb.unsqueeze(1) + cond_njt  # NJT (B, <n>, D)
            adaln_out = self.adaln_mlp(base_njt)  # NJT (B, <n>, n*D)
            if self.met_cross_attention:
                g1, b1, a1, g_c, a_c, g2, b2, a2 = adaln_out.chunk(8, dim=-1)
            else:
                g1, b1, a1, g2, b2, a2 = adaln_out.chunk(6, dim=-1)

            # Pre-attention: NJT norm + per-particle modulation (inline, not compiled)
            x_norm = self.norm1(x_njt) * (1 + g1) + b1

            # Attention: drop to flat, varlen, wrap back
            attn_out = self.attn.forward_geometric(
                x_norm.values(), cu_seqlens, max_seqlen
            )
            attn_njt = torch.nested.nested_tensor_from_jagged(attn_out, offsets=offsets)

            # Cross-attention (if applicable)
            if self.met_cross_attention and met_context is not None:
                x_njt = x_njt + a1 * attn_njt
                x_cn = self.cross_norm(x_njt) * (1 + g_c)
                cross_out = self.cross_attn.forward_geometric(
                    x_cn.values(), met_context, cu_seqlens, max_seqlen
                )
                cross_njt = torch.nested.nested_tensor_from_jagged(
                    cross_out, offsets=offsets
                )
                x_njt = x_njt + a_c * cross_njt
                x_njt = x_njt + a2 * self.mlp(self.norm2(x_njt) * (1 + g2) + b2)
                return x_njt

            x_njt = x_njt + a1 * attn_njt
            x_njt = x_njt + a2 * self.mlp(self.norm2(x_njt) * (1 + g2) + b2)
            return x_njt

        # Legacy per-event path — uses compiled helpers
        adaln_out = self.adaln_mlp(time_emb)
        if self.met_cross_attention:
            g1, b1, a1, g_c, a_c, g2, b2, a2 = (
                v.unsqueeze(1) for v in adaln_out.chunk(8, dim=-1)
            )
        else:
            g1, b1, a1, g2, b2, a2 = (
                v.unsqueeze(1) for v in adaln_out.chunk(6, dim=-1)
            )

        # Pre-attention: NJT norm + modulation
        x_norm = self._njt_pre_attn(x_njt, g1, b1)

        # Attention: drop to flat, varlen, wrap back
        attn_out = self.attn.forward_geometric(x_norm.values(), cu_seqlens, max_seqlen)
        attn_njt = torch.nested.nested_tensor_from_jagged(attn_out, offsets=offsets)

        # Cross-attention (if applicable)
        if self.met_cross_attention and met_context is not None:
            # Need intermediate residual before cross-attn
            x_njt = x_njt + a1 * attn_njt
            x_cn = self._njt_pre_cross_attn(x_njt, g_c)
            cross_out = self.cross_attn.forward_geometric(
                x_cn.values(), met_context, cu_seqlens, max_seqlen
            )
            cross_njt = torch.nested.nested_tensor_from_jagged(
                cross_out, offsets=offsets
            )
            x_njt = x_njt + a_c * cross_njt
            # Post-attention FFN (a1 already applied above)
            x_njt = x_njt + a2 * self.mlp(self.norm2(x_njt) * (1 + g2) + b2)
            return x_njt

        # Post-attention: residual + FFN (no cross-attention)
        return self._njt_post_attn(x_njt, a1, attn_njt, g2, b2, a2)
