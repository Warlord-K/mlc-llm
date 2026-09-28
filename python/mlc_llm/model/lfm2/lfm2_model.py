"""Implementation of the LFM2 hybrid architecture."""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
from tvm import relax as R
from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LFM2Config(ConfigBase):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    norm_eps: float
    rope_theta: float
    vocab_size: int
    layer_types: List[str]  # noqa: UP006
    conv_L_cache: int
    conv_bias: bool
    block_auto_adjust_ff_dim: bool = False
    block_ffn_dim_multiplier: Optional[float] = None
    block_multiple_of: int = 1
    tie_embedding: Optional[bool] = None
    tie_word_embeddings: Optional[bool] = None
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    head_dim: int = 0
    dtype: str = "float32"
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):
        if self.tie_embedding is None:
            self.tie_embedding = (
                self.tie_word_embeddings if self.tie_word_embeddings is not None else False
            )
        self.tie_word_embeddings = self.tie_embedding
        if self.tensor_parallel_shards != 1:
            raise ValueError("LFM2 currently supports a single device.")
        if self.conv_bias:
            raise ValueError("LFM2 convolution bias is not supported.")
        if len(self.layer_types) != self.num_hidden_layers or any(
            layer_type not in ("conv", "full_attention") for layer_type in self.layer_types
        ):
            raise ValueError("layer_types must identify every LFM2 layer as conv or full_attention")
        if self.head_dim == 0:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError("Unable to determine the maximum sequence length.")
        if self.prefill_chunk_size == 0 or self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = min(self.context_window_size, 2048)

    @property
    def num_attention_layers(self) -> int:
        return self.layer_types.count("full_attention")

    @property
    def num_conv_layers(self) -> int:
        return self.layer_types.count("conv")

    @property
    def mlp_intermediate_size(self) -> int:
        if not self.block_auto_adjust_ff_dim:
            return self.intermediate_size
        intermediate_size = int(2 * self.intermediate_size / 3)
        if self.block_ffn_dim_multiplier is not None:
            intermediate_size = int(self.block_ffn_dim_multiplier * intermediate_size)
        return self.block_multiple_of * (
            (intermediate_size + self.block_multiple_of - 1) // self.block_multiple_of
        )


class LFM2Embedding(nn.Embedding):
    def lm_head_forward(self, x: Tensor):
        return op.matmul(x, op.permute_dims(self.weight), out_dtype="float32")


class LFM2MLP(nn.Module):
    def __init__(self, config: LFM2Config):
        self.w1 = nn.Linear(config.hidden_size, config.mlp_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.mlp_intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.mlp_intermediate_size, bias=False)

    def forward(self, x: Tensor):
        return self.w2(op.silu(self.w1(x)) * self.w3(x))


class LFM2Attention(nn.Module):
    def __init__(self, config: LFM2Config):
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.c_attn = nn.Linear(
            config.hidden_size,
            (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_layernorm = nn.RMSNorm(config.head_dim, -1, config.norm_eps, bias=False)
        self.k_layernorm = nn.RMSNorm(config.head_dim, -1, config.norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        qkv = op.reshape(self.c_attn(hidden_states), (b, s, h_q + 2 * h_kv, d))
        q, k, v = op.split(qkv, [h_q, h_q + h_kv], axis=2)
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)
        qkv = op.concat([q, k, v], dim=2)
        output = paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, h_q, sm_scale=d**-0.5)
        return self.out_proj(op.reshape(output, (b, s, h_q * d)))


class LFM2ShortConv(nn.Module):
    def __init__(self, config: LFM2Config, conv_layer_id: int):
        self.hidden_size = config.hidden_size
        self.kernel_size = config.conv_L_cache
        self.conv_layer_id = conv_layer_id
        self.dtype = config.dtype
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.conv_bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.conv_bias)
        self.conv_weight = nn.Parameter((config.hidden_size, 1, config.conv_L_cache))

    def forward(self, hidden_states: Tensor, state: RNNState) -> Tuple[Tensor, RNNState]:  # noqa: UP006
        b, _, _ = hidden_states.shape
        projected = self.in_proj(hidden_states)
        B, C, x = op.split(projected, 3, axis=-1)
        hidden_states = B * x
        conv_state = state.get(
            self.conv_layer_id,
            0,
            (b, self.kernel_size - 1, self.hidden_size),
            self.dtype,
        )
        hidden_states, new_conv_state = self._causal_conv(hidden_states, conv_state)
        state = state.set(self.conv_layer_id, 0, new_conv_state)
        return self.out_proj(C * hidden_states), state

    def _causal_conv(self, x: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        kernel_size = self.kernel_size

        def _update_state(old_state: te.Tensor, values: te.Tensor):
            state_len = old_state.shape[1]
            seq_len = values.shape[1]
            return te.compute(
                old_state.shape,
                lambda bi, ti, di: tirx.if_then_else(
                    seq_len + ti < state_len,
                    old_state[bi, seq_len + ti, di],
                    values[bi, seq_len + ti - state_len, di],
                ),
                name="update_conv_state",
            )

        new_state = op.tensor_expr_op(_update_state, "update_conv_state", [conv_state, x])

        def _depthwise_conv(old_state: te.Tensor, values: te.Tensor, weight: te.Tensor):
            state_len = old_state.shape[1]
            seq_len = values.shape[1]
            kernel_index = te.reduce_axis((0, kernel_size), name="kernel_index")

            def _compute(bi, si, di):
                result = te.sum(
                    tirx.if_then_else(
                        si + kernel_index < state_len,
                        old_state[bi, si + kernel_index, di],
                        values[bi, si + kernel_index - state_len, di],
                    )
                    * weight[di, 0, kernel_index],
                    axis=kernel_index,
                )
                return result

            return te.compute(
                (values.shape[0], seq_len, values.shape[2]),
                _compute,
                name="depthwise_conv1d",
            )

        result = op.tensor_expr_op(
            _depthwise_conv,
            "depthwise_conv1d",
            [conv_state, x, self.conv_weight],
            attrs={"op_pattern": 8},
        )
        return result, new_state

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype


class LFM2DecoderLayer(nn.Module):
    def __init__(self, config: LFM2Config, layer_id: int, category_id: int):
        self.layer_type = config.layer_types[layer_id]
        if self.layer_type == "full_attention":
            self.self_attn = LFM2Attention(config)
        else:
            self.conv = LFM2ShortConv(config, category_id)
        self.category_id = category_id
        self.feed_forward = LFM2MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, state: RNNState):
        residual = hidden_states
        hidden_states = self.operator_norm(hidden_states)
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(hidden_states, paged_kv_cache, self.category_id)
        else:
            hidden_states, state = self.conv(hidden_states, state)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, state


class LFM2Model(nn.Module):
    def __init__(self, config: LFM2Config):
        self.embed_tokens = LFM2Embedding(config.vocab_size, config.hidden_size)
        attention_id = 0
        conv_id = 0
        layers = []
        for layer_id, layer_type in enumerate(config.layer_types):
            category_id = attention_id if layer_type == "full_attention" else conv_id
            layers.append(LFM2DecoderLayer(config, layer_id, category_id))
            if layer_type == "full_attention":
                attention_id += 1
            else:
                conv_id += 1
        self.layers = nn.ModuleList(layers)
        self.embedding_norm = nn.RMSNorm(config.hidden_size, -1, config.norm_eps, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache, state: RNNState):
        hidden_states = inputs
        for layer in self.layers:
            hidden_states, state = layer(hidden_states, paged_kv_cache, state)
        return self.embedding_norm(hidden_states), state


class LFM2ForCausalLM(nn.Module):
    def __init__(self, config: LFM2Config):
        self.config = config
        self.model = LFM2Model(config)
        self.tie_embedding = config.tie_embedding
        if not config.tie_embedding:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.num_attention_layers = config.num_attention_layers
        self.num_conv_layers = config.num_conv_layers
        self.kernel_size = config.conv_L_cache

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        return self.model.embed_tokens(input_ids)

    def _forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states, rnn_state = self.model(input_embeds, paged_kv_cache, rnn_state)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        if self.tie_embedding:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache, rnn_state

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state, logit_positions)

    def batch_decode(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state)

    def batch_verify(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state)

    def create_rnn_state(self, max_batch_size: tirx.Var, max_history: tirx.Var) -> RNNState:
        return RNNState.create(
            max_batch_size=max_batch_size,
            num_hidden_layers=self.num_conv_layers,
            max_history=max_history,
            init_values=[R.const(np.zeros((self.kernel_size - 1, self.hidden_size), self.dtype))],
        )

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
