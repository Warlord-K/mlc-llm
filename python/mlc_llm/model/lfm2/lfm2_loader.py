"""Hugging Face parameter mapping for LFM2."""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .lfm2_model import LFM2Config, LFM2ForCausalLM


def huggingface(model_config: LFM2Config, quantization: Quantization) -> ExternMapping:
    model = LFM2ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    named_parameters = dict(named_params)
    mapping = ExternMapping()

    for layer_id, layer_type in enumerate(model_config.layer_types):
        if layer_type == "full_attention":
            name = f"model.layers.{layer_id}.self_attn.c_attn.weight"
            parameter = named_parameters[name]
            prefix = f"model.layers.{layer_id}.self_attn"
            mapping.add_mapping(
                name,
                [
                    f"{prefix}.q_proj.weight",
                    f"{prefix}.k_proj.weight",
                    f"{prefix}.v_proj.weight",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=str(parameter.dtype),
                ),
            )
        else:
            name = f"model.layers.{layer_id}.conv.conv_weight"
            parameter = named_parameters[name]
            mapping.add_mapping(
                name,
                [f"model.layers.{layer_id}.conv.conv.weight"],
                functools.partial(lambda x, dtype: x.astype(dtype), dtype=str(parameter.dtype)),
            )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name == "lm_head.weight" and model_config.tie_embedding:
            continue
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(lambda x, dtype: x.astype(dtype), dtype=str(mlc_param.dtype)),
            )
    return mapping
