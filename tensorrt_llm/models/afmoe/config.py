# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import sys
from pathlib import Path
from typing import Optional, Union

from ...layers import MoeConfig
from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class AfmoeConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 mlp_bias: bool = False,
                 attn_bias: bool = False,  # Qwen3: no attention bias
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 sliding_window: int = 2048,
                 global_attn_every_n_layers: int = 4,
                 layer_types: Optional[list] = None,
                 moe: Optional[Union[MoeConfig, dict]] = None,
                 num_key_value_heads: int = 32,
                 head_dim: int = 64,
                 # AFMoE specific parameters from HF config
                 moe_intermediate_size: int = 1408,
                 num_dense_layers: int = 1,
                 n_group: int = 1,
                 topk_group: int = 1,
                 score_func: str = "sigmoid",
                 route_norm: bool = True,
                 route_scale: float = 1.0,
                 mup_enabled: bool = False,
                 seq_length: int = 8192,  # Qwen3 sequence length
                 qwen_type: str = "afmoe",  # Identify as AFMoE variant
                 **kwargs):
        self.mlp_bias = mlp_bias
        self.attn_bias = attn_bias
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.sliding_window = sliding_window
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.layer_types = layer_types or []
        
        # GQA configuration
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        
        # AFMoE specific parameters
        self.moe_intermediate_size = moe_intermediate_size
        self.num_dense_layers = num_dense_layers
        self.n_group = n_group
        self.topk_group = topk_group
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.mup_enabled = mup_enabled
        self.seq_length = seq_length
        self.qwen_type = qwen_type
        
        # Initialize MoE configuration if provided
        if moe is None:
            # Legacy MOE config fields
            moe = MoeConfig(
                num_experts=kwargs.pop('moe_num_experts', 0),
                top_k=kwargs.pop('moe_top_k', 0),
                normalization_mode=kwargs.pop(
                    'moe_normalization_mode',
                    MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE))
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()
        
        # Standard LLaMA-style configuration
        self.fc_after_embed = False
        self.use_input_layernorm_in_first_layer = True
        self.use_last_layernorm = True
        self.layer_idx_offset = 0
        self.has_partial_lora_mask = False

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in AfmoeConfig
        output['mlp_bias'] = self.mlp_bias
        output['attn_bias'] = self.attn_bias
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output['sliding_window'] = self.sliding_window
        output['global_attn_every_n_layers'] = self.global_attn_every_n_layers
        output['layer_types'] = self.layer_types
        output['num_key_value_heads'] = self.num_key_value_heads
        output['head_dim'] = self.head_dim
        output['moe_intermediate_size'] = self.moe_intermediate_size
        output['num_dense_layers'] = self.num_dense_layers
        output['n_group'] = self.n_group
        output['topk_group'] = self.topk_group
        output['score_func'] = self.score_func
        output['route_norm'] = self.route_norm
        output['route_scale'] = self.route_scale
        output['mup_enabled'] = self.mup_enabled
        output['seq_length'] = self.seq_length
        output['qwen_type'] = self.qwen_type
        output['fc_after_embed'] = self.fc_after_embed
        output['use_input_layernorm_in_first_layer'] = self.use_input_layernorm_in_first_layer
        output['use_last_layernorm'] = self.use_last_layernorm
        output['layer_idx_offset'] = self.layer_idx_offset
        output['moe'] = self.moe.to_dict()
        return output

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        trust_remote_code = kwargs.pop('trust_remote_code', True)

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=trust_remote_code)

        # Extract AFMoE-specific configurations
        num_key_value_heads = getattr(hf_config, "num_key_value_heads", 32)
        head_dim = getattr(hf_config, "head_dim", 64)
        sliding_window = getattr(hf_config, "sliding_window", 2048)
        global_attn_every_n_layers = getattr(hf_config, "global_attn_every_n_layers", 4)
        layer_types = getattr(hf_config, "layer_types", [])
        rotary_scaling = getattr(hf_config, "rope_scaling", None)
        rotary_base = getattr(hf_config, "rope_theta", 10000.0)
        attn_bias = getattr(hf_config, 'bias', False) or getattr(hf_config, 'attention_bias', False)
        
        # AFMoE specific parameters from HF config
        moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", 1408)
        num_dense_layers = getattr(hf_config, "num_dense_layers", 1)
        n_group = getattr(hf_config, "n_group", 1)
        topk_group = getattr(hf_config, "topk_group", 1)
        score_func = getattr(hf_config, "score_func", "sigmoid")
        route_norm = getattr(hf_config, "route_norm", True)
        route_scale = getattr(hf_config, "route_scale", 1.0)
        mup_enabled = getattr(hf_config, "mup_enabled", False)
        seq_length = getattr(hf_config, "seq_length", 8192)

        # MoE configuration
        moe_num_experts = getattr(hf_config, "num_local_experts", 0)
        moe_top_k = getattr(hf_config, "num_experts_per_tok", 0)
        moe_config = MoeConfig(num_experts=moe_num_experts,
                               top_k=moe_top_k,
                               normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE)
        moe_config.validate()

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))
        tie_word_embeddings = getattr(hf_config, 'tie_word_embeddings', False)

        return cls(
            architecture=hf_config.architectures[0],
            dtype=dtype,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_key_value_heads=num_key_value_heads,
            head_size=head_dim,
            vocab_size=hf_config.vocab_size,
            position_embedding_type='rope_gpt_neox',
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_act=hf_config.hidden_act,
            norm_epsilon=getattr(hf_config, 'rms_norm_eps', 1e-5),
            attn_bias=attn_bias,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            sliding_window=sliding_window,
            global_attn_every_n_layers=global_attn_every_n_layers,
            layer_types=layer_types,
            moe_intermediate_size=moe_intermediate_size,
            num_dense_layers=num_dense_layers,
            n_group=n_group,
            topk_group=topk_group,
            score_func=score_func,
            route_norm=route_norm,
            route_scale=route_scale,
            mup_enabled=mup_enabled,
            seq_length=seq_length,
            moe=moe_config,
            mapping=mapping,
            quantization=quant_config,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs)
