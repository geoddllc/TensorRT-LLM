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

from typing import Optional, Union

from ..._utils import pad_vocab_size
from ...functional import (Tensor, is_gated_activation, non_gated_version, recv,
                           send, LayerNormType)
from ...layers import (MLP, MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, LayerNorm, MoeConfig,
                       PositionEmbeddingType, RmsNorm, SharedMoE)
from ...lora_helper import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ...quantization import QuantMode
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .config import AfmoeConfig
from .convert import load_hf_afmoe, load_weights_from_hf_model


def MLPFactory(hidden_size,
               ffn_hidden_size,
               hidden_act,
               bias=True,
               dtype=None,
               moe_config: MoeConfig = MoeConfig(),
               tp_group=None,
               tp_size=1,
               mapping=Mapping(),
               quant_mode=QuantMode(0),
               inner_layernorm=False,
               eps=1e-05,
               # Phase 3: SharedMoE parameters
               use_shared_moe=False,
               shared_expert_intermediate_size=1408,
               use_shared_gate=True):
    """Factory function for creating MLP or MoE layers"""
    if moe_config.has_moe():
        if use_shared_moe and moe_config.shared_expert_intermediate_size > 0:
            # Use SharedMoE for AFMoE
            return SharedMoE(
                moe_config=moe_config,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                hidden_act=hidden_act,
                mapping=mapping,
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                quant_mode=quant_mode,
                use_shared_gate=use_shared_gate,
                use_side_stream=False  # Can be enabled for performance
            )
        else:
            # Use standard MOE
            return MOE(moe_config,
                       hidden_size,
                       ffn_hidden_size,
                       hidden_act,
                       mapping=mapping,
                       bias=bias,
                       dtype=dtype,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       quant_mode=quant_mode)
    else:
        # Use dense MLP
        MLPClass = GatedMLP if is_gated_activation(hidden_act) else MLP
        hidden_act = non_gated_version(hidden_act)
        return MLPClass(
            hidden_size,
            ffn_hidden_size,
            hidden_act,
            bias,
            dtype,
            tp_group,
            tp_size,
            quant_mode,
            inner_layernorm=inner_layernorm,
            eps=eps,
        )


class AfmoeDecoderLayer(Module):

    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank

        # Use RmsNorm like Qwen3
        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        inner_layernorm = config.inner_layernorm if hasattr(
            config, "inner_layernorm") else False
        
        # Determine attention type for this layer
        use_sliding_attention = self._should_use_sliding_attention()
        
        # Qwen3-style attention with GQA support
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,  # Use actual KV heads for GQA
            max_seqlen_for_logn_scaling=config.seq_length,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_rank=tp_rank,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
            # Qwen3: Add Q/K layer normalization
            qk_layernorm=True,  # AFMoE uses Q/K normalization like Qwen3
            layernorm_type=LayerNormType.RmsNorm)

        # Phase 3: Configure MoE parameters for AFMoE
        mlp_hidden_size = config.moe_intermediate_size if hasattr(config, 'moe_intermediate_size') else config.intermediate_size
        self.norm_before_bmm1 = config.norm_before_bmm1 if hasattr(
            config, "norm_before_bmm1") else False

        # Determine if this is a dense layer or MoE layer
        num_dense_layers = getattr(config, 'num_dense_layers', 0)
        is_dense_layer = layer_idx < num_dense_layers
        
        if is_dense_layer:
            # Use dense MLP for the first num_dense_layers
            self.mlp = MLPFactory(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                dtype=config.dtype,
                bias=config.mlp_bias,
                moe_config=MoeConfig(),  # No MoE
                tp_group=tp_group,
                tp_size=tp_size,
                mapping=config.mapping,
                quant_mode=config.quant_mode,
                inner_layernorm=inner_layernorm,
                eps=config.norm_epsilon,
            )
        else:
            # Create MoE configuration for AFMoE
            # AFMoE uses: num_experts=128, top_k=8, num_shared_experts=1, sigmoid routing
            moe_config = MoeConfig(
                num_experts=128,
                top_k=8,
                shared_expert_intermediate_size=getattr(config, 'shared_expert_intermediate_size', 1408),
                normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
            )
            moe_config.validate()

            # Phase 3: Replace dense MLP with SharedMoE
            self.mlp = MLPFactory(
                hidden_size=config.hidden_size,
                ffn_hidden_size=mlp_hidden_size,
                hidden_act=config.hidden_act,
                dtype=config.dtype,
                bias=config.mlp_bias,
                moe_config=moe_config,
                tp_group=tp_group,
                tp_size=tp_size,
                mapping=config.mapping,
                quant_mode=config.quant_mode,
                inner_layernorm=inner_layernorm,
                eps=config.norm_epsilon,
                # Phase 3: SharedMoE parameters
                use_shared_moe=True,
                shared_expert_intermediate_size=getattr(config, 'shared_expert_intermediate_size', 1408),
                use_shared_gate=False  # Disable - HF model doesn't have this weight
            )

        # Use RmsNorm like Qwen3
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

    def _should_use_sliding_attention(self) -> bool:
        """Determine if this layer should use sliding window attention"""
        if not hasattr(self.config, 'layer_types') or not self.config.layer_types:
            # Default: use sliding attention unless it's a global attention layer
            return (self.layer_idx % self.config.global_attn_every_n_layers) != 0
        else:
            # Use explicit layer_types if provided
            if self.layer_idx < len(self.config.layer_types):
                return self.config.layer_types[self.layer_idx] != 'full_attention'
            return True  # Default to sliding attention

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None,
                spec_decoding_params=None):

        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
            norm_before_bmm1=self.norm_before_bmm1)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        # Phase 3: MoE forward pass with routing tensors
        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)

        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class AfmoeModel(Module):

    def __init__(self, config: AfmoeConfig):
        super().__init__()
        self.mapping = config.mapping
        self.position_embedding_type = config.position_embedding_type
        if config.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

            self.embedding_scale = config.embedding_scale

            if config.position_embedding_type == PositionEmbeddingType.learned_absolute:
                self.position_embedding = Embedding(
                    num_embeddings=config.max_position_embeddings,
                    embedding_dim=config.hidden_size,
                    dtype=config.dtype)

        self.layers = DecoderLayerList(AfmoeDecoderLayer, config)

        if config.mapping.is_last_pp_rank():
            # Use RmsNorm like Qwen3
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None,
                lora_params=None,
                spec_decoding_params=None):
        if self.mapping.is_first_pp_rank():
            ptuning_args = [
                prompt_embedding_table, prompt_tasks, prompt_vocab_size
            ] if prompt_embedding_table is not None else []
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
            if self.embedding_scale is not None:
                hidden_states *= self.embedding_scale
            if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
                hidden_states = hidden_states + self.position_embedding(
                    position_ids)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params,
                                    lora_params=lora_params,
                                    spec_decoding_params=spec_decoding_params)
        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class AfmoeForCausalLM(DecoderModelForCausalLM):
    config_class = AfmoeConfig

    def __init__(self, config: AfmoeConfig):
        transformer = AfmoeModel(config)

        if config.mapping.is_last_pp_rank():
            vocab_size_padded = pad_vocab_size(config.vocab_size,
                                               config.mapping.tp_size)
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.trtllm_modules_to_hf_modules = {
            "attn_q": "q_proj",
            "attn_k": "k_proj",
            "attn_v": "v_proj",
            "attn_dense": "o_proj",
            "mlp_h_to_4h": "c_fc",
            "mlp_4h_to_h": "c_proj",
        }
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        ''' Create a AfmoeForCausalLM object from give parameters
        '''
        import transformers

        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
        is_prequantized_to_fp8 = kwargs.pop('is_prequantized_to_fp8', False)

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = AfmoeConfig.from_hugging_face(hf_config_or_dir,
                                               dtype=dtype,
                                               mapping=mapping,
                                               quant_config=quant_config,
                                               **kwargs)
        if is_prequantized_to_fp8:
            # TODO: Implement prequantized weight loading for AFMoE
            raise NotImplementedError("Prequantized weight loading not implemented for AFMoE")
        else:
            if use_preloading:
                weights = {}  # Dummy weights for now
                model = cls(config)
                model.load(weights)
            else:
                hf_model = load_hf_afmoe(hf_model_dir, load_model_on_cpu)
                weights = load_weights_from_hf_model(hf_model, config)
                model = cls(config)
                model.load(weights)
                return model
        return model

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)
