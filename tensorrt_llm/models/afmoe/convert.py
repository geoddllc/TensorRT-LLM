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
import copy
import functools
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..._utils import pad_vocab_size, release_gc, str_dtype_to_torch
from ...logger import logger
from ...quantization import QuantAlgo
from ...quantization.quantize import (qserve_pack_reorder_per_channel,
                                      qserve_pack_reorder_per_group,
                                      qserve_quantize_weight_per_channel,
                                      qserve_quantize_weight_per_group)
from ..convert_utils import (dup_kv_weight, generate_int8,
                             get_tllm_linear_weight, iterate_shard_files,
                             load_calib_dataset, load_state_dict,
                             retrieved_layer_index_from_name, smooth_gemm,
                             smooth_gemm_fc1_gate, split, split_matrix_tp,
                             split_qkv_bias_tp, split_qkv_tp)
from ..modeling_utils import PretrainedConfig
from .config import AfmoeConfig


def get_weight(named_params, prefix, dtype):
    """Get weight from named parameters and ensure correct dtype"""
    if prefix + '.weight' not in named_params:
        return None
    weight = named_params[prefix + '.weight']
    if weight.dtype != dtype:
        weight = weight.to(dtype)
    return weight.detach()


def get_bias(named_params, prefix, dtype):
    """Get bias from named parameters and ensure correct dtype"""
    if prefix + '.bias' not in named_params:
        return None
    bias = named_params[prefix + '.bias']
    if bias.dtype != dtype:
        bias = bias.to(dtype)
    return bias.detach()


def get_weight_and_scale(named_params, prefix, dtype, mapping=None, split_scale=False):
    """Get weight and scale with optional TP splitting"""
    if prefix + '.weight_scale' not in named_params:
        return get_weight(named_params, prefix, dtype), None
    else:
        assert named_params[prefix + '.weight'].dtype == torch.float8_e4m3fn
        assert named_params[prefix + '.weight_scale'].dtype == torch.float32
        weight_scale = named_params[prefix + '.weight_scale'].detach()
        if split_scale:
            weight_scale = split(weight_scale,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 dim=0)
        return named_params[prefix +
                            '.weight'].detach(), weight_scale.reshape(-1)


def get_weight_and_bias(named_params, prefix, dtype):
    """Get both weight and bias from named parameters"""
    return get_weight(named_params, prefix, dtype), get_bias(named_params, prefix, dtype)


def get_prefix_and_param_name_map(architecture, use_safetensors=False):
    """Get parameter name mapping for AFMoE model"""
    key_postfix = ""
    if use_safetensors:
        key_postfix = ".weight"

    # AFMoE uses direct naming without model prefix
    model_prefix = "model"
    param_name_map = {
        "vocab_embedding": "embed_tokens" + key_postfix,  # vocab_embedding
        "lm_head": "lm_head" + key_postfix,  # lm_head
        "ln_f": "norm" + key_postfix,  # ln_f
    }
    layer_prefix = 'layers'

    return model_prefix, layer_prefix, param_name_map


def load_hf_afmoe(model_dir: str, load_model_on_cpu: bool = False):
    """Load AFMoE model from HuggingFace"""
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map='auto' if not load_model_on_cpu else 'cpu',
        dtype='auto',
        trust_remote_code=True,
    )
    return model


def validate_tensor_shapes(weights, config: AfmoeConfig):
    """Validate tensor shapes before saving to catch issues early"""
    logger.info("Validating tensor shapes...")

    # Check expected shapes for key tensors
    expected_shapes = {
        'vocab_embedding.weight': (config.vocab_size, config.hidden_size),
        'lm_head.weight': (config.vocab_size, config.hidden_size),
        'ln_f.weight': (config.hidden_size,),
    }

    for weight_name, expected_shape in expected_shapes.items():
        if weight_name in weights:
            actual_shape = weights[weight_name].shape
            if actual_shape != expected_shape:
                logger.warning(f"Shape mismatch for {weight_name}: expected {expected_shape}, got {actual_shape}")

    # Validate MoE-specific shapes if MoE is configured
    if config.moe.has_moe():
        num_experts = config.moe.num_experts
        intermediate_size = config.moe_intermediate_size

        logger.info(f"Validating MoE shapes: {num_experts} experts, intermediate_size={intermediate_size}")

        # Check if expert weights have expected stacked shapes
        for layer_idx in range(config.num_hidden_layers):
            tllm_prex = f'transformer.layers.{layer_idx}.'

            # MoE expert weights should be stacked
            fc_weight_name = tllm_prex + 'mlp.fc.weight'
            proj_weight_name = tllm_prex + 'mlp.proj.weight'

            # Check if this is a dense layer or MoE layer
            is_dense_layer = layer_idx < config.num_dense_layers

            if fc_weight_name in weights:
                fc_shape = weights[fc_weight_name].shape
                if is_dense_layer:
                    # Dense layer: fc_weight shape is (hidden_size, 2 * 3 * intermediate_size) = (1024, 6144)
                    expected_fc_shape = (config.hidden_size, 6 * intermediate_size)
                else:
                    # MoE layer: fc_weight shape is (num_experts, hidden_size, 2 * intermediate_size)
                    expected_fc_shape = (num_experts, config.hidden_size, 2 * intermediate_size)
                if fc_shape != expected_fc_shape:
                    logger.warning(f"MoE FC shape mismatch in layer {layer_idx}: expected {expected_fc_shape}, got {fc_shape}")

            if proj_weight_name in weights:
                proj_shape = weights[proj_weight_name].shape
                if is_dense_layer:
                    # Dense layer: proj_weight shape is (3 * intermediate_size, hidden_size) = (3072, 1024)
                    expected_proj_shape = (3 * intermediate_size, config.hidden_size)
                else:
                    # MoE layer: proj_weight shape is (num_experts, intermediate_size, hidden_size)
                    expected_proj_shape = (num_experts, intermediate_size, config.hidden_size)
                if proj_shape != expected_proj_shape:
                    logger.warning(f"MoE proj shape mismatch in layer {layer_idx}: expected {expected_proj_shape}, got {proj_shape}")


def load_weights_from_hf_model(hf_model,
                               config: AfmoeConfig,
                               act_range: Optional[dict] = None,
                               qkv_para: Optional[dict] = None,
                               smoother: Optional[dict] = None):
    """
    Load weights from HuggingFace AFMoE model to TRT-LLM format

    This function handles:
    - QKV concatenation
    - Expert stacking (128 experts)
    - Shared expert weights
    - Router weights (never quantized)
    - TP + EP splits
    - Tensor shape validation
    """
    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None
    use_gemm_woq_plugin = (not config.disable_weight_only_quant_plugin)
    use_fp8_rowwise = quant_algo in [QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN]

    use_smooth_quant = config.quantization._use_plugin_sq
    per_channel = use_smooth_quant and 'PER_CHANNEL' in quant_algo
    per_token = use_smooth_quant and 'PER_TOKEN' in quant_algo
    int8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.INT8
    fp8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.FP8
    if use_smooth_quant or int8_kv_cache:
        assert act_range is not None
        assert qkv_para is not None
        assert smoother is not None

    weights = {}
    tik = time.time()
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, config.dtype)

    mapping = config.mapping
    moe_config = config.moe
    mha_mode = (config.num_key_value_heads == config.num_attention_heads)
    layers_range = config.mapping.pp_layers(config.num_hidden_layers)
    exclude_layers_id = [0, config.num_hidden_layers - 1]

    print(f"DEBUG: num_hidden_layers={config.num_hidden_layers}")
    print(f"DEBUG: pp_size={config.mapping.pp_size}")
    print(f"DEBUG: layers_range={layers_range}")
    print(f"DEBUG: num_dense_layers={config.num_dense_layers}")

    model_prefix, layer_prefix, param_name_map = get_prefix_and_param_name_map(
        config.architecture)

    def convert_layer(l):
        """Convert a single layer from HF to TRT-LLM format"""
        prefix = f'{model_prefix}.{layer_prefix}.{l}.'
        tllm_prex = f'transformer.layers.{l}.'
        has_moe_experts = l >= config.num_dense_layers  # Layers 2+ have MoE experts
        has_shared_experts = l >= config.num_dense_layers  # Layers 2+ have shared_experts
        has_router = l >= config.num_dense_layers  # Layers 2+ have router

        # ==============================================
        # ATTENTION LAYER CONVERSION
        # ==============================================

        # QKV weights
        q_weight = get_weight(model_params, prefix + 'self_attn.q_proj', dtype)
        k_weight = get_weight(model_params, prefix + 'self_attn.k_proj', dtype)
        v_weight = get_weight(model_params, prefix + 'self_attn.v_proj', dtype)

        # Handle attention: GQA with 8 Q heads and 2 KV heads
        # For GQA, we split Q, K, V separately and concatenate
        wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
        wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
        wv = split(v_weight, mapping.tp_size, mapping.tp_rank)
        split_v = torch.concat((wq, wk, wv))

        # Handle QKV bias if present
        if prefix + 'self_attn.q_proj.bias' in model_params:
            q_bias = get_bias(model_params, prefix + 'self_attn.q_proj', dtype)
            k_bias = get_bias(model_params, prefix + 'self_attn.k_proj', dtype)
            v_bias = get_bias(model_params, prefix + 'self_attn.v_proj', dtype)
            qkv_bias = torch.cat((q_bias, k_bias, v_bias))
            split_bias_v = split_qkv_bias_tp(qkv_bias,
                                             config.num_attention_heads,
                                             config.hidden_size,
                                             mapping.tp_size, mapping.tp_rank)
        else:
            split_bias_v = None

        # Store QKV weights
        weights[tllm_prex + 'attention.qkv.weight'] = split_v
        if split_bias_v is not None:
            weights[tllm_prex + 'attention.qkv.bias'] = split_bias_v

        # Attention dense layer
        attn_dense_weight = get_weight(model_params, prefix + 'self_attn.o_proj', dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)
        weights[tllm_prex + 'attention.dense.weight'] = split_v

        # Q/K norm (attention layer normalization - NOT q_layernorm)
        q_norm_weight = get_weight(model_params, prefix + 'self_attn.q_norm', dtype)
        k_norm_weight = get_weight(model_params, prefix + 'self_attn.k_norm', dtype)
        if q_norm_weight is not None:
            weights[tllm_prex + 'attention.q_layernorm.weight'] = q_norm_weight
        if k_norm_weight is not None:
            weights[tllm_prex + 'attention.k_layernorm.weight'] = k_norm_weight

        # ==============================================
        # MLP LAYER CONVERSION (AFMoE Specific)
        # ==============================================

        if has_moe_experts:
            logger.info(f"Converting MoE layer {l} with {moe_config.num_experts} experts")

            # Get rank experts for this layer
            rank_experts = list(range(moe_config.num_experts))
            if mapping.has_moe_ep():
                rank_experts = mapping.ep_experts(moe_config.num_experts)

            # ==============================================
            # EXPERT WEIGHTS STACKING (128 experts)
            # ==============================================

            # Stack expert weights: gate_proj, down_proj, up_proj for each expert
            expert_w1_weights = []  # gate_proj (first MLP)
            expert_w2_weights = []  # down_proj (output projection)
            expert_w3_weights = []  # up_proj (second MLP)

            for expert_id in rank_experts:
                # HF naming: model.layers.X.mlp.experts.{expert_id}.{weight_name}.weight
                expert_prefix = f'{prefix}mlp.experts.{expert_id}.'

                # Get expert weights
                # HF weights are already in correct shape for TRT-LLM:
                # - gate_proj.weight: (256, 1024) = (intermediate_size, hidden_size) ✓
                # - down_proj.weight: (1024, 256) = (hidden_size, intermediate_size) ✓
                # - up_proj.weight: (256, 1024) = (intermediate_size, hidden_size) ✓
                w1_weight = get_weight(model_params, expert_prefix + 'gate_proj', dtype)  # (intermediate, hidden)
                w2_weight = get_weight(model_params, expert_prefix + 'down_proj', dtype)  # (hidden, intermediate)
                w3_weight = get_weight(model_params, expert_prefix + 'up_proj', dtype)  # (intermediate, hidden)

                expert_w1_weights.append(w1_weight)
                expert_w2_weights.append(w2_weight)
                expert_w3_weights.append(w3_weight)

            # Stack expert weights: [num_experts, intermediate_size, hidden_size]
            # w1 (gate_proj): (256, 1024) -> (128, 256, 1024) ✓
            # w2 (down_proj): (1024, 256) -> (128, 1024, 256) ✓
            # w3 (up_proj): (256, 1024) -> (128, 256, 1024) ✓
            stacked_w1 = torch.stack(expert_w1_weights, dim=0)  # [num_experts, intermediate_size, hidden_size]
            stacked_w2 = torch.stack(expert_w2_weights, dim=0)  # [num_experts, hidden_size, intermediate_size]
            stacked_w3 = torch.stack(expert_w3_weights, dim=0)  # [num_experts, intermediate_size, hidden_size]

            # Apply tensor parallelism to expert weights
            if mapping.has_moe_tp():
                # Split along intermediate_size dimension for TP
                stacked_w1 = split(stacked_w1, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)
                stacked_w2 = split(stacked_w2, mapping.moe_tp_size, mapping.moe_tp_rank, dim=2)
                stacked_w3 = split(stacked_w3, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)

            # GatedMLP uses separate fc and gate weights
            weights[tllm_prex + 'mlp.fc.weight'] = stacked_w1.contiguous()
            weights[tllm_prex + 'mlp.gate.weight'] = stacked_w3.contiguous()
            weights[tllm_prex + 'mlp.proj.weight'] = stacked_w2.contiguous()

        else:
            # Dense MLP for first num_dense_layers (layers 0-1)
            # HF naming: mlp.gate_proj, mlp.up_proj, mlp.down_proj
            # HF weights are already in correct shape for TRT-LLM:
            # - gate_proj.weight: (3072, 1024) = (ffn_hidden_size, hidden_size) ✓
            # - up_proj.weight: (3072, 1024) = (ffn_hidden_size, hidden_size) ✓
            # - down_proj.weight: (1024, 3072) = (hidden_size, ffn_hidden_size) ✓
            gate_weight = get_weight(model_params, prefix + 'mlp.gate_proj', dtype)
            up_weight = get_weight(model_params, prefix + 'mlp.up_proj', dtype)
            down_weight = get_weight(model_params, prefix + 'mlp.down_proj', dtype)

            # Apply TP to fc and gate (split along output dimension)
            if mapping.has_moe_tp():
                gate_weight = split(gate_weight, mapping.moe_tp_size, mapping.moe_tp_rank, dim=0)
                up_weight = split(up_weight, mapping.moe_tp_size, mapping.moe_tp_rank, dim=0)
                down_weight = split(down_weight, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)

            # Store separate fc and gate weights for GatedMLP
            weights[tllm_prex + 'mlp.fc.weight'] = gate_weight.contiguous()
            weights[tllm_prex + 'mlp.gate.weight'] = up_weight.contiguous()
            weights[tllm_prex + 'mlp.proj.weight'] = down_weight.contiguous()

        # ==============================================
        # SHARED EXPERT WEIGHTS (Only for MoE layers)
        # ==============================================

        if has_shared_experts:
            # AFMoE has shared experts that are always active
            # HF naming: mlp.shared_experts.gate_proj, mlp.shared_experts.up_proj, mlp.shared_experts.down_proj
            # HF weights are already in correct shape:
            # - gate_proj.weight: (256, 1024) = (intermediate_size, hidden_size) ✓
            # - up_proj.weight: (256, 1024) = (intermediate_size, hidden_size) ✓
            # - down_proj.weight: (1024, 256) = (hidden_size, intermediate_size) ✓
            # SharedMoE uses regular MLP with fused fc (gate+up concatenated)
            shared_expert_gate_weight = get_weight(
                model_params, prefix + 'mlp.shared_experts.gate_proj', dtype)
            shared_expert_fc_weight = get_weight(
                model_params, prefix + 'mlp.shared_experts.up_proj', dtype)
            shared_expert_proj_weight = get_weight(
                model_params, prefix + 'mlp.shared_experts.down_proj', dtype)

            # Apply TP to shared expert weights
            if mapping.has_moe_tp():
                # fc is split along output dim (ffn_hidden_size * 2)
                shared_expert_gate_weight = split(shared_expert_gate_weight, mapping.moe_tp_size, mapping.moe_tp_rank, dim=0)
                shared_expert_fc_weight = split(shared_expert_fc_weight, mapping.moe_tp_size, mapping.moe_tp_rank, dim=0)
                # proj is split along output dim (hidden_size)
                shared_expert_proj_weight = split(shared_expert_proj_weight, mapping.moe_tp_size, mapping.moe_tp_rank, dim=0)

            # Concatenate gate and up for gate-up fusion: (2 * intermediate, hidden)
            shared_fc_weight = torch.concat([shared_expert_gate_weight, shared_expert_fc_weight], dim=0)

            # Store shared expert weights
            weights[tllm_prex + 'mlp.shared_expert.fc.weight'] = shared_fc_weight.contiguous()
            weights[tllm_prex + 'mlp.shared_expert.proj.weight'] = shared_expert_proj_weight.contiguous()

        # ==============================================
        # ROUTER WEIGHTS (Only for MoE layers)
        # ==============================================

        if has_router:
            # Router weights (never quantized) - HF naming: mlp.router.gate.weight
            router_weight = get_weight(
                model_params, prefix + 'mlp.router.gate', dtype)
            weights[tllm_prex + 'mlp.router.weight'] = router_weight.contiguous()

        # ==============================================
        # LAYER NORMS
        # ==============================================

        # Input layernorm
        input_layernorm_weight = get_weight(
            model_params, prefix + 'input_layernorm', dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_layernorm_weight

        # Post attention layernorm (named post_attention_layernorm in HF, maps to post_layernorm in TRT-LLM)
        post_attention_layernorm_weight = get_weight(
            model_params, prefix + 'post_attention_layernorm', dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_attention_layernorm_weight

    # pp_layers returns a list of layer indices, iterate directly
    for l in tqdm(layers_range, desc="Converting layers"):
        convert_layer(l)

    # ==============================================
    # EMBEDDINGS AND OUTPUT
    # ==============================================

    # Vocab embedding - HF uses "model.embed_tokens"
    vocab_embedding_weight = get_weight(
        model_params, "model.embed_tokens", dtype)
    if vocab_embedding_weight is not None:
        weights['transformer.vocab_embedding.weight'] = vocab_embedding_weight

    # Final layernorm - HF uses "model.norm"
    ln_f_weight = get_weight(
        model_params, "model.norm", dtype)
    if ln_f_weight is not None:
        weights['transformer.ln_f.weight'] = ln_f_weight

    # LM head - HF uses "lm_head"
    lm_head_weight = get_weight(
        model_params, "lm_head", dtype)
    if lm_head_weight is not None:
        weights['lm_head.weight'] = lm_head_weight

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')

    # Validate tensor shapes
    validate_tensor_shapes(weights, config)

    return weights
