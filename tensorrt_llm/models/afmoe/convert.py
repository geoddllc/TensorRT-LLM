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

    # AFMoE uses similar naming to LLaMA but with different MoE structure
    model_prefix = "model"
    param_name_map = {
        "vocab_embedding": "embed_tokens" + key_postfix,  # vocab_embedding
        "lm_head": "lm_head" + key_postfix,  # lm_head
        "ln_f": "norm" + key_postfix,  # ln_f
        "attention.qkv": "self_attn",  # attention.qkv
        "qkv_suffix": "_proj" + key_postfix,  # qkv suffix
        "attention.dense": "self_attn.o_proj" + key_postfix,  # attention.dense
        "mlp.gate": "mlp.up_proj" + key_postfix,  # mlp.gate
        "mlp.proj": "mlp.down_proj" + key_postfix,  # mlp.proj
        "mlp.fc": "mlp.gate_proj" + key_postfix,  # mlp.fc
        "input_layernorm": "input_layernorm" + key_postfix,  # input_layernorm
        "post_layernorm": "post_attention_layernorm" + key_postfix,  # post_layernorm
        # AFMoE specific
        "moe_router": "block_sparse_moe.gate" + key_postfix,  # MoE router (never quantized)
        "moe_shared_expert_gate": "block_sparse_moe.shared_expert_gate" + key_postfix,  # Shared expert gate
        "moe_shared_expert_fc": "block_sparse_moe.shared_expert.up_proj" + key_postfix,  # Shared expert FC
        "moe_shared_expert_proj": "block_sparse_moe.shared_expert.down_proj" + key_postfix,  # Shared expert proj
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
            
            if fc_weight_name in weights:
                fc_shape = weights[fc_weight_name].shape
                expected_fc_shape = (num_experts, config.hidden_size, 2 * intermediate_size)
                if fc_shape != expected_fc_shape:
                    logger.warning(f"MoE FC shape mismatch in layer {layer_idx}: expected {expected_fc_shape}, got {fc_shape}")
            
            if proj_weight_name in weights:
                proj_shape = weights[proj_weight_name].shape
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

    model_prefix, layer_prefix, param_name_map = get_prefix_and_param_name_map(
        config.architecture)

    def convert_layer(l):
        """Convert a single layer from HF to TRT-LLM format"""
        prefix = f'{model_prefix}.{layer_prefix}.{l}.'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}.'
        
        # ==============================================
        # ATTENTION LAYER CONVERSION
        # ==============================================
        
        # QKV concatenation (same as LLaMA)
        q_weight = get_weight(
            model_params, prefix + f'{param_name_map["attention.qkv"]}.q_proj',
            dtype)
        k_weight = get_weight(
            model_params, prefix + f'{param_name_map["attention.qkv"]}.k_proj',
            dtype)
        v_weight = get_weight(
            model_params, prefix + f'{param_name_map["attention.qkv"]}.v_proj',
            dtype)

        # Handle GQA for AFMoE
        if not mha_mode:
            if config.num_key_value_heads < mapping.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, config.num_key_value_heads,
                                         mapping.tp_size)
                v_weight = dup_kv_weight(v_weight, config.num_key_value_heads,
                                         mapping.tp_size)
            assert (k_weight.shape[0] %
                    (mapping.tp_size * config.head_size)) == 0
            assert (v_weight.shape[0] %
                    (mapping.tp_size * config.head_size)) == 0

            wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
            wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
            wv = split(v_weight, mapping.tp_size, mapping.tp_rank)

            split_v = torch.concat((wq, wk, wv))
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            split_v = split_qkv_tp(qkv_weight, config.num_attention_heads,
                                   config.hidden_size, mapping.tp_size,
                                   mapping.tp_rank)

        # Handle QKV bias if present
        if prefix + f'{param_name_map["attention.qkv"]}.q_proj.bias' in model_params:
            q_bias = get_bias(
                model_params,
                prefix + f'{param_name_map["attention.qkv"]}.q_proj', dtype)
            k_bias = get_bias(
                model_params,
                prefix + f'{param_name_map["attention.qkv"]}.k_proj', dtype)
            v_bias = get_bias(
                model_params,
                prefix + f'{param_name_map["attention.qkv"]}.v_proj', dtype)
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
        attn_dense_weight = get_weight(
            model_params, prefix + param_name_map["attention.dense"], dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)
        weights[tllm_prex + 'attention.dense.weight'] = split_v

        # ==============================================
        # MoE LAYER CONVERSION (AFMoE Specific)
        # ==============================================
        
        if moe_config.has_moe():
            logger.info(f"Converting MoE layer {l} with {moe_config.num_experts} experts")
            
            # Get rank experts for this layer
            rank_experts = list(range(moe_config.num_experts))
            if mapping.has_moe_ep():
                rank_experts = mapping.ep_experts(moe_config.num_experts)
            
            # ==============================================
            # EXPERT WEIGHTS STACKING (128 experts)
            # ==============================================
            
            # Stack expert weights: w1, w2, w3 for each expert
            expert_w1_weights = []  # gate_proj (first MLP)
            expert_w2_weights = []  # down_proj (output projection)  
            expert_w3_weights = []  # up_proj (second MLP)
            
            for expert_id in rank_experts:
                # HF naming: model.layers.X.block_sparse_moe.experts.{expert_id}.{weight_name}.weight
                expert_prefix = f'{prefix}block_sparse_moe.experts.{expert_id}.'
                
                # Get expert weights
                w1_weight = get_weight(model_params, expert_prefix + 'w1.weight', dtype)  # gate_proj
                w2_weight = get_weight(model_params, expert_prefix + 'w2.weight', dtype)  # down_proj
                w3_weight = get_weight(model_params, expert_prefix + 'w3.weight', dtype)  # up_proj
                
                expert_w1_weights.append(w1_weight)
                expert_w2_weights.append(w2_weight) 
                expert_w3_weights.append(w3_weight)
            
            # Stack expert weights: [num_experts, hidden_size, intermediate_size]
            stacked_w1 = torch.stack(expert_w1_weights, dim=0)  # [num_experts, hidden_size, intermediate_size]
            stacked_w2 = torch.stack(expert_w2_weights, dim=0)  # [num_experts, intermediate_size, hidden_size]
            stacked_w3 = torch.stack(expert_w3_weights, dim=0)  # [num_experts, hidden_size, intermediate_size]
            
            # Apply tensor parallelism to expert weights
            if mapping.has_moe_tp():
                # Split along intermediate_size dimension for TP
                stacked_w1 = split(stacked_w1, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)
                stacked_w2 = split(stacked_w2, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)  
                stacked_w3 = split(stacked_w3, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)
            
            # Concatenate w3 and w1 for gate-up fusion: [num_experts, hidden_size, 2*intermediate_size]
            fc_weight = torch.concat([stacked_w3, stacked_w1], dim=-1)  # gate-up fusion
            proj_weight = stacked_w2
            
            # Store MoE expert weights
            weights[tllm_prex + 'mlp.fc.weight'] = fc_weight.contiguous()
            weights[tllm_prex + 'mlp.proj.weight'] = proj_weight.contiguous()
            
            # ==============================================
            # SHARED EXPERT WEIGHTS (AFMoE Specific)
            # ==============================================
            
            # AFMoE has shared experts that are always active
            shared_expert_fc_weight = get_weight(
                model_params, prefix + 'block_sparse_moe.shared_expert.up_proj', dtype)
            shared_expert_proj_weight = get_weight(
                model_params, prefix + 'block_sparse_moe.shared_expert.down_proj', dtype)
            shared_expert_gate_weight = get_weight(
                model_params, prefix + 'block_sparse_moe.shared_expert_gate', dtype)
            
            # Apply TP to shared expert weights
            if mapping.has_moe_tp():
                shared_expert_fc_weight = split(shared_expert_fc_weight, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)
                shared_expert_proj_weight = split(shared_expert_proj_weight, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1
