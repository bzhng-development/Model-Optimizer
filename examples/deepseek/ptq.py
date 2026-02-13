# Adapted from https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model
from tqdm import tqdm
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export.model_config import KV_CACHE_FP8
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils import (
    is_quantized_column_parallel_linear,
    is_quantized_parallel_linear,
    is_quantized_row_parallel_linear,
)
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader
from modelopt.torch.utils.distributed import ParallelState

deekseep_model = None
weight_dequant = None
act_quant = None
fp8_gemm = None

# HF config field name → DeepSeek ModelArgs field name
_HF_TO_MODELARGS = {
    "hidden_size": "dim",
    "intermediate_size": "inter_dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "first_k_dense_replace": "n_dense_layers",
    "num_attention_heads": "n_heads",
    "num_experts_per_tok": "n_activated_experts",
    "n_group": "n_expert_groups",
    "topk_group": "n_limited_groups",
    "routed_scaling_factor": "route_scale",
    "scoring_func": "score_func",
    "max_position_embeddings": "max_seq_len",
}

# Weight name mapping from HF → DeepSeek (target_name, shard_dim)
_HF_WEIGHT_MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
    "wq_b": ("wq_b", None),
    "wk": ("wk", None),
    "k_norm": ("k_norm", None),
    "weights_proj": ("weights_proj", None),
}


def setup_inference_modules(inference_path=None):
    """Set up the DeepSeek inference modules by adding the inference code to sys.path.

    Args:
        inference_path: Explicit path to the inference code directory. If None,
            auto-detects from DeepSeek-V3.2-Exp/inference or DeepSeek-V3/inference
            relative to this script.
    """
    global deekseep_model, weight_dequant, act_quant, fp8_gemm

    if inference_path is not None:
        path = Path(inference_path)
        if not path.exists():
            raise ValueError(f"Inference path does not exist: {path}")
        sys.path.append(str(path))
    else:
        ds_v3_path = Path(__file__).resolve().parent / "DeepSeek-V3/inference"
        ds_v3_2_path = Path(__file__).resolve().parent / "DeepSeek-V3.2-Exp/inference"
        if ds_v3_2_path.exists():
            sys.path.append(str(ds_v3_2_path))
        elif ds_v3_path.exists():
            sys.path.append(str(ds_v3_path))
        else:
            raise ValueError(
                f"DeepSeek-V3 or DeepSeek-V3.2-Exp not found in {Path(__file__).resolve().parent}."
                " Use --inference_path to specify the inference code directory."
            )

    import model as _deekseep_model
    from ds_kernel import weight_dequant as _weight_dequant
    from kernel import act_quant as _act_quant
    from kernel import fp8_gemm as _fp8_gemm

    deekseep_model = _deekseep_model
    weight_dequant = _weight_dequant
    act_quant = _act_quant
    fp8_gemm = _fp8_gemm


def map_hf_config(config_dict):
    """Translate HF config fields to DeepSeek ModelArgs fields.

    If the config already uses DeepSeek naming (no ``hidden_size`` key), it is
    returned as-is.  Otherwise, known HF field names are mapped and
    ``rope_theta`` is extracted from nested rope config structures.
    """
    if "hidden_size" not in config_dict:
        return config_dict

    mapped = {}
    for key, value in config_dict.items():
        if key in _HF_TO_MODELARGS:
            mapped[_HF_TO_MODELARGS[key]] = value
        elif key in (
            "vocab_size",
            "n_routed_experts",
            "n_shared_experts",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        ):
            # Fields that share the same name in both formats
            mapped[key] = value
        elif key.startswith("index_"):
            # index_n_heads, index_head_dim, index_topk, etc.
            mapped[key] = value

    # Extract rope_theta from nested structures
    if "rope_theta" in config_dict:
        mapped["rope_theta"] = config_dict["rope_theta"]
    elif "rope_scaling" in config_dict and isinstance(config_dict["rope_scaling"], dict):
        rope_cfg = config_dict["rope_scaling"]
        if "rope_theta" in rope_cfg:
            mapped["rope_theta"] = rope_cfg["rope_theta"]

    return mapped


def monkey_patch_deepseek_model():
    gemm_impl: Literal["bf16", "fp8"] = "bf16"
    block_size = 128

    def linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        act_quantizer: TensorQuantizer | None = None,
        weight_quantizer: TensorQuantizer | None = None,
    ) -> torch.Tensor:
        if weight.element_size() > 1:
            if act_quantizer is not None:
                x = act_quantizer(x)
            if weight_quantizer is not None:
                weight = weight_quantizer(weight)
            return F.linear(x, weight, bias)
        elif gemm_impl == "bf16":
            weight = weight_dequant(weight, weight.scale)
            if act_quantizer is not None:
                x = act_quantizer(x)
            if weight_quantizer is not None:
                weight = weight_quantizer(weight)

            return F.linear(x, weight, bias)
        else:
            assert weight_quantizer is None
            assert act_quantizer is None
            x, scale = act_quant(x, block_size)
            y = fp8_gemm(x, scale, weight, weight.scale)
            if bias is not None:
                y += bias
            return y

    class QuantColumnParallelLinear(deekseep_model.ColumnParallelLinear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._setup()

        def _setup(self):
            self.input_quantizer = TensorQuantizer()
            self.weight_quantizer = TensorQuantizer()
            # Use TP parallel state
            self._parallel_state = ParallelState(data_parallel_group=-1, tensor_parallel_group=None)
            self._is_column_parallel = True

            assert is_quantized_column_parallel_linear(self)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = linear(
                x,
                self.weight,
                self.bias,
                act_quantizer=self.input_quantizer,
                weight_quantizer=self.weight_quantizer,
            )
            return y

    class QuantRowParallelLinear(deekseep_model.RowParallelLinear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._setup()

        def _setup(self):
            self.input_quantizer = TensorQuantizer()
            self.weight_quantizer = TensorQuantizer()
            # Use TP parallel state
            self._parallel_state = ParallelState(data_parallel_group=-1, tensor_parallel_group=None)
            self._is_row_parallel = True

            assert is_quantized_row_parallel_linear(self)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = linear(
                x,
                self.weight,
                act_quantizer=self.input_quantizer,
                weight_quantizer=self.weight_quantizer,
            )
            if deekseep_model.world_size > 1:
                dist.all_reduce(y)
            if self.bias is not None:
                y += self.bias
            return y

    class QuantLinear(deekseep_model.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._setup()

        def _setup(self):
            self.input_quantizer = TensorQuantizer()
            self.weight_quantizer = TensorQuantizer()
            # No parallel state.
            self._parallel_state = ParallelState(data_parallel_group=-1, tensor_parallel_group=-1)

            assert not is_quantized_parallel_linear(self)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = linear(
                x,
                self.weight,
                self.bias,
                act_quantizer=self.input_quantizer,
                weight_quantizer=self.weight_quantizer,
            )
            return y

    class QuantMLA(deekseep_model.MLA):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._setup()

        def _setup(self):
            self.kv_bmm_quantizer = TensorQuantizer()
            self.pe_bmm_quantizer = TensorQuantizer()

    class CalibMoe(deekseep_model.MoE):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._setup()

        def _setup(self):
            self._original_topk = self.gate.topk
            self._original_topk_groups = self.gate.topk_groups

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Forward all tokens to all experts for calibration
            self.gate.topk = self.n_routed_experts
            self.gate.topk_groups = self.gate.n_groups
            super().forward(x)
            # Restore the original topk and topk_groups
            self.gate.topk = self._original_topk
            self.gate.topk_groups = self._original_topk_groups

            return super().forward(x)

    mtq.register(
        original_cls=deekseep_model.RowParallelLinear,
        quantized_cls=QuantRowParallelLinear,
    )
    mtq.register(
        original_cls=deekseep_model.ColumnParallelLinear,
        quantized_cls=QuantColumnParallelLinear,
    )
    mtq.register(original_cls=deekseep_model.Linear, quantized_cls=QuantLinear)
    mtq.register(original_cls=deekseep_model.MLA, quantized_cls=QuantMLA)
    mtq.register(original_cls=deekseep_model.MoE, quantized_cls=CalibMoe)


def _convert_hf_name(name):
    """Convert a single HF weight name to DeepSeek format.

    Returns (converted_name, shard_dim) or None if the name should be skipped.
    """
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    key = name.split(".")[-2]
    if key not in _HF_WEIGHT_MAPPING:
        return None
    new_key, dim = _HF_WEIGHT_MAPPING[key]
    name = name.replace(key, new_key)
    return name, dim


def _load_hf_sharded_checkpoint(model, model_path, world_size, rank):
    """Load an HF sharded safetensors checkpoint, converting names and sharding in-memory."""
    from safetensors.torch import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    n_layers = len(model.layers)
    n_routed_experts = model.args.n_routed_experts
    n_local_experts = n_routed_experts // world_size

    # Collect unique shard files
    shard_files = sorted(set(index["weight_map"].values()))

    state_dict = {}
    for shard_file in tqdm(shard_files, desc="Loading HF shards"):
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_name in f.keys():
                # Skip MTP layers (layers beyond the main model layers)
                layer_match = _extract_layer_idx(hf_name)
                if layer_match is not None and layer_match >= n_layers:
                    continue

                result = _convert_hf_name(hf_name)
                if result is None:
                    continue
                name, dim = result
                param = f.get_tensor(hf_name)

                # Shard for model parallelism
                if "experts" in name and "shared_experts" not in name:
                    idx = int(name.split(".")[-3])
                    if idx < rank * n_local_experts or idx >= (rank + 1) * n_local_experts:
                        continue
                elif dim is not None and world_size > 1:
                    assert param.size(dim) % world_size == 0
                    shard_size = param.size(dim) // world_size
                    param = param.narrow(dim, rank * shard_size, shard_size).contiguous()

                state_dict[name] = param

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected}")
    if missing:
        # Filter out known-optional missing keys (e.g. gate bias for non-7168 dim)
        important_missing = [k for k in missing if "freq" not in k]
        if important_missing:
            print(f"Warning: missing keys in checkpoint: {important_missing}")


def _extract_layer_idx(name):
    """Extract the layer index from a parameter name like 'model.layers.42.attn...'."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def load_deepseek_model(model_config: str, model_path: str, batch_size: int):
    """Loads the deepseek model to memory."""
    # get distributed info
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    # run with bf16
    torch.set_default_dtype(torch.bfloat16)

    # get config and build model
    with open(model_config) as f:
        config_dict = json.load(f)
    config_dict = map_hf_config(config_dict)
    model_args = deekseep_model.ModelArgs(**config_dict)
    model_args.max_batch_size = max(batch_size, model_args.max_batch_size)
    with torch.device("cuda"):
        model = deekseep_model.Transformer(model_args)

    # monkey path the model definition for quantization
    monkey_patch_deepseek_model()

    # Detect checkpoint format
    hf_index = os.path.join(model_path, "model.safetensors.index.json")
    per_rank_ckpt = os.path.join(model_path, f"model{rank}-mp{world_size}.safetensors")

    if os.path.exists(per_rank_ckpt):
        # DeepSeek per-rank format
        print(f"Loading {per_rank_ckpt}")

        # Temporary fix for fp32 params
        fp32_params = {}
        for name, param in model.named_parameters():
            if param.dtype == torch.float32 and (
                "head.weight" in name or "attn.indexer.weights_proj.weight" in name
            ):
                param.data = param.data.to(torch.get_default_dtype())
                fp32_params[name] = param
        load_model(model, per_rank_ckpt)
        for param in fp32_params.values():
            param.data = param.data.to(torch.float32)
        print(f"Loaded {per_rank_ckpt}")
    elif os.path.exists(hf_index):
        # HF sharded safetensors format
        print(f"Loading HF sharded checkpoint from {model_path}")
        _load_hf_sharded_checkpoint(model, model_path, world_size, rank)
        print(f"Loaded HF sharded checkpoint from {model_path}")
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {model_path}. Expected either "
            f"'{os.path.basename(per_rank_ckpt)}' (DeepSeek format) or "
            f"'model.safetensors.index.json' (HF format)."
        )

    return model


def ptq(
    model,
    tokenizer,
    quant_cfg: str,
    batch_size: int,
    calib_size: int,
    mla_quant: str | None = None,
):
    """Runs Deepseek model PTQ and returns the quantized model."""

    # quantize the model
    ## create dataset
    device = next(model.parameters()).device
    calib_dataset = get_dataset_dataloader(
        dataset_name=["cnn_dailymail", "nemotron-post-training-dataset-v2"],
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=[calib_size, calib_size],
        device=device,
    )

    ## define calib loop
    def calibrate_loop(model):
        for data in tqdm(calib_dataset):
            model(data["input_ids"])

    ## handle DeepSeek model structures
    transformer = model.model if hasattr(model, "model") else model

    # make sure all processes are ready before starting the calibration
    dist.barrier()

    ## quant config
    mtq_cfg = getattr(mtq, quant_cfg)

    # disable head that corresponds to lm_head (for the huggingface checkpoint)
    mtq_cfg["quant_cfg"]["*head*"] = {"enable": False}

    allowed_mla_quant = [None, "per_tensor_fp8", "nvfp4"]
    assert mla_quant in allowed_mla_quant, f"mla_quant must be {allowed_mla_quant}"

    if not mla_quant:
        mtq_cfg["quant_cfg"]["*attn*"] = {"enable": False}
    elif mla_quant == "per_tensor_fp8":
        mtq_cfg["quant_cfg"]["*attn*weight_quantizer"] = {"num_bits": (4, 3), "axis": None}
        mtq_cfg["quant_cfg"]["*attn*input_quantizer"] = {"num_bits": (4, 3), "axis": None}
    elif mla_quant == "nvfp4":  # for DeepSeek-R1-0528-NVFP4-Turbo
        mla_linear_layers = ["*wq_a*", "*wq_b*", "*wkv_a*", "*wkv_b*", "*wo*"]
        mla_nvfp4_linear_layers = ["*wq_a*", "*wkv_a*", "*wq_b*", "*wo*"]
        for layer in mla_linear_layers:
            if layer in mla_nvfp4_linear_layers:
                # wq_a, wkv_a, wq_b, wo use NVFP4 quantization
                mtq_cfg["quant_cfg"][layer + "_quantizer"] = {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    "axis": None,
                    "enable": True,
                }
            else:
                mtq_cfg["quant_cfg"][layer + "_quantizer"] = {"enable": False}

        # Disable BMM quantizers
        mtq_cfg["quant_cfg"]["*attn.kv_bmm_quantizer*"] = {"enable": False}
        mtq_cfg["quant_cfg"]["*attn.pe_bmm_quantizer*"] = {"enable": False}

    if not args.disable_wo_quant and "FP4" in quant_cfg:
        mtq_cfg["quant_cfg"]["*wo*weight_quantizer"] = mtq_cfg["quant_cfg"]["*input_quantizer"]
        mtq_cfg["quant_cfg"]["*wo*input_quantizer"] = mtq_cfg["quant_cfg"]["*weight_quantizer"]

    ## ptq
    transformer = mtq.quantize(transformer, mtq_cfg, calibrate_loop)

    if int(os.environ["LOCAL_RANK"]) == 0:
        mtq.print_quant_summary(transformer)

    return model


def save_amax_and_quant_config(model, output_path: str, enable_fp8_kvcache: bool = True):
    """Saves the amax values of the model to the output path."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))

    if rank == 0 and not os.path.exists(output_path):
        os.mkdir(output_path)

    dist.barrier()

    # save amax
    def state_dict_filter(state_dict):
        return {key: value for key, value in state_dict.items() if "amax" in key or "quant" in key}

    # save quantization results
    torch.save(
        state_dict_filter(model.state_dict()),
        os.path.join(output_path, f"amax_dict_rank{rank}-mp{world_size}.pt"),
    )

    # if rank == 0:
    #     with open("expert_activation_counts.txt", "w") as f:
    #         for name, module in model.named_modules():
    #             if isinstance(module, deekseep_model.MoE):
    #                 counts = module.activated_expert_counts()
    #                 f.writelines(f"{name}: {count}\n" for count in counts)

    quant_config = get_quant_config(model)

    if enable_fp8_kvcache:
        quant_config["quantization"]["kv_cache_quant_algo"] = KV_CACHE_FP8

    all_quant_configs = [None] * dist.get_world_size()
    dist.all_gather_object(all_quant_configs, quant_config)

    if rank == 0:
        exclude_modules = set()
        quantized_layers = {}

        for quant_config_rank in all_quant_configs:
            assert quant_config_rank is not None
            if "exclude_modules" in quant_config_rank["quantization"]:
                exclude_modules.update(quant_config_rank["quantization"]["exclude_modules"])
            if "quantized_layers" in quant_config_rank["quantization"]:
                quantized_layers.update(quant_config_rank["quantization"]["quantized_layers"])

        if exclude_modules:
            quant_config["quantization"]["exclude_modules"] = sorted(exclude_modules)
            # add the last layer to the exclude module as the mtp is not loaded in the quantized model
            quant_config["quantization"]["exclude_modules"].append(f"layers.{len(model.layers)}*")
        if quantized_layers:
            quant_config["quantization"]["quantized_layers"] = quantized_layers

        with open(os.path.join(output_path, "hf_quant_config.json"), "w") as f:
            json.dump(quant_config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model_path", type=str, required=True, help="path to converted FP8 ckpt")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_671B.json",
        help="config file for the model.",
    )
    parser.add_argument("--quant_cfg", type=str, required=True, help="target quantization config.")
    parser.add_argument(
        "--output_path", type=str, required=True, help="target quantization config."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for quantization.")
    parser.add_argument("--calib_size", type=int, default=512, help="samples for calibration.")
    parser.add_argument("--disable_fp8_kvcache", action="store_true", help="disable fp8 kvcache.")
    parser.add_argument("--disable_wo_quant", action="store_true", help="disable MLA wo quant.")
    parser.add_argument("--trust_remote_code", action="store_true", help="trust remote code.")
    parser.add_argument(
        "--mla_quant",
        type=str,
        default=None,
        help="MLA quantization type: None (disable), per_tensor_fp8, nvfp4",
    )
    parser.add_argument(
        "--inference_path",
        type=str,
        default=None,
        help="Path to the DeepSeek inference code directory (containing model.py, kernel.py). "
        "If not provided, auto-detects from DeepSeek-V3.2-Exp/inference or DeepSeek-V3/inference.",
    )

    args = parser.parse_args()
    setup_inference_modules(args.inference_path)
    model = load_deepseek_model(args.config, args.model_path, args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model = ptq(model, tokenizer, args.quant_cfg, args.batch_size, args.calib_size, args.mla_quant)
    save_amax_and_quant_config(model, args.output_path, not args.disable_fp8_kvcache)
