# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddlenlp.transformers import AutoConfig, PretrainedConfig


def activate_memory(config: PretrainedConfig, B=1, S=1024):
    H = config.hidden_size
    H_ = config.intermediate_size
    L = config.num_hidden_layers
    A = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    G = A / num_kv_heads
    base_activate_memory = L * ((32 + 8 / G) * B * S * H + 8 * B * S * H_ + 8 * B * S + 4 * B * A * S)
    activate_memory_size = base_activate_memory * 2  # activate memory
    activate_memory_size = activate_memory_size / pow(2, 30)
    return activate_memory_size


def sft_memory(config: PretrainedConfig, return_base_model_state=False):
    H = config.hidden_size
    H_ = config.intermediate_size
    L = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    G = num_attention_heads / num_kv_heads
    vocab_size = config.vocab_size

    base_model_state = 2 * vocab_size * H + L * (
        2 * H + (2 + 2 / G) * H * H + 3 * H * H_  # layernorm  # attention projection
    )  # mlp

    if return_base_model_state:
        return base_model_state

    model_state = (
        base_model_state * 2  # model prameters fp16 or bf16
        + base_model_state * 2  # model grad
        + base_model_state * 4  # optimizer 1-order momentum fp32
        + base_model_state * 4  # optimizer 2-order momentum fp32
        + base_model_state * 4
    )  # optimizer master weight fp32
    model_state = model_state / pow(2, 30)
    return model_state


def lora_memory(config: PretrainedConfig, R=128):
    """_summary_

    Args:
        config (PretrainedConfig): _description_
        R (int, optional): lora size. Defaults to 128.

    Returns:
        _type_: _description_
    """
    H = config.hidden_size
    H_ = config.intermediate_size
    L = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    G = num_attention_heads / num_kv_heads

    base_model_state = sft_memory(config, return_base_model_state=True)

    base_lora_state = L * (
        2 * H + (2 + 2 / G) * (H * R + R * H) + 3 * (H * R + R * H_)  # layernorm  # attention projection
    )  # mlp

    model_state = (
        base_model_state * 2  # model prameters fp16 or bf16
        + base_lora_state * 2  # lora prameters
        + base_lora_state * 2  # model grad
        + base_lora_state * 4  # optimizer 1-order momentum fp32
        + base_lora_state * 4  # optimizer 2-order momentum fp32
        + base_lora_state * 4
    )  # optimizer master weight fp32
    model_state = model_state / pow(2, 30)
    return model_state


def qlora_memory(config: PretrainedConfig, R=128, algorithm="weight_only_int8"):
    """_summary_

    Args:
        config (PretrainedConfig): _description_
        r_size (int, optional): _description_. Defaults to 128.
        algorithm (str, optional): fp4, nf4, weight_only_int8. Defaults to 'weight_only_int8'.
    """
    H = config.hidden_size
    H_ = config.intermediate_size
    L = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    G = num_attention_heads / num_kv_heads

    base_model_state = sft_memory(config, return_base_model_state=True)

    base_lora_state = L * (
        2 * H + (2 + 2 / G) * (H * R + R * H) + 3 * (H * R + R * H_)  # layernorm  # attention projection
    )  # mlp

    model_state = (
        base_model_state * 2  # model prameters fp16 or bf16
        #    + base_lora_state * 2  # lora prameters
        + base_lora_state * 2  # model grad
        + base_lora_state * 4  # optimizer 1-order momentum fp32
        + base_lora_state * 4  # optimizer 2-order momentum fp32
        + base_lora_state * 4
    )  # optimizer master weight fp32

    if algorithm == "fp4":
        model_state += base_lora_state * 0.5
    elif algorithm == "nf4":
        model_state += base_lora_state * 0.5
    elif algorithm == "weight_only_int8":
        model_state += base_lora_state * 1.0

    model_state = model_state / pow(2, 30)
    return model_state


config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-70B")
sft_model_size = sft_memory(config)
lora_model_size = lora_memory(config, R=128)
qlora_model_size = qlora_memory(config, R=128, algorithm="weight_only_int8")
activate_memory_size = activate_memory(config, B=1, S=512)
print("SFT Model Size:", f"{sft_model_size:.4f}GB")
