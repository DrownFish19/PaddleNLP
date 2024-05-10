# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 Databricks Mosaic Research and The HuggingFace Inc. team. All rights reserved.
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
""" DBRX model configuration """

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "DbrxConfig",
]


class DbrxConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DbrxModel`]. It is used to instantiate a Dbrx model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a different configuration to that of the [databricks/dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Dbrx model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DbrxModel`].
                hidden_size (d_model) (`int`, *optional*, defaults to 2048):
            Dimensionality of the embeddings and hidden states.
        intermediate_size (ffn_hidden_size) (`int`, defaults to 3584):
            The hidden size of the feedforward network.
        num_hidden_layers (n_layers) (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (n_heads) (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (ffn_act_fn) (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
            A dict specifying activation function for the FFN.
            The dict should have a key 'name' with the value being the name of the activation function along with
            any additional keyword arguments. If `None`, then set to `{"name": "silu"}`.
        max_position_embeddings (max_seq_len) (`int`, *optional*, defaults to 2048):
            The maximum sequence length of the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
            rope_theta (`float`, defaults to 10000.0):
            The base frequency for rope.
        sliding_window (`int`, *optional*):
            Sliding window attention window size.
        attention_probs_dropout_prob (attn_pdrop) (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        hidden_dropout_prob (resid_pdrop) (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to the attention output before combining with residual.
        num_experts_per_tok (moe_top_k) (`int`, defaults to 1):
            The number of experts to use in the mixture of experts layer.
        num_local_experts (moe_num_experts) (`int`, defaults to 4):
            The number of experts in the mixture of experts layer.
        moe_jitter_eps (`float`, *optional*, defaults to `None`):
            If not `None`, the jitter epsilon for the mixture of experts layer.
        router_aux_loss_coef (moe_loss_weight) (`float`, defaults to 0.01):
            The loss weight for the mixture of experts layer.
        moe_normalize_expert_weights (`float`, *optional*, defaults to 1.0):
            The normalization factor for the expert weights.
            output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details.
        use_fused_rope(`bool`, *optional*, defaults to False):
            Enable rope fusion or not.
        num_key_value_heads (kv_n_heads) (`Optional[int]`, defaults to 1):
            For grouped_query_attention only, allow user to specify number of kv heads.
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.

        clip_qkv (`float`, *optional*):
            If set, clip the queries, keys, and values in the attention layer to this value.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the embedding layer.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    Example:
    ```python
    >>> from paddlenlp.transformer import DbrxConfig, DbrxModel

    >>> # Initializing a Dbrx configuration
    >>> configuration = DbrxConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DbrxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "dbrx"
    attribute_map: Dict[str, str] = {
        "ffn_hidden_size": "intermediate_size",
        "n_layers": "num_hidden_layers",
        "num_attention_heads": "n_heads",
        "hidden_act": "ffn_act_fn",
        "max_position_embeddings": "max_seq_len",
        "attention_probs_dropout_prob": "attn_pdrop",
        "hidden_dropout_prob": "resid_pdrop",
        "num_experts_per_tok": "moe_top_k",
        "moe_num_experts": "num_local_experts",
        "moe_loss_weight": "router_aux_loss_coef",
        "kv_n_heads": "num_key_value_heads",
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=100352,
        hidden_size=6144,
        intermediate_size=10752,
        num_hidden_layers=40,
        num_attention_heads=48,
        hidden_act="silu",
        max_position_embeddings=32768,
        seq_length=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        use_recompute=False,
        recompute_granularity="full",
        no_recompute_layers=None,
        use_flash_attention=False,
        use_fused_rope=False,
        rope_theta=5e5,
        tensor_parallel_output=True,
        sequence_parallel=False,
        fuse_sequence_parallel_allreduce=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        attention_probs_dropout_prob=0.0,
        num_local_experts=16,
        num_experts_per_tok=4,
        moe_jitter_eps=None,
        router_aux_loss_coef=0.05,
        moe_normalize_expert_weights=1.0,
        clip_qkv=8,
        num_key_value_heads=8,
        resid_pdrop=0.0,
        emb_pdrop=0.0,
        output_router_logits=False,
        sliding_window=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.no_recompute_layers = no_recompute_layers
        self.use_flash_attention = use_flash_attention
        self.tensor_parallel_output = tensor_parallel_output
        self.sequence_parallel = sequence_parallel
        self.fuse_sequence_parallel_allreduce = fuse_sequence_parallel_allreduce

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.use_fused_rope = use_fused_rope
        self.rope_theta = rope_theta

        # ----------------- Experts -------------------- #
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_normalize_expert_weights = moe_normalize_expert_weights

        self.sliding_window = sliding_window

        # ----------------- Others -------------------- #
        self.clip_qkv = clip_qkv
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop

        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        if tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported for DBRX models.")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            tensor_parallel_output=tensor_parallel_output,
            **kwargs,
        )
