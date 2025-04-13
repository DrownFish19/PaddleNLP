# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict
from typing import Tuple

import numpy as np
import paddle

from ..utils.comm_utils import masked_whiten


def compute_gae_advantage_return(
    token_level_rewards: paddle.Tensor,
    values: paddle.Tensor,
    sequence_mask: paddle.Tensor,
    start: int,
    gamma: paddle.Tensor,
    lam: paddle.Tensor,
    use_tgt_len_return: bool = True,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
    # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
    lastgaelam = 0.0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]

    values = values * sequence_mask
    token_level_rewards = token_level_rewards * sequence_mask
    if use_tgt_len_return and start > 0:
        # consistent with Beaver
        # values length is src+tgt-1, start is src-1, return length is tgt
        pass
    elif use_tgt_len_return:
        # values length is tgt, start is 0, return length is tgt
        assert start == 0
    else:
        # values length is src+tgt-1, start is src-1, return length is src+tgt-1
        pass
    for t in reversed(range(start, gen_len)):  # pylint: disable=invalid-name
        next_values = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * next_values - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = paddle.stack(advantages_reversed[::-1], axis=1)

    returns = advantages + values[:, start:].contiguous()

    if not use_tgt_len_return:
        advantages = paddle.concat(
            [
                paddle.zeros([advantages.shape[0], start], dtype=advantages.dtype),
                advantages,
            ],
            axis=-1,
        )
        returns = paddle.concat(
            [
                paddle.zeros([returns.shape[0], start], dtype=returns.dtype),
                returns,
            ],
            axis=-1,
        )

    return advantages.detach(), returns


@paddle.no_grad()
def compute_grpo_advantages(
    rewards: paddle.Tensor,
    index: np.ndarray,
    sequence_mask: paddle.Tensor,
    response_length: int,
    epsilon: float = 1e-6,
):
    """
    计算每个prompt的GRPO优势。

    Args:
        rewards (paddle.Tensor, shape=[batch_size]): 回报，单位为float。
        index (np.ndarray, shape=[batch_size]): 每个样本对应的prompt索引，类型为int。
        sequence_mask (paddle.Tensor, shape=[batch_size, response_length]): 序列掩码，用于标记每个时间步是否有效，类型为bool。
        response_length (int): 每个样本的响应长度。
        epsilon (float, optional, default=1e-6): 避免除以0的值，默认为1e-6。

    Returns:
        rewards (paddle.Tensor, shape=[batch_size, response_length]): GRPO优势，单位为float。

    Raises:
        ValueError (ValueError): 如果没有在给定的prompt索引中有分数。
    """
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    batch_size = rewards.shape[0]

    for i in range(batch_size):
        id2score[index[i]].append(rewards[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = paddle.to_tensor(0.0, dtype=rewards.dtype)
            id2std[idx] = paddle.to_tensor(1.0, dtype=rewards.dtype)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = paddle.mean(paddle.stack(id2score[idx]))
            id2std[idx] = paddle.std(paddle.stack(id2score[idx]))
        else:
            raise ValueError(f"No score in prompt index: {idx}")
    for i in range(batch_size):
        rewards[i] = (rewards[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    rewards = rewards.unsqueeze(-1).tile([1, response_length]) * sequence_mask
    return rewards


@paddle.no_grad()
def compute_reinforce_plus_plus_advantages_and_returns(
    rewards: paddle.Tensor,
    eos_mask: paddle.Tensor,
    gamma: float,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Compute reinforce_plus_plus_advantages_and_returns."""
    length = rewards.shape[-1]
    returns = paddle.zeros_like(rewards)
    running_return = 0
    for t in reversed(range(length)):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        running_return = running_return * eos_mask[:, t]

    advantages = masked_whiten(returns, eos_mask)
    advantages = advantages * eos_mask
    return advantages, returns


def add_kl_divergence_regularization(
    prompt: paddle.Tensor,  # size = (B, S) # pylint: disable=unused-argument
    log_probs: paddle.Tensor,  # size = (B, L)
    ref_log_probs: paddle.Tensor,  # size = (B, L)
    reward_score: paddle.Tensor,  # size = (B,)
    sequence_mask: paddle.Tensor,  # size = (B, L)
    kl_coeff: float,
    clip_range_score: float,
) -> paddle.Tensor:
    """
        计算KL散度迭代增益，并将其添加到回报中。
    参数：
        prompt (paddle.Tensor, shape=(B, S)): 输入序列的prompt，未使用。
        log_probs (paddle.Tensor, shape=(B, L)): 当前预测的log概率分布。
        ref_log_probs (paddle.Tensor, shape=(B, L)): 基线预测的log概率分布。
        reward_score (paddle.Tensor, shape=(B,)): 基于prompt和输出序列的基本奖励得分。
        sequence_mask (paddle.Tensor, shape=(B, L)): 序列的mask，用于确定序列的长度。
    返回值（paddle.Tensor, shape=(B, L)}：
        包含KL散度迭代增益的向量。
    """

    kl_divergence_estimate = -kl_coeff * (log_probs - ref_log_probs)  # size = (B, L)
    rewards = kl_divergence_estimate  # size = (B, L)
    reward_clip = paddle.clip(  # size = (B,)
        reward_score,
        min=-clip_range_score,
        max=clip_range_score,
    )
    # TODO(guosheng): use scatter_add/put_along_axis
    index = paddle.cumsum(sequence_mask.cast(paddle.int64), axis=-1).argmax(-1, keepdim=True)

    rewards = paddle.put_along_axis(
        rewards,
        index,
        reward_clip.unsqueeze(axis=-1),
        axis=-1,
        reduce="add",
    )
    return rewards, kl_divergence_estimate
