# 分布式模型专区 (演示界面)

大家好，PaddleNLP 团队在这里为大家整理了各个模型的分布式结构，方便大家对比参考。


## 1. 模型结构
|  Model  |   Attention    | LMHead | DecoderLayer | PretrainingCriterion | XXX | XXX | XXX | XXX |
|:-------:|:--------------:|:------:|:------------:|:--------------------:|:---:|:---:|:---:|:---:|
|  LLaMA  | LLaMA (TP,SEP) |        |              |        LLaMA         |     |     |     |     |
| LLaMA2  | LLaMA (TP,SEP) |        |              |        LLaMA         |     |     |     |     |
| LLaMA3  | LLaMA (TP,SEP) |        |              |        LLaMA         |     |     |     |     |
|  Qwen   |     LLaMA      |        |              |        LLaMA         |     |     |     |     |
| Qwen1.5 |   LLaMA (TP)   |        |              |        LLaMA         |     |     |     |     |
|  Qwen2  |   LLaMA (TP)   |        |              |        LLaMA         |     |     |     |     |



## 2. 模型模块

### LLaMA Attention
```python
class LlamaAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layerwise_recompute: bool = False):
        super().__init__()

        self.xxx = config.xxxx

    def forward(
        self,
        hidden_states,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        alibi: Optional[paddle.Tensor] = None,
        npu_is_casual: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:

        xxxxxxxx
        xxxxxxxx

```
