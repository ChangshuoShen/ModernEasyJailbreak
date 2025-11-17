# alphasteer_model.py
"""
AlphaSteerModel wrapper: LLaMA + AlphaLlamaForCausalLM + steering,
with the same high-level interface as HuggingfaceModel.
"""

import logging
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, LlamaConfig

# 根据你的工程实际路径调整导入
from .AlphaLlama import AlphaLlamaForCausalLM
from .huggingface_model import HuggingfaceModel

logger = logging.getLogger(__name__)


class AlphaSteerModel(HuggingfaceModel):
    """
    AlphaSteerModel: 和 HuggingfaceModel 拥有基本一致的接口，
    但内部模型是 AlphaLlamaForCausalLM（带 steering）。
    """

    def __init__(
        self,
        model: AlphaLlamaForCausalLM,
        tokenizer,
        model_name: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        # 直接复用 HuggingfaceModel 的所有逻辑（chat template, generate, batch_generate, etc.）
        super().__init__(model, tokenizer, model_name=model_name, generation_config=generation_config)

    # 给外部一个显式的 steering 设置入口
    def set_steering_parameters(
        self,
        steering_matrix: Optional[torch.Tensor] = None,
        strength: Optional[List[float]] = None,
    ):
        """
        直接转发到内部 AlphaLlamaForCausalLM 的 set_steering_parameters。
        """
        if not hasattr(self.model, "set_steering_parameters"):
            raise RuntimeError("Underlying model has no `set_steering_parameters` method.")
        self.model.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)


def from_pretrained(
    model_name_or_path: str,
    model_name: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    steering_matrix: Optional[torch.Tensor] = None,
    strength: Optional[List[float]] = None,
    **generation_config: Dict[str, Any],
) -> AlphaSteerModel:
    """
    加载带 AlphaSteer 的 LLaMA 模型，并封装成 AlphaSteerModel。

    参数：
        model_name_or_path: HF 上的 LLaMA 权重路径，例如 'meta-llama/Llama-3.1-8B-Instruct'
        model_name: 用于日志和标识的名字，不填就用 model_name_or_path
        tokenizer_name_or_path: tokenizer 路径，不填就等于 model_name_or_path
        dtype: torch.dtype，比如 torch.bfloat16；None 时传给 HF 的 'auto'
        steering_matrix: 你的 alpha steering 矩阵（一般 shape = [num_layers, D, K] 或类似）
        strength: list[float]，每一层一个 strength
        generation_config: 其他传给 generate 的默认配置，例如 max_new_tokens, temperature, top_p 等

    返回：
        AlphaSteerModel 实例，接口和 HuggingfaceModel 一致。
    """

    if dtype is None:
        dtype = "auto"

    # 1) 加载 config（可选，用于检查 model_type）
    config = LlamaConfig.from_pretrained(model_name_or_path)
    if getattr(config, "model_type", None) != "llama":
        logger.warning(
            f"AlphaSteer is intended for LLaMA-family models, "
            f"but got model_type={config.model_type}. Proceed at your own risk."
        )

    # 2) 加载 AlphaLlamaForCausalLM（带 steering）
    # 注意：这里直接用你的 AlphaLlamaForCausalLM.from_pretrained
    model: AlphaLlamaForCausalLM = AlphaLlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        steering_matrix=steering_matrix,
        strength=strength,
    ).eval()

    # 3) 加载 tokenizer
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    # 尽量和 HuggingfaceModel 保持一致：
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    # 4) 推断 model_name
    if model_name is None:
        model_name = model_name_or_path

    # 5) 封装到 AlphaSteerModel 中
    alphasteer_model = AlphaSteerModel(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        generation_config=generation_config,
    )

    return alphasteer_model
