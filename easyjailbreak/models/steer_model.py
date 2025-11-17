"""steer_model.py

SteerModel wrapper: LLaMA + SteerLlamaForCausalLM + steering vector injection,
并保持与 HuggingfaceModel 相同的高层接口，方便在 EasyJailbreak 中复用。
"""

import logging
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, LlamaConfig

from .SteerLlama import SteerLlamaForCausalLM
from .huggingface_model import HuggingfaceModel

logger = logging.getLogger(__name__)


class SteerModel(HuggingfaceModel):
    """
    SteerModel: 继承 HuggingfaceModel，但内部模型是 SteerLlamaForCausalLM，
    通过 steering vector 直接在隐藏状态进行干预。
    """

    def __init__(
        self,
        model: SteerLlamaForCausalLM,
        tokenizer,
        model_name: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model, tokenizer, model_name=model_name, generation_config=generation_config)

    def set_steering_parameters(
        self,
        steering_vector: Optional[torch.Tensor] = None,
        strength: Optional[List[float]] = None,
    ):
        """
        对外暴露设置 steering vector 的接口，直接转发给底层 SteerLlamaForCausalLM。
        """
        if not hasattr(self.model, "set_steering_parameters"):
            raise RuntimeError("Underlying model has no `set_steering_parameters` method.")
        self.model.set_steering_parameters(steering_vector=steering_vector, strength=strength)


def from_pretrained(
    model_name_or_path: str,
    model_name: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    steering_vector: Optional[torch.Tensor] = None,
    strength: Optional[List[float]] = None,
    **generation_config: Dict[str, Any],
) -> SteerModel:
    """
    加载带 SteerLlama steering 的模型，并封装成 SteerModel。
    """

    if dtype is None:
        dtype = "auto"

    config = LlamaConfig.from_pretrained(model_name_or_path)
    if getattr(config, "model_type", None) != "llama":
        logger.warning(
            "SteerModel 目前主要用于 LLaMA 系列模型，"
            f"当前 model_type={config.model_type}，请确认兼容性。"
        )

    model: SteerLlamaForCausalLM = SteerLlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        steering_vector=steering_vector,
        strength=strength,
    ).eval()

    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    if model_name is None:
        model_name = model_name_or_path

    return SteerModel(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        generation_config=generation_config,
    )

