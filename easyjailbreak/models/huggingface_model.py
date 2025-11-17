"""
This file contains a wrapper for Huggingface models, implementing various methods used in downstream tasks.
It includes the HuggingfaceModel class that extends the functionality of the WhiteBoxModelBase class.
"""

import sys
from .model_base import WhiteBoxModelBase
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
import functools
import torch
from typing import Optional, Dict, List, Any
import logging


class HuggingfaceModel(WhiteBoxModelBase):
    """
    HuggingfaceModel is a wrapper for Huggingface's transformers models.
    It extends the WhiteBoxModelBase class and provides additional functionality specifically
    for handling conversation generation tasks with various models.
    This class supports custom conversation templates and formatting,
    and offers configurable options for generation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the HuggingfaceModel with a specified model, tokenizer, and generation configuration.

        :param Any model: A huggingface model.
        :param Any tokenizer: A huggingface tokenizer.
        :param Optional[str] model_name: The name of the model being used. If None, will be inferred from model config or tokenizer name.
        :param Optional[Dict[str, Any]] generation_config: A dictionary containing configuration settings for text generation.
            If None, a default configuration is used.
        """
        super().__init__(model, tokenizer)
        
        # Auto-infer model_name if not provided
        if model_name is None:
            # Try to get from model config
            if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
                model_name = model.config._name_or_path
            elif hasattr(model, 'name_or_path'):
                model_name = model.name_or_path
            elif hasattr(tokenizer, 'name_or_path'):
                model_name = tokenizer.name_or_path
            else:
                # Fallback: use a generic name
                model_name = "unknown_model"
        
        self.model_name = model_name
        self.system_message: Optional[str] = None        # NEW: store system message explicitly

        # NEW: detect whether we can use HF chat templates
        self.use_hf_chat_template: bool = bool(
            hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None) is not None
        )  # MODIFIED: use HF chat_template flag

        if not self.use_hf_chat_template:
            logging.warning(
                f"HuggingfaceModel[{model_name}]: tokenizer has no usable chat_template; "
                f"falling back to plain text prompts."
            )

        # We keep a simple format_str only for format()/format_instance fallback
        self.format_str = "{prompt}\n{response}"          # MODIFIED

        if generation_config is None:
            generation_config = {}
        self.generation_config = generation_config

    def set_system_message(self, system_message: str):
        r"""
        Sets a system message to be used in the conversation.

        :param str system_message: The system message to be set for the conversation.
        """
        # MODIFIED: previously wrote to self.conversation.system_message
        self.system_message = system_message

    # REMOVED: create_format_str()
    # def create_format_str(self): ...

    def _build_chat_messages(self, messages) -> List[Dict[str, str]]:
        """
        INTERNAL:
        Build a list of HF-style chat messages from raw text messages.

        messages: str | list[str], alternating user/assistant if len > 1.
        """
        # NEW helper
        if isinstance(messages, str):
            messages = [messages]

        chat: List[Dict[str, str]] = []
        if self.system_message:
            chat.append({"role": "system", "content": self.system_message})

        for idx, msg in enumerate(messages):
            role = "user" if idx % 2 == 0 else "assistant"
            chat.append({"role": role, "content": msg})
        return chat

    def create_conversation_prompt(self, messages, clear_old_history: bool = True):
        r"""
        Constructs a conversation prompt that includes the conversation history.

        :param list[str] messages: A list of messages that form the conversation history.
                                   Messages from the user and the assistant should alternate.
        :param bool clear_old_history: Kept for API compatibility; no effect now.
        :return: A string representing the conversation prompt including the history.
        """
        # MODIFIED:  HF chat_template or simple concat
        chat_messages = self._build_chat_messages(messages)

        if self.use_hf_chat_template:
            # Use tokenizer's Jinja chat template
            prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,  # we want the model to continue as assistant
            )
            return prompt

        # Fallback: very simple plain-text format if no chat_template available
        lines = []
        for m in chat_messages:
            prefix = m["role"].upper() + ": "
            lines.append(prefix + m["content"])
        # add generation prompt marker for assistant
        lines.append("ASSISTANT: ")
        return "\n".join(lines)

    def clear_conversation(self):
        r"""
        Clears the current conversation history.
        """
        # MODIFIED: no persistent history now, kept for API compatibility
        pass

    def generate(self, messages, input_field_name='input_ids', clear_old_history=True, **kwargs):
        r"""
        Generates a response for the given messages within a single conversation.

        :param list[str]|str messages: The text input by the user. Can be a list of messages or a single message.
        :param str input_field_name: The parameter name for the input message in the model's generation function.
        :param bool clear_old_history: If True, clears the conversation history before generating a response.
                                       (No effect now; kept for API compatibility.)
        :param dict kwargs: Optional parameters for the model's generation function, such as 'temperature' and 'top_p'.
        :return: A string representing the pure response from the model, containing only the text of the response.
        """
        # (logic mostly kept, only prompt construction changed)
        if isinstance(messages, str):
            messages = [messages]
        prompt = self.create_conversation_prompt(messages, clear_old_history=clear_old_history)

        # MODIFIED: safer device handling
        device = self.model.device
        input_tokens = self.tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=False
        ).to(device)
        if input_field_name not in input_tokens:
            if 'input_ids' not in input_tokens:
                raise KeyError(f"Tokenizer output missing '{input_field_name}' and 'input_ids'.")
            input_field_name = 'input_ids'
        input_ids = input_tokens[input_field_name]
        input_length = input_ids.shape[1]

        kwargs[input_field_name] = input_ids
        for key in ('attention_mask', 'token_type_ids'):
            if key in input_tokens and key not in kwargs:
                kwargs[key] = input_tokens[key]

        output_ids = self.model.generate(**kwargs, **self.generation_config)
        output = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

        return output

    def batch_generate(self, conversations, **kwargs) -> List[str]:
        r"""
        Generates responses for a batch of conversations.

        :param list[list[str]]|list[str] conversations: A list of conversations. Each conversation can be a list of messages
                                                         or a single message string. If a single string is provided, a warning
                                                         is issued, and the string is treated as a single-message conversation.
        :param dict kwargs: Optional parameters for the model's generation function.
        :return: A list of responses, each corresponding to one of the input conversations.
        """
        # MODIFIED: prompt construction now via create_conversation_prompt / HF chat template
        prompt_list = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn('If you want the model to generate batches based on several conversations, '
                              'please construct a list[str] for each conversation, or they will be divided into individual sentences. '
                              'Switch input type of batch_generate() to list[list[str]] to avoid this warning.')
                conversation = [conversation]
            prompt_list.append(self.create_conversation_prompt(conversation))

        input_tokens = self.tokenizer(
            prompt_list,
            return_tensors='pt',
            padding=True,
            add_special_tokens=False
        )
        device = self.model.device
        input_tokens = {k: v.to(device) for k, v in input_tokens.items()}
        kwargs.update(**input_tokens)

        output_ids = self.model.generate(
            **kwargs, 
            **self.generation_config
        )
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, input_tokens["input_ids"].shape[1]:]
        output_list = self.batch_decode(output_ids, skip_special_tokens=True)

        return output_list

    def __call__(self, *args, **kwargs):
        r"""
        Allows the HuggingfaceModel instance to be called like a function, which internally calls the model's
        __call__ method.

        :return: The output from the model's __call__ method.
        """
        return self.model(*args, **kwargs)

    def tokenize(self, *args, **kwargs):
        r"""
        Tokenizes the input using the model's tokenizer.

        :return: The tokenized output.
        """
        return self.tokenizer.tokenize(*args, **kwargs)

    def batch_encode(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def batch_decode(self, *args, **kwargs) -> List[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def format(self, **kwargs):
        # MODIFIED: simple fallback formatter
        return self.format_str.format(**kwargs)

    def format_instance(self, query, jailbreak_prompt, response):
        """
        Build the *full* prompt+response text for logging / evaluation.

        For HF chat template, we mirror the real conversation format.
        """
        # MODIFIED: support HF chat_template
        prompt_text = jailbreak_prompt.replace('{query}', query)  # 原逻辑保留

        if self.use_hf_chat_template:
            chat_messages: List[Dict[str, str]] = []
            if self.system_message:
                chat_messages.append({"role": "system", "content": self.system_message})
            chat_messages.append({"role": "user", "content": prompt_text})
            chat_messages.append({"role": "assistant", "content": response})

            return self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False,  # 这里是完整 QA，不再需要 generation prompt
            )

        # Fallback: plain text format
        return self.format(prompt=prompt_text, response=response)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    # @functools.cache  # 记忆结果，避免重复计算。不影响子类的重载。
    def embed_layer(self) -> Optional[torch.nn.Embedding]:
        """
        Retrieve the embedding layer object of the model.

        This method provides two commonly used approaches to search for the embedding layer in Hugging Face models.
        If these methods are not effective, users should consider manually overriding this method.

        Returns:
            torch.nn.Embedding or None: The embedding layer object if found, otherwise None.
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                return module

        for module in self.model.modules():
            if all(hasattr(module, attr) for attr in ['bos_token_id', 'eos_token_id', 'encode', 'decode', 'tokenize']):
                return module
        return None

    @property
    # @functools.cache
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.

        This method provides two commonly used approaches for obtaining the vocabulary size in Hugging Face models.
        If these methods are not effective, users should consider manually overriding this method.

        Returns:
            int: The size of the vocabulary.
        """
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
            return self.model.config.vocab_size

        embed = self.embed_layer
        if embed is not None:
            return embed.weight.size(0)
        
        # Fallback: try tokenizer vocab size
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        
        raise ValueError("Unable to determine vocabulary size. Please override vocab_size property.")


def from_pretrained(
    model_name_or_path: str, 
    model_name: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None, 
    **generation_config: Dict[str, Any]
) -> HuggingfaceModel:
    """
    Imports a Hugging Face model and tokenizer with a single function call.

    :param str model_name_or_path: The identifier or path for the pre-trained model.
    :param Optional[str] model_name: The name of the model, used for logging and special model detection.
        If None, will default to `model_name_or_path`. This parameter is mainly for backward compatibility
        and special model identification (e.g., 'chatglm', 'qwen').
    :param Optional[str] tokenizer_name_or_path: The identifier or path for the pre-trained tokenizer.
        Defaults to `model_name_or_path` if not specified separately.
    :param Optional[torch.dtype] dtype: The data type to which the model should be cast.
        Defaults to None (which becomes 'auto').
    :param generation_config: Additional configuration options for model generation.
    :type generation_config: dict

    :return HuggingfaceModel: An instance of the HuggingfaceModel class containing the imported model and tokenizer.

    .. note::
        The model is loaded for evaluation by default. If `dtype` is specified, the model is cast to the specified data type.
        The `tokenizer.padding_side` is set to 'right' if not already specified.
        If the tokenizer has no specified pad token, it is set to the EOS token, and the model's config is updated accordingly.

    **Example**

    .. code-block:: python

        # Simple usage - model_name will default to model_name_or_path
        model = from_pretrained('meta-llama/Llama-3.1-8B-Instruct', dtype=torch.bfloat16, max_length=512)
        
        # Explicit model_name for special model detection
        model = from_pretrained('meta-llama/Llama-3.1-8B-Instruct', model_name='llama-3.1', dtype=torch.bfloat16)
    """
    if dtype is None:
        dtype = 'auto'
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype
    ).eval()
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=True)

    if tokenizer.padding_side is None:
        tokenizer.padding_side = 'right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    # Auto-infer model_name if not provided
    if model_name is None:
        model_name = model_name_or_path

    return HuggingfaceModel(model, tokenizer, model_name=model_name, generation_config=generation_config)
