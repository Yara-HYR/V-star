# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import copy
import json
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sized, Union
from unittest.mock import patch

import torch
import numpy as np
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl import apply_chat_template, is_conversational, maybe_apply_chat_template
# from trl import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl import SyncRefModelCallback
from trl import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

import random

from transformers import (
        is_wandb_available, 
        AutoTokenizer, 
        AutoModelForCausalLM,
        TemperatureLogitsWarper, 
        LogitsProcessorList,
        Trainer
    )

from LogitProcessor import ConstrainedLogitsProcessor
from transformers.generation import LogitsProcessor
import math

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_vllm_available():
    # from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class ValueHead(nn.Module):
    """Lightweight Transformer encoder that produces per-token scalar values.

    Architecture: prepend a learnable CLS token, run through a shallow
    TransformerEncoder, then project every position to a scalar.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.value_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)
        cls = self.cls_embed.expand(batch_size, -1, -1)
        enc_inp = torch.cat([cls, hidden_states], dim=1)
        cls_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
        attn_mask = torch.cat([cls_mask, attention_mask], dim=1)
        src_key_padding_mask = attn_mask == 0
        enc_out = self.encoder(enc_inp, src_key_padding_mask=src_key_padding_mask)
        token_values = self.value_proj(enc_out[:, 1:, :]).squeeze(-1)
        cls_value = self.value_proj(enc_out[:, 0, :]).squeeze(-1)
        return token_values, cls_value


class EmaValueHead(nn.Module):
    """Value head variant used for EMA-based target value estimation.

    Same encoder backbone as *ValueHead* but uses a two-layer MLP for the
    value projection (controlled by *hidden_mult*).
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        hidden_mult: int,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        hidden_mid = hidden_size * max(1, int(hidden_mult))
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_mid),
            nn.GELU(),
            nn.Linear(hidden_mid, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)
        cls = self.cls_embed.expand(batch_size, -1, -1)
        enc_inp = torch.cat([cls, hidden_states], dim=1)
        cls_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
        attn_mask = torch.cat([cls_mask, attention_mask], dim=1)
        src_key_padding_mask = attn_mask == 0
        enc_out = self.encoder(enc_inp, src_key_padding_mask=src_key_padding_mask)
        token_values = self.value_proj(enc_out[:, 1:, :]).squeeze(-1)
        cls_value = self.value_proj(enc_out[:, 0, :]).squeeze(-1)
        return token_values, cls_value


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class ReReTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method adapted to recommendation. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        base_model: str,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,

        #* sample
        add_gt: bool = False,
        dynamic_sampling: bool = False,
        beam_search: bool = False,
        length_penalty: float = 0.0,
        #* eval
        test_during_training: bool = True,
        test_beam: int = 20,

        #*loss
        dapo: bool = False,
        gspo: bool = False,

        #* value head (from mcts2)
        value_head_layers: int = 2,
        value_head_heads: int = 4,
        value_head_dropout: float = 0.1,
        value_head_weight: float = 0.0,
        value_td_gamma: float = 0.99,
        freeze_lm: bool = False,
        value_index_path: Optional[str] = None,
        value_emb_path: Optional[str] = None,
        ema_value_head_weight: float = 0.0,
        ema_value_head_hidden_mult: int = 2,

        #* sibling-grpo (V-STAR Sec 4.3)
        use_sibling_grpo: bool = False,
        sibling_loss_weight: float = 1.0,

        #* VED (V-STAR Sec 4.2)
        use_ved: bool = False,
        ved_budget_multiplier: float = 1.0,
        ved_init_beam: int = 8,
        ved_lambda: float = 0.1,
        ved_beta_ucb: float = 1.0,

        #* others
        info_file: str = None,
        # logits_processor: Optional[LogitsProcessor] = None,
        prompt2history: dict[str, str] = None,
        history2target: dict[str, str] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        self.base_model = base_model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(self.base_model, padding_side="left")
            processing_class.pad_token = processing_class.eos_token


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        print(f"max_completion_length: {self.max_completion_length}")
        self.num_generations = args.num_generations  # = G in the GRPO paper 
        self.use_vllm = args.use_vllm

        self.beta = args.beta
        

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)   
        self.log_completions = args.log_completions

        # Value head setup (from mcts2)
        self.value_head_layers = int(max(1, value_head_layers))
        self.value_head_heads = int(max(1, value_head_heads))
        self.value_head_dropout = float(value_head_dropout)
        self.value_head_weight = float(value_head_weight)
        self.value_td_gamma = float(value_td_gamma)
        self.freeze_lm = bool(freeze_lm)
        self.value_index_path = value_index_path
        self.value_emb_path = value_emb_path
        self.ema_value_head_weight = float(ema_value_head_weight)
        self.ema_value_head_hidden_mult = int(max(1, ema_value_head_hidden_mult))
        self._value_item_index = None
        self._value_item_emb = None
        self._value_item_mean = None

        # Build value head on the model before accelerator wrapping
        if self.value_head_weight > 0:
            self._build_value_head(model)
            if isinstance(model_id, str):
                self._load_value_head_weights(model, model_id)

        # Freeze backbone if requested (only value head remains trainable)
        if self.freeze_lm:
            for p in model.parameters():
                p.requires_grad_(False)
            head = getattr(model, "value_head", None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad_(True)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.prompt2history = prompt2history
        self.history2target = history2target
        self.add_gt = add_gt
        self.beam_search = beam_search
        self.info_file = info_file
        self.temperature = args.temperature
        self.length_penalty = length_penalty
        self.test_during_training = test_during_training
        self.test_beam = test_beam
        self.dynamic_sampling = dynamic_sampling
        self.dapo = dapo
        self.gspo = gspo

        # Initialize value embeddings (needs accelerator, so after super().__init__)
        if self.value_head_weight > 0 and self.value_index_path and self.value_emb_path:
            self._init_value_embeddings()

        # Sibling-GRPO setup
        self.use_sibling_grpo = bool(use_sibling_grpo)
        self.sibling_loss_weight = float(sibling_loss_weight)

        # VED setup (V-STAR Sec 4.2)
        self.use_ved = bool(use_ved)
        self.ved_budget_multiplier = float(ved_budget_multiplier)
        self.ved_init_beam = int(ved_init_beam)
        self.ved_lambda = float(ved_lambda)
        self.ved_beta_ucb = float(ved_beta_ucb)

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            if self.beam_search:
                 #* temperature 默认为 1.0
                print(f"self.temperature: {self.temperature}")
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    length_penalty=self.length_penalty,
                    num_beams=self.num_generations,
                    num_return_sequences=self.num_generations,
                    pad_token_id=processing_class.pad_token_id,
                    eos_token_id=processing_class.eos_token_id,
                    top_k=None,
                    top_p=None,
                    temperature=self.temperature,
                    do_sample=True,
                    # temperature=self.temperature,
                    # do_sample=True, # if self.temperature > 1.0 else False,
                )
            else:
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    length_penalty=self.length_penalty,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=processing_class.pad_token_id,
                    eos_token_id=processing_class.eos_token_id,
                )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            # print("Sync Begin")
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        
        with open(self.info_file, 'r') as f:
            info = f.readlines()
            # Parse new format: semantic_id \t item_title \t item_id
            semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
            item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
            
            # Format for tokenization
            info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
            info_titles = [f'''### Response:\n{_}''' for _ in item_titles]

            info = info_semantic

        # with open(self.info_file, 'r') as f:
        #     info = f.readlines()
        #     info = ["\"" + _[:-len(_.split('\t')[-1])].strip() + "\"\n" for _ in info]
        #     info = [f'''### Response:\n{_}''' for _ in info]

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.base_model.lower().find("llama") > -1: 
            prefixID = [tokenizer(_).input_ids[1:] for _ in info]
        else:
            prefixID = [tokenizer(_).input_ids for _ in info]
        
        if self.base_model.lower().find("gpt2") > -1:
            prefix_index = 4
        else:
            prefix_index = 3
            
        self.hash_dict = dict()
        # sasrec_dict = dict()
        for index, ID in enumerate(prefixID):
            ID.append(tokenizer.eos_token_id)
            for i in range(prefix_index, len(ID)):
                if i == prefix_index:
                    hash_number = self.get_hash(ID[:i])
                else:
                    hash_number = self.get_hash(ID[prefix_index:i])
                if hash_number not in self.hash_dict:
                    self.hash_dict[hash_number] = set()
                    # sasrec_dict[hash_number] = set()
                self.hash_dict[hash_number].add(ID[i])

        for key in self.hash_dict.keys():
            self.hash_dict[key] = list(self.hash_dict[key])

        self.test_generation_config = GenerationConfig(max_new_tokens=self.max_completion_length,
                                                            length_penalty=self.length_penalty,
                                                            num_beams=self.test_beam,
                                                            num_return_sequences=self.test_beam,
                                                            do_sample=False,
                                                            top_k=None,
                                                            top_p=None,
                                                            pad_token_id=self.processing_class.pad_token_id,
                                                            eos_token_id=self.processing_class.eos_token_id,)

    def get_hash(self, x):
            x = [str(_) for _ in x]
            return '-'.join(x)

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
            hash_number = self.get_hash(input_ids)
            if hash_number in self.hash_dict:
                return self.hash_dict[hash_number]
            return []
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # def _get_train_sampler(self,  *args, **kwargs) -> Sampler:
    #     # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
    #     # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
    #     # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
    #     # preventing discrepancies in group formation.
    #     sampler = super()._get_train_sampler(*args, **kwargs)
    #     return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)
    
    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        if train_dataset is None:
            train_dataset = self.train_dataset
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # ── Value Model methods (from mcts2) ──────────────────────────────────

    def _build_value_head(self, backbone: PreTrainedModel):
        """Attach a ValueHead to the backbone model."""
        if hasattr(backbone, "value_head"):
            return
        hidden = int(backbone.config.hidden_size)
        num_heads = min(self.value_head_heads, max(1, hidden // 8))
        value_head = ValueHead(
            hidden_size=hidden,
            num_layers=self.value_head_layers,
            num_heads=num_heads,
            dropout=self.value_head_dropout,
        )
        value_head.to(next(backbone.parameters()).device)
        for p in value_head.parameters():
            p.requires_grad_(True)
        backbone.value_head = value_head

    def _build_ema_value_head(self, backbone: PreTrainedModel):
        """Attach an EmaValueHead to the backbone model."""
        if hasattr(backbone, "ema_value_head"):
            return
        hidden = int(backbone.config.hidden_size)
        num_heads = min(self.value_head_heads, max(1, hidden // 8))
        ema_value_head = EmaValueHead(
            hidden_size=hidden,
            num_layers=self.value_head_layers,
            num_heads=num_heads,
            dropout=self.value_head_dropout,
            hidden_mult=self.ema_value_head_hidden_mult,
        )
        ema_value_head.to(next(backbone.parameters()).device)
        for p in ema_value_head.parameters():
            p.requires_grad_(True)
        backbone.ema_value_head = ema_value_head

    def _init_value_embeddings(self) -> None:
        """Load item embeddings and SID-token-to-item-index mapping for dense supervision."""
        if self.value_index_path is None or self.value_emb_path is None:
            return
        with open(self.value_index_path, "r") as f:
            index_data = json.load(f)
        emb = np.load(self.value_emb_path)
        emb_tensor = torch.tensor(emb, device=self.accelerator.device, dtype=torch.float32)
        self._value_item_emb = emb_tensor
        self._value_item_mean = emb_tensor.mean(dim=0)
        value_item_index = {}
        for key, tokens in index_data.items():
            if not isinstance(tokens, list) or len(tokens) < 3:
                continue
            tok_ids = self.processing_class.convert_tokens_to_ids(tokens[:3])
            if any(tok is None for tok in tok_ids):
                continue
            value_item_index[tuple(int(tok) for tok in tok_ids)] = int(key)
        self._value_item_index = value_item_index
        print(
            f"[VALUE] Loaded embeddings={tuple(emb_tensor.shape)} "
            f"index_entries={len(self._value_item_index)}"
        )

    def _lookup_item_embeddings(self, token_triplets: torch.Tensor) -> torch.Tensor:
        """Map SID token triplets to item embeddings via the pre-loaded index."""
        if self._value_item_emb is None or self._value_item_mean is None or self._value_item_index is None:
            emb_dim = int(self._value_item_emb.size(1)) if self._value_item_emb is not None else 1
            return torch.zeros(
                (token_triplets.size(0), emb_dim),
                device=token_triplets.device,
            )
        embeddings: List[torch.Tensor] = []
        for row in token_triplets.tolist():
            idx = self._value_item_index.get(tuple(int(tok) for tok in row))
            if idx is None:
                embeddings.append(self._value_item_mean)
            else:
                embeddings.append(self._value_item_emb[idx])
        return torch.stack(embeddings, dim=0)

    def _load_value_head_weights(self, model: PreTrainedModel, model_id: str) -> None:
        """Try to load pre-trained value_head weights from a checkpoint directory."""
        if not model_id or not os.path.isdir(model_id):
            return
        safetensors_path = os.path.join(model_id, "model.safetensors")
        pt_path = os.path.join(model_id, "pytorch_model.bin")
        state_dict = None
        if os.path.exists(safetensors_path):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
            except Exception:
                state_dict = None
        if state_dict is None and os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location="cpu")
        if not state_dict:
            return
        value_state = {k: v for k, v in state_dict.items() if k.startswith("value_head.")}
        if value_state:
            missing, unexpected = model.load_state_dict(value_state, strict=False)
            print(
                f"[VALUE] Loaded value_head weights from {model_id} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    # ── End Value Model methods ───────────────────────────────────────────

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                unwrapped_model.unmerge_adapter()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        targets = None
        if self.add_gt or self.test_during_training or self.dynamic_sampling or self.value_head_weight > 0.0:
            histories = [self.prompt2history[x["prompt"]] for x in inputs]
            targets = [self.history2target[x] for x in histories]
            num_categories = len(set(targets))
        
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        ccc = ConstrainedLogitsProcessor(
                # guidance_scale=1.0,
                # cf_logits=None,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                # cf_dict=sasrec_dict,
                # unconditional_ids=None,
                num_beams=self.num_generations if self.beam_search else 1,
                base_model=self.base_model,
                eos_token_id=self.processing_class.eos_token_id
            )
        self.logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=self.temperature), ccc])
        self.test_lp_list = LogitsProcessorList([ccc])

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                topk = [3, 5, 10, 20]
                ndcg = [0 , 0, 0, 0]
                hr = [0, 0, 0, 0]

                if self.test_during_training:
                    dedup_prompt = []
                    dedup_mask = []
                    dedup_target = []

                    for i in range(len(prompt_ids)):
                        if i % self.num_generations == 0:
                            dedup_prompt.append(prompt_ids[i])
                            dedup_mask.append(prompt_mask[i])
                            dedup_target.append(targets[i])
                    
                    dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                    dedup_prompt_mask = torch.stack(dedup_mask).to(device)
                    # print(f"dedup_prompt_ids: {dedup_prompt_ids.shape}")
                
                    # print(f"test_beam: {self.test_beam}")
                    with torch.no_grad():
                        test_completion_ids = unwrapped_model.generate(
                            dedup_prompt_ids, attention_mask=dedup_prompt_mask, generation_config=self.test_generation_config,
                            logits_processor=self.test_lp_list,
                        )
                    
                    # print(f"test_completion_ids: {test_completion_ids.shape}")
                    if self.base_model.lower().find("llama")>-1:
                        test_completions = self.processing_class.batch_decode(test_completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    else:
                        test_completions = self.processing_class.batch_decode(test_completion_ids, skip_special_tokens=True)
                    test_completions = [_.split("Response:\n")[-1] for _ in test_completions]
                    test_comp_lis = [test_completions[i:i+self.test_beam] for i in range(0, len(test_completions), self.test_beam)]
                    for i, comp_lis in enumerate(test_comp_lis):
                        target = dedup_target[i]
                        for j in range(len(comp_lis)):
                            if comp_lis[j].strip("\n\"") == target.strip("\n\""):
                                for index, k in enumerate(topk):
                                    if j < k:
                                        hr[index] += 1
                                        ndcg[index] += 1 / math.log2(j+2) 
                                break
                    hr = [elm/len(dedup_target) for elm in hr]
                    ndcg = [elm/len(dedup_target) for elm in ndcg]

               

                if self.use_ved and self.value_head_weight > 0:
                    # VED: Value-Guided Efficient Decoding (V-STAR Sec 4.2)
                    from ved import ved_decode_batch
                    dedup_prompt = []
                    dedup_mask = []
                    for i in range(len(prompt_ids)):
                        if i % self.num_generations == 0:
                            dedup_prompt.append(prompt_ids[i])
                            dedup_mask.append(prompt_mask[i])
                    dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                    dedup_prompt_mask = torch.stack(dedup_mask).to(device)

                    value_head = getattr(unwrapped_model, "value_head", None)
                    # Build prompt_prefix_ids for each prompt (tokens before SID)
                    prefix_index = 3 if self.base_model.lower().find("gpt2") == -1 else 4
                    prompt_prefix_ids_list = []
                    for i in range(dedup_prompt_ids.size(0)):
                        p = dedup_prompt_ids[i]
                        mask = dedup_prompt_mask[i]
                        # The prefix is the last prefix_index non-pad tokens of the prompt
                        valid_len = mask.sum().item()
                        prefix_ids = p[int(valid_len) - prefix_index : int(valid_len)].tolist()
                        prompt_prefix_ids_list.append(prefix_ids)

                    all_candidates = ved_decode_batch(
                        model=unwrapped_model,
                        value_head=value_head,
                        prompt_ids=dedup_prompt_ids,
                        prompt_mask=dedup_prompt_mask,
                        prompt_prefix_ids_list=prompt_prefix_ids_list,
                        hash_dict=self.hash_dict,
                        get_hash_fn=self.get_hash,
                        num_generations=self.num_generations,
                        sid_length=3,
                        init_beam_width=self.ved_init_beam,
                        lambda_explore=self.ved_lambda,
                        beta_ucb=self.ved_beta_ucb,
                        budget_multiplier=self.ved_budget_multiplier,
                        prefix_index=prefix_index,
                    )

                    # Convert SID token lists to prompt_completion_ids
                    all_completion_ids = []
                    for prompt_idx, candidates in enumerate(all_candidates):
                        p = dedup_prompt_ids[prompt_idx]
                        if not candidates:
                            # VED found no valid candidates for this prompt;
                            # fill with a dummy SID (first valid token repeated) + EOS
                            dummy_prefix = self.get_hash(prompt_prefix_ids_list[prompt_idx])
                            fallback_tokens = self.hash_dict.get(dummy_prefix, [self.processing_class.eos_token_id])
                            dummy_sid = [fallback_tokens[0]] * 3
                            for _ in range(self.num_generations):
                                eos = torch.tensor([self.processing_class.eos_token_id], device=device, dtype=p.dtype)
                                sid_tensor = torch.tensor(dummy_sid, device=device, dtype=p.dtype)
                                comp = torch.cat([p, sid_tensor, eos])
                                all_completion_ids.append(comp)
                        else:
                            for sid_tokens in candidates:
                                eos = torch.tensor([self.processing_class.eos_token_id], device=device, dtype=p.dtype)
                                sid_tensor = torch.tensor(sid_tokens, device=device, dtype=p.dtype)
                                comp = torch.cat([p, sid_tensor, eos])
                                all_completion_ids.append(comp)
                    prompt_completion_ids = pad(all_completion_ids, padding_value=self.processing_class.pad_token_id)
                    prompt_completion_ids = prompt_completion_ids.to(device)
                    # Rebuild prompt_mask to match VED output batch size
                    prompt_mask = (prompt_completion_ids[:, :prompt_ids.size(1)] != self.processing_class.pad_token_id).long()

                elif self.beam_search:
                    dedup_prompt = []
                    dedup_mask = []
                    for i in range(len(prompt_ids)):
                        if i % self.num_generations == 0:
                            dedup_prompt.append(prompt_ids[i])
                            dedup_mask.append(prompt_mask[i])
                    dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                    dedup_prompt_mask = torch.stack(dedup_mask).to(device)
                    prompt_completion_ids = unwrapped_model.generate(
                        dedup_prompt_ids, attention_mask=dedup_prompt_mask, generation_config=self.generation_config,
                        logits_processor=self.logits_processor,
                    )
                else:
                    if self.dynamic_sampling:
                        lis1 = []
                        lis2 = []
                        extended_targets = []
                        for i in range(0, len(prompt_ids), self.num_generations):
                            lis1.extend([prompt_ids[i]]*int(1.5*self.num_generations))
                            lis2.extend([prompt_mask[i]]*int(1.5*self.num_generations))
                            extended_targets.extend([targets[i]]*int(1.5*self.num_generations))
                        extended_prompt_ids = torch.stack(lis1).to(device)
                        extended_prompt_mask = torch.stack(lis2).to(device)
                        # print(f"extended_prompt_ids: {extended_prompt_ids.shape}")
                        # print(f"extended_prompt_mask: {extended_prompt_mask.shape}")
                        prompt_completion_ids = unwrapped_model.generate(
                            extended_prompt_ids, attention_mask=extended_prompt_mask, generation_config=self.generation_config,
                            logits_processor=self.logits_processor,
                        )
                        prompt_length = prompt_ids.size(1)
                        extended_completion_ids = prompt_completion_ids[:, prompt_length:]
                        if self.base_model.lower().find("llama")>-1:
                            extended_completions_text = self.processing_class.batch_decode(extended_completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        else:
                            extended_completions_text = self.processing_class.batch_decode(extended_completion_ids, skip_special_tokens=True)
                        # print(f"extended_completions_text: {extended_completions_text}")

                        def select_completion(completions, target):
                            from collections import Counter
                            selected = []
                            completion_times = Counter(completions)
                            completion_times = dict(sorted(completion_times.items(), key=lambda x: x[1], reverse=True))
                            if target in completions:
                                selected.extend([target]*min(completion_times[target], self.num_generations))
                            if len(selected) == self.num_generations:
                                return selected
                            for item in completion_times:
                                if item != target and completion_times[item] > 0:
                                    selected.append(item)
                                    completion_times[item] -= 1
                                    if len(selected) == self.num_generations:
                                        return selected
                            while len(selected) < self.num_generations:
                                for item in completion_times:
                                    if item != target and completion_times[item] > 0:
                                        selected.append(item)
                                        completion_times[item] -= 1
                                        if len(selected) == self.num_generations:
                                            return selected
                        selected_completion = []
                        for i in range(0, len(extended_completions_text), int(self.num_generations*1.5)):
                            selected_completion.extend(select_completion(extended_completions_text[i:i+int(self.num_generations*1.5)], extended_targets[i]))
                        # print(f"selected_completion: {len(selected_completion)}")
                        selected_completion_ids = self.processing_class(selected_completion, return_tensors="pt", padding=True, padding_side="right", \
                            add_special_tokens=True)["input_ids"].to(device)
                        # print(f"selected_completion_ids: {selected_completion_ids.shape}")
                        prompt_completion_ids = torch.cat([prompt_ids, selected_completion_ids], dim=1)
                        # print(f"dynSam_prompt_completion_ids: {prompt_completion_ids.shape}")
                            
                    else:
                        prompt_completion_ids = unwrapped_model.generate(
                            prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config,
                            logits_processor=self.logits_processor,
                        )

            if self.add_gt:
                repeat = len(prompts) // num_categories
                new_prompt_completions = []
                flag = False
                # rep_ind = [random.randint(i, i+repeat-1) for i in range(0, len(prompts), repeat)]
                for i in range(len(prompts)):
                    if (i+1)%repeat == 0:
                        target_ids = self.processing_class(targets[i], return_tensors="pt", padding=True, padding_side="left", \
                            add_special_tokens=True)["input_ids"].squeeze()
                        # print(f"target_ids: {target_ids.shape}")
                        # print(f"prompt_ids: {prompt_ids[idx].shape}")
                        target_ids = target_ids.to(device)
                        added_ids = torch.cat([prompt_ids[i], target_ids], dim=0)
                        # print(f"added_ids: {added_ids.shape}")
                        new_prompt_completions.append(added_ids)
                    else:
                        new_prompt_completions.append(prompt_completion_ids[i])
                prompt_completion_ids = pad(new_prompt_completions, padding_value=self.processing_class.pad_token_id)
                
                    
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            
            completion_ids = prompt_completion_ids[:, prompt_length:]


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print(completions_text)
        
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        if self.base_model.lower().find("llama")>-1:
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print(completions_text)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        div_lis = [len(set(completions_text[i:i+self.num_generations]))/self.num_generations for i in range(0, len(completions_text), self.num_generations)]
        # cate_diversity = len(set(completions_text))/len(completions_text)
        cate_diversity = sum(div_lis)/len(div_lis)
        completion_ids_cpu = completion_ids.cpu().numpy()
        total_ids = set()
        num_tokens = 0
        for ids in completion_ids_cpu:
            ids = ids[ids != self.processing_class.pad_token_id]
            total_ids.update(set(ids))
            num_tokens += len(ids)        
        num_unique_tokens = len(total_ids)
        token_diversity = num_unique_tokens / num_tokens if num_tokens > 0 else 0.0

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # print(f"advantages: {advantages}")

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        sliced_rewards = rewards[process_slice]



        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())


        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["categorical_diversity"].append(cate_diversity)
        self._metrics["token_diversity"].append(token_diversity)

        if self.test_during_training:
            for i in range(len(topk)):
                self._metrics[f"NDCG@{topk[i]}"].append(ndcg[i])
                self._metrics[f"HR@{topk[i]}"].append(hr[i])

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        # Encode targets for value loss computation
        target_ids = None
        target_lens = None
        if targets is not None:
            target_inputs = self.processing_class(
                targets,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            target_ids = target_inputs["input_ids"].to(device)
            pad_id = self.processing_class.pad_token_id
            if pad_id is None:
                pad_id = self.processing_class.eos_token_id
            if pad_id is not None:
                target_lens = (target_ids != pad_id).sum(dim=1)
            else:
                target_lens = torch.full(
                    (target_ids.size(0),),
                    target_ids.size(1),
                    device=target_ids.device,
                    dtype=torch.long,
                )

        # Compute sibling advantages if enabled
        sibling_adv_tensor = None
        if self.use_sibling_grpo:
            from sibling_grpo import compute_sibling_advantages, build_sibling_advantage_tensor
            # Extract SID tokens from completions (first 3 tokens)
            sid_tokens = completion_ids[:, :3]  # [B*G, 3]
            sibling_advs = compute_sibling_advantages(
                candidate_sids=sid_tokens,
                rewards=sliced_rewards,
                sid_length=3,
            )
            sibling_adv_tensor = build_sibling_advantage_tensor(
                candidate_sids=sid_tokens,
                sibling_advantages=sibling_advs,
                sid_length=3,
            )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "sliced_rewards": sliced_rewards,
            "target_ids": target_ids,
            "target_lens": target_lens,
            "sibling_adv_tensor": sibling_adv_tensor,
        }
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")


        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Check if we need hidden states for value head
        value_head = getattr(model, "value_head", None)
        need_hidden = value_head is not None and self.value_head_weight > 0

        if need_hidden:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:]
            input_ids_slice = input_ids[:, -logits_to_keep:]
            per_token_logps = selective_log_softmax(logits, input_ids_slice)
        else:
            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
            outputs = None

        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        if self.dapo:
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        elif self.gspo:
            per_token_ratio = per_token_logps - per_token_logps.detach()
            s_score = torch.exp((per_token_ratio*completion_mask).sum(dim=1)/completion_mask.sum(dim=1)) 
            sequence_kl = (per_token_kl * completion_mask).sum(dim=1)/completion_mask.sum(dim=1)
            loss = -(s_score*advantages - self.beta*sequence_kl).mean()
        else:
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # Log the metrics

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # ── Value loss (dense supervision + TD learning, from mcts2) ──
        target_ids = inputs.get("target_ids")
        target_lens = inputs.get("target_lens")
        if (
            value_head is not None
            and self.value_head_weight > 0
            and outputs is not None
            and target_ids is not None
            and target_lens is not None
        ):
            hidden_states = outputs.hidden_states[-1]
            prompt_len = prompt_ids.size(1)
            comp_len = completion_ids.size(1)
            max_sid_steps = 3

            target_ids = target_ids.to(completion_ids.device)
            target_lens = target_lens.to(completion_ids.device)
            prefix_weights = torch.tensor(
                [0.3, 0.5, 1.0], device=completion_ids.device, dtype=hidden_states.dtype
            )
            pad_val = -1

            # Pad target prefix to max_sid_steps
            if target_ids.size(1) < max_sid_steps:
                pad = torch.full(
                    (target_ids.size(0), max_sid_steps - target_ids.size(1)),
                    pad_val, device=target_ids.device, dtype=target_ids.dtype,
                )
                tgt_prefix = torch.cat([target_ids, pad], dim=1)[:, :max_sid_steps]
            else:
                tgt_prefix = target_ids[:, :max_sid_steps]

            # Pad completion prefix to max_sid_steps
            if completion_ids.size(1) < max_sid_steps:
                pad = torch.full(
                    (completion_ids.size(0), max_sid_steps - completion_ids.size(1)),
                    pad_val, device=completion_ids.device, dtype=completion_ids.dtype,
                )
                comp_prefix = torch.cat([completion_ids, pad], dim=1)[:, :max_sid_steps]
            else:
                comp_prefix = completion_ids[:, :max_sid_steps]

            # Prefix matching (cumulative product for hierarchical SID matching)
            match = (comp_prefix == tgt_prefix) & (tgt_prefix != pad_val) & (comp_prefix != pad_val)
            prefix_ok = match.cumprod(dim=1).to(dtype=hidden_states.dtype)

            # Semantic-aware distance using item embeddings
            if self._value_item_emb is not None and self._value_item_index is not None:
                pred_emb = self._lookup_item_embeddings(comp_prefix)
                gt_emb = self._lookup_item_embeddings(tgt_prefix)
                cosine = F.cosine_similarity(pred_emb, gt_emb, dim=1, eps=1e-8)
                cosine = cosine.clamp(-1.0, 1.0)
                dist = (1.0 - cosine).to(hidden_states.dtype)
            else:
                dist = torch.zeros(
                    (completion_ids.size(0),), device=completion_ids.device, dtype=hidden_states.dtype
                )

            # Dense step rewards
            dist = dist.unsqueeze(1)
            step_rewards = (
                prefix_ok * prefix_weights.unsqueeze(0)
                + (1.0 - prefix_ok) * (-dist * prefix_weights.unsqueeze(0))
            )

            # Value loss with TD bootstrap
            valid_steps = min(comp_len, max_sid_steps)

            def compute_value_loss(
                token_values: torch.Tensor, weight: float, metric_key: str
            ) -> torch.Tensor:
                comp_values = token_values[:, prompt_len : prompt_len + comp_len]
                value_targets = torch.zeros_like(comp_values)
                value_mask = torch.zeros_like(comp_values)
                if valid_steps > 0:
                    value_targets[:, :valid_steps] = step_rewards[:, :valid_steps]
                    value_mask[:, :valid_steps] = 1
                    if valid_steps > 1:
                        has_prefix = prefix_ok[:, 0] > 0
                        no_prefix = ~has_prefix
                        if no_prefix.any():
                            bootstrap = comp_values[:, 1:valid_steps].detach()
                            value_targets[no_prefix, :valid_steps - 1] = (
                                value_targets[no_prefix, :valid_steps - 1]
                                + self.value_td_gamma * bootstrap[no_prefix]
                            )
                value_loss = (comp_values - value_targets).pow(2)
                value_loss = (value_loss * value_mask * completion_mask).sum() / (
                    (value_mask * completion_mask).sum().clamp_min(1)
                )
                self._metrics[metric_key].append(
                    self.accelerator.gather_for_metrics(value_loss).mean().item()
                )
                return value_loss * weight

            token_values, _cls_value = value_head(hidden_states, attention_mask)
            loss = loss + compute_value_loss(token_values, self.value_head_weight, "value_loss")

        # ── Sibling-GRPO loss (V-STAR Sec 4.3) ──
        sibling_adv_tensor = inputs.get("sibling_adv_tensor")
        if self.use_sibling_grpo and sibling_adv_tensor is not None:
            from sibling_grpo import sibling_grpo_loss
            loss_sibling = sibling_grpo_loss(
                per_token_logps=per_token_logps,
                ref_per_token_logps=ref_per_token_logps,
                sibling_adv_tensor=sibling_adv_tensor,
                completion_mask=completion_mask,
                sid_length=3,
                beta=self.beta,
            )
            self._metrics["sibling_loss"].append(
                self.accelerator.gather_for_metrics(loss_sibling).mean().item()
            )
            loss = loss + self.sibling_loss_weight * loss_sibling

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
