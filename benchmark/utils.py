"""
Common utility functions.
"""

import argparse
import json
import os
import random
import resource
from typing import Union

import numpy as np
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from config import AIRGAPPED, DATASET


def get_dataset(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerBase
) -> list[tuple[str, int, int]]:
    if args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    return input_requests


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
) -> list[tuple[str, int, int]]:
    """Sample random prompts from the ShareGPT dataset.
    
    Returns:
        A list of tuple (prompt, input_len, output_len), where prompt is the
        input of an LLM, input_len is the prompt length, and output_len is the
        expeceted output length.
    """

    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens
    if not os.path.exists(dataset_path):
        raise ValueError("Please provide an existing dataset_path to ShareGPT data.")

    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns
    dataset = [
        data for data in dataset if len(data.get("conversations", [])) >= 2
    ]
    # Only keep the first two turns of each conversation
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset
    ]
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    input_requests: list[tuple[str, int, int]] = []
    prompt_cnt = 0
    for data in dataset:
        if prompt_cnt == num_prompts:
            break

        # Tokenize the prompts and completions.
        prompt = data[0]
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)

        # Skip empty prompt
        if prompt_len == 0:
            continue

        input_len = input_lens[prompt_cnt]
        output_len = output_lens[prompt_cnt]
        if prompt_len > input_len:
            # Truncate the prompt
            input_ids = prompt_token_ids[:input_len]
        else:
            # Repeat the prompt
            ratio = (input_len + prompt_len - 1) // prompt_len
            input_ids = (prompt_token_ids * ratio)[: input_len]
        prompt = tokenizer.decode(input_ids)
        input_requests.append((prompt, int(input_len), int(output_len)))

        prompt_cnt += 1

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")

    return input_requests


def _get_model(pretrained_model_name_or_path: str) -> str:
    if (
        # Short circuit with AIRGAPPED
        not AIRGAPPED 
        and os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() == "true"
    ):
        import huggingface_hub.constants
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
        )

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(pretrained_model_name_or_path: str) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if (
        not AIRGAPPED
        and (pretrained_model_name_or_path.endswith(".json")
             or pretrained_model_name_or_path.endswith(".model"))
    ):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        return get_tokenizer(pretrained_model_name_or_path)
    elif pretrained_model_name_or_path is None or not os.path.exists(pretrained_model_name_or_path):
        raise ValueError("Please provide an existing filepath to the pretrained model.")
    
    pretrained_model_name_or_path = _get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        # Note the downside of trust_remote_code (mainly security issues)
        pretrained_model_name_or_path, trust_remote_code=False
    )


def set_ulimit(target_soft_limit=65535):
    """Set the soft limit of the number of open file descriptors."""
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


def check_chat_template(model_path: str) -> bool:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return "chat_template" in tokenizer.init_kwargs
    except Exception as e:
        print(f"Fail to load tokenizer config with error={e}")
        return False


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def remove_suffix(text: str, suffix: str) -> str:
    return text[:-len(suffix)] if text.endswith(suffix) else text



if __name__ == "__main__":
    dataset_path="./ShareGPT_V3_unfiltered_cleaned_split.json"
    pretrained_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    args = argparse.Namespace(
        dataset_name="random",
        dataset_path=dataset_path,
        random_input_len=1024,
        random_output_len=512,
        num_prompts=2,
        random_range_ratio=0.5,
    )
    dataset = get_dataset(args, tokenizer=tokenizer)
    for prompt, input_len, output_len in dataset:
        print(f"Prompt len: {len(prompt)}")
        token_ids = tokenizer.encode(prompt)
        print(f"#Token ids: {len(token_ids)}")
        print(input_len, output_len)