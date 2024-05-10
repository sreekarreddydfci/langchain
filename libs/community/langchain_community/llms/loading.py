"""Base interface for loading large language model APIs."""
import json
from pathlib import Path
from typing import Any, Union

import yaml
from langchain_core.language_models.llms import BaseLLM

from langchain_community.llms import get_type_to_cls_dict

_ALLOW_DANGEROUS_DESERIALIZATION_ARG = "allow_dangerous_deserialization"

def convert_config(old_config):
    # Adjusted to ignore 'base_url'
    new_config = {
        'model': old_config.get('model', 'llama2'),
        'mirostat': old_config.get('options', {}).get('mirostat', None),
        'mirostat_eta': old_config.get('options', {}).get('mirostat_eta', None),
        'mirostat_tau': old_config.get('options', {}).get('mirostat_tau', None),
        'num_ctx': old_config.get('options', {}).get('num_ctx', None),
        'num_gpu': old_config.get('options', {}).get('num_gpu', None),
        'num_thread': old_config.get('options', {}).get('num_thread', None),
        'num_predict': old_config.get('options', {}).get('num_predict', None),
        'repeat_last_n': old_config.get('options', {}).get('repeat_last_n', None),
        'repeat_penalty': old_config.get('options', {}).get('repeat_penalty', None),
        'temperature': old_config.get('options', {}).get('temperature', None),
        'stop': old_config.get('options', {}).get('stop', None),
        'tfs_z': old_config.get('options', {}).get('tfs_z', None),
        'top_k': old_config.get('options', {}).get('top_k', None),
        'top_p': old_config.get('options', {}).get('top_p', None),
        'system': old_config.get('system', None),
        'template': old_config.get('template', None),
        'format': old_config.get('format', None),
        'keep_alive': old_config.get('keep_alive', None),
        'headers': old_config.get('headers', None),
        'timeout': old_config.get('timeout', None),
    }
    return new_config

def load_llm_from_config(config: dict, **kwargs: Any) -> BaseLLM:
    """Load LLM from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify an LLM Type in config")
    config_type = config.pop("_type")

    if config_type == "ollama":
        config = convert_config(config)

    type_to_cls_dict = get_type_to_cls_dict()

    if config_type not in type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")

    llm_cls = type_to_cls_dict[config_type]()

    load_kwargs = {}
    if _ALLOW_DANGEROUS_DESERIALIZATION_ARG in llm_cls.__fields__:
        load_kwargs[_ALLOW_DANGEROUS_DESERIALIZATION_ARG] = kwargs.get(
            _ALLOW_DANGEROUS_DESERIALIZATION_ARG, False
        )

    return llm_cls(**config, **load_kwargs)

def load_llm(file: Union[str, Path], **kwargs: Any) -> BaseLLM:
    """Load LLM from a file."""
    # Convert file to Path object.
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    # Load from either json or yaml.
    if file_path.suffix == ".json":
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix.endswith((".yaml", ".yml")):
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("File type must be json or yaml")
    # Load the LLM from the config now.
    return load_llm_from_config(config, **kwargs)
