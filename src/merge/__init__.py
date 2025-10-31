"""
Model merging algorithms and utilities.

This module contains implementations of various model merging methods
for large language models, including equal-weight averaging and other
advanced merging techniques.
"""

from .main_merging import main
from .merging_methods import (
    MergingMethod,
    FlopsCounter
)
from .utils import (
    set_random_seed,
    save_state_and_model_for_hf_trainer,
    load_state_and_model_for_hf_trainer,
    get_param_names_to_merge,
    get_modules_to_merge,
    smart_tokenizer_and_embedding_resize
)
from .config import get_hf_config, HFConfig

__all__ = [
    "main",
    "MergingMethod",
    "FlopsCounter",
    "set_random_seed",
    "save_state_and_model_for_hf_trainer",
    "load_state_and_model_for_hf_trainer",
    "get_param_names_to_merge",
    "get_modules_to_merge",
    "smart_tokenizer_and_embedding_resize",
    "get_hf_config",
    "HFConfig"
]