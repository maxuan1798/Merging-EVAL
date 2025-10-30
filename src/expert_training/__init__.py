"""
Expert model training for specialized domains.

This module contains utilities for training expert language models
on specific domains and tasks.
"""

from .train_expert import train_expert_model
from .domain_specialization import (
    prepare_domain_data,
    compute_domain_metrics,
    DomainSpecializationConfig
)

__all__ = [
    "train_expert_model",
    "prepare_domain_data",
    "compute_domain_metrics",
    "DomainSpecializationConfig"
]