"""
Configuration module for Hugging Face authentication and settings.

This module handles HF token configuration from multiple sources:
- Environment variables (HF_TOKEN, HUGGINGFACE_HUB_TOKEN)
- .env files
- Command-line arguments
- Direct parameter passing
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class HFConfig:
    """Hugging Face configuration manager."""

    def __init__(self, token: Optional[str] = None, use_auth: bool = False):
        """
        Initialize HF configuration.

        Args:
            token: Explicit HF token (highest priority)
            use_auth: Whether to use authentication (enables token lookup)
        """
        self.token = token
        self.use_auth = use_auth

    def get_token(self) -> Optional[str]:
        """
        Get HF token with priority order:
        1. Explicit token passed to constructor
        2. HF_TOKEN environment variable
        3. HUGGINGFACE_HUB_TOKEN environment variable
        4. None if no token found

        Returns:
            HF token string or None
        """
        if self.token:
            return self.token

        # Check environment variables
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        return token if token else None

    def should_use_auth(self) -> bool:
        """
        Determine if authentication should be used.

        Returns:
            True if authentication should be used
        """
        return self.use_auth or bool(self.get_token())

    def get_model_loading_kwargs(self) -> dict:
        """
        Get keyword arguments for model loading with authentication.

        Returns:
            Dictionary of kwargs for from_pretrained methods
        """
        kwargs = {}

        if self.should_use_auth():
            token = self.get_token()
            if token:
                kwargs['token'] = token
            else:
                # Use default authentication (will use cached token or prompt)
                kwargs['token'] = True

        return kwargs

    def get_tokenizer_loading_kwargs(self) -> dict:
        """
        Get keyword arguments for tokenizer loading with authentication.

        Returns:
            Dictionary of kwargs for from_pretrained methods
        """
        return self.get_model_loading_kwargs()


def get_hf_config(token: Optional[str] = None, use_auth: bool = False) -> HFConfig:
    """
    Convenience function to create HFConfig instance.

    Args:
        token: Explicit HF token
        use_auth: Whether to use authentication

    Returns:
        HFConfig instance
    """
    return HFConfig(token=token, use_auth=use_auth)