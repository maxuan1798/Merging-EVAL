"""
File-based chat template system for model merging.

This module provides a flexible system for loading chat templates from files
based on model types, with fallback to default templates.
"""

import os
import json
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ChatTemplateManager:
    """Manager for file-based chat templates."""

    def __init__(self, templates_dir: str = None):
        """Initialize the template manager.

        Args:
            templates_dir: Directory containing template files. If None, uses default location.
        """
        if templates_dir is None:
            # Default to project root templates directory
            self.templates_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'templates'
            )
        else:
            self.templates_dir = templates_dir

        self._templates: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self):
        """Load all template files from the templates directory."""
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.jinja') or filename.endswith('.template'):
                template_name = os.path.splitext(filename)[0]
                template_path = os.path.join(self.templates_dir, filename)

                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        self._templates[template_name] = f.read()
                    logger.info(f"Loaded chat template: {template_name}")
                except Exception as e:
                    logger.error(f"Failed to load template {filename}: {e}")

    def get_template(self, model_name: str) -> Optional[str]:
        """Get the appropriate chat template for a model.

        Args:
            model_name: Name or path of the model

        Returns:
            Chat template string or None if no suitable template found
        """
        # Try exact model name match first
        if model_name in self._templates:
            return self._templates[model_name]

        # Try model family matching (e.g., "llama" for "llama-2-7b-chat")
        model_lower = model_name.lower()

        # Check for common model families
        model_families = ['llama', 'qwen', 'chatglm', 'baichuan', 'yi', 'mistral', 'phi']

        for family in model_families:
            if family in model_lower:
                if family in self._templates:
                    logger.info(f"Using {family} template for model {model_name}")
                    return self._templates[family]

        # Try to extract base model name from path
        if '/' in model_name:
            base_name = model_name.split('/')[-1]
            if base_name in self._templates:
                return self._templates[base_name]

        logger.warning(f"No suitable chat template found for model: {model_name}")
        return None

    def list_templates(self) -> Dict[str, str]:
        """List all available templates."""
        return self._templates.copy()

    def add_template(self, name: str, template: str, save_to_file: bool = False):
        """Add a template to the manager.

        Args:
            name: Template name
            template: Template content
            save_to_file: Whether to save to file in templates directory
        """
        self._templates[name] = template

        if save_to_file:
            template_path = os.path.join(self.templates_dir, f"{name}.jinja")
            try:
                os.makedirs(self.templates_dir, exist_ok=True)
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template)
                logger.info(f"Saved template to file: {template_path}")
            except Exception as e:
                logger.error(f"Failed to save template {name} to file: {e}")


# Global template manager instance
_template_manager: Optional[ChatTemplateManager] = None


def get_template_manager() -> ChatTemplateManager:
    """Get the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = ChatTemplateManager()
    return _template_manager


def inject_default_chat_template(tokenizer, model_name: str = None) -> Any:
    """
    Inject appropriate chat template into tokenizer based on model type.

    This function replaces the hardcoded template injection with a file-based
    system that can load model-specific templates.

    Args:
        tokenizer: Hugging Face tokenizer
        model_name: Model name or path for template selection

    Returns:
        Tokenizer with injected chat template
    """
    # Check if tokenizer already has a chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        logger.info("Tokenizer already has chat_template, skipping injection")
        return tokenizer

    # Determine model name for template selection
    if model_name is None:
        if hasattr(tokenizer, 'name_or_path'):
            model_name = tokenizer.name_or_path
        else:
            model_name = "unknown"

    # Get template manager and find appropriate template
    template_manager = get_template_manager()
    template = template_manager.get_template(model_name)

    if template is None:
        # Fallback to default template (current hardcoded one)
        logger.warning(f"No template found for {model_name}, using default template")
        from .default_templates import get_default_template
        template = get_default_template()

    # Inject the template
    tokenizer.chat_template = template
    logger.info(f"Injected chat template for model: {model_name}")

    # Save to tokenizer_config.json if possible
    _save_template_to_config(tokenizer, template)

    return tokenizer


def _save_template_to_config(tokenizer, template: str):
    """Save chat template to tokenizer_config.json if possible."""
    if not hasattr(tokenizer, 'save_pretrained'):
        return

    try:
        # Get the tokenizer config path
        tokenizer_config_path = None
        if hasattr(tokenizer, 'name_or_path'):
            tokenizer_config_path = os.path.join(tokenizer.name_or_path, 'tokenizer_config.json')

        if tokenizer_config_path and os.path.exists(tokenizer_config_path):
            # Load existing config
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Add chat_template to config
            config['chat_template'] = template

            # Save updated config
            with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"Updated tokenizer_config.json with chat_template at {tokenizer_config_path}")
    except Exception as e:
        logger.warning(f"Could not update tokenizer_config.json: {e}")