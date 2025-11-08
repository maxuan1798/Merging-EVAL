"""
Default chat templates for fallback when no file-based template is found.
"""


import os


def _load_template_from_file(template_name: str) -> str:
    """Load a template from file."""
    templates_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'templates'
    )
    template_path = os.path.join(templates_dir, f"{template_name}.jinja")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load template from {template_path}: {e}")


def get_default_template() -> str:
    """Get the default chat template from file."""
    return _load_template_from_file('default')


def get_llama_template() -> str:
    """Get the Llama chat template from file."""
    return _load_template_from_file('llama')


def get_template_by_name(name: str) -> str:
    """Get a template by name."""
    # Try to load the template from file
    try:
        return _load_template_from_file(name)
    except FileNotFoundError:
        # If specific template not found, fallback to default
        return _load_template_from_file('default')