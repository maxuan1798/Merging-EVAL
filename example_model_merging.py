#!/usr/bin/env python3
"""
Example: Model Merging with merging-eval Package

This script demonstrates how to use the merging-eval package
for model merging in Python code after pip installation.

Usage:
  pip install merging-eval
  python example_model_merging.py
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment variables as recommended
os.environ['TRANSFORMERS_NO_TORCHVISION'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def example_basic_merging():
    """Example of basic model merging using the package"""
    print("Example: Basic Model Merging")
    print("=" * 50)

    try:
        # Import the merging-eval package
        from merge import MergingMethod

        print("1. Loading models...")
        # Note: Replace these paths with actual model paths
        # For demonstration, we'll use placeholder paths
        base_model_path = "/path/to/base/model"
        model1_path = "/path/to/model1"
        model2_path = "/path/to/model2"
        output_dir = "./merged_model_output"

        print(f"   Base model: {base_model_path}")
        print(f"   Model 1: {model1_path}")
        print(f"   Model 2: {model2_path}")
        print(f"   Output: {output_dir}")

        # In a real scenario, you would load actual models:
        # base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        # model1 = AutoModelForCausalLM.from_pretrained(model1_path)
        # model2 = AutoModelForCausalLM.from_pretrained(model2_path)
        # models_to_merge = [model1, model2]

        print("\n2. Creating merging engine...")
        # Choose your merging method
        merging_method = "task_arithmetic"  # or "average_merging", "ties_merging", etc.
        merging_engine = MergingMethod(merging_method_name=merging_method)
        print(f"   Using method: {merging_method}")

        print("\n3. Performing model merging...")
        # In a real scenario, you would call:
        # merged_model = merging_engine.get_merged_model(
        #     merged_model=base_model,
        #     models_to_merge=models_to_merge,
        #     exclude_param_names_regex=[],
        #     scaling_coefficient=0.2,
        #     use_gpu=True  # if GPU available
        # )

        print("   Parameters used:")
        print("     - exclude_param_names_regex: [] (no parameters excluded)")
        print("     - scaling_coefficient: 0.2")
        print("     - use_gpu: True (if available)")

        print("\n4. Saving merged model...")
        # In a real scenario:
        # merged_model.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)
        print(f"   Model would be saved to: {output_dir}")

        print("\n✅ Example completed successfully!")
        print("   Replace placeholder paths with actual model paths to run.")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

def example_advanced_merging():
    """Example of advanced model merging with different methods"""
    print("\n\nExample: Advanced Model Merging")
    print("=" * 50)

    try:
        from merge import MergingMethod

        print("Available merging methods:")
        methods = [
            ("average_merging", "Equal-weight averaging of models"),
            ("task_arithmetic", "Task vector arithmetic merging"),
            ("ties_merging", "TIES merging with parameter pruning"),
            ("ties_merging_dare", "TIES merging with DARE variant"),
            ("mask_merging", "Mask-based merging with weight masking")
        ]

        for method, description in methods:
            print(f"  • {method}: {description}")

        print("\nExample usage with different parameters:")
        print("\nFor TIES merging:")
        print("  merging_engine = MergingMethod(merging_method_name='ties_merging')")
        print("  merged_model = merging_engine.get_merged_model(")
        print("      merged_model=base_model,")
        print("      models_to_merge=models_to_merge,")
        print("      param_value_mask_rate=0.8,  # Mask 80% of smallest parameters")
        print("      scaling_coefficient=0.3")
        print("  )")

        print("\nFor mask merging:")
        print("  merging_engine = MergingMethod(merging_method_name='mask_merging')")
        print("  merged_model = merging_engine.get_merged_model(")
        print("      merged_model=base_model,")
        print("      models_to_merge=models_to_merge,")
        print("      weight_format='delta_weight',")
        print("      weight_mask_rates=[0.5, 0.5],  # Mask rates for each model")
        print("      mask_strategy='magnitude'")
        print("  )")

    except Exception as e:
        print(f"❌ Advanced example failed: {e}")

def example_cli_equivalent():
    """Show the CLI equivalent of the Python code"""
    print("\n\nCLI Equivalent Commands")
    print("=" * 50)

    print("The same merging operation can be done via CLI:")
    print()
    print("For basic task arithmetic:")
    print("  merging-eval \\")
    print("    --merge_method task_arithmetic \\")
    print("    --output_dir ./merged_model_output \\")
    print("    --base_model /path/to/base/model \\")
    print("    --models_to_merge \"/path/to/model1,/path/to/model2\" \\")
    print("    --scaling_coefficient 0.2 \\")
    print("    --use_gpu")
    print()

    print("For TIES merging:")
    print("  merging-eval \\")
    print("    --merge_method ties_merging \\")
    print("    --output_dir ./merged_model_output \\")
    print("    --base_model /path/to/base/model \\")
    print("    --models_to_merge \"/path/to/model1,/path/to/model2\" \\")
    print("    --param_value_mask_rate 0.8 \\")
    print("    --scaling_coefficient 0.3 \\")
    print("    --use_gpu")

if __name__ == "__main__":
    print("Merging-EVAL Package Usage Examples")
    print("=" * 60)
    print("This demonstrates how to use the package after:")
    print("  pip install merging-eval")
    print()

    example_basic_merging()
    example_advanced_merging()
    example_cli_equivalent()

    print("\n" + "=" * 60)
    print("Summary:")
    print("  • Install: pip install merging-eval")
    print("  • Use in Python: import merge")
    print("  • Use CLI: merging-eval --help")
    print("  • Multiple merging methods available")
    print("  • Both CPU and GPU support")
    print()
    print("The package is ready for production use! 🚀")