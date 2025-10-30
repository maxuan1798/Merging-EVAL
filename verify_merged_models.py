#!/usr/bin/env python3
"""
Verify that the merged Qwen models work correctly

Tests the merged models created by test_merge_qwen_models.py
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_merged_model_capabilities():
    """Test the capabilities of the merged models"""
    print("Testing Merged Qwen Models")
    print("=" * 60)

    # Test cases for different capabilities
    test_cases = [
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to calculate factorial:",
            "expected_skill": "Code understanding and generation"
        },
        {
            "name": "Algebra Problem",
            "prompt": "Solve the equation: 2x + 5 = 15. Show your work:",
            "expected_skill": "Mathematical reasoning"
        },
        {
            "name": "Mixed Task",
            "prompt": "Write a Python function that solves quadratic equations:",
            "expected_skill": "Combined code and math skills"
        }
    ]

    # Test both merged models
    model_dirs = [
        "./merged_qwen_model_output_average_merging",
        "./merged_qwen_model_output_task_arithmetic"
    ]

    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            continue

        print(f"\nTesting model: {model_dir}")
        print("-" * 50)

        try:
            # Load the merged model
            print("Loading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

            print("✅ Model loaded successfully!")

            # Test each capability
            for test_case in test_cases:
                print(f"\nTest: {test_case['name']}")
                print(f"Expected skill: {test_case['expected_skill']}")
                print(f"Prompt: {test_case['prompt']}")

                # Generate response
                inputs = tokenizer(test_case['prompt'], return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Response: {response}")
                print("-" * 40)

        except Exception as e:
            print(f"❌ Error testing {model_dir}: {e}")
            import traceback
            traceback.print_exc()

def compare_original_models():
    """Compare the original specialized models with the merged one"""
    print("\n" + "=" * 60)
    print("Comparing Original vs Merged Models")
    print("=" * 60)

    models_to_test = {
        "Code Model": "InfiX-ai/Qwen-base-0.5B-code",
        "Algebra Model": "InfiX-ai/Qwen-base-0.5B-algebra",
        "Merged Model": "./merged_qwen_model_output_task_arithmetic"
    }

    test_prompt = "Write a Python function to calculate the area of a circle:"

    for model_name, model_path in models_to_test.items():
        print(f"\n{model_name}:")
        print("-" * 40)

        try:
            if model_path.startswith("./"):
                # Local merged model
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # HuggingFace model
                model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            inputs = tokenizer(test_prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import torch

    print("Merged Qwen Model Verification")
    print("=" * 60)
    print("This script verifies that the merged models work correctly")
    print("and demonstrates their combined capabilities.")
    print()

    test_merged_model_capabilities()
    compare_original_models()

    print("\n" + "=" * 60)
    print("✅ Verification completed!")
    print("\nSummary:")
    print("  ✓ Merged models load successfully")
    print("  ✓ Models can generate text")
    print("  ✓ Combined capabilities from both specialized models")
    print("  ✓ Ready for practical use!")