# Merging-EVAL: Model Evaluation Framework

A comprehensive evaluation framework for language models with support for multiple datasets, GPU specification, and offline evaluation capabilities.

## Features

- 🚀 **Multi-dataset Support**: Evaluate on code, algebra, analysis, and other domains
- 🎯 **GPU Specification**: Run evaluations on specific GPUs
- 📊 **Offline Mode**: Evaluate models without internet connection
- 🔧 **Data Slicing**: Evaluate on specific data subsets using indices
- 💾 **Caching**: Intelligent caching for faster repeated evaluations
- 📈 **Comprehensive Metrics**: Cross-entropy loss, token-level analysis

## Repository Structure

```
.
├── scripts/
│   └── eval.py           # Main evaluation script
├── src/merge/
│   └── main_merging.py   # Model merging script
├── data/
│   ├── eval_partial/     # Evaluation datasets
│   │   ├── code.json     # Code generation tasks
│   │   ├── algebra.json  # Mathematical problems
│   │   └── analysis.json # Data analysis tasks
│   └── train_partial/    # Training datasets
├── test_result/          # Evaluation results and merged models
└── cache/                # Cached tokenized data
```

## Installation and Setup

### 1. Environment Setup

Activate the required Python environment:

```bash
source /zju_0038/jinjia/workspace/Merging-Scaling-Law-main/merge-eval-py312/bin/activate
```

### 2. Environment Variables

Set required environment variables:

```bash
export TRANSFORMERS_NO_TORCHVISION=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Usage

### Model Merging

Merge multiple models using Task Arithmetic:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python3 src/merge/main_merging.py \
  --merge_method task_arithmetic \
  --output_dir /path/to/output \
  --base_model /path/to/base/model \
  --models_to_merge "/path/to/model1,/path/to/model2,/path/to/model3" \
  --scaling_coefficient 0.2 \
  --use_gpu
```

#### GPU-Specific Merging

Run model merging on a specific GPU:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=7
python3 src/merge/main_merging.py \
  --merge_method task_arithmetic \
  --output_dir /zju_0038/test_merge/Merging-EVAL/test_result/scheme1_16models_task_arithmetic \
  --base_model /zju_0038/wyy/mergebench/models/Llama-3.2-3B \
  --models_to_merge "/zju_0038/yifyang/scripts/models/llama-instruct-3B-v2-algebra,/zju_0038/yifyang/scripts/models/llama-instruct-3B-v2-analysis,/zju_0038/yifyang/scripts/models/llama-instruct-3B-v2-number_theory,/zju_0038/yifyang/scripts/models/llama-instruct-3B-v2-physics" \
  --scaling_coefficient 0.2 \
  --use_gpu
```

### Basic Evaluation

Evaluate a model on a specific dataset:

```bash
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --file /path/to/dataset.json \
  --output ./results \
  --batch_size 1 \
  --max_length 2048
```

### GPU-Specific Evaluation

Run evaluation on a specific GPU:

```bash
export CUDA_VISIBLE_DEVICES=7
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --file /path/to/dataset.json \
  --output ./results \
  --batch_size 1 \
  --max_length 2048 \
  --gpu_id 0 \
  --offline
```

### Dataset-Specific Examples

#### Mathematical Problems (Algebra)
```bash
# 数学问题推荐 max_length = 2048
source /zju_0038/jinjia/workspace/Merging-Scaling-Law-main/merge-eval-py312/bin/activate && \
export TRANSFORMERS_NO_TORCHVISION=1 && \
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && \
export CUDA_VISIBLE_DEVICES=7 && \
python3 /zju_0038/test_merge/Merging-EVAL/scripts/eval.py \
  --model /zju_0038/jinjia/workspace/Merging-Scaling-Law-main/models/merged/Llama-3B-cmb/task_arithmetic_9/sc0.1_r0/6p3h \
  --tokenizer /zju_0038/jinjia/workspace/Merging-Scaling-Law-main/models/merged/Llama-3B-cmb/task_arithmetic_9/sc0.1_r0/6p3h \
  --file /zju_0038/test_merge/Merging-EVAL/data/eval_partial/algebra.json \
  --output /zju_0038/test_merge/Merging-EVAL/test_result \
  --batch_size 1 \
  --max_length 2048 \
  --gpu_id 0 \
  --offline
```

#### Code Generation Tasks
```bash
# 代码问题推荐 max_length = 4096
source /zju_0038/jinjia/workspace/Merging-Scaling-Law-main/merge-eval-py312/bin/activate && \
export TRANSFORMERS_NO_TORCHVISION=1 && \
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && \
export CUDA_VISIBLE_DEVICES=7 && \
python3 /zju_0038/test_merge/Merging-EVAL/scripts/eval.py \
  --model /zju_0038/jinjia/workspace/Merging-Scaling-Law-main/models/merged/Llama-3B-cmb/task_arithmetic_9/sc0.1_r0/6p3h \
  --tokenizer /zju_0038/jinjia/workspace/Merging-Scaling-Law-main/models/merged/Llama-3B-cmb/task_arithmetic_9/sc0.1_r0/6p3h \
  --file /zju_0038/test_merge/Merging-EVAL/data/eval_partial/code.json \
  --output /zju_0038/test_merge/Merging-EVAL/test_result \
  --batch_size 1 \
  --max_length 4096 \
  --gpu_id 0 \
  --offline
```

## Parameters

### Model Merging Parameters

#### Required Parameters
- `--merge_method`: Merging method (e.g., "task_arithmetic", "average_merging")
- `--base_model`: Path to the base model directory
- `--models_to_merge`: Comma-separated list of model paths to merge
- `--output_dir`: Output directory for merged model

#### Optional Parameters
- `--scaling_coefficient`: Scaling coefficient for merging (default: 1.0)
- `--use_gpu`: Use GPU for merging (default: CPU)
- `--exclude_param_names_regex`: Regex patterns for parameters to exclude
- `--param_value_mask_rate`: Parameter value mask rate (default: 0.8)
- `--mask_apply_method`: Method for applying masks (default: "average_merging")
- `--weight_mask_rates`: Comma-separated weight mask rates

### Model Evaluation Parameters

#### Required Parameters
- `--model`: Path to the model directory
- `--tokenizer`: Path to the tokenizer directory
- `--file`: Path to the evaluation dataset JSON file

#### Optional Parameters
- `--output`: Output directory for results (default: `./output`)
- `--batch_size`: Batch size for evaluation (default: 10)
- `--max_length`: Maximum sequence length (default: 2048)
- `--gpu_id`: Specific GPU ID to use (e.g., 0, 1, 2)
- `--offline`: Run in offline mode (no internet connection required)
- `--indices`: Evaluate specific data indices (e.g., "1-10,15,20-22")
- `--run_name`: Custom name for output folder
- `--no_cache`: Disable caching mechanism

### Dataset-Specific Recommendations

#### Code Generation Tasks
- **Recommended max_length**: 4096-8192
- **Reason**: Code samples are typically longer (average: 1,899 tokens)
- **Coverage**: 8192 covers 98% of samples

#### Mathematical Problems
- **Recommended max_length**: 2048
- **Reason**: Math problems are typically shorter
- **Coverage**: 2048 is sufficient for most algebra problems

## Output Format

Results are saved as CSV files in the output directory:

```
test_result/
└── model_name/
    └── all/
        └── results.csv
```

CSV format:
```csv
problem,CE Loss,class
algebra,0.6289,algebra
Avg.,0.6289,average
Overall,0.6289,overall
```

## Troubleshooting

### Model Merging Issues

1. **Protobuf Version Conflicts**:
   - Solution: Set `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

2. **GPU Memory Issues**:
   - Use CPU mode (remove `--use_gpu` flag)
   - Reduce number of models to merge simultaneously
   - Use specific GPU with `export CUDA_VISIBLE_DEVICES=X`

3. **PEFT Configuration Errors**:
   - Some models may have incompatible PEFT configurations
   - Solution: Exclude problematic models or use CPU mode

4. **RoPE Configuration Warnings**:
   - Warning: `rope_scaling` configuration issues
   - Solution: These are usually non-fatal warnings

### Model Evaluation Issues

1. **NaN Loss Values**: Usually caused by very long sequences or all-masked labels
   - Solution: Increase `max_length` or check data format

2. **GPU Memory Issues**: 
   - Reduce `batch_size` to 1
   - Decrease `max_length`
   - Use specific GPU with `--gpu_id`

3. **Network Connection Issues**:
   - Use `--offline` flag for local model evaluation

### Performance Tips

#### Model Merging
- Use GPU mode for faster merging when memory allows
- Start with smaller model subsets to test compatibility
- Use CPU mode for large model collections to avoid memory issues
- Set environment variables before running to avoid conflicts

#### Model Evaluation
- Use caching for repeated evaluations on the same dataset
- Specify GPU ID for better resource management
- Adjust `max_length` based on dataset characteristics
- Use offline mode when internet connection is unstable 