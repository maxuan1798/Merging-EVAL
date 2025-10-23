#!/bin/bash
# 自动化模型评估脚本
# 自动从GitHub下载代码，从HuggingFace下载模型，并执行评估
# 支持Conda虚拟环境

set -e

# ========================================
# 用户配置 - 修改这些参数
# ========================================
# Conda环境配置
CONDA_ENV_NAME="dev_eval_env"              # Conda环境名称，留空则不使用conda
CREATE_ENV_IF_NOT_EXISTS=true          # 如果环境不存在是否自动创建
PYTHON_VERSION="3.10"                  # 创建新环境时使用的Python版本

# HuggingFace模型仓库地址
MODEL_URL="microsoft/DialoGPT-medium"  # 修改为您的模型地址

# 评估数据集 (可选: algebra, analysis, discrete, geometry, number_theory, all)
DATASET="all"

# Callback配置
CALLBACK_URL=""
TASK_ID="eval_task_$(date +%s)"
MODEL_ID=""  # 留空则自动从MODEL_URL提取，或手动指定模型ID
BENCHMARK_ID="math_problems"
API_KEY=""  # 留空使用默认值，或设置为您的API Key

# 评估参数
BATCH_SIZE=8
MAX_LENGTH=2048

# SwanLab模式配置 (可选: local, cloud, disabled)
SWANLAB_MODE="local"  # local=本地存储, cloud=云端存储, disabled=禁用SwanLab

# PIP镜像源配置 (可选，用于加速下载)
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"  # 留空则使用默认源

# SwanLab自定义源配置 (可选，用于从私有源安装swanlab)
SWANLAB_INDEX_URL=""  # 留空则使用PIP_INDEX_URL或默认源
# SwanLab API Key配置
export SWANLAB_API_KEY=""

# =======================================
# 自动执行部分 - 无需修改
# ========================================
REPO_URL="https://github.com/maxuan1798/Merging-EVAL.git"
WORK_DIR="./eval_workspace"
REPO_DIR="$WORK_DIR/Merging-EVAL"

# Conda环境设置
if [ -n "$CONDA_ENV_NAME" ]; then
    echo "Setting up Conda environment: $CONDA_ENV_NAME"

    # 初始化conda（支持多种安装方式）
    CONDA_BASE=""
    CONDA_SH=""

    # 方法1: 从CONDA_EXE环境变量获取（最可靠）
    if [ -n "$CONDA_EXE" ] && [ -f "$CONDA_EXE" ]; then
        CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
        CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
    fi

    # 方法2: 尝试从PATH中找到conda
    if [ -z "$CONDA_SH" ] || [ ! -f "$CONDA_SH" ]; then
        CONDA_PATH=$(which conda 2>/dev/null || true)
        if [ -n "$CONDA_PATH" ] && [ -f "$CONDA_PATH" ]; then
            CONDA_BASE=$(dirname $(dirname "$CONDA_PATH"))
            CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
        fi
    fi

    # 方法3: 检查常见安装位置
    if [ -z "$CONDA_SH" ] || [ ! -f "$CONDA_SH" ]; then
        for base_dir in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniconda" "$HOME/anaconda" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
            if [ -f "${base_dir}/etc/profile.d/conda.sh" ]; then
                CONDA_BASE="$base_dir"
                CONDA_SH="${base_dir}/etc/profile.d/conda.sh"
                break
            fi
        done
    fi

    # 验证是否找到conda.sh
    if [ -z "$CONDA_SH" ] || [ ! -f "$CONDA_SH" ]; then
        echo "Error: Could not locate conda installation."
        echo "Please ensure conda is installed and either:"
        echo "  1. Run 'conda init bash' and restart your shell, or"
        echo "  2. Set CONDA_ENV_NAME to empty to use system Python"
        exit 1
    fi

    # Source conda.sh
    echo "Found conda at: $CONDA_BASE"
    source "$CONDA_SH"

    # 检查环境是否存在
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Activating existing environment: $CONDA_ENV_NAME"
        conda activate "$CONDA_ENV_NAME"
    else
        if [ "$CREATE_ENV_IF_NOT_EXISTS" = true ]; then
            echo "Creating new Conda environment: $CONDA_ENV_NAME with Python $PYTHON_VERSION"
            conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
            conda activate "$CONDA_ENV_NAME"
        else
            echo "Error: Conda environment '$CONDA_ENV_NAME' not found."
            echo "Set CREATE_ENV_IF_NOT_EXISTS=true to auto-create it."
            exit 1
        fi
    fi

    PYTHON_CMD="python"
    echo "Using Conda Python: $(which python)"
else
    # 不使用conda，使用系统Python
    PYTHON_CMD="python3"
    command -v python3 &> /dev/null || PYTHON_CMD="python"
    echo "Using system Python: $(which $PYTHON_CMD)"
fi

# 创建工作目录
mkdir -p "$WORK_DIR"

# 克隆或更新代码仓库
if [ -d "$REPO_DIR" ]; then
    echo "Updating repository..."
    cd "$REPO_DIR"
    git pull -q origin main 2>/dev/null || (cd .. && rm -rf Merging-EVAL && git clone -q "$REPO_URL")
    cd - > /dev/null
else
    echo "Cloning repository..."
    cd "$WORK_DIR"
    git clone -q "$REPO_URL"
    cd - > /dev/null
fi

# 安装依赖（使用 requirements-minimal.txt）
PIP_INSTALL_CMD="$PYTHON_CMD -m pip install "
if [ -n "$PIP_INDEX_URL" ]; then
    echo "Using pip mirror: $PIP_INDEX_URL"
    PIP_INSTALL_CMD="$PIP_INSTALL_CMD -i $PIP_INDEX_URL"
fi

if [ -f "$REPO_DIR/requirements-minimal.txt" ]; then
    echo "Installing dependencies from requirements-minimal.txt..."
    $PIP_INSTALL_CMD -r "$REPO_DIR/requirements-minimal.txt" 2>/dev/null || true
else
    echo "Warning: requirements-minimal.txt not found, installing packages individually..."
    $PIP_INSTALL_CMD torch transformers datasets pandas tqdm requests 2>/dev/null || true
fi

# 安装swanlab（可能从自定义源）
if [ -n "$SWANLAB_INDEX_URL" ]; then
    echo "Installing swanlab from custom index: $SWANLAB_INDEX_URL"

    # Extract host from URL for trusted-host flag
    SWANLAB_HOST=$(echo "$SWANLAB_INDEX_URL" | sed -E 's|^https?://([^/:]+).*|\1|')

    # Try to install swanlab from custom index with fallback to PyPI
    # Use --trusted-host for HTTP sources and --extra-index-url to allow fallback
    if [[ "$SWANLAB_INDEX_URL" == http://* ]]; then
        echo "Using HTTP source, adding trusted-host: $SWANLAB_HOST"
        $PYTHON_CMD -m pip install --trusted-host "$SWANLAB_HOST" \
            --extra-index-url "$SWANLAB_INDEX_URL" swanlab==0.8.0 || \
        $PYTHON_CMD -m pip install swanlab
    else
        # HTTPS source
        $PYTHON_CMD -m pip install --extra-index-url "$SWANLAB_INDEX_URL" swanlab==0.8.0 || \
        $PYTHON_CMD -m pip install swanlab
    fi
else
    echo "Installing swanlab from standard source..."
    $PIP_INSTALL_CMD swanlab 2>/dev/null || true
fi

# 准备路径
EVAL_SCRIPT="$REPO_DIR/scripts/eval.py"
DATA_DIR="$REPO_DIR/data/eval_partial"
OUTPUT_DIR="$REPO_DIR/output/$TASK_ID"
CACHE_DIR="$REPO_DIR/cache/$TASK_ID"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# 显示配置信息
echo "Starting evaluation..."
echo "Model: $MODEL_URL"
echo "Dataset: $DATASET"
echo "Task ID: $TASK_ID"

# 构建可选参数
API_KEY_ARG=""
if [ -n "$API_KEY" ]; then
    API_KEY_ARG="--api_key $API_KEY"
fi

MODEL_ID_ARG=""
if [ -n "$MODEL_ID" ]; then
    MODEL_ID_ARG="--model_id $MODEL_ID"
fi

# 执行评估
if [ "$DATASET" = "all" ]; then
    echo "Evaluating all datasets..."
    # 评估所有数据集
    $PYTHON_CMD "$EVAL_SCRIPT" \
        --model "$MODEL_URL" \
        --tokenizer "$MODEL_URL" \
        --dataset "$DATA_DIR" \
        --output "$OUTPUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --experiment_name "$TASK_ID" \
        --callback_url "$CALLBACK_URL" \
        --task_id "$TASK_ID" \
        $MODEL_ID_ARG \
        --benchmark_id "$BENCHMARK_ID" \
        $API_KEY_ARG \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --use_swanlab \
        --swanlab_mode "$SWANLAB_MODE" # 2>&1 | grep -E "(Evaluation|Callback|Error|✅|❌|Final)"
else
    echo "Evaluating single dataset: $DATASET"
    # 评估单个数据集
    $PYTHON_CMD "$EVAL_SCRIPT" \
        --model "$MODEL_URL" \
        --tokenizer "$MODEL_URL" \
        --file "$DATA_DIR/${DATASET}.json" \
        --output "$OUTPUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --experiment_name "$TASK_ID" \
        --callback_url "$CALLBACK_URL" \
        --task_id "$TASK_ID" \
        $MODEL_ID_ARG \
        --benchmark_id "$BENCHMARK_ID" \
        $API_KEY_ARG \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --use_swanlab \
        --swanlab_mode "$SWANLAB_MODE" # 2>&1 | grep -E "(Evaluation|Callback|Error|✅|❌|Final)"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed!"
    echo "Results: $OUTPUT_DIR"
else
    echo "Evaluation failed"
    exit 1
fi
