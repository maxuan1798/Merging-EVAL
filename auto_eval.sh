#!/bin/bash
# 自动化模型评估脚本
# 自动从GitHub下载代码，从HuggingFace下载模型，并执行评估
# 支持UV虚拟环境

set -e

# ========================================
# 用户配置 - 修改这些参数
# ========================================
# UV环境配置
UV_ENV_NAME="dev_eval_env"              # UV环境名称
CREATE_ENV_IF_NOT_EXISTS=true          # 如果环境不存在是否自动创建
PYTHON_VERSION="3.10"                  # 创建新环境时使用的Python版本
UV_ENV_DIR="./.venv"                   # UV环境目录（相对于脚本路径）

# HuggingFace模型仓库地址
MODEL_URL="https://huggingface.co/microsoft/DialoGPT-medium"  # 修改为您的模型地址

# 评估数据集 (可选: algebra, analysis, discrete, geometry, number_theory, all)
DATASET="all"

# Callback配置
CALLBACK_URL=""
TASK_ID=""
MODEL_ID=""  # 留空则自动从MODEL_URL提取，或手动指定模型ID
BENCHMARK_ID=""
BENCHMARK_INDICES=""
API_KEY=""  # 留空使用默认值，或设置为您的API Key

# 评估参数
BATCH_SIZE=8
MAX_LENGTH=2048

# SwanLab模式配置 (可选: local, cloud, disabled)
SWANLAB_MODE=""  # local=本地存储, cloud=云端存储, disabled=禁用SwanLab

# PIP镜像源配置 (可选，用于加速下载)
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"  # 留空则使用默认源

# SwanLab自定义源配置 (可选，用于从私有源安装swanlab)
SWANLAB_INDEX_URL=""  # 留空则使用PIP_INDEX_URL或默认源
# SwanLab API Key配置
export SWANLAB_API_KEY=""
export SWANLAB_BASE_URL=""

# HuggingFace镜像配置 (可选，使用 hf-mirror 加速下载)
USE_HF_MIRROR=true  # 设置为 true 使用 hf-mirror，设置为 false 使用官方源
HF_MIRROR_URL="https://hf-mirror.com"  # hf-mirror 地址

# =======================================
# 自动执行部分 - 无需修改
# ========================================
REPO_URL="https://github.com/maxuan1798/Merging-EVAL.git"
WORK_DIR="./eval_workspace"
REPO_DIR="$WORK_DIR/Merging-EVAL"

# UV环境设置
if [ -n "$UV_ENV_NAME" ]; then
    echo "Setting up UV environment: $UV_ENV_NAME"

    # 检查uv是否安装
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed."
        echo "Please install uv first:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  or visit: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    # 获取脚本所在目录的绝对路径
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    UV_ENV_FULL_PATH="$SCRIPT_DIR/$UV_ENV_DIR"

    # 检查环境是否存在
    if [ -d "$UV_ENV_FULL_PATH" ]; then
        echo "Activating existing UV environment: $UV_ENV_FULL_PATH"
        source "$UV_ENV_FULL_PATH/bin/activate"
    else
        if [ "$CREATE_ENV_IF_NOT_EXISTS" = true ]; then
            echo "Creating new UV environment: $UV_ENV_FULL_PATH with Python $PYTHON_VERSION"
            cd "$SCRIPT_DIR"
            uv venv "$UV_ENV_DIR" --python "$PYTHON_VERSION"
            source "$UV_ENV_FULL_PATH/bin/activate"
            cd - > /dev/null
        else
            echo "Error: UV environment '$UV_ENV_FULL_PATH' not found."
            echo "Set CREATE_ENV_IF_NOT_EXISTS=true to auto-create it."
            exit 1
        fi
    fi

    PYTHON_CMD="python"
    echo "Using UV Python: $(which python)"
else
    # 不使用uv，使用系统Python
    PYTHON_CMD="python3"
    command -v python3 &> /dev/null || PYTHON_CMD="python"
    echo "Using system Python: $(which $PYTHON_CMD)"
fi

# 创建工作目录
mkdir -p "$WORK_DIR"

# 克隆或更新代码仓库
if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists, skipping update to preserve modifications..."
    # cd "$REPO_DIR"
    # git pull -q origin main 2>/dev/null || (cd .. && rm -rf Merging-EVAL && git clone -q "$REPO_URL")
    # cd - > /dev/null
else
    echo "Cloning repository..."
    cd "$WORK_DIR"
    git clone -q "$REPO_URL"
    cd - > /dev/null
fi

# 安装依赖（使用 requirements-minimal.txt）
if [ -n "$UV_ENV_NAME" ]; then
    # 使用uv安装包
    UV_INSTALL_CMD="uv pip install"
    if [ -n "$PIP_INDEX_URL" ]; then
        echo "Using pip mirror: $PIP_INDEX_URL"
        UV_INSTALL_CMD="$UV_INSTALL_CMD -i $PIP_INDEX_URL"
    fi
    PIP_INSTALL_CMD="$UV_INSTALL_CMD"
else
    # 使用传统pip安装包
    PIP_INSTALL_CMD="$PYTHON_CMD -m pip install "
    if [ -n "$PIP_INDEX_URL" ]; then
        echo "Using pip mirror: $PIP_INDEX_URL"
        PIP_INSTALL_CMD="$PIP_INSTALL_CMD -i $PIP_INDEX_URL"
    fi
fi

# 函数：验证关键依赖是否安装成功
check_critical_dependencies() {
    echo "Verifying critical dependencies..."
    local missing_deps=()
    
    # 检查 torch
    if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
        missing_deps+=("torch")
    fi
    
    # 检查 transformers
    if ! $PYTHON_CMD -c "import transformers" 2>/dev/null; then
        missing_deps+=("transformers")
    fi
    
    # 检查 datasets（可选，跳过检查）
    echo "Note: datasets library skipped to avoid pyarrow dependency issues"
    
    # 检查 accelerate
    if ! $PYTHON_CMD -c "import accelerate" 2>/dev/null; then
        missing_deps+=("accelerate")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo "Error: Missing critical dependencies: ${missing_deps[*]}"
        return 1
    else
        echo "All critical dependencies are installed."
        return 0
    fi
}

# 跳过 requirements-minimal.txt 安装（避免 pyarrow 依赖问题）
echo "Skipping requirements-minimal.txt installation to avoid pyarrow dependency issues..."
echo "Installing critical packages individually..."

# 直接安装关键包
DEPENDENCIES_INSTALLED=false

# 安装 PyTorch (需要 2.6+ 版本解决安全漏洞)
echo "Installing PyTorch (version 2.6+ for security)..."
if [ -n "$UV_ENV_NAME" ]; then
    # 使用uv安装PyTorch，先尝试官方源
    if ! uv pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
        echo "Failed to install PyTorch from official source, trying with mirror..."
        if [ -n "$PIP_INDEX_URL" ]; then
            uv pip install torch>=2.6.0 torchvision torchaudio -i $PIP_INDEX_URL || {
                echo "Error: Failed to install PyTorch. Please check your internet connection and try again."
                exit 1
            }
        else
            uv pip install "torch>=2.6.0" torchvision torchaudio || {
                echo "Error: Failed to install PyTorch. Please check your internet connection and try again."
                exit 1
            }
        fi
    fi
else
    # 使用传统pip安装
    if ! $PYTHON_CMD -m pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
        echo "Failed to install PyTorch from official source, trying with mirror..."
        $PIP_INSTALL_CMD "torch>=2.6.0" torchvision torchaudio || {
            echo "Error: Failed to install PyTorch. Please check your internet connection and try again."
            exit 1
        }
    fi
fi

# 安装其他关键依赖（跳过 pyarrow，使用 pandas 内置的 parquet 支持）
echo "Installing other dependencies..."
$PIP_INSTALL_CMD transformers pandas tqdm requests numpy scipy accelerate || {
    echo "Error: Failed to install some dependencies. Please check your internet connection and try again."
    exit 1
}

# 跳过 datasets 安装（避免 pyarrow 依赖问题，对于模型评估不是必需的）
echo "Skipping datasets installation (not required for model evaluation)..."
echo "Note: Will use direct JSON file loading instead of datasets library"

# 跳过 PyArrow 安装（避免构建问题，对于模型评估不是必需的）
echo "Skipping PyArrow installation (not required for model evaluation)..."
echo "Note: Some datasets features may be limited without PyArrow"

DEPENDENCIES_INSTALLED=true

# 验证关键依赖
if ! check_critical_dependencies; then
    echo "Error: Critical dependencies are missing. Please check the installation logs above."
    exit 1
fi

# 安装swanlab（可能从自定义源）
if [ -n "$SWANLAB_INDEX_URL" ]; then
    echo "Installing swanlab from custom index: $SWANLAB_INDEX_URL"

    # Extract host from URL for trusted-host flag
    SWANLAB_HOST=$(echo "$SWANLAB_INDEX_URL" | sed -E 's|^https?://([^/:]+).*|\1|')

    # Try to install swanlab from custom index with fallback to PyPI
    if [ -n "$UV_ENV_NAME" ]; then
        # 使用uv安装
        if [[ "$SWANLAB_INDEX_URL" == http://* ]]; then
            echo "Using HTTP source, adding trusted-host: $SWANLAB_HOST"
            uv pip install --trusted-host "$SWANLAB_HOST" \
                --extra-index-url "$SWANLAB_INDEX_URL" "swanlab[dashboard]" --prerelease=allow || \
            uv pip install "swanlab[dashboard]" --prerelease=allow || \
            uv pip install swanlab
        else
            # HTTPS source
            uv pip install --extra-index-url "$SWANLAB_INDEX_URL" "swanlab[dashboard]" --prerelease=allow || \
            uv pip install "swanlab[dashboard]" --prerelease=allow || \
            uv pip install swanlab
        fi
    else
        # 使用传统pip安装
        if [[ "$SWANLAB_INDEX_URL" == http://* ]]; then
            echo "Using HTTP source, adding trusted-host: $SWANLAB_HOST"
            $PYTHON_CMD -m pip install --trusted-host "$SWANLAB_HOST" \
                --extra-index-url "$SWANLAB_INDEX_URL" "swanlab[dashboard]" --prerelease=allow || \
            $PYTHON_CMD -m pip install "swanlab[dashboard]" --prerelease=allow || \
            $PYTHON_CMD -m pip install swanlab
        else
            # HTTPS source
            $PYTHON_CMD -m pip install --extra-index-url "$SWANLAB_INDEX_URL" "swanlab[dashboard]" --prerelease=allow || \
            $PYTHON_CMD -m pip install "swanlab[dashboard]" --prerelease=allow || \
            $PYTHON_CMD -m pip install swanlab
        fi
    fi
else
    echo "Installing swanlab from standard source..."
    if [ -n "$UV_ENV_NAME" ]; then
        # 使用uv安装swanlab[dashboard]，支持预发布版本
        echo "Installing swanlab[dashboard] with uv..."
        uv pip install "swanlab[dashboard]" --prerelease=allow 2>/dev/null || {
            echo "Warning: SwanLab[dashboard] installation failed, trying without dashboard..."
            uv pip install swanlab 2>/dev/null || true
        }
    else
        # 使用传统pip安装
        $PIP_INSTALL_CMD "swanlab[dashboard]" --prerelease=allow 2>/dev/null || {
            echo "Warning: SwanLab[dashboard] installation failed, trying without dashboard..."
            $PIP_INSTALL_CMD swanlab 2>/dev/null || true
        }
    fi
fi

# 准备路径
EVAL_SCRIPT="$REPO_DIR/scripts/eval.py"
DATA_DIR="$REPO_DIR/data/eval_partial"
OUTPUT_DIR="$REPO_DIR/output/$TASK_ID"
CACHE_DIR="$REPO_DIR/cache/$TASK_ID"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# 设置 HuggingFace 镜像（如果需要）
if [ "$USE_HF_MIRROR" = true ]; then
    echo "Using HuggingFace mirror: $HF_MIRROR_URL"
    export HF_ENDPOINT="$HF_MIRROR_URL"
    
    # 将 MODEL_URL 从 huggingface.co 转换为 hf-mirror.com
    if [[ "$MODEL_URL" == *"huggingface.co"* ]]; then
        MODEL_URL="${MODEL_URL//huggingface.co/hf-mirror.com}"
        echo "Converted MODEL_URL to mirror: $MODEL_URL"
    fi
else
    echo "Using official HuggingFace source"
fi

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
        --swanlab_mode "$SWANLAB_MODE" \
        --indices $BENCHMARK_INDICES # 2>&1 | grep -E "(Evaluation|Callback|Error|✅|❌|Final)"
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
        --swanlab_mode "$SWANLAB_MODE" \
        --indices $BENCHMARK_INDICES # 2>&1 | grep -E "(Evaluation|Callback|Error|✅|❌|Final)"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed!"
    echo "Results: $OUTPUT_DIR"
else
    echo "Evaluation failed"
    exit 1
fi