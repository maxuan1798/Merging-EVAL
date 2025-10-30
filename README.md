# Model Merging Scaling Laws in Large Language Models

This repository contains the official implementation for the paper **"Model Merging Scaling Laws in Large Language Models"**.

> **Abstract**: We study empirical scaling laws for language model merging measured by cross-entropy. Despite the wide use of merging in practice, there is no quantitative rule that predicts what happens as we merge more experts or scale the base model. We show that the expected cross-entropy under equal-weight merging follows a simple, two-factor law: at a fixed base size, the loss decreases roughly in proportion to the inverse of the number of merged experts, and when varying the base size, the loss decreases according to the same power-law trend observed in pretraining. The law has two interpretable components: a baseline floor determined by the base model and its shared direction across domains, and a merging gain that reflects how dispersed the task vectors are under the model's curvature. Although individual merge outcomes can fluctuate unpredictably, the average over random subsets reveals a clear inverse relationship with the number of experts, and the variability shrinks as more experts are included. This law turns trial-and-error into predictive planning for budget-aware expert counts, saturation detection, and rule comparison across model scales.

## Repository Structure

```
.
├── src/                    # Core source code
│   ├── expert_training/   # Code for expert model training
│   └── merge/             # Code for model merging algorithms
├── scripts/               # Execution scripts (SLURM-based)
│   ├── train.sh          # Example training script
│   ├── merge.sh          # Example model merging script
│   ├── eval.sh           # Example evaluation script
│   └── ...               # Additional dependency files
├── data/                  # Example data files
└── ENV.txt               # Python environment configuration
```

## Installation and Setup

### 1. Python Environment

Create a conda environment using the provided specification:

```bash
conda create --name merging-laws --file ENV.txt
conda activate merging-laws
```

### 2. Additional Dependencies

Our expert training implementation builds upon the **OpenRLHF** framework. After setting up the base environment, install OpenRLHF:

```bash
pip install openrlhf
```

*Note: Please refer to the official [OpenRLHF repository](https://github.com/OpenRLHF/OpenRLHF) for any additional installation requirements or troubleshooting.*

## Usage

### Expert Model Training

To train expert models on specialized domains:

```bash
cd scripts/
sbatch train.sh
```

Key configuration parameters in `train.sh`.

### Model Merging

Execute model merging with equal-weight averaging:

```bash
sbatch merge.sh
```

Configuration parameters in `merge.sh`.

### Evaluation

Evaluate merged models on cross-entropy metrics:

```bash
sbatch eval.sh
```

## Data Format

The `data/` directory contains example datasets in the expected format.

## PyPI Package Installation

This project is also available as a Python package on PyPI:

### Install from PyPI

```bash
pip install merging-eval
```

### Install with full dependencies

```bash
pip install "merging-eval[full]"
```

### Install in development mode

```bash
git clone https://github.com/Merging-EVAL/Merging-EVAL.git
cd Merging-EVAL
pip install -e .
```

## Package Publishing

### Local Development

To build and test the package locally:

```bash
./build_and_test.sh
```

### Publishing to PyPI

#### 1. Setup PyPI Credentials

Copy the template configuration file:

```bash
cp .pypirc.template ~/.pypirc
```

Edit `~/.pypirc` and replace the placeholder tokens with your actual PyPI API tokens:

- Get your PyPI API token from [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
- Get your TestPyPI API token from [test.pypi.org/manage/account/token/](https://test.pypi.org/manage/account/token/)

#### 2. Build and Publish

**Test PyPI (Recommended for testing):**
```bash
./publish_to_pypi.sh test
```

**Production PyPI:**
```bash
./publish_to_pypi.sh production
```

#### 3. Manual Publishing (Alternative)

If you prefer manual control:

```bash
# Build the package
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### Automated Publishing

This repository includes GitHub Actions workflow that automatically publishes to PyPI when a new release is created. The workflow is triggered on:
- New releases created in GitHub
- Tags pushed to the repository

To create a new release:
1. Update the version in `pyproject.toml`
2. Create a new release in GitHub
3. The workflow will automatically build and publish the package

### Package Structure

The package provides the following modules:

```python
import merge
from merge import MergingMethod, FlopsCounter

# Available merging methods
merging_methods = [
    "average_merging",      # Equal-weight averaging
    "task_arithmetic",      # Task vector arithmetic
    "ties_merging",         # TIES merging algorithm
    "ties_merging_dare",    # TIES with DARE variant
    "mask_merging"          # Mask-based merging
]
```

## Package Structure

```
merging-eval/
├── merge/                    # Model merging algorithms
│   ├── main_merging.py      # Main merging entry point
│   ├── merging_methods.py   # Various merging techniques
│   └── utils.py             # Utility functions
├── expert_training/         # Expert model training
│   ├── train_expert.py      # Expert training
│   └── domain_specialization.py # Domain specialization
└── __init__.py              # Package initialization
``` 

