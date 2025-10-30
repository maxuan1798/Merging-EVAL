# Model Merging API

A Flask-based REST API for merging language models using the `merging-eval` package.

## Features

- 🚀 **Multiple Merging Methods**: Support for average_merging, task_arithmetic, ties_merging, ties_merging_dare, mask_merging
- 📊 **Async Job Processing**: Background job execution with status tracking
- 🔧 **Flexible Parameters**: Configurable scaling coefficients, mask rates, and exclusion patterns
- 💾 **Model Persistence**: Automatic saving of merged models
- 📱 **RESTful Interface**: Standard HTTP endpoints for easy integration
- 🔍 **Job Management**: Track and monitor merging jobs

## Quick Start

### 1. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python run_api.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check

**GET** `/api/health`

Check API status and availability.

```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "merging_available": true,
  "version": "1.0.0"
}
```

### Get Available Methods

**GET** `/api/methods`

Get list of available merging methods and their parameters.

```bash
curl http://localhost:5000/api/methods
```

**Response:**
```json
{
  "methods": [
    {
      "name": "task_arithmetic",
      "description": "Task vector arithmetic merging",
      "parameters": {
        "scaling_coefficient": {
          "type": "float",
          "default": 0.3,
          "description": "Scaling coefficient for task vectors"
        }
      }
    }
  ]
}
```

### Merge Models

**POST** `/api/merge`

Start a model merging job.

**Request Body:**
```json
{
  "base_model": "Qwen/Qwen2.5-0.5B",
  "models_to_merge": [
    "InfiX-ai/Qwen-base-0.5B-code",
    "InfiX-ai/Qwen-base-0.5B-algebra"
  ],
  "merge_method": "task_arithmetic",
  "output_dir": "./merged_output",
  "scaling_coefficient": 0.3,
  "use_gpu": true,
  "param_value_mask_rate": 0.8,
  "exclude_param_names_regex": []
}
```

**Example using curl:**
```bash
curl -X POST http://localhost:5000/api/merge \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "Qwen/Qwen2.5-0.5B",
    "models_to_merge": [
      "InfiX-ai/Qwen-base-0.5B-code",
      "InfiX-ai/Qwen-base-0.5B-algebra"
    ],
    "merge_method": "task_arithmetic",
    "scaling_coefficient": 0.3,
    "use_gpu": true
  }'
```

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "message": "Merging job started",
  "check_status": "/api/jobs/123e4567-e89b-12d3-a456-426614174000"
}
```

### Get Job Status

**GET** `/api/jobs/{job_id}`

Check the status of a merging job.

```bash
curl http://localhost:5000/api/jobs/123e4567-e89b-12d3-a456-426614174000
```

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00.000Z",
  "started_at": "2024-01-15T10:30:01.000Z",
  "completed_at": "2024-01-15T10:35:00.000Z",
  "error_message": null,
  "output_path": "./api/output/123e4567-e89b-12d3-a456-426614174000",
  "params": {
    "base_model": "Qwen/Qwen2.5-0.5B",
    "models_to_merge": ["model1", "model2"],
    "merge_method": "task_arithmetic"
  }
}
```

### List All Jobs

**GET** `/api/jobs`

Get list of all merging jobs.

```bash
curl http://localhost:5000/api/jobs
```

### Download Merged Model

**GET** `/api/download/{job_id}`

Get information about downloading the merged model.

```bash
curl http://localhost:5000/api/download/123e4567-e89b-12d3-a456-426614174000
```

### Get Example Request

**GET** `/api/example`

Get an example request payload.

```bash
curl http://localhost:5000/api/example
```

## Parameters Reference

### Required Parameters

- `base_model` (string): Path to the base model (HuggingFace model identifier or local path)
- `models_to_merge` (array): List of model paths to merge
- `merge_method` (string): Merging method to use

### Optional Parameters

- `output_dir` (string): Output directory for merged model (default: auto-generated)
- `scaling_coefficient` (float): Scaling coefficient for merging (default: 1.0)
- `use_gpu` (boolean): Use GPU for merging (default: false)
- `param_value_mask_rate` (float): Parameter value mask rate for TIES methods (default: 0.8)
- `weight_mask_rates` (array): Weight mask rates for mask_merging (default: [0.5, 0.5])
- `exclude_param_names_regex` (array): Regex patterns for parameters to exclude (default: [])

## Merging Methods

### average_merging
Equal-weight averaging of models. Simple but effective for similar models.

### task_arithmetic
Task vector arithmetic merging. Creates task vectors and applies scaling.

### ties_merging
TIES merging with parameter pruning. Prunes conflicting parameters before merging.

### ties_merging_dare
TIES merging with DARE variant. Enhanced version with different pruning strategy.

### mask_merging
Mask-based merging with weight masking. Uses magnitude-based masking.

## Usage Examples

### Python Client

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:5000/api"

# Start a merge job
payload = {
    "base_model": "Qwen/Qwen2.5-0.5B",
    "models_to_merge": [
        "InfiX-ai/Qwen-base-0.5B-code",
        "InfiX-ai/Qwen-base-0.5B-algebra"
    ],
    "merge_method": "task_arithmetic",
    "scaling_coefficient": 0.3,
    "use_gpu": True
}

response = requests.post(f"{BASE_URL}/merge", json=payload)
job_data = response.json()
job_id = job_data['job_id']

print(f"Started job: {job_id}")

# Check job status
while True:
    status_response = requests.get(f"{BASE_URL}/jobs/{job_id}")
    job_status = status_response.json()

    if job_status['status'] == 'completed':
        print(f"Job completed! Output: {job_status['output_path']}")
        break
    elif job_status['status'] == 'failed':
        print(f"Job failed: {job_status['error_message']}")
        break
    else:
        print(f"Job status: {job_status['status']}")
        time.sleep(5)
```

### JavaScript Client

```javascript
const startMerge = async () => {
    const payload = {
        base_model: "Qwen/Qwen2.5-0.5B",
        models_to_merge: [
            "InfiX-ai/Qwen-base-0.5B-code",
            "InfiX-ai/Qwen-base-0.5B-algebra"
        ],
        merge_method: "task_arithmetic",
        scaling_coefficient: 0.3,
        use_gpu: true
    };

    const response = await fetch('http://localhost:5000/api/merge', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    });

    const jobData = await response.json();
    console.log(`Started job: ${jobData.job_id}`);

    return jobData.job_id;
};

const checkJobStatus = async (jobId) => {
    const response = await fetch(`http://localhost:5000/api/jobs/${jobId}`);
    const jobStatus = await response.json();
    return jobStatus;
};
```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "run_api.py"]
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**: Ensure model paths are correct and accessible
2. **GPU Memory Issues**: Set `use_gpu: false` or reduce batch size
3. **Import Errors**: Verify all dependencies are installed
4. **Timeout Issues**: Increase timeout for large models

### Logs

Check the Flask logs for detailed error information:

```bash
tail -f api.log
```

## License

MIT License - See main project LICENSE file.