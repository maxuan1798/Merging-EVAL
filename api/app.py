#!/usr/bin/env python3
"""
Flask API for Model Merging

This API provides endpoints for merging language models using the merging-eval package.
Supports multiple merging methods and parameter configurations.
"""
import os
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Set environment variables for transformers
os.environ['TRANSFORMERS_NO_TORCHVISION'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = './api/uploads'
app.config['OUTPUT_FOLDER'] = './api/output'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Import merging functionality
try:
    import torch
    from merge import MergingMethod
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from merge.config import get_hf_config
    MERGING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import merging modules: {e}")
    MERGING_AVAILABLE = False

# In-memory storage for job status (in production, use a database)
jobs = {}

class MergeJob:
    """Represents a model merging job"""

    def __init__(self, job_id, params):
        self.job_id = job_id
        self.params = params
        self.status = 'pending'  # pending, running, completed, failed
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.output_path = None

    def to_dict(self):
        """Convert job to dictionary for JSON response"""
        return {
            'job_id': self.job_id,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'output_path': self.output_path,
            'params': self.params
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'merging_available': MERGING_AVAILABLE,
        'version': '1.0.0'
    }
    return jsonify(status)

@app.route('/api/methods', methods=['GET'])
def get_merging_methods():
    """Get available merging methods"""
    if not MERGING_AVAILABLE:
        return jsonify({'error': 'Merging functionality not available'}), 503

    methods = [
        {
            'name': 'average_merging',
            'description': 'Equal-weight averaging of models',
            'parameters': {
                'scaling_coefficient': {'type': 'float', 'default': 1.0, 'description': 'Scaling coefficient'}
            }
        },
        {
            'name': 'task_arithmetic',
            'description': 'Task vector arithmetic merging',
            'parameters': {
                'scaling_coefficient': {'type': 'float', 'default': 0.3, 'description': 'Scaling coefficient for task vectors'}
            }
        },
        {
            'name': 'ties_merging',
            'description': 'TIES merging with parameter pruning',
            'parameters': {
                'scaling_coefficient': {'type': 'float', 'default': 0.3, 'description': 'Scaling coefficient'},
                'param_value_mask_rate': {'type': 'float', 'default': 0.8, 'description': 'Mask rate for smallest parameters'}
            }
        },
        {
            'name': 'ties_merging_dare',
            'description': 'TIES merging with DARE variant',
            'parameters': {
                'scaling_coefficient': {'type': 'float', 'default': 0.3, 'description': 'Scaling coefficient'},
                'param_value_mask_rate': {'type': 'float', 'default': 0.8, 'description': 'Mask rate for smallest parameters'}
            }
        },
        {
            'name': 'mask_merging',
            'description': 'Mask-based merging with weight masking',
            'parameters': {
                'weight_mask_rates': {'type': 'list[float]', 'default': [0.5, 0.5], 'description': 'Mask rates for each model'},
                'mask_strategy': {'type': 'string', 'default': 'magnitude', 'description': 'Masking strategy'}
            }
        }
    ]

    return jsonify({'methods': methods})

@app.route('/api/merge', methods=['POST'])
def merge_models():
    """
    Merge models endpoint

    Expected JSON payload:
    {
        "base_model": "huggingface/model/path",
        "models_to_merge": ["model1/path", "model2/path"],
        "merge_method": "task_arithmetic",
        "output_dir": "optional/output/path",
        "scaling_coefficient": 0.3,
        "use_gpu": true,
        "param_value_mask_rate": 0.8,
        "weight_mask_rates": [0.5, 0.5],
        "exclude_param_names_regex": [],
        "hf_token": "optional_hf_token",
        "use_hf_auth": false,
        "local_files_only": false
    }
    """
    if not MERGING_AVAILABLE:
        return jsonify({'error': 'Merging functionality not available'}), 503

    try:
        data = request.get_json()

        # Validate required parameters
        required_params = ['base_model', 'models_to_merge', 'merge_method']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Missing required parameter: {param}'}), 400

        # Create job
        job_id = str(uuid.uuid4())
        job = MergeJob(job_id, data)
        jobs[job_id] = job

        # Start merging in background (in production, use a task queue)
        import threading
        thread = threading.Thread(target=execute_merging, args=(job,))
        thread.daemon = True
        thread.start()

        logger.info(f"Started merging job {job_id}")

        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Merging job started',
            'check_status': f'/api/jobs/{job_id}'
        }), 202

    except Exception as e:
        logger.error(f"Error starting merge job: {e}")
        return jsonify({'error': str(e)}), 500

def execute_merging(job):
    """Execute model merging in background"""
    try:
        job.status = 'running'
        job.started_at = datetime.now()

        params = job.params

        # Set output directory
        output_dir = params.get('output_dir')
        if not output_dir:
            output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job.job_id)

        logger.info(f"Starting merge for job {job.job_id}: {params['merge_method']}")

        # Initialize HF configuration
        hf_config = get_hf_config(
            token=params.get('hf_token'),
            use_auth=params.get('use_hf_auth', False)
        )

        # Prepare model loading kwargs
        model_kwargs = {
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True
        }

        # Add authentication kwargs if needed
        if not params.get('local_files_only', False):
            model_kwargs.update(hf_config.get_model_loading_kwargs())
        else:
            model_kwargs['local_files_only'] = True
            logger.info("Running in local files only mode")

        # Add authentication info to log
        if hf_config.should_use_auth():
            logger.info("Using Hugging Face authentication")
            if hf_config.get_token():
                logger.info("HF token provided")

        # Load models
        device = "cuda" if params.get('use_gpu', False) else "cpu"

        logger.info(f"Loading base model: {params['base_model']}")
        base_model = AutoModelForCausalLM.from_pretrained(
            params['base_model'],
            **model_kwargs
        ).to(device)

        # Load tokenizer
        tokenizer_kwargs = {'trust_remote_code': True}
        if not params.get('local_files_only', False):
            tokenizer_kwargs.update(hf_config.get_tokenizer_loading_kwargs())
        else:
            tokenizer_kwargs['local_files_only'] = True

        tokenizer = AutoTokenizer.from_pretrained(
            params['base_model'],
            **tokenizer_kwargs
        )

        # Load models to merge
        models_to_merge = []
        for i, model_path in enumerate(params['models_to_merge']):
            logger.info(f"Loading model {i+1}: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            ).to(device)
            models_to_merge.append(model)

        # Create merging engine
        merging_engine = MergingMethod(merging_method_name=params['merge_method'])

        # Prepare merging parameters
        merge_kwargs = {
            'merged_model': base_model,
            'models_to_merge': models_to_merge,
            'exclude_param_names_regex': params.get('exclude_param_names_regex', []),
            'scaling_coefficient': params.get('scaling_coefficient', 1.0)
        }

        # Add method-specific parameters
        if params['merge_method'] in ['ties_merging', 'ties_merging_dare']:
            merge_kwargs['param_value_mask_rate'] = params.get('param_value_mask_rate', 0.8)
        elif params['merge_method'] == 'mask_merging':
            merge_kwargs['weight_mask_rates'] = params.get('weight_mask_rates', [0.5, 0.5])

        # Perform merging
        logger.info(f"Performing {params['merge_method']} merging...")
        merged_model = merging_engine.get_merged_model(**merge_kwargs)

        # Save merged model
        logger.info(f"Saving merged model to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        merged_model = merged_model.to(torch.bfloat16)
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Update job status
        job.status = 'completed'
        job.completed_at = datetime.now()
        job.output_path = output_dir

        logger.info(f"Merge job {job.job_id} completed successfully")

    except Exception as e:
        logger.error(f"Merge job {job.job_id} failed: {e}")
        job.status = 'failed'
        job.error_message = str(e)
        job.completed_at = datetime.now()

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of a merging job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(jobs[job_id].to_dict())

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all merging jobs"""
    job_list = [job.to_dict() for job in jobs.values()]
    return jsonify({'jobs': job_list})

@app.route('/api/download/<job_id>', methods=['GET'])
def download_model(job_id):
    """Download merged model files"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job.status != 'completed':
        return jsonify({'error': 'Job not completed'}), 400

    # In production, you might want to create a zip file
    # For now, just return the path
    return jsonify({
        'job_id': job_id,
        'output_path': job.output_path,
        'message': 'Model files are available at the specified path'
    })

@app.route('/api/example', methods=['GET'])
def get_example_request():
    """Get example request payload"""
    example = {
        "base_model": "Qwen/Qwen2.5-0.5B",
        "models_to_merge": [
            "InfiX-ai/Qwen-base-0.5B-code",
            "InfiX-ai/Qwen-base-0.5B-algebra"
        ],
        "merge_method": "task_arithmetic",
        "output_dir": "./merged_output",
        "scaling_coefficient": 0.3,
        "use_gpu": True,
        "param_value_mask_rate": 0.8,
        "exclude_param_names_regex": []
    }
    return jsonify(example)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    import torch

    # Check if merging is available
    if not MERGING_AVAILABLE:
        logger.warning("Merging functionality not available - API will run in limited mode")

    # Start Flask development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )