#!/usr/bin/env python3
"""
Client Example for Model Merging API

This script demonstrates how to use the API from a client application.
"""
import requests
import json
import time

class ModelMergingClient:
    """Client for the Model Merging API"""

    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"

    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.api_url}/health")
        return response.json()

    def get_methods(self):
        """Get available merging methods"""
        response = requests.get(f"{self.api_url}/methods")
        return response.json()

    def start_merge(self, params):
        """Start a model merging job"""
        response = requests.post(f"{self.api_url}/merge", json=params)
        if response.status_code == 202:
            return response.json()
        else:
            raise Exception(f"Failed to start merge: {response.text}")

    def get_job_status(self, job_id):
        """Get status of a merging job"""
        response = requests.get(f"{self.api_url}/jobs/{job_id}")
        return response.json()

    def wait_for_completion(self, job_id, poll_interval=5):
        """Wait for job completion"""
        while True:
            status = self.get_job_status(job_id)

            if status['status'] == 'completed':
                print(f"✅ Job completed! Output: {status['output_path']}")
                return status
            elif status['status'] == 'failed':
                print(f"❌ Job failed: {status['error_message']}")
                return status
            else:
                print(f"⏳ Job status: {status['status']}")
                time.sleep(poll_interval)

    def list_jobs(self):
        """List all jobs"""
        response = requests.get(f"{self.api_url}/jobs")
        return response.json()

def main():
    """Example usage of the client"""
    client = ModelMergingClient()

    print("Model Merging API Client Example")
    print("=" * 50)

    # Check API health
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Merging Available: {health['merging_available']}")

    # Get available methods
    methods = client.get_methods()
    print(f"\nAvailable Methods: {len(methods['methods'])}")
    for method in methods['methods']:
        print(f"  - {method['name']}: {method['description']}")

    # Example merge request
    merge_params = {
        "base_model": "Qwen/Qwen2.5-0.5B",
        "models_to_merge": [
            "InfiX-ai/Qwen-base-0.5B-code",
            "InfiX-ai/Qwen-base-0.5B-algebra"
        ],
        "merge_method": "task_arithmetic",
        "scaling_coefficient": 0.3,
        "use_gpu": False
    }

    print(f"\nStarting merge job...")
    print(f"Base model: {merge_params['base_model']}")
    print(f"Models to merge: {merge_params['models_to_merge']}")
    print(f"Method: {merge_params['merge_method']}")

    try:
        # Start merge job
        job_info = client.start_merge(merge_params)
        job_id = job_info['job_id']
        print(f"✅ Job started: {job_id}")

        # Wait for completion
        print(f"\nWaiting for job completion...")
        final_status = client.wait_for_completion(job_id)

        if final_status['status'] == 'completed':
            print(f"\n🎉 Merge completed successfully!")
            print(f"Output directory: {final_status['output_path']}")
        else:
            print(f"\n❌ Merge failed")

    except Exception as e:
        print(f"❌ Error: {e}")

    # List all jobs
    print(f"\nAll jobs:")
    jobs = client.list_jobs()
    for job in jobs['jobs']:
        print(f"  - {job['job_id']}: {job['status']}")

if __name__ == "__main__":
    main()