#!/usr/bin/env python3
"""
Test script for the Model Merging API

This script tests the API endpoints to ensure they work correctly.
"""
import requests
import json
import time
import sys

# API base URL
BASE_URL = "http://localhost:5001/api"

def test_health():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_methods():
    """Test methods endpoint"""
    print("\nTesting methods endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/methods")
        if response.status_code == 200:
            data = response.json()
            methods = data.get('methods', [])
            print(f"✅ Methods endpoint passed: {len(methods)} methods available")
            for method in methods:
                print(f"   - {method['name']}: {method['description']}")
            return True
        else:
            print(f"❌ Methods endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Methods endpoint error: {e}")
        return False

def test_example():
    """Test example endpoint"""
    print("\nTesting example endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/example")
        if response.status_code == 200:
            data = response.json()
            print("✅ Example endpoint passed")
            print(f"   Example payload keys: {list(data.keys())}")
            return True
        else:
            print(f"❌ Example endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Example endpoint error: {e}")
        return False

def test_merge_job():
    """Test starting a merge job"""
    print("\nTesting merge job creation...")

    # Use a simple test payload (won't actually merge due to model loading)
    payload = {
        "base_model": "Qwen/Qwen2.5-0.5B",
        "models_to_merge": [
            "InfiX-ai/Qwen-base-0.5B-code",
            "InfiX-ai/Qwen-base-0.5B-algebra"
        ],
        "merge_method": "task_arithmetic",
        "scaling_coefficient": 0.3,
        "use_gpu": False  # Use CPU for testing
    }

    try:
        response = requests.post(f"{BASE_URL}/merge", json=payload)
        if response.status_code == 202:
            data = response.json()
            job_id = data.get('job_id')
            print(f"✅ Merge job created: {job_id}")
            print(f"   Status: {data.get('status')}")
            print(f"   Check URL: {data.get('check_status')}")

            # Test job status endpoint
            return test_job_status(job_id)
        else:
            print(f"❌ Merge job creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Merge job creation error: {e}")
        return False

def test_job_status(job_id):
    """Test job status endpoint"""
    print(f"\nTesting job status for {job_id}...")

    try:
        response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Job status retrieved")
            print(f"   Status: {data.get('status')}")
            print(f"   Created: {data.get('created_at')}")
            return True
        else:
            print(f"❌ Job status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Job status error: {e}")
        return False

def test_list_jobs():
    """Test listing all jobs"""
    print("\nTesting jobs list endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/jobs")
        if response.status_code == 200:
            data = response.json()
            jobs = data.get('jobs', [])
            print(f"✅ Jobs list retrieved: {len(jobs)} jobs")
            return True
        else:
            print(f"❌ Jobs list failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Jobs list error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Model Merging API Test Suite")
    print("=" * 60)
    print("Make sure the API server is running at http://localhost:5000")
    print()

    # Run tests
    tests = [
        test_health,
        test_methods,
        test_example,
        test_merge_job,
        test_list_jobs
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed! API is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the API server.")
        return 1

if __name__ == "__main__":
    sys.exit(main())