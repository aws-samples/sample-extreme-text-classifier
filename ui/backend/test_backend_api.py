#!/usr/bin/env python3
"""
Test script for the FastAPI backend server.

This script tests the main API endpoints to ensure they work correctly.
"""

import requests
import json
import time
from pathlib import Path


BASE_URL = "http://127.0.0.1:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data['status']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure it's running on port 8000")
        return False


def test_list_datasets():
    """Test listing datasets."""
    print("Testing dataset listing...")
    try:
        response = requests.get(f"{BASE_URL}/datasets")
        if response.status_code == 200:
            datasets = response.json()
            print(f"✓ Found {len(datasets)} datasets")
            for dataset in datasets:
                print(f"  - {dataset['name']} ({dataset['id']})")
            return datasets
        else:
            print(f"✗ Dataset listing failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"✗ Error listing datasets: {e}")
        return []


def test_get_dataset(dataset_id):
    """Test getting a specific dataset."""
    print(f"Testing dataset retrieval for {dataset_id}...")
    try:
        response = requests.get(f"{BASE_URL}/datasets/{dataset_id}")
        if response.status_code == 200:
            dataset = response.json()
            print(f"✓ Retrieved dataset: {dataset['name']}")
            print(f"  Classes: {len(dataset['classes'])}")
            print(f"  Embeddings: {dataset['embeddingsGenerated']}")
            return dataset
        else:
            print(f"✗ Dataset retrieval failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error retrieving dataset: {e}")
        return None


def test_generate_embeddings(dataset_id):
    """Test generating embeddings for a dataset."""
    print(f"Testing embedding generation for {dataset_id}...")
    try:
        response = requests.post(f"{BASE_URL}/datasets/{dataset_id}/embeddings")
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Embeddings generated: {result['message']}")
            return True
        else:
            print(f"✗ Embedding generation failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error generating embeddings: {e}")
        return False


def test_text_classification(dataset_id):
    """Test text classification."""
    print(f"Testing text classification with {dataset_id}...")
    
    # Test documents for different classes
    test_texts = [
        "Invoice #INV-2024-001 - Amount Due: $1,500.00 - Payment Terms: Net 30 days",
        "John Doe - Senior Software Engineer with 8 years experience in Python and machine learning",
        "Subject: Team Meeting - Hi everyone, we have a team meeting scheduled for tomorrow at 3 PM"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"  Test {i}: {text[:50]}...")
        
        try:
            payload = {
                "text": text,
                "dataset_id": dataset_id,
                "config": {
                    "use_reranking": False,
                    "use_attribute_validation": False,
                    "top_k_candidates": 5
                }
            }
            
            response = requests.post(f"{BASE_URL}/classify/text", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                predicted_class = result['predicted_class']['name']
                confidence = result.get('effective_score', 0)
                print(f"    ✓ Predicted: {predicted_class} (confidence: {confidence:.2%})")
            else:
                print(f"    ✗ Classification failed: {response.status_code}")
                print(f"      Response: {response.text}")
        
        except Exception as e:
            print(f"    ✗ Error in classification: {e}")


def test_attributes(dataset_id):
    """Test attribute management."""
    print(f"Testing attribute management for {dataset_id}...")
    
    try:
        # Get existing attributes
        response = requests.get(f"{BASE_URL}/attributes/{dataset_id}")
        if response.status_code == 200:
            attributes = response.json()
            print(f"✓ Retrieved attributes: {len(attributes.get('attributes', []))} classes")
        else:
            print(f"✗ Attribute retrieval failed: {response.status_code}")
        
        # Test attribute generation
        print(f"  Testing attribute generation...")
        response = requests.post(
            f"{BASE_URL}/attributes/{dataset_id}/generate",
            data={'domain': 'business documents'}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Generated attributes for {len(result.get('attributes', []))} classes")
        else:
            print(f"  ✗ Attribute generation failed: {response.status_code}")
            print(f"    Response: {response.text}")
    
    except Exception as e:
        print(f"✗ Error testing attributes: {e}")


def main():
    """Main test function."""
    print("FastAPI Backend API Test Suite")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("Server is not responding. Please start the server first:")
        print("cd ui/backend && python start_backend.py")
        return
    
    print()
    
    # Test dataset operations
    datasets = test_list_datasets()
    print()
    
    if datasets:
        # Test with the first available dataset
        dataset_id = datasets[0]['id']
        
        # Get dataset details
        dataset = test_get_dataset(dataset_id)
        print()
        
        if dataset:
            # Generate embeddings if not already generated
            if not dataset['embeddingsGenerated']:
                test_generate_embeddings(dataset_id)
                print()
            
            # Test classification
            test_text_classification(dataset_id)
            print()
            
            # Test attributes
            test_attributes(dataset_id)
            print()
    
    print("Test suite completed!")


if __name__ == "__main__":
    main()