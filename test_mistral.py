#!/usr/bin/env python3
"""
Test script to check mistral_nemo API separately and diagnose issues.
"""

import requests
import json
import time
from app.utils.llm_utils import LLM_ENDPOINTS

def test_mistral_recommendations():
    """Test mistral_nemo recommendations API call."""
    print("=== Testing mistral_nemo recommendations ===")
    
    endpoint = LLM_ENDPOINTS['mistral_nemo']
    url = endpoint['url']
    headers = endpoint['headers']
    
    columns_description = """
    Dataset columns: SUSP_RACE, SUSP_SEX, VIC_RACE, VIC_SEX, LAW_CAT_CD, OFNS_DESC, 
    SUSP_AGE_GROUP, VIC_AGE_GROUP, BORO_NM, PRECINCT, JURISDICTION_CODE, 
    X_COORD_CD, Y_COORD_CD, Latitude, Longitude, Lon_Lat, TRANSIT_DISTRICT
    """
    
    df_sample = {
        'SUSP_RACE': ['BLACK', 'WHITE', 'HISPANIC'],
        'SUSP_SEX': ['M', 'F', 'M'],
        'VIC_RACE': ['WHITE', 'BLACK', 'HISPANIC'],
        'LAW_CAT_CD': ['FELONY', 'MISDEMEANOR', 'VIOLATION']
    }
    
    prompt = f"""
You are a data scientist analyzing a criminal justice dataset. Based on the following dataset description and sample data, please identify:

1. The target variable (what we want to predict)
2. Protected attributes (demographic variables that should not cause bias)
3. Columns to exclude from the model
4. The race column (if any)
5. Privileged and unprivileged groups for the race column

Dataset Description:
{columns_description}

Sample Data:
{df_sample}

Please respond with a JSON object containing:
{{
    "target_column": "column_name",
    "protected_columns": ["col1", "col2"],
    "excluded_columns": ["col1", "col2"],
    "race_column": "column_name",
    "privileged_list": ["group1", "group2"],
    "unprivileged_list": ["group1", "group2"]
}}

Only identify columns related to race and gender for bias estimation.
"""
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    print(f"Making request to: {url}")
    print(f"Headers: {headers}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return False
    except Exception as e:
        print(f"General exception: {e}")
        return False

def test_mistral_bias_check():
    """Test mistral_nemo bias check API call."""
    print("\n=== Testing mistral_nemo bias check ===")
    
    endpoint = LLM_ENDPOINTS['mistral_nemo']
    url = endpoint['url']
    headers = endpoint['headers']
    
    analysis_summary = """
    === Overall Classification Report (protected_attr = SUSP_RACE) ===
                 precision    recall
    FELONY        0.575221  0.067288
    MISDEMEANOR   0.579388  0.952124
    VIOLATION     0.453901  0.176309
    """
    
    prompt = f"""
You are a fairness and bias analysis expert. Here is the bias and classification analysis for the protected attribute 'SUSP_RACE':

{analysis_summary}

Please only detect and describe any potential biases or fairness issues. Do not provide recommendations or mitigation steps.
"""
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    print(f"Making request to: {url}")
    print(f"Headers: {headers}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return False
    except Exception as e:
        print(f"General exception: {e}")
        return False

def test_mistral_with_delays():
    """Test mistral_nemo with delays to simulate multi-LLM scenario."""
    print("\n=== Testing mistral_nemo with delays (simulating multi-LLM) ===")
    
    for i in range(3):
        print(f"\n--- Attempt {i+1} ---")
        
        if i > 0:
            delay = 5 * i
            print(f"Waiting {delay} seconds before attempt {i+1}...")
            time.sleep(delay)
        
        success = test_mistral_recommendations()
        if success:
            print(f"Attempt {i+1} succeeded!")
        else:
            print(f"Attempt {i+1} failed!")
        
        if i < 2:
            print("Waiting 3 seconds before next attempt...")
            time.sleep(3)

def check_mistral_endpoint():
    """Check if mistral_nemo endpoint is accessible."""
    print("=== Checking mistral_nemo endpoint ===")
    
    endpoint = LLM_ENDPOINTS['mistral_nemo']
    url = endpoint['url']
    headers = endpoint['headers']
    
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    
    # Try a simple GET request to check if endpoint is reachable
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"GET request status: {response.status_code}")
        print(f"GET response: {response.text[:200]}...")
    except Exception as e:
        print(f"GET request failed: {e}")

if __name__ == "__main__":
    print("Mistral Nemo API Test Script")
    print("=" * 50)
    
    # Check endpoint first
    check_mistral_endpoint()
    
    # Test individual calls
    test_mistral_recommendations()
    test_mistral_bias_check()
    
    # Test with delays (simulating multi-LLM scenario)
    test_mistral_with_delays()
    
    print("\n" + "=" * 50)
    print("Test completed!") 