import requests
import json
import time

# Mistral Nemo endpoint configuration (from your config)
MISTRAL_ENDPOINT = {
    'url': 'https://zmtfbikxvnuldgj6piuxg7qf.agents.do-ai.run/api/v1/chat/completions',
    'headers': {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer Cko36hAqnC8xfUkCnT2-Up47IzaHvQdf'
    }
}

def test_mistral_simple():
    """Test mistral_nemo with a simple prompt."""
    print("=== Testing mistral_nemo simple prompt ===")
    
    url = MISTRAL_ENDPOINT['url']
    headers = MISTRAL_ENDPOINT['headers']
    
    prompt = "Hello, this is a test message. Please respond with 'Hello back!'"
    
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

def test_mistral_with_retries():
    """Test mistral_nemo with retries to simulate multi-LLM scenario."""
    print("\n=== Testing mistral_nemo with retries ===")
    
    for i in range(3):
        print(f"\n--- Attempt {i+1} ---")
        
        if i > 0:
            delay = 5 * i
            print(f"Waiting {delay} seconds before attempt {i+1}...")
            time.sleep(delay)
        
        success = test_mistral_simple()
        if success:
            print(f"Attempt {i+1} succeeded!")
        else:
            print(f"Attempt {i+1} failed!")
        
        if i < 2:
            print("Waiting 3 seconds before next attempt...")
            time.sleep(3)

def test_mistral_recommendations():
    """Test mistral_nemo with the actual recommendations prompt."""
    print("\n=== Testing mistral_nemo recommendations prompt ===")
    
    url = MISTRAL_ENDPOINT['url']
    headers = MISTRAL_ENDPOINT['headers']
    
    columns_description = """
    Dataset columns: SUSP_RACE, SUSP_SEX, VIC_RACE, VIC_SEX, LAW_CAT_CD, OFNS_DESC, 
    SUSP_AGE_GROUP, VIC_AGE_GROUP, BORO_NM, PRECINCT, JURISDICTION_CODE, 
    X_COORD_CD, Y_COORD_CD, Latitude, Longitude, Lon_Lat, TRANSIT_DISTRICT
    """
    
    prompt = f"""The dataset has these columns (with context):
{columns_description}

Identify which columns are specifically related to race and gender for bias estimation.
RESPOND WITH ONLY THE COLUMN NAMES, SEPARATED BY COMMAS. NO EXPLANATIONS."""
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant. You must respond with ONLY column names separated by commas. No explanations, no reasoning, just the column names."},
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

if __name__ == "__main__":
    print("Mistral Nemo API Test Script (Simple)")
    print("=" * 50)
    
    # Test simple call
    test_mistral_simple()
    
    # Test recommendations prompt
    test_mistral_recommendations()
    
    # Test with retries
    test_mistral_with_retries()
    
    print("\n" + "=" * 50)
    print("Test completed!") 