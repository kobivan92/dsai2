import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast
import json
import time
from app.config import LLM_ENDPOINTS

def extract_column_name(response_text):
    """Extract just the column name from verbose LLM responses."""
    # Clean the response
    text = response_text.strip()
    
    # If it's just a column name, return it
    if ',' not in text and len(text.split()) <= 3:
        return text
    
    # Look for patterns that indicate the final answer
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('Therefore') and not line.startswith('So') and not line.startswith('In conclusion'):
            # Check if this line contains just a column name
            if ',' not in line and len(line.split()) <= 3:
                return line
    
    # If no clear pattern, return the last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    
    return text

def extract_column_list(response_text):
    """Extract comma-separated column names from verbose LLM responses."""
    # Clean the response
    text = response_text.strip()
    
    # If it's already a comma-separated list, return it
    if ',' in text and len(text.split(',')) > 1:
        # Check if it's just a list of column names
        parts = [part.strip() for part in text.split(',')]
        if all(len(part.split()) <= 3 for part in parts):
            return text
    
    # Look for the final list in the response
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and ',' in line:
            # Check if this line looks like a column list
            parts = [part.strip() for part in line.split(',')]
            if all(len(part.split()) <= 3 for part in parts):
                return line
    
    # If no clear list found, try to extract from the last meaningful line
    for line in reversed(lines):
        if line.strip() and not line.startswith('Therefore') and not line.startswith('So'):
            return line.strip()
    
    return text

def get_llm_recommendations(columns_description, df, llm_model='llama_3_3'):
    """Get target variable, protected columns, excluded columns, plus race column and privileged/unprivileged lists from LLM."""
    endpoint = LLM_ENDPOINTS[llm_model]
    url = endpoint['url']
    headers = endpoint['headers']
    
    # 1. Identify protected/bias-related columns
    payload1 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant. You must respond with ONLY column names separated by commas. No explanations, no reasoning, just the column names."},
            {"role": "user", "content": f"""The dataset has these columns (with context):
{columns_description}

Identify which columns are specifically related to race and gender for bias estimation.
RESPOND WITH ONLY THE COLUMN NAMES, SEPARATED BY COMMAS. NO EXPLANATIONS."""}
        ],
        "stream": False
    }
    response1 = requests.post(url, headers=headers, json=payload1)
    if response1.status_code != 200:
        raise Exception(f"HTTP {response1.status_code} error from {llm_model}: {response1.text}")
    response1_json = response1.json()
    if 'choices' not in response1_json or not response1_json['choices']:
        raise Exception(f"Invalid response from {llm_model}: {response1_json}")
    protected_columns = extract_column_list(response1_json['choices'][0]['message']['content'].strip())
    
    # 2. Identify the target variable (assumed to be LAW_CAT_CD or similar)
    second_prompt = f"""I have identified these protected/bias-related columns: {protected_columns}

Among ALL the original dataset columns, which ONE column should be the target variable for bias evaluation?
RESPOND WITH ONLY THE COLUMN NAME. NO EXPLANATIONS."""
    payload2 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant. Respond with ONLY the column name. No explanations."},
            {"role": "user", "content": columns_description},
            {"role": "assistant", "content": protected_columns},
            {"role": "user", "content": second_prompt}
        ],
        "stream": False
    }
    response2 = requests.post(url, headers=headers, json=payload2)
    if response2.status_code != 200:
        raise Exception(f"HTTP {response2.status_code} error from {llm_model}: {response2.text}")
    response2_json = response2.json()
    if 'choices' not in response2_json or not response2_json['choices']:
        raise Exception(f"Invalid response from {llm_model}: {response2_json}")
    target_column = extract_column_name(response2_json['choices'][0]['message']['content'].strip())
    
    # 3. Calculate correlations and identify columns to exclude
    exclude = {target_column} | set(c.strip() for c in protected_columns.split(','))
    feature_columns = [col for col in df.columns if col not in exclude]
    
    # Encode target if categorical
    y = df[target_column].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Compute per-feature correlation with target
    corr_dict = {}
    numeric_feats = df[feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_feats:
        corr_val = abs(df[col].corr(pd.Series(y_enc, index=df.index)))
        corr_dict[col] = corr_val
    
    categorical_feats = df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_feats:
        dummies = pd.get_dummies(df[col], prefix=col)
        max_corr = max(
            abs(dummies[c].corr(pd.Series(y_enc, index=df.index))) 
            for c in dummies.columns
        )
        corr_dict[col] = max_corr
    
    corr_series = pd.Series(corr_dict).sort_values(ascending=False)
    top10 = corr_series.head(10)
    corr_snippet = "\n".join(f"{var}={val:.3f}" for var, val in top10.items())
    
    third_prompt = f"""Dataset columns: {columns_description}

Protected features (DO NOT EXCLUDE): {protected_columns}
Target variable: {target_column}
Top correlated features: {corr_snippet}

Which columns should be excluded due to high correlation with target?
RESPOND WITH ONLY THE COLUMN NAMES, SEPARATED BY COMMAS. NO EXPLANATIONS."""
    payload3 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant. Respond with ONLY column names separated by commas. No explanations."},
            {"role": "user", "content": third_prompt}
        ],
        "stream": False
    }
    response3 = requests.post(url, headers=headers, json=payload3)
    if response3.status_code != 200:
        raise Exception(f"HTTP {response3.status_code} error from {llm_model}: {response3.text}")
    response3_json = response3.json()
    if 'choices' not in response3_json or not response3_json['choices']:
        raise Exception(f"Invalid response from {llm_model}: {response3_json}")
    excluded_columns = extract_column_list(response3_json['choices'][0]['message']['content'].strip())
    
    # 4. Identify which column is the race column for suspects
    fourth_prompt = f"""Dataset columns: {columns_description}

Protected columns: {protected_columns}

Which column represents suspect's race?
RESPOND WITH ONLY THE COLUMN NAME. NO EXPLANATIONS."""
    payload4 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant. Respond with ONLY the column name. No explanations."},
            {"role": "user", "content": fourth_prompt}
        ],
        "stream": False
    }
    response4 = requests.post(url, headers=headers, json=payload4)
    if response4.status_code != 200:
        raise Exception(f"HTTP {response4.status_code} error from {llm_model}: {response4.text}")
    response4_json = response4.json()
    if 'choices' not in response4_json or not response4_json['choices']:
        raise Exception(f"Invalid response from {llm_model}: {response4_json}")
    race_column = extract_column_name(response4_json['choices'][0]['message']['content'].strip())
    
    # 5. Fetch unique values for that race column, then ask LLM to split into privileged/unprivileged lists
    unique_races = df[race_column].dropna().unique().tolist()
    fifth_prompt = f"""Column "{race_column}" has these values: {unique_races}

Split into privileged and unprivileged groups.
Return ONLY a JSON object with "privileged_list" and "unprivileged_list" keys.
NO EXPLANATIONS, ONLY THE JSON."""
    payload5 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant. Return ONLY valid JSON. No explanations."},
            {"role": "user", "content": fifth_prompt}
        ],
        "stream": False
    }
    response5 = requests.post(url, headers=headers, json=payload5)
    if response5.status_code != 200:
        raise Exception(f"HTTP {response5.status_code} error from {llm_model}: {response5.text}")
    response5_json = response5.json()
    if 'choices' not in response5_json or not response5_json['choices']:
        raise Exception(f"Invalid response from {llm_model}: {response5_json}")
    lists_json = response5_json['choices'][0]['message']['content'].strip()
    
    # Parse the JSON response from the LLM
    try:
        # First try to clean the response if it contains extra text
        cleaned_json = lists_json.strip()
        # Remove any text before the first {
        if '{' in cleaned_json:
            cleaned_json = cleaned_json[cleaned_json.find('{'):]
        # Remove any text after the last }
        if '}' in cleaned_json:
            cleaned_json = cleaned_json[:cleaned_json.rfind('}')+1]
        
        parsed_lists = json.loads(cleaned_json)
        privileged_list = parsed_lists.get("privileged_list", [])
        unprivileged_list = parsed_lists.get("unprivileged_list", [])
    except json.JSONDecodeError as e:
        print(f"JSON decode error for {llm_model}: {e}")
        print(f"Raw response: {lists_json}")
        # If the LLM returns Python list syntax instead of JSON, try ast.literal_eval
        try:
            # Try to extract Python dict syntax
            if 'privileged_list' in lists_json and 'unprivileged_list' in lists_json:
                # Find the dict part
                start = lists_json.find('{')
                end = lists_json.rfind('}') + 1
                if start != -1 and end != 0:
                    dict_str = lists_json[start:end]
                    parsed_py = ast.literal_eval(dict_str)
                    privileged_list = parsed_py.get("privileged_list", [])
                    unprivileged_list = parsed_py.get("unprivileged_list", [])
                else:
                    privileged_list = []
                    unprivileged_list = []
            else:
                privileged_list = []
                unprivileged_list = []
        except (SyntaxError, ValueError) as e2:
            print(f"AST literal_eval error for {llm_model}: {e2}")
            # If both parsing attempts fail, use empty lists
            privileged_list = []
            unprivileged_list = []
    
    return {
        'protected_columns': protected_columns,
        'target_column': target_column,
        'excluded_columns': excluded_columns,
        'correlations': corr_snippet,
        'race_column': race_column,
        'privileged_list': privileged_list,
        'unprivileged_list': unprivileged_list
    }

def get_llm_bias_check(protected_attr, analysis_summary, shap_table=None, llm_model='llama_3_3'):
    """Send the analysis summary and SHAP table for a protected attribute to the LLM and get a bias detection response."""
    endpoint = LLM_ENDPOINTS[llm_model]
    url = endpoint['url']
    headers = endpoint['headers']
    
    prompt = f"""
You are a fairness and bias analysis expert. Here is the bias and classification analysis for the protected attribute '{protected_attr}':

{analysis_summary}
"""
    if shap_table is not None:
        prompt += f"\nSHAP Feature Importance Table:\n{shap_table}\n"
        prompt += "\nPlease analyze also the SHAP table for evidence of bias."
    prompt += "\nPlease only detect and describe any potential biases or fairness issues. Do not provide recommendations or mitigation steps."
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code} error from {llm_model}: {response.text}")
    response_json = response.json()
    if 'choices' not in response_json or not response_json['choices']:
        raise Exception(f"Invalid response from {llm_model}: {response_json}")
    return response_json['choices'][0]['message']['content'].strip()

def get_llm_bias_check_multi(protected_attr, analysis_summary, shap_table=None, llm_models=['llama_3_3', 'deepseek_r1', 'mistral_nemo']):
    """Send the analysis summary and SHAP table for a protected attribute to multiple LLMs and get bias detection responses."""
    results = {}
    
    # Skip mistral_nemo if it's causing issues (can be enabled later)
    skip_mistral = False  # Set to True to skip mistral_nemo
    
    for i, model in enumerate(llm_models):
        # Skip mistral_nemo if flag is set
        if skip_mistral and model == 'mistral_nemo':
            print(f"Skipping {model} bias check due to known issues...")
            results[model] = {'error': f'{model} skipped due to known 500 error issues'}
            continue
            
        max_retries = 3
        retry_delay = 3  # Start with 3 seconds for bias checks
        
        # Special handling for mistral_nemo - longer delay
        if model == 'mistral_nemo':
            print(f"Special handling for mistral_nemo bias check - adding extra delay...")
            time.sleep(10)  # 10 second delay before mistral_nemo bias check
        
        for attempt in range(max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                if i > 0 or attempt > 0:
                    print(f"Waiting {retry_delay} seconds before bias check from {model} (attempt {attempt + 1})...")
                    time.sleep(retry_delay)
                    
                print(f"Getting bias check from {model} (attempt {attempt + 1})...")
                result = get_llm_bias_check(protected_attr, analysis_summary, shap_table, model)
                results[model] = result
                print(f"Successfully got bias check from {model}")
                break  # Success, exit retry loop
                
            except requests.exceptions.RequestException as e:
                print(f"Request error getting bias check from {model} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    results[model] = {'error': f"Network error for {model} after {max_retries} attempts: {str(e)}"}
            except Exception as e:
                print(f"Error getting bias check from {model} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    results[model] = {'error': str(e)}
                break  # Don't retry for non-network errors
        
        # Add extra delay between different LLMs to prevent rate limiting
        if i < len(llm_models) - 1:  # Not the last model
            print(f"Waiting 5 seconds before next LLM bias check...")
            time.sleep(5)
    
    return results

def get_llm_recommendations_multi(columns_description, df, llm_models=['llama_3_3', 'deepseek_r1', 'mistral_nemo']):
    """Get target variable, protected columns, excluded columns, plus race column and privileged/unprivileged lists from multiple LLMs."""
    results = {}
    
    # Skip mistral_nemo if it's causing issues (can be enabled later)
    skip_mistral = False  # Set to True to skip mistral_nemo
    
    for i, model in enumerate(llm_models):
        # Skip mistral_nemo if flag is set
        if skip_mistral and model == 'mistral_nemo':
            print(f"Skipping {model} due to known issues...")
            results[model] = {'error': f'{model} skipped due to known 500 error issues'}
            continue
            
        max_retries = 3
        retry_delay = 5  # Start with 5 seconds
        
        # Special handling for mistral_nemo - longer delay
        if model == 'mistral_nemo':
            print(f"Special handling for mistral_nemo - adding extra delay...")
            time.sleep(15)  # 15 second delay before mistral_nemo
        
        for attempt in range(max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                if i > 0 or attempt > 0:
                    print(f"Waiting {retry_delay} seconds before {model} (attempt {attempt + 1})...")
                    time.sleep(retry_delay)
                
                print(f"Getting recommendations from {model} (attempt {attempt + 1})...")
                result = get_llm_recommendations(columns_description, df, model)
                results[model] = result
                print(f"Successfully got recommendations from {model}")
                break  # Success, exit retry loop
                
            except requests.exceptions.RequestException as e:
                print(f"Request error getting recommendations from {model} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    results[model] = {'error': f"Network error for {model} after {max_retries} attempts: {str(e)}"}
            except Exception as e:
                print(f"Error getting recommendations from {model} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    results[model] = {'error': str(e)}
                break  # Don't retry for non-network errors
        
        # Add extra delay between different LLMs to prevent rate limiting
        if i < len(llm_models) - 1:  # Not the last model
            print(f"Waiting 8 seconds before next LLM...")
            time.sleep(8)
    
    return results