import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast
import json

def get_llm_recommendations(columns_description, df):
    """Get target variable, protected columns, excluded columns, plus race column and privileged/unprivileged lists from LLM."""
    url = "https://xoxof3kdzvlwkyk5hfaajacb.agents.do-ai.run/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer _V__VxUKW6o9wnCPGh8YYgof_Rknl-XQ"
    }
    
    # 1. Identify protected/bias-related columns
    payload1 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant that only lists column names."},
            {"role": "user", "content": f"""The dataset has these columns (with context):
{columns_description}

Please identify which of these columns should be used in code to estimate bias (i.e., protected attributes like gender and race or contextual factors relevant for fairness analysis).  
Respond with only the column names, separated by commas."""}
        ],
        "stream": False,
        "include_functions_info": False,
        "include_retrieval_info": False,
        "include_guardrails_info": False
    }
    response1 = requests.post(url, headers=headers, json=payload1)
    protected_columns = response1.json()['choices'][0]['message']['content'].strip()
    
    # 2. Identify the target variable (assumed to be LAW_CAT_CD or similar)
    second_prompt = f"""I have identified the following columns as potential protected/bias‐related features:

{protected_columns}

Now, among **all** the original dataset columns, provide only one column that should be treated as the **target variable** when computing precision and recall for bias evaluation?  
Respond with only the column name."""
    payload2 = {
        "messages": [
            {"role": "user", "content": columns_description},
            {"role": "assistant", "content": protected_columns},
            {"role": "user", "content": second_prompt}
        ],
        "stream": False
    }
    response2 = requests.post(url, headers=headers, json=payload2)
    target_column = response2.json()['choices'][0]['message']['content'].strip()
    
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
    
    third_prompt = f"""
This dataset is for a machine learning classification task.
The dataset includes the following columns (with context):
{columns_description}

Protected or bias-related features (should not be excluded):
{protected_columns}

Target variable:
{target_column}

Top 10 features most correlated with the target (formatted as variable=correlation):
{corr_snippet}

Please respond with **only** the column names that should be excluded due to high dependence on the target. Protected or bias-related features should not be excluded. Return the column names separated by commas—no additional text.
""".strip()
    payload3 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant that only lists column names."},
            {"role": "user", "content": third_prompt}
        ]
    }
    response3 = requests.post(url, headers=headers, json=payload3)
    excluded_columns = response3.json()['choices'][0]['message']['content'].strip()
    
    # 4. Identify which column is the race column for suspects
    fourth_prompt = f"""The dataset has these columns (with context):
{columns_description}

You previously identified the protected/bias-related columns as:
{protected_columns}

Among those, which single column represents the suspect's race ?  
Respond with only the column name."""
    payload4 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant that only lists column names."},
            {"role": "user", "content": fourth_prompt}
        ],
        "stream": False
    }
    response4 = requests.post(url, headers=headers, json=payload4)
    race_column = response4.json()['choices'][0]['message']['content'].strip()
    
    # 5. Fetch unique values for that race column, then ask LLM to split into privileged/unprivileged lists
    unique_races = df[race_column].dropna().unique().tolist()
    fifth_prompt = f"""The column "{race_column}" has these unique values:
{unique_races}

Please separate these values into two Python lists:
1. privileged_list 
2. unprivileged_list

Return a JSON object with keys "privileged_list" and "unprivileged_list", where each value is a list of strings. Do not include any extra text."""
    payload5 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant that outputs valid JSON."},
            {"role": "user", "content": fifth_prompt}
        ],
        "stream": False
    }
    response5 = requests.post(url, headers=headers, json=payload5)
    lists_json = response5.json()['choices'][0]['message']['content'].strip()
    
    # Parse the JSON response from the LLM
    try:
        parsed_lists = json.loads(lists_json)
        privileged_list = parsed_lists.get("privileged_list", [])
        unprivileged_list = parsed_lists.get("unprivileged_list", [])
    except json.JSONDecodeError:
        # If the LLM returns Python list syntax instead of JSON, try ast.literal_eval
        try:
            parsed_py = ast.literal_eval(lists_json)
            privileged_list = parsed_py.get("privileged_list", [])
            unprivileged_list = parsed_py.get("unprivileged_list", [])
        except (SyntaxError, ValueError):
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

def get_llm_bias_check(protected_attr, analysis_summary, shap_table=None):
    """Send the analysis summary and SHAP table for a protected attribute to the LLM and get a bias detection response."""
    url = "https://xoxof3kdzvlwkyk5hfaajacb.agents.do-ai.run/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer _V__VxUKW6o9wnCPGh8YYgof_Rknl-XQ"
    }
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
    return response.json()['choices'][0]['message']['content'].strip()