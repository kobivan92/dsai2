import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_llm_recommendations(columns_description, df):
    """Get target variable, protected columns, and excluded columns recommendations from LLM."""
    url = "https://xoxof3kdzvlwkyk5hfaajacb.agents.do-ai.run/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer _V__VxUKW6o9wnCPGh8YYgof_Rknl-XQ"
    }
    
    # First prompt to identify protected attributes
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
    
    # Second prompt to identify target variable
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
    
    # Calculate correlations
    exclude = {target_column} | set(protected_columns.split(','))
    
    feature_columns = [col for col in df.columns if col not in exclude]
    
    # Encode target if categorical
    y = df[target_column].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Compute per-feature correlation with target
    corr_dict = {}
    
    # Numeric features: Pearson correlation
    numeric_feats = df[feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_feats:
        corr_val = abs(df[col].corr(pd.Series(y_enc, index=df.index)))
        corr_dict[col] = corr_val
    
    # Categorical features: max correlation among dummies
    categorical_feats = df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_feats:
        dummies = pd.get_dummies(df[col], prefix=col)
        max_corr = max(abs(dummies[c].corr(pd.Series(y_enc, index=df.index))) for c in dummies.columns)
        corr_dict[col] = max_corr
    
    # Create sorted series and take top 10
    corr_series = pd.Series(corr_dict).sort_values(ascending=False)
    top10 = corr_series.head(10)
    
    # Format correlation snippet
    corr_snippet = "\n".join(f"{var}={val:.3f}" for var, val in top10.items())
    
    # Third prompt to identify columns to exclude
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

Please respond with **only** the column names that should be excluded due to high dependence on the target.Protected or bias-related features should not be excluded.Return the column names separated by commas—no additional text.
""".strip()
    
    payload3 = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant that only lists column names."},
            {"role": "user", "content": third_prompt}
        ]
    }
    
    response3 = requests.post(url, headers=headers, json=payload3)
    excluded_columns = response3.json()['choices'][0]['message']['content'].strip()
    
    return {
        'protected_columns': protected_columns,
        'target_column': target_column,
        'excluded_columns': excluded_columns,
        'correlations': corr_snippet
    } 