from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json
import requests

app = Flask(__name__)

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
            {"role": "user", "content": columns_description}
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
    exclude = {target_column} | set(protected_columns.split(',')) | {
        'CMPLNT_NUM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM',
        'CMPLNT_TO_DT', 'CMPLNT_TO_TM',
        'Lat_Lon', 'Latitude', 'Longitude'
    }
    
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
I have these protected/bias-related features (do NOT include them in the correlation step):
{protected_columns}

My target variable is:
{target_column}

Here are the top 10 features most correlated with the target, formatted as variable=correlation:
{corr_snippet}

Please respond **only** with the column names that should be excluded due to high dependence on the target—separated by commas, no extra text.
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

def evaluate_model_bias(df: pd.DataFrame,
                        target_col: str,
                        protected_attr: str,
                        features: list = None,
                        test_size: float = 0.3,
                        random_state: int = 42,
                        max_iter: int = 1000):
    """
    Trains a logistic regression (binary or multinomial) to predict a categorical target 
    and computes precision & recall per protected group for each class.
    """
    # Drop missing in target and protected_attr
    data = df.dropna(subset=[target_col, protected_attr])
    
    # Auto-select features
    if features is None:
        exclude = {target_col, protected_attr,
                   'CMPLNT_NUM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM',
                   'CMPLNT_TO_DT', 'CMPLNT_TO_TM',
                   'Lat_Lon', 'Latitude', 'Longitude'}
        features = [c for c in df.columns if c not in exclude]
    
    X = data[features]
    y = data[target_col].astype(str)
    groups = data[protected_attr].astype(str)
    
    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(range(len(le.classes_)))
    class_names = le.classes_
    
    # Pipelines
    num_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_feats = X.select_dtypes(include=['object','category']).columns.tolist()
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_feats),
        ('cat', categorical_pipeline, cat_feats)
    ])
    
    X_proc = preprocessor.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X_proc, y_enc, groups, test_size=test_size,
        stratify=y_enc, random_state=random_state
    )
    
    # Train
    model = LogisticRegression(max_iter=max_iter, multi_class='auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Overall report
    overall = pd.DataFrame(
        classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    ).transpose()[['precision','recall','f1-score']]
    
    # Group-wise report
    rows = []
    for grp in sorted(g_test.unique()):
        mask = g_test == grp
        if mask.sum() == 0:
            continue
        grp_dict = classification_report(
            y_test[mask], y_pred[mask],
            labels=classes, target_names=class_names, output_dict=True
        )
        for cls_name in class_names:
            cls_metrics = grp_dict.get(cls_name, {})
            rows.append({
                protected_attr: grp,
                'class': cls_name,
                'precision': cls_metrics.get('precision', 0.0),
                'recall': cls_metrics.get('recall', 0.0),
                'support': cls_metrics.get('support', 0)
            })
    group_report = pd.DataFrame(rows)
    
    return {
        'overall': overall.to_dict(),
        'group_report': group_report.to_dict(orient='records'),
        'class_names': class_names.tolist()
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get parameters from request
        file = request.files['file']
        columns_description = request.form['columns_description']
        n_rows = int(request.form['n_rows'])
        test_size = float(request.form['test_size'])
        max_iter = int(request.form['max_iter'])
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Limit rows if specified
        if n_rows > 0:
            df = df.head(n_rows)
        
        # Get LLM recommendations
        llm_recommendations = get_llm_recommendations(columns_description, df)
        target_column = llm_recommendations['target_column']
        protected_attributes = llm_recommendations['protected_columns'].split(',')
        excluded_columns = llm_recommendations['excluded_columns'].split(',')
        
        # Remove excluded columns from the dataset
        df = df.drop(columns=[col.strip() for col in excluded_columns if col.strip() in df.columns])
        
        # Process each protected attribute
        results = {}
        for attr in protected_attributes:
            attr = attr.strip()
            if attr in df.columns:
                result = evaluate_model_bias(
                    df,
                    target_col=target_column,
                    protected_attr=attr,
                    test_size=test_size,
                    max_iter=max_iter
                )
                results[attr] = result
        
        return jsonify({
            'status': 'success',
            'results': results,
            'llm_recommendations': llm_recommendations
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 
