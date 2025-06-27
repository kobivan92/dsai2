#!/usr/bin/env python3
"""
Standalone Global Explanations Report Generator
Creates an offline HTML report with SHAP Global Explanations for uploaded datasets
"""

import pandas as pd
import os
import json
from datetime import datetime
from app.models.bias_evaluator import evaluate_model_bias
from app.utils.llm_utils import get_llm_recommendations
import argparse
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def calculate_shap_global_explanations(df, target_column, protected_attributes, test_size=0.2, max_categories=50):
    """Calculate SHAP Global Explanations for each class"""
    
    print("Calculating SHAP Global Explanations...")
    
    # Prepare data
    data = df.dropna(subset=[target_column, protected_attributes[0]]).copy()
    
    # Restrict features to only the specified columns if they exist
    allowed_features = [
        'SUSP_RACE', 'SUSP_SEX', 'SUSP_AGE_GROUP',
        'VIC_RACE', 'VIC_SEX', 'VIC_AGE_GROUP'
    ]
    feature_cols = [col for col in allowed_features if col in data.columns]
    if not feature_cols:
        raise ValueError("None of the specified analysis columns found in the dataset.")
    print(f"Using only these features for SHAP analysis: {feature_cols}")
    X = data[feature_cols].copy()
    y = data[target_column].copy()
    
    print(f"Initial X shape: {X.shape}")
    print(f"Feature columns: {feature_cols}")
    
    # Handle categorical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        if X[col].nunique() <= max_categories:
            try:
                le = LabelEncoder()
                # Handle any non-string values
                X[col] = X[col].astype(str)
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
                print(f"Encoded {col}: {X[col].nunique()} unique values")
            except Exception as e:
                print(f"Error encoding {col}: {e}")
                X = X.drop(columns=[col])
                feature_cols.remove(col)
        else:
            # Drop high cardinality features
            print(f"Dropping high cardinality feature: {col} ({X[col].nunique()} unique values)")
            X = X.drop(columns=[col])
            feature_cols.remove(col)
    
    # Fill missing values
    X = X.fillna(X.median())
    
    # Ensure all columns are numeric
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"Converting {col} to numeric")
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(0)
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"Final X shape: {X.shape}")
    print(f"Final feature columns: {X.columns.tolist()}")
    
    # Ensure we have features
    if X.shape[1] == 0:
        raise ValueError("No valid features found for SHAP analysis")
    
    # Train Random Forest model
    print("Training Random Forest model for SHAP analysis...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    try:
        explainer = shap.TreeExplainer(rf_model)
        
        # Use a smaller sample for SHAP calculation to avoid memory issues
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        print(f"Using {sample_size} samples for SHAP calculation")
        
        # Try different approaches for SHAP calculation
        try:
            # Method 1: Direct SHAP values
            shap_values = explainer.shap_values(X_sample)
            print(f"SHAP values shape: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"SHAP values list length: {len(shap_values)}")
                for i, sv in enumerate(shap_values):
                    print(f"  Class {i} SHAP shape: {sv.shape}")
            else:
                print(f"SHAP values array shape: {shap_values.shape}")
        except Exception as e1:
            print(f"Direct SHAP calculation failed: {e1}")
            # Method 2: Use expected values approach
            try:
                print("Trying expected values approach...")
                expected_value = explainer.expected_value
                shap_values = explainer.shap_values(X_sample, check_additivity=False)
                print(f"Expected values SHAP shape: {type(shap_values)}")
            except Exception as e2:
                print(f"Expected values approach failed: {e2}")
                # Method 3: Use feature importance directly
                print("Using feature importance as SHAP approximation...")
                return calculate_feature_importance_fallback(X, y, rf_model)
                
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}")
        # Fallback to feature importance
        print("Falling back to feature importance...")
        return calculate_feature_importance_fallback(X, y, rf_model)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Calculate global explanations for each class
    global_explanations = {}
    
    if isinstance(shap_values, list):
        # Multi-class case
        for class_idx, class_shap_values in enumerate(shap_values):
            class_name = rf_model.classes_[class_idx]
            print(f"Processing SHAP values for class: {class_name}")
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap_values = np.abs(class_shap_values).mean(axis=0)
            
            # Create DataFrame with feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap_values,
                'rank': range(1, len(feature_names) + 1)
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
            feature_importance['rank'] = range(1, len(feature_importance) + 1)
            
            global_explanations[class_name] = feature_importance
            
    else:
        # Single array case (binary or single output)
        print("Processing single SHAP values array...")
        
        # Handle different array shapes
        if len(shap_values.shape) == 2:
            # 2D array: (samples, features)
            mean_shap_values = np.abs(shap_values).mean(axis=0)
        elif len(shap_values.shape) == 3:
            # 3D array: could be (samples, features, classes) or (classes, samples, features)
            print(f"3D SHAP array shape: {shap_values.shape}")
            
            # Check if it's (samples, features, classes) format
            if shap_values.shape[0] == sample_size:
                # Format: (samples, features, classes)
                print("Detected (samples, features, classes) format")
                # Transpose to get (classes, samples, features)
                shap_values = np.transpose(shap_values, (2, 0, 1))
                print(f"Transposed SHAP array shape: {shap_values.shape}")
            
            # Now process as (classes, samples, features)
            mean_shap_values = np.abs(shap_values).mean(axis=1)  # Shape: (classes, features)
            
            # Process each class
            for class_idx in range(shap_values.shape[0]):
                class_name = rf_model.classes_[class_idx] if hasattr(rf_model, 'classes_') else f"Class_{class_idx}"
                print(f"Processing class {class_idx}: {class_name}")
                
                class_importance = mean_shap_values[class_idx]
                
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': class_importance,
                    'rank': range(1, len(feature_names) + 1)
                })
                
                feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
                feature_importance['rank'] = range(1, len(feature_importance) + 1)
                
                global_explanations[class_name] = feature_importance
        else:
            print(f"Unexpected SHAP array shape: {shap_values.shape}")
            # Fallback to feature importance
            print("Falling back to feature importance due to unexpected SHAP shape...")
            return calculate_feature_importance_fallback(X, y, rf_model)
        
        # If we didn't process multiple classes above, process as single class
        if not global_explanations:
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap_values,
                'rank': range(1, len(feature_names) + 1)
            })
            
            feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
            feature_importance['rank'] = range(1, len(feature_importance) + 1)
            
            global_explanations['Overall'] = feature_importance
    
    print(f"SHAP Global Explanations calculated for {len(global_explanations)} classes")
    for cls, exp_df in global_explanations.items():
        print(f"  Class {cls}: {len(exp_df)} features")
    
    return global_explanations, False

def calculate_feature_importance_fallback(X, y, model):
    """Fallback to feature importance when SHAP fails"""
    print("Using feature importance fallback...")
    
    feature_names = X.columns.tolist()
    importances = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'rank': range(1, len(feature_names) + 1)
    })
    
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    feature_importance['rank'] = range(1, len(feature_importance) + 1)
    
    global_explanations = {'Overall': feature_importance}
    
    print(f"Feature importance calculated for {len(global_explanations)} classes")
    return global_explanations, True

def generate_html_report(df, target_column, protected_attributes, global_explanations, filename, output_dir="reports", used_fallback=False):
    """Generate a standalone HTML report with SHAP Global Explanations"""
    
    # Create reports directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"shap_global_explanations_report_{timestamp}.html"
    report_path = os.path.join(output_dir, report_filename)
    
    # Determine method used
    method_text = "Feature Importance (SHAP fallback)" if used_fallback else "SHAP TreeExplainer with Random Forest"
    title_text = "Feature Importance Report" if used_fallback else "SHAP Global Explanations Report"
    badge_text = "Feature Importance" if used_fallback else "Powered by SHAP"
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_text} - {filename}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .feature-table {{
            font-size: 0.9rem;
        }}
        .importance-high {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .importance-medium {{
            background-color: #fff3cd !important;
        }}
        .importance-low {{
            background-color: #f8f9fa !important;
        }}
        .class-header {{
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 15px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 0;
        }}
        .report-header {{
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .stats-card {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .shap-badge {{
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .fallback-badge {{
            background: linear-gradient(135deg, #6c757d, #495057);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="report-header text-center">
            <h1><i class="fas fa-chart-bar"></i> {title_text}</h1>
            <p class="lead mb-0">{'Feature Importance Analysis' if used_fallback else 'SHAP Feature Importance Analysis'} for Machine Learning Model</p>
            <span class="{'fallback-badge' if used_fallback else 'shap-badge'} mt-2">{badge_text}</span>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="stats-card">
                    <h5><i class="fas fa-file-alt"></i> Dataset Information</h5>
                    <p><strong>Filename:</strong> {filename}</p>
                    <p><strong>Total Rows:</strong> {len(df):,}</p>
                    <p><strong>Total Columns:</strong> {len(df.columns)}</p>
                    <p><strong>Target Column:</strong> <code>{target_column}</code></p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="stats-card">
                    <h5><i class="fas fa-shield-alt"></i> Protected Attributes</h5>
                    <p><strong>Analyzed Attributes:</strong></p>
                    <ul>
                        {''.join([f'<li><code>{attr}</code></li>' for attr in protected_attributes])}
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <div class="stats-card">
                    <h5><i class="fas fa-info-circle"></i> Analysis Summary</h5>
                    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p><strong>Classes Analyzed:</strong> {len(global_explanations)}</p>
                    <p><strong>Classes:</strong> {', '.join(global_explanations.keys())}</p>
                    <p><strong>Method:</strong> {method_text}</p>
                    {f'<p><strong>Note:</strong> SHAP calculation failed, using feature importance fallback</p>' if used_fallback else ''}
                </div>
            </div>
        </div>
    """
    
    # Add SHAP Global Explanations for each class
    for class_name, explanations in global_explanations.items():
        if isinstance(explanations, pd.DataFrame) and not explanations.empty:
            html_content += f"""
        <div class="row mb-5">
            <div class="col-12">
                <div class="card shadow">
                    <div class="class-header">
                        <h3 class="mb-0"><i class="fas fa-list-ol"></i> {class_name}</h3>
                        <small>SHAP Feature Importance - Top {min(10, len(explanations))} Most Important Features</small>
                    </div>
                    <div class="card-body p-0">
                        <div class="table-responsive">
                            <table class="table table-hover feature-table mb-0">
                                <thead class="table-dark">
                                    <tr>
                                        <th width="8%">Rank</th>
                                        <th width="50%">Feature Name</th>
                                        <th width="20%">SHAP Importance</th>
                                        <th width="22%">Percentage</th>
                                    </tr>
                                </thead>
                                <tbody>
            """
            
            # Calculate total importance for percentage
            total_importance = explanations['importance'].sum()
            
            # Add feature rows
            for idx, row in explanations.head(10).iterrows():
                importance = row['importance']
                percentage = (importance / total_importance) * 100 if total_importance > 0 else 0
                
                # Determine row class based on importance
                if percentage > 15:
                    row_class = "importance-high"
                elif percentage > 8:
                    row_class = "importance-medium"
                else:
                    row_class = "importance-low"
                
                html_content += f"""
                                    <tr class="{row_class}">
                                        <td><span class="badge bg-secondary">{row['rank']}</span></td>
                                        <td><code class="fs-6">{row['feature']}</code></td>
                                        <td><strong>{importance:.4f}</strong></td>
                                        <td><span class="badge bg-primary">{percentage:.1f}%</span></td>
                                    </tr>
                """
            
            html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    """
    
    # Add footer with SHAP explanation
    html_content += """
        <div class="row mt-5">
            <div class="col-12">
                <div class="alert alert-info">
                    <h5><i class="fas fa-lightbulb"></i> Understanding SHAP Feature Importance</h5>
                    <ul class="mb-0">
                        <li><strong>SHAP Values:</strong> Show how much each feature contributes to model predictions</li>
                        <li><strong>High Importance (>15%):</strong> Features that heavily influence model predictions</li>
                        <li><strong>Medium Importance (8-15%):</strong> Features with moderate influence</li>
                        <li><strong>Low Importance (<8%):</strong> Features with minimal influence</li>
                        <li><strong>Method:</strong> TreeExplainer with Random Forest classifier</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <footer class="text-center text-muted mt-5">
            <p>Generated by SHAP Global Explanations Report Generator</p>
            <p><small>This report shows SHAP-based feature importance for predicting the target variable across different classes.</small></p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    
    # Write HTML file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path

def analyze_dataset(filepath, n_rows=0, test_size=0.2, max_categories=50):
    """Analyze a dataset and generate SHAP Global Explanations report"""
    
    print(f"Loading dataset: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Limit rows if specified
    if n_rows > 0:
        df = df.head(n_rows)
        print(f"Limited to {n_rows} rows")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get LLM recommendations for column analysis
    print("Getting LLM recommendations...")
    columns_description = f"Dataset with {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns)}"
    
    try:
        llm_recommendations = get_llm_recommendations(columns_description, df)
        target_column = llm_recommendations['target_column']
        protected_attributes = [col.strip() for col in llm_recommendations['protected_columns'].split(',') if col.strip()]
        excluded_columns = llm_recommendations['excluded_columns'].split(',')
        
        print(f"Target column: {target_column}")
        print(f"Protected attributes: {protected_attributes}")
        print(f"Excluded columns: {excluded_columns}")
        
        # Remove excluded columns
        df = df.drop(columns=[col.strip() for col in excluded_columns if col.strip() in df.columns])
        
        # Calculate SHAP Global Explanations
        global_explanations, used_fallback = calculate_shap_global_explanations(
            df, target_column, protected_attributes, test_size, max_categories
        )
        
        # Generate HTML report
        filename = os.path.basename(filepath)
        report_path = generate_html_report(df, target_column, protected_attributes, global_explanations, filename, used_fallback=used_fallback)
        
        print(f"\n✅ SHAP Report generated successfully!")
        print(f"📄 Report saved to: {report_path}")
        print(f"🌐 Open the HTML file in your browser to view the report")
        
        return report_path
        
    except Exception as e:
        print(f"❌ Error generating SHAP report: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate SHAP Global Explanations HTML Report')
    parser.add_argument('filepath', help='Path to the CSV file to analyze')
    parser.add_argument('--n-rows', type=int, default=0, help='Number of rows to analyze (0 for all)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test size for model evaluation')
    parser.add_argument('--max-categories', type=int, default=50, help='Maximum categories for categorical features')
    parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"❌ File not found: {args.filepath}")
        return
    
    try:
        report_path = analyze_dataset(
            args.filepath, 
            n_rows=args.n_rows,
            test_size=args.test_size,
            max_categories=args.max_categories
        )
        
        # Try to open the report automatically
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(report_path)}')
            print("🌐 Opening SHAP report in default browser...")
        except:
            print("💡 To view the report, open the HTML file in your web browser")
            
    except Exception as e:
        print(f"❌ Failed to generate SHAP report: {str(e)}")

if __name__ == "__main__":
    main() 