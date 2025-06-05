from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import os
from app.utils.llm_utils import get_llm_recommendations
from app.models.bias_evaluator import evaluate_model_bias
import logging

bp = Blueprint('main', __name__)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@bp.route('/')
def home():
    return render_template('index.html')

@bp.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get parameters from request
        file = request.files['file']
        columns_description = request.form['columns_description']
        n_rows = int(request.form['n_rows'])
        test_size = float(request.form['test_size'])
        max_categories = int(request.form['max_categories'])
        
        # Check if file exists in uploads directory
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        is_cached = os.path.exists(filepath)
        
        # Save file if not cached
        if not is_cached:
            file.save(filepath)
        
        # Read the CSV file
        df = pd.read_csv(filepath, low_memory=False)
        
        # Limit rows if specified
        if n_rows > 0:
            df = df.head(n_rows)
        
        # Get LLM recommendations
        llm_recommendations = get_llm_recommendations(columns_description, df)
        target_column = llm_recommendations['target_column']
        protected_attributes = llm_recommendations['protected_columns'].split(',')
        excluded_columns = llm_recommendations['excluded_columns'].split(',')
        race_col = llm_recommendations.get('race_column')
        privileged_list = llm_recommendations.get('privileged_list')
        unprivileged_list = llm_recommendations.get('unprivileged_list')
        
        # Remove excluded columns from the dataset
        df = df.drop(columns=[col.strip() for col in excluded_columns if col.strip() in df.columns])
        
        # Process each protected attribute
        results = {}
        for attr in protected_attributes:
            attr = attr.strip()
            if attr in df.columns:
                model, preprocessor, overall, group_report, shap_tables, bias_metrics = evaluate_model_bias(
                    df,
                    target_col=target_column,
                    protected_attr=attr,
                    test_size=test_size,
                    max_categories=max_categories,
                    race_col=race_col,
                    privileged_list=privileged_list,
                    unprivileged_list=unprivileged_list
                )
                results[attr] = {
                    'overall': overall.to_dict(),
                    'group_report': group_report.to_dict('records'),
                    'shap_tables': {k: v.to_dict('records') for k, v in shap_tables.items()},
                    'bias_metrics': bias_metrics.to_dict('records') if bias_metrics is not None else None
                }
        
        return jsonify({
            'status': 'success',
            'results': results,
            'llm_recommendations': llm_recommendations,
            'is_cached': is_cached
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }) 