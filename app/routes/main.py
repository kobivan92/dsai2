from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import os
from app.utils.llm_utils import get_llm_recommendations, get_llm_bias_check
from app.models.bias_evaluator import evaluate_model_bias
import logging
from app.config import LLM_ENDPOINTS

bp = Blueprint('main', __name__)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@bp.route('/')
def home():
    return render_template('index.html', llm_models=list(LLM_ENDPOINTS.keys()))

@bp.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get parameters from request
        file = request.files['file']
        columns_description = request.form['columns_description']
        n_rows = int(request.form['n_rows'])
        test_size = float(request.form['test_size'])
        max_categories = int(request.form['max_categories'])
        llm_model = request.form.get('llm_model', 'llama_3_3')  # Get selected LLM model
        
        # Validate LLM model selection
        if llm_model not in LLM_ENDPOINTS:
            return jsonify({
                'status': 'error',
                'message': f'Invalid LLM model selected. Available models: {", ".join(LLM_ENDPOINTS.keys())}'
            })
        
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
        llm_recommendations = get_llm_recommendations(columns_description, df, llm_model=llm_model)
        target_column = llm_recommendations['target_column']
        protected_attributes = llm_recommendations['protected_columns'].split(',')
        excluded_columns = llm_recommendations['excluded_columns'].split(',')
        race_col = llm_recommendations.get('race_column')
        privileged_list = llm_recommendations.get('privileged_list')
        unprivileged_list = llm_recommendations.get('unprivileged_list')
        
        # Remove excluded columns from the dataset
        df = df.drop(columns=[col.strip() for col in excluded_columns if col.strip() in df.columns])
        
        # Calculate SHAP tables once for all features and samples 0-10
        model, preprocessor, overall, group_report, shap_tables, _ = evaluate_model_bias(
            df,
            target_col=target_column,
            protected_attr=protected_attributes[0].strip(),
            test_size=test_size,
            max_categories=max_categories
        )
        global_shap = {'shap_tables': shap_tables}
        # Process each protected attribute
        results = {}
        aif_metrics_for_llm = None
        for attr in protected_attributes:
            attr = attr.strip()
            if attr in df.columns:
                model, preprocessor, overall, group_report, _, bias_metrics = evaluate_model_bias(
                    df,
                    target_col=target_column,
                    protected_attr=attr,
                    test_size=test_size,
                    max_categories=max_categories,
                    race_col=race_col,
                    privileged_list=privileged_list,
                    unprivileged_list=unprivileged_list,
                    global_shap=None
                )
                if attr == race_col and bias_metrics is not None:
                    aif_metrics_for_llm = bias_metrics.to_dict('records')
                summary = f"Overall:\n{overall.to_string()}\n\nGroup-wise:\n{group_report.to_string()}"
                if bias_metrics is not None:
                    summary += f"\n\nBias Metrics:\n{bias_metrics.to_string()}"
                shap_table_str = ""
                if shap_tables:
                    for cls, shap_df in shap_tables.items():
                        shap_table_str += f"Class: {cls}\n{pd.DataFrame(shap_df).to_string(index=False)}\n\n"
                llm_bias_check = get_llm_bias_check(attr, summary, shap_table=shap_table_str if shap_table_str else None, llm_model=llm_model)
                results[attr] = {
                    'overall': overall.to_dict(),
                    'group_report': group_report.to_dict('records'),
                    'shap_tables': {k: v.to_dict('records') for k, v in shap_tables.items()},
                    'bias_metrics': bias_metrics.to_dict('records') if bias_metrics is not None else None,
                    'llm_bias_check': llm_bias_check
                }
        
        if aif_metrics_for_llm is not None:
            llm_recommendations['aif_metrics'] = aif_metrics_for_llm
            
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