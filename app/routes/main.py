from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import os
from app.utils.llm_utils import get_llm_recommendations, get_llm_bias_check, get_llm_recommendations_multi, get_llm_bias_check_multi
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
        protected_attributes = [col.strip() for col in llm_recommendations['protected_columns'].split(',') if col.strip()]
        excluded_columns = llm_recommendations['excluded_columns'].split(',')
        race_col = llm_recommendations.get('race_column')
        privileged_list = llm_recommendations.get('privileged_list')
        unprivileged_list = llm_recommendations.get('unprivileged_list')
        
        # Validate that we have protected attributes
        if not protected_attributes:
            return jsonify({
                'status': 'error',
                'message': 'No protected attributes found in LLM recommendations. Please check your data and try again.'
            })
        
        # Remove excluded columns from the dataset
        df = df.drop(columns=[col.strip() for col in excluded_columns if col.strip() in df.columns])
        
        # Calculate Global Explanations once for all features
        model, preprocessor, overall, group_report, global_explanations, _ = evaluate_model_bias(
            df,
            target_col=target_column,
            protected_attr=protected_attributes[0].strip(),
            test_size=test_size,
            max_categories=max_categories,
            protected_columns=llm_recommendations['protected_columns']
        )
        global_explanations_data = {'global_explanations': global_explanations}
        
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
                    global_shap=None,
                    protected_columns=llm_recommendations['protected_columns']
                )
                if attr == race_col and bias_metrics is not None:
                    aif_metrics_for_llm = bias_metrics.to_dict('records')
                summary = f"Overall:\n{overall.to_string()}\n\nGroup-wise:\n{group_report.to_string()}"
                if bias_metrics is not None:
                    summary += f"\n\nBias Metrics:\n{bias_metrics.to_string()}"
                
                # Format global explanations for LLM
                global_explanations_str = ""
                if global_explanations:
                    for cls, exp_df in global_explanations.items():
                        global_explanations_str += f"\nClass: {cls}\n{exp_df.to_string(index=False)}\n"
                
                llm_bias_check = get_llm_bias_check(attr, summary, global_explanations=global_explanations_str if global_explanations_str else None, llm_model=llm_model)
                results[attr] = {
                    'overall': overall.to_dict(),
                    'group_report': group_report.to_dict('records'),
                    'global_explanations': {k: v.to_dict('records') for k, v in global_explanations.items()},
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

@bp.route('/analyze_multi', methods=['POST'])
def analyze_multi():
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
        
        # Get LLM recommendations from all 3 models
        print("Starting multi-LLM analysis...")
        llm_recommendations_multi = get_llm_recommendations_multi(columns_description, df)
        print(f"LLM recommendations received: {list(llm_recommendations_multi.keys())}")
        
        # Use the first successful result for the main analysis
        successful_recommendations = None
        for model, result in llm_recommendations_multi.items():
            if 'error' not in result:
                successful_recommendations = result
                print(f"Using recommendations from {model} for main analysis")
                break
        
        if successful_recommendations is None:
            return jsonify({
                'status': 'error',
                'message': 'All LLM models failed to provide recommendations'
            })
        
        target_column = successful_recommendations['target_column']
        protected_attributes = [col.strip() for col in successful_recommendations['protected_columns'].split(',') if col.strip()]
        excluded_columns = successful_recommendations['excluded_columns'].split(',')
        race_col = successful_recommendations.get('race_column')
        privileged_list = successful_recommendations.get('privileged_list')
        unprivileged_list = successful_recommendations.get('unprivileged_list')
        
        # Validate that we have protected attributes
        if not protected_attributes:
            return jsonify({
                'status': 'error',
                'message': 'No protected attributes found in LLM recommendations. Please check your data and try again.'
            })
        
        # Remove excluded columns from the dataset
        df = df.drop(columns=[col.strip() for col in excluded_columns if col.strip() in df.columns])
        
        # Calculate Global Explanations once for all features
        model, preprocessor, overall, group_report, global_explanations, _ = evaluate_model_bias(
            df,
            target_col=target_column,
            protected_attr=protected_attributes[0].strip(),
            test_size=test_size,
            max_categories=max_categories,
            protected_columns=successful_recommendations['protected_columns']
        )
        global_explanations_data = {'global_explanations': global_explanations}
        
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
                    global_shap=None,
                    protected_columns=successful_recommendations['protected_columns']
                )
                if attr == race_col and bias_metrics is not None:
                    aif_metrics_for_llm = bias_metrics.to_dict('records')
                summary = f"Overall:\n{overall.to_string()}\n\nGroup-wise:\n{group_report.to_string()}"
                if bias_metrics is not None:
                    summary += f"\n\nBias Metrics:\n{bias_metrics.to_string()}"
                
                # Format global explanations for LLM
                global_explanations_str = ""
                if global_explanations:
                    for cls, exp_df in global_explanations.items():
                        global_explanations_str += f"\nClass: {cls}\n{exp_df.to_string(index=False)}\n"
                
                # Get bias analysis from all LLMs
                print(f"Getting multi-LLM bias analysis for {attr}...")
                llm_bias_check_multi = get_llm_bias_check_multi(attr, summary, global_explanations=global_explanations_str if global_explanations_str else None)
                llm_bias_check = llm_bias_check_multi
                
                results[attr] = {
                    'overall': overall.to_dict(),
                    'group_report': group_report.to_dict('records'),
                    'global_explanations': {k: v.to_dict('records') for k, v in global_explanations.items()},
                    'bias_metrics': bias_metrics.to_dict('records') if bias_metrics is not None else None,
                    'llm_bias_check': llm_bias_check
                }
        
        if aif_metrics_for_llm is not None:
            for model in llm_recommendations_multi:
                if 'error' not in llm_recommendations_multi[model]:
                    llm_recommendations_multi[model]['aif_metrics'] = aif_metrics_for_llm
            
        return jsonify({
            'status': 'success',
            'results': results,
            'llm_recommendations': llm_recommendations_multi,
            'is_cached': is_cached,
            'use_multi_llm': True
        })
    
    except Exception as e:
        print(f"Error in analyze_multi: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }) 