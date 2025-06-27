#!/usr/bin/env python3
"""
Test script to verify that AIF360 bias levels are computed for all protected attributes.
This script tests the updated compute_bias_metrics function.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from app.models.bias_evaluator import compute_bias_metrics, evaluate_model_bias

def test_aif360_bias_levels():
    """Test AIF360 bias level computation for all protected attributes."""
    
    # Create a sample dataset for testing
    data = {
        'SUSP_RACE': ['WHITE', 'BLACK', 'HISPANIC', 'WHITE', 'BLACK', 'HISPANIC'] * 50,
        'SUSP_SEX': ['M', 'F', 'M', 'F', 'M', 'F'] * 50,
        'LAW_CAT_CD': ['FELONY', 'MISDEMEANOR', 'VIOLATION', 'FELONY', 'MISDEMEANOR', 'VIOLATION'] * 50,
        'VIC_RACE': ['WHITE', 'BLACK', 'HISPANIC', 'WHITE', 'BLACK', 'HISPANIC'] * 50,
        'VIC_SEX': ['M', 'F', 'M', 'F', 'M', 'F'] * 50
    }
    
    df = pd.DataFrame(data)
    
    print("🧪 Testing AIF360 Bias Levels for All Protected Attributes")
    print("=" * 60)
    
    # Test parameters
    target_col = 'LAW_CAT_CD'
    privileged_list = ['WHITE', 'M']  # White and Male as privileged
    unprivileged_list = ['BLACK', 'HISPANIC', 'F']  # Others as unprivileged
    
    # Test for different protected attributes
    protected_attributes = ['SUSP_RACE', 'SUSP_SEX', 'VIC_RACE', 'VIC_SEX']
    
    for protected_attr in protected_attributes:
        print(f"\n📊 Testing protected attribute: {protected_attr}")
        
        try:
            # Compute bias metrics
            bias_metrics = compute_bias_metrics(
                df, 
                target_col, 
                protected_attr, 
                privileged_list, 
                unprivileged_list
            )
            
            if bias_metrics is not None and not bias_metrics.empty:
                print("✅ AIF360 Bias Metrics computed successfully:")
                print(bias_metrics.to_string(index=False))
                
                # Check if bias levels are present
                if 'Bias_Level' in bias_metrics.columns:
                    print(f"✅ Bias levels found: {bias_metrics['Bias_Level'].tolist()}")
                else:
                    print("❌ No Bias_Level column found")
            else:
                print("❌ No bias metrics computed")
                
        except Exception as e:
            print(f"❌ Error computing bias metrics for {protected_attr}: {str(e)}")
    
    # Test the full evaluate_model_bias function
    print(f"\n🔧 Testing full evaluate_model_bias function...")
    try:
        model, preprocessor, overall, group_report, global_explanations, bias_metrics = evaluate_model_bias(
            df,
            target_col=target_col,
            protected_attr='SUSP_RACE',
            race_col='SUSP_RACE',
            privileged_list=privileged_list,
            unprivileged_list=unprivileged_list
        )
        
        if bias_metrics is not None and not bias_metrics.empty:
            print("✅ Full evaluation successful - Bias metrics:")
            print(bias_metrics.to_string(index=False))
        else:
            print("❌ No bias metrics from full evaluation")
            
    except Exception as e:
        print(f"❌ Error in full evaluation: {str(e)}")
    
    print("\n🎯 Test completed!")
    print("\nExpected behavior:")
    print("1. ✅ AIF360 bias metrics should be computed for all protected attributes")
    print("2. ✅ Each attribute should have bias levels for each target category")
    print("3. ✅ Bias levels should be LOW, MEDIUM, HIGH, or CRITICAL")
    print("4. ✅ The comparison table should show AIF360 bias levels for all attributes")

if __name__ == "__main__":
    test_aif360_bias_levels() 