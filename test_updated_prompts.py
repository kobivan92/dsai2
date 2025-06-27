#!/usr/bin/env python3
"""
Test script to verify the updated LLM prompts work correctly.
This script tests the new Key Findings section with bias level, definition, and argumentation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.llm_utils import get_llm_bias_check
import pandas as pd

def test_updated_prompts():
    """Test the updated LLM bias check prompts."""
    
    # Sample analysis summary for testing
    analysis_summary = """
    Protected Attribute: SUSP_RACE
    
    Overall Performance Metrics:
    - Accuracy: 0.85
    - Macro F1: 0.82
    
    Group-wise Analysis:
    - White suspects: Precision=0.88, Recall=0.90, F1=0.89
    - Black suspects: Precision=0.82, Recall=0.78, F1=0.80
    - Hispanic suspects: Precision=0.84, Recall=0.81, F1=0.82
    
    Bias Metrics:
    - Statistical Parity Difference: 0.12
    - Disparate Impact: 0.87
    - Mean Difference: 0.08
    
    Bias Level Classification: MEDIUM
    """
    
    # Sample global explanations
    global_explanations = """
    Feature Importance Analysis:
    
    FELONY Class:
    1. SUSP_RACE (23.5%)
    2. SUSP_SEX (18.9%)
    3. VIC_RACE (15.7%)
    
    MISDEMEANOR Class:
    1. SUSP_SEX (21.2%)
    2. SUSP_RACE (19.8%)
    3. VIC_SEX (14.3%)
    
    VIOLATION Class:
    1. SUSP_RACE (25.1%)
    2. VIC_RACE (17.6%)
    3. SUSP_SEX (16.2%)
    """
    
    print("🧪 Testing Updated LLM Prompts")
    print("=" * 50)
    
    try:
        # Test with llama_3_3
        print("\n📊 Testing llama_3_3...")
        result = get_llm_bias_check(
            protected_attr="SUSP_RACE",
            analysis_summary=analysis_summary,
            global_explanations=global_explanations,
            llm_model='llama_3_3'
        )
        
        print("\n✅ llama_3_3 Response:")
        print("-" * 30)
        print(result)
        
        # Check if response contains Key Findings section
        if "## KEY FINDINGS" in result:
            print("\n🎯 SUCCESS: Key Findings section found!")
            
            # Check for bias level
            if "**Bias Level**:" in result:
                print("✅ Bias Level found")
            else:
                print("❌ Bias Level missing")
                
            # Check for definition
            if "**Definition**:" in result:
                print("✅ Definition found")
            else:
                print("❌ Definition missing")
                
            # Check for argumentation
            if "**Argumentation**:" in result:
                print("✅ Argumentation found")
            else:
                print("❌ Argumentation missing")
                
        else:
            print("\n❌ Key Findings section not found in response")
            
    except Exception as e:
        print(f"\n❌ Error testing llama_3_3: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_updated_prompts() 
 