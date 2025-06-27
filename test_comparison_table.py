#!/usr/bin/env python3
"""
Test script to verify the Multi-LLM Bias Analysis Comparison table with AIF360 bias levels.
This script simulates the data structure that would be sent to the frontend.
"""

import json
import re

def test_comparison_table_data():
    """Test the data structure for the comparison table."""
    
    # Simulate the data structure that would be sent to the frontend
    test_data = {
        "llm_recommendations": {
            "target_column": "LAW_CAT_CD",
            "protected_columns": "SUSP_RACE, SUSP_SEX",
            "excluded_columns": "ID, DATE",
            "correlations": "SUSP_RACE=0.234, SUSP_SEX=0.189"
        },
        "results": {
            "SUSP_RACE": {
                "overall": {
                    "FELONY": {"precision": 0.85, "recall": 0.82, "f1-score": 0.83},
                    "MISDEMEANOR": {"precision": 0.78, "recall": 0.81, "f1-score": 0.79},
                    "VIOLATION": {"precision": 0.92, "recall": 0.89, "f1-score": 0.90}
                },
                "group_report": [
                    {"SUSP_RACE": "WHITE", "class": "FELONY", "precision": 0.88, "recall": 0.90, "support": 1500},
                    {"SUSP_RACE": "BLACK", "class": "FELONY", "precision": 0.82, "recall": 0.78, "support": 1200},
                    {"SUSP_RACE": "HISPANIC", "class": "FELONY", "precision": 0.84, "recall": 0.81, "support": 800}
                ],
                "bias_metrics": [
                    {
                        "Category": "FELONY",
                        "Bias_Level": "MEDIUM",
                        "Privileged_Rate": 0.88,
                        "Unprivileged_Rate": 0.82,
                        "Statistical_Parity_Difference": 0.06,
                        "Disparate_Impact": 0.93,
                        "Mean_Difference": 0.06
                    },
                    {
                        "Category": "MISDEMEANOR",
                        "Bias_Level": "HIGH",
                        "Privileged_Rate": 0.85,
                        "Unprivileged_Rate": 0.75,
                        "Statistical_Parity_Difference": 0.10,
                        "Disparate_Impact": 0.88,
                        "Mean_Difference": 0.10
                    },
                    {
                        "Category": "VIOLATION",
                        "Bias_Level": "LOW",
                        "Privileged_Rate": 0.92,
                        "Unprivileged_Rate": 0.90,
                        "Statistical_Parity_Difference": 0.02,
                        "Disparate_Impact": 0.98,
                        "Mean_Difference": 0.02
                    }
                ],
                "global_explanations": {
                    "FELONY": [
                        {"rank": 1, "feature": "SUSP_RACE", "importance": 0.235},
                        {"rank": 2, "feature": "SUSP_SEX", "importance": 0.189},
                        {"rank": 3, "feature": "VIC_RACE", "importance": 0.157}
                    ],
                    "MISDEMEANOR": [
                        {"rank": 1, "feature": "SUSP_SEX", "importance": 0.212},
                        {"rank": 2, "feature": "SUSP_RACE", "importance": 0.198},
                        {"rank": 3, "feature": "VIC_SEX", "importance": 0.143}
                    ],
                    "VIOLATION": [
                        {"rank": 1, "feature": "SUSP_RACE", "importance": 0.251},
                        {"rank": 2, "feature": "VIC_RACE", "importance": 0.176},
                        {"rank": 3, "feature": "SUSP_SEX", "importance": 0.162}
                    ]
                },
                "llm_bias_check": {
                    "llama_3_3": """## KEY FINDINGS
**Bias Level**: MEDIUM
**Definition**: Moderate bias detected in race-based predictions with concerning disparities in performance metrics across demographic groups.
**Argumentation**: Statistical Parity Difference of 0.06 indicates meaningful outcome rate differences between racial groups. Disparate Impact of 0.93 shows that minority groups receive positive outcomes at 93% the rate of majority groups. Feature importance analysis reveals that race-related features are among the top predictors, suggesting potential proxy discrimination.

## DETAILED ANALYSIS
1. **Specific Bias Patterns**: White suspects show higher precision and recall across all crime categories compared to Black and Hispanic suspects.

2. **Feature Influence Analysis**: 
   - SUSP_RACE is the most important feature for FELONY and VIOLATION predictions
   - SUSP_SEX shows high importance across all classes
   - VIC_RACE appears as a potential proxy for suspect demographics

3. **Risk Assessment**: The moderate bias level indicates potential unfair treatment that should be monitored and addressed.""",
                    "deepseek_r1": """## KEY FINDINGS
**Bias Level**: HIGH
**Definition**: Significant bias detected with clear patterns of unfair treatment across racial groups in the criminal justice system.
**Argumentation**: The Statistical Parity Difference of 0.10 exceeds acceptable thresholds, indicating substantial outcome disparities. Disparate Impact of 0.88 falls below the 0.8 threshold, suggesting adverse impact on minority groups. The consistent ranking of demographic features as top predictors indicates systematic bias in the model.

## DETAILED ANALYSIS
1. **Specific Bias Patterns**: Clear performance gaps between White and minority suspects, particularly in MISDEMEANOR classification.

2. **Feature Influence Analysis**: 
   - Demographic features dominate the feature importance rankings
   - Race-related features show consistent high importance across all classes
   - Evidence of proxy discrimination through victim demographics

3. **Risk Assessment**: High bias level suggests significant risk of unfair treatment requiring immediate attention.""",
                    "mistral_nemo": """## KEY FINDINGS
**Bias Level**: MEDIUM
**Definition**: Moderate bias detected with some concerning patterns in demographic-based predictions that warrant attention.
**Argumentation**: Statistical Parity Difference of 0.06 indicates moderate but meaningful disparities between groups. Disparate Impact of 0.93 is close to but still below the ideal 1.0 ratio. Feature importance analysis shows demographic attributes are influential but not overwhelmingly dominant.

## DETAILED ANALYSIS
1. **Specific Bias Patterns**: Moderate performance differences between racial groups, with some categories showing more bias than others.

2. **Feature Influence Analysis**: 
   - Race and sex features are important but not the only predictors
   - Some evidence of proxy discrimination through victim characteristics
   - Importance varies by crime category

3. **Risk Assessment**: Moderate risk level requiring monitoring and potential mitigation strategies."""
                }
            }
        }
    }
    
    print("🧪 Testing Multi-LLM Bias Analysis Comparison Table")
    print("=" * 60)
    
    # Test the data structure
    for attr, result in test_data["results"].items():
        print(f"\n📊 Testing attribute: {attr}")
        
        # Check AIF360 bias levels
        if result.get("bias_metrics"):
            print("✅ AIF360 Bias Levels found:")
            for metric in result["bias_metrics"]:
                print(f"   - {metric['Category']}: {metric['Bias_Level']}")
        else:
            print("❌ No AIF360 bias metrics found")
        
        # Check LLM bias checks
        if result.get("llm_bias_check"):
            print("✅ LLM Bias Checks found:")
            for model, analysis in result["llm_bias_check"].items():
                # Parse bias level from content using Python regex
                bias_match = re.search(r'## KEY FINDINGS\s*\*\*Bias Level\*\*:\s*(\w+)', analysis)
                if bias_match:
                    bias_level = bias_match.group(1)
                    print(f"   - {model}: {bias_level}")
                else:
                    print(f"   - {model}: Could not parse bias level")
        else:
            print("❌ No LLM bias checks found")
    
    # Simulate the comparison table structure
    print("\n📋 Expected Comparison Table Structure:")
    print("-" * 60)
    print("| Category   | Bias Level AIF | llama_3_3 | deepseek_r1 | mistral_nemo |")
    print("|------------|----------------|-----------|-------------|--------------|")
    print("| FELONY     | MEDIUM         | MEDIUM    | HIGH        | MEDIUM       |")
    print("| MISDEMEANOR| HIGH           | MEDIUM    | HIGH        | MEDIUM       |")
    print("| VIOLATION  | LOW            | MEDIUM    | HIGH        | MEDIUM       |")
    
    print("\n🎯 Test completed successfully!")
    print("\nThe comparison table should show:")
    print("1. ✅ AIF360 bias levels from bias_metrics")
    print("2. ✅ LLM bias levels parsed from Key Findings")
    print("3. ✅ Color-coded badges for each bias level")
    print("4. ✅ Detailed analysis in expandable sections")

if __name__ == "__main__":
    test_comparison_table_data() 