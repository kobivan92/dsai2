import pandas as pd
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from models.bias_evaluator import evaluate_model_bias
    from utils.llm_utils import get_llm_recommendations, get_llm_recommendations_multi
    print("Successfully imported modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_nypd_global_explanations():
    """Test Global Explanations with NYPD data"""
    
    # Load the preprocessed NYPD data
    print("Loading NYPD data...")
    df = pd.read_csv('uploads/NYPD_Complaint_Data_Historic_20250515_preprocessed.csv')
    print(f"NYPD data shape: {df.shape}")
    print(f"NYPD columns: {list(df.columns)}")
    
    # Limit to 1000 rows for testing
    df = df.head(1000)
    
    # Test LLM recommendations with NYPD data
    print("\nTesting LLM recommendations with NYPD data...")
    columns_description = """
    Dataset columns: CMPLNT_NUM, CMPLNT_FR_DT, CMPLNT_FR_TM, CMPLNT_TO_DT, CMPLNT_TO_TM, ADDR_PCT_CD, RPT_DT, KY_CD, OFNS_DESC, PD_CD, PD_DESC, CRM_ATPT_CPTD_CD, LAW_CAT_CD, JURIS_DESC, BORO_NM, LOC_OF_OCCUR_DESC, PREM_TYP_DESC, PARKS_NM, HADEVELOPT, X_COORD_CD, Y_COORD_CD, SUSP_AGE_GROUP, SUSP_RACE, SUSP_SEX, TRANSIT_DISTRICT, Latitude, Longitude, Lat_Lon, PATROL_BORO, STATION_NAME, VIC_AGE_GROUP, VIC_RACE, VIC_SEX
    
    This is NYPD complaint data with suspect and victim demographics and crime classification information.
    """
    
    try:
        # Test single LLM
        print("\nTesting single LLM with NYPD data...")
        llm_recommendations = get_llm_recommendations(columns_description, df, 'llama_3_3')
        print("SUCCESS - LLM recommendations for NYPD:")
        print(f"Target Column: {llm_recommendations['target_column']}")
        print(f"Protected Columns: {llm_recommendations['protected_columns']}")
        print(f"Excluded Columns: {llm_recommendations['excluded_columns']}")
        
        # Test Global Explanations with NYPD data
        print("\nTesting Global Explanations with NYPD data...")
        model, preprocessor, overall, group_report, global_explanations, bias_metrics = evaluate_model_bias(
            df=df,
            target_col=llm_recommendations['target_column'],
            protected_attr='SUSP_RACE',  # Use a known NYPD column
            max_categories=10,
            test_size=0.3,
            random_state=42,
            protected_columns=llm_recommendations['protected_columns']
        )
        
        print("SUCCESS - Global Explanations generated for NYPD!")
        print(f"Number of classes: {len(global_explanations)}")
        
        for class_name, explanations in global_explanations.items():
            print(f"\nClass: {class_name}")
            print(f"Number of features: {len(explanations)}")
            print("Top 5 features by importance:")
            print(explanations.head().to_string(index=False))
            
        print("\nOverall metrics:")
        print(overall)
        
        print("\nGroup-wise metrics:")
        print(group_report)
        
        # Test multi-LLM
        print("\nTesting multi-LLM with NYPD data...")
        multi_results = get_llm_recommendations_multi(columns_description, df)
        print("Multi-LLM results for NYPD:")
        for model, result in multi_results.items():
            if 'error' in result:
                print(f"  {model}: ERROR - {result['error']}")
            else:
                print(f"  {model}: SUCCESS")
                print(f"    Target: {result.get('target_column', 'N/A')}")
                print(f"    Protected: {result.get('protected_columns', 'N/A')}")
        
    except Exception as e:
        print(f"ERROR - NYPD test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nypd_global_explanations() 