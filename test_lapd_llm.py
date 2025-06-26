import pandas as pd
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from utils.llm_utils import get_llm_recommendations, get_llm_recommendations_multi
    print("Successfully imported LLM functions")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_lapd_llm():
    """Test LLM recommendations with LAPD data"""
    
    # Load the preprocessed LAPD data
    print("Loading LAPD data...")
    df = pd.read_csv('uploads/Crime_Data_from_2020_to_Present_20250626_preprocessed.csv')
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create a columns description
    columns_description = """
    Dataset columns: DR_NO, Date Rptd, DATE OCC, TIME OCC, AREA, AREA NAME, Rpt Dist No, Part 1-2, 
    Crm Cd, Crm Cd Desc, Mocodes, Vict Age, Vict Sex, Vict Descent, Premis Cd, Premis Desc, 
    Weapon Used Cd, Weapon Desc, Status, Status Desc, Crm Cd 1, Crm Cd 2, Crm Cd 3, Crm Cd 4, 
    LOCATION, Cross Street, LAT, LON
    
    This is LAPD crime data with victim demographics and crime classification information.
    """
    
    print("\nTesting LLM recommendations...")
    print("Columns description:")
    print(columns_description)
    
    try:
        # Test with llama_3_3 first
        print("\nTesting with llama_3_3...")
        result = get_llm_recommendations(columns_description, df, 'llama_3_3')
        print("Success! Result:")
        print(f"Target Column: {result['target_column']}")
        print(f"Protected Columns: {result['protected_columns']}")
        print(f"Excluded Columns: {result['excluded_columns']}")
        print(f"Race Column: {result['race_column']}")
        print(f"Privileged List: {result['privileged_list']}")
        print(f"Unprivileged List: {result['unprivileged_list']}")
        
    except Exception as e:
        print(f"Error with llama_3_3: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test multi-LLM
        print("\nTesting multi-LLM recommendations...")
        multi_result = get_llm_recommendations_multi(columns_description, df)
        print("Multi-LLM Result:")
        for model, result in multi_result.items():
            if 'error' in result:
                print(f"{model}: ERROR - {result['error']}")
            else:
                print(f"{model}: SUCCESS")
                print(f"  Target: {result.get('target_column', 'N/A')}")
                print(f"  Protected: {result.get('protected_columns', 'N/A')}")
                print(f"  Race: {result.get('race_column', 'N/A')}")
        
    except Exception as e:
        print(f"Error with multi-LLM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lapd_llm() 