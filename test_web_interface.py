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

def test_web_interface_simulation():
    """Simulate exactly what the web interface does"""
    
    # Simulate the web interface parameters
    filename = 'Crime_Data_from_2020_to_Present_20250626_preprocessed.csv'
    filepath = os.path.join('uploads', filename)
    columns_description = "LAPD crime data with victim demographics and crime classification information"
    n_rows = 1000  # Simulate limiting rows
    test_size = 0.3
    max_categories = 10
    
    print(f"Simulating web interface with file: {filepath}")
    print(f"Columns description: {columns_description}")
    print(f"n_rows: {n_rows}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} does not exist!")
        return
    
    # Read the CSV file (same as web interface)
    print("Reading CSV file...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Original data shape: {df.shape}")
    
    # Limit rows if specified (same as web interface)
    if n_rows > 0:
        df = df.head(n_rows)
        print(f"Limited data shape: {df.shape}")
    
    print(f"Columns: {list(df.columns)}")
    
    # Test single LLM (same as /analyze route)
    print("\n=== Testing Single LLM (like /analyze route) ===")
    try:
        llm_recommendations = get_llm_recommendations(columns_description, df, llm_model='llama_3_3')
        print("SUCCESS - Single LLM recommendations:")
        print(f"Target Column: {llm_recommendations['target_column']}")
        print(f"Protected Columns: {llm_recommendations['protected_columns']}")
        print(f"Excluded Columns: {llm_recommendations['excluded_columns']}")
        
        # Test the protected_attributes parsing (same as web interface)
        protected_attributes = [col.strip() for col in llm_recommendations['protected_columns'].split(',') if col.strip()]
        print(f"Parsed protected attributes: {protected_attributes}")
        
        if not protected_attributes:
            print("ERROR: No protected attributes found!")
        else:
            print(f"First protected attribute: {protected_attributes[0]}")
            
    except Exception as e:
        print(f"ERROR - Single LLM failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test multi-LLM (same as /analyze_multi route)
    print("\n=== Testing Multi-LLM (like /analyze_multi route) ===")
    try:
        llm_recommendations_multi = get_llm_recommendations_multi(columns_description, df)
        print(f"Multi-LLM results received: {list(llm_recommendations_multi.keys())}")
        
        # Find successful recommendations (same as web interface)
        successful_recommendations = None
        for model, result in llm_recommendations_multi.items():
            if 'error' not in result:
                successful_recommendations = result
                print(f"Using recommendations from {model} for main analysis")
                break
        
        if successful_recommendations is None:
            print("ERROR: All LLM models failed to provide recommendations")
            print("Individual results:")
            for model, result in llm_recommendations_multi.items():
                if 'error' in result:
                    print(f"  {model}: {result['error']}")
                else:
                    print(f"  {model}: SUCCESS")
        else:
            print("SUCCESS - Multi-LLM recommendations:")
            print(f"Target Column: {successful_recommendations['target_column']}")
            print(f"Protected Columns: {successful_recommendations['protected_columns']}")
            print(f"Excluded Columns: {successful_recommendations['excluded_columns']}")
            
            # Test the protected_attributes parsing (same as web interface)
            protected_attributes = [col.strip() for col in successful_recommendations['protected_columns'].split(',') if col.strip()]
            print(f"Parsed protected attributes: {protected_attributes}")
            
            if not protected_attributes:
                print("ERROR: No protected attributes found!")
            else:
                print(f"First protected attribute: {protected_attributes[0]}")
                
    except Exception as e:
        print(f"ERROR - Multi-LLM failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_web_interface_simulation() 