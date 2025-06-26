import pandas as pd
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from models.bias_evaluator import evaluate_model_bias
    print("Successfully imported bias evaluator")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_global_explanations():
    """Test Global Explanations with LAPD data"""
    
    # Load the preprocessed LAPD data
    print("Loading LAPD data...")
    df = pd.read_csv('uploads/Crime_Data_from_2020_to_Present_20250626_preprocessed.csv')
    print(f"Data shape: {df.shape}")
    
    # Limit to 1000 rows for testing
    df = df.head(1000)
    
    # Test the bias evaluator with Global Explanations
    print("\nTesting Global Explanations...")
    try:
        model, preprocessor, overall, group_report, global_explanations, bias_metrics = evaluate_model_bias(
            df=df,
            target_col='Status',
            protected_attr='Vict Sex',
            max_categories=10,
            test_size=0.3,
            random_state=42,
            protected_columns='Vict Sex, Vict Descent, Vict Age'
        )
        
        print("SUCCESS - Global Explanations generated!")
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
        
    except Exception as e:
        print(f"ERROR - Global Explanations failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_global_explanations() 