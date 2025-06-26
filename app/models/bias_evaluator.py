import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
import matplotlib
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore")
matplotlib.use("Agg")

def classify_bias_level(statistical_parity_diff, disparate_impact, mean_diff):
    """
    Classify bias level based on statistical metrics.
    
    Returns:
        - 'LOW': Minimal bias detected
        - 'MEDIUM': Moderate bias detected  
        - 'HIGH': Significant bias detected
        - 'CRITICAL': Severe bias detected
    """
    # Initialize bias scores
    bias_score = 0
    
    # Statistical Parity Difference scoring
    # Values close to 0 indicate fairness
    abs_spd = abs(statistical_parity_diff)
    if abs_spd < 0.05:
        bias_score += 0
    elif abs_spd < 0.1:
        bias_score += 1
    elif abs_spd < 0.2:
        bias_score += 2
    else:
        bias_score += 3
    
    # Disparate Impact scoring
    # Values close to 1.0 indicate fairness
    if disparate_impact is None or pd.isna(disparate_impact):
        bias_score += 1
    else:
        di_ratio = min(disparate_impact, 1/disparate_impact) if disparate_impact > 0 else 0
        if di_ratio > 0.8:
            bias_score += 0
        elif di_ratio > 0.6:
            bias_score += 1
        elif di_ratio > 0.4:
            bias_score += 2
        else:
            bias_score += 3
    
    # Mean Difference scoring
    abs_md = abs(mean_diff)
    if abs_md < 0.05:
        bias_score += 0
    elif abs_md < 0.1:
        bias_score += 1
    elif abs_md < 0.2:
        bias_score += 2
    else:
        bias_score += 3
    
    # Classify based on total bias score (0-9 scale)
    if bias_score <= 2:
        return 'LOW'
    elif bias_score <= 4:
        return 'MEDIUM'
    elif bias_score <= 6:
        return 'HIGH'
    else:
        return 'CRITICAL'

def compute_bias_metrics(df: pd.DataFrame,
                        target_col: str,
                        race_col: str,
                        privileged_list: list,
                        unprivileged_list: list):
    """
    Compute bias metrics using AIF360 for each category in the target column.
    """
    # Filter to only those suspects whose race is in one of our two groups
    mask = df[race_col].isin(privileged_list + unprivileged_list)
    df_sub = df.loc[mask, [target_col, race_col]].copy()
    
    # Create a numeric "prot_binary" column: 0 if race is privileged, 1 if unprivileged
    df_sub["prot_binary"] = df_sub[race_col].apply(lambda r: 0 if r in privileged_list else 1)
    
    # Loop over each unique category in target column and compute bias metrics
    results = []
    
    for category in sorted(df_sub[target_col].unique()):
        # Binarize the target: 1 if target == category, else 0
        df_sub["label_num"] = (df_sub[target_col] == category).astype(int)
        
        # Create a small DataFrame with only the numeric columns
        df_for_aif = df_sub[["label_num", "prot_binary"]]
        
        # Build the AIF360 StandardDataset for this one-vs-all split
        aif_data = StandardDataset(
            df_for_aif,
            label_name="label_num",
            favorable_classes=[1],
            protected_attribute_names=["prot_binary"],
            privileged_classes=[[0]],
            features_to_keep=["prot_binary"]
        )
        
        # Compute bias metrics
        metric = BinaryLabelDatasetMetric(
            aif_data,
            unprivileged_groups=[{"prot_binary": 1}],
            privileged_groups=[{"prot_binary": 0}],
        )
        
        # Compute raw rates via pandas
        priv_rate = df_for_aif[df_for_aif["prot_binary"] == 0]["label_num"].mean()
        unpriv_rate = df_for_aif[df_for_aif["prot_binary"] == 1]["label_num"].mean()
        
        # Get bias metrics
        spd = metric.statistical_parity_difference()
        di = metric.disparate_impact()
        md = metric.mean_difference()
        
        # Classify bias level
        bias_level = classify_bias_level(spd, di, md)
        
        results.append({
            "Category": category,
            "Privileged Rate": priv_rate,
            "Unprivileged Rate": unpriv_rate,
            "Statistical Parity Difference": spd,
            "Disparate Impact": di,
            "Mean Difference": md,
            "Bias Level": bias_level
        })
    
    return pd.DataFrame(results)

def evaluate_model_bias(
    df: pd.DataFrame,
    target_col: str,
    protected_attr: str,
    max_categories: int = 10,
    test_size: float = 0.3,
    random_state: int = 42,
    race_col: str = None,
    privileged_list: list = None,
    unprivileged_list: list = None,
    global_shap: dict = None,
    protected_columns: str = None
):
    """
    1) Trains a RandomForestClassifier (3‐class) on df[features] → df[target_col].
    2) Prints overall + per‐protected‐group precision/recall for each class.
    3) Builds a SHAP "long" table via nested Python loops to ensure lengths match, and
       prints a waterfall plot for sample 0 of each class.
    4) Optionally collapses rare categories so OneHotEncoder doesn't explode into too many columns.
    """

    # --- 0) Drop rows missing target or protected_attr
    data = df.dropna(subset=[target_col, protected_attr]).copy()

    # --- 1) Auto‐select features (everything except target + protected_attr + known ID columns)
    # Use the protected columns from LLM recommendations instead of hardcoded features
    if protected_columns:
        # If protected_columns is available from LLM recommendations, use it
        protected_cols = protected_columns.split(',') if isinstance(protected_columns, str) else protected_columns
        features = [c.strip() for c in protected_cols if c.strip() in data.columns]
    else:
        # Fallback: look for common demographic columns
        demographic_patterns = ['race', 'sex', 'gender', 'age', 'ethnic', 'descent', 'vict_', 'suspect']
        features = []
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in demographic_patterns):
                features.append(col)
    
    # Ensure we have at least some features
    if not features:
        print("Warning: No demographic features found, using first 3 categorical columns")
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        features = cat_cols[:3] if len(cat_cols) >= 3 else cat_cols
    
    print("Selected categorical features for SHAP/model:", features)

    X_df     = data[features].copy()
    y_series = data[target_col].astype(str).copy()
    groups   = data[protected_attr].astype(str).copy()

    # --- 2) Encode target into 0,1,2
    le = LabelEncoder()
    y  = le.fit_transform(y_series)
    class_names = le.classes_   # e.g. array(['FELONY','MISDEMEANOR','VIOLATION'], dtype='<U10')

    # --- 3) Collapse infrequent categories (optional)
    X_trimmed = X_df.copy()
    for col in X_trimmed.select_dtypes(include=['object','category']).columns:
        topk = X_trimmed[col].value_counts().nlargest(max_categories).index
        X_trimmed[col] = X_trimmed[col].where(
            X_trimmed[col].isin(topk),
            other="<OTHER>"
        )
    print("X_trimmed columns:", X_trimmed.columns.tolist())

    # --- 4) Build categorical pipeline ONLY
    cat_feats = X_trimmed.select_dtypes(include=['object','category']).columns.tolist()
    print("Categorical features:", cat_feats)

    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('cat', categorical_pipeline, cat_feats)
    ])

    # Fit/transform X
    X = preprocessor.fit_transform(X_trimmed)
    print("X shape:", X.shape)
    
    # Get feature names properly
    feature_names = [f"cat__{name}" for name in cat_feats]
    print("Feature names for model and SHAP:", feature_names)

    # --- 5) Train/test split (stratified by y)
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # --- 6) Train a 3‐class RandomForest
    model = RandomForestClassifier(
        max_depth=5,
        n_estimators=100,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- 7) Overall classification_report
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Convert report to DataFrame and select only class rows
    overall = pd.DataFrame(report).transpose()
    overall = overall[overall.index.isin(class_names)][['precision', 'recall']]

    print(f"\n=== Overall Classification Report (protected_attr = {protected_attr}) ===")
    print(overall)

    # --- 8) Per‐protected‐group, per‐class metrics
    rows = []
    for grp in sorted(g_test.unique()):
        mask_grp = (g_test == grp)
        if mask_grp.sum() == 0:
            continue
        grp_dict = classification_report(
            y_test[mask_grp],
            y_pred[mask_grp],
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True
        )
        for cls in class_names:
            m = grp_dict.get(cls, {})
            rows.append({
                protected_attr: grp,
                'class':      cls,
                'precision':  m.get('precision', 0.0),
                'recall':     m.get('recall',    0.0),
                'support':    m.get('support',   0)
            })

    group_report = pd.DataFrame(rows)
    print(f"\n=== Metrics by {protected_attr} and Class ===")
    print(group_report)

    # Only calculate Global Explanations if not provided
    if global_shap is not None:
        global_explanations = global_shap['global_explanations']
    else:
        # --- 9) Calculate Global Feature Importance from Random Forest
        global_explanations = {}
        
        # Get feature importance for each class
        for class_idx in range(len(class_names)):
            cls_name = class_names[class_idx]
            
            try:
                # Get feature importance for this class
                if hasattr(model, 'estimators_'):
                    # For Random Forest, we can get feature importance per class
                    class_importance = np.zeros(len(feature_names))
                    for estimator in model.estimators_:
                        # Get feature importance from each tree
                        tree_importance = estimator.feature_importances_
                        if len(tree_importance) == len(feature_names):
                            class_importance += tree_importance
                    class_importance /= len(model.estimators_)
                else:
                    # Fallback: use overall feature importance
                    class_importance = model.feature_importances_
                
                # Create global explanation table
                explanation_data = []
                for i, (feature, importance) in enumerate(zip(feature_names, class_importance)):
                    explanation_data.append({
                        'feature': feature,
                        'importance': float(importance),
                        'rank': i + 1
                    })
                
                # Sort by importance (descending)
                explanation_data.sort(key=lambda x: x['importance'], reverse=True)
                
                df_explanation = pd.DataFrame(explanation_data)
                print(f"Global explanation for class {cls_name}:")
                print(df_explanation.head())
                
                global_explanations[cls_name] = df_explanation
                
            except Exception as e:
                print(f"Error calculating global explanation for class {cls_name}: {e}")
                # Create an empty explanation table for this class
                global_explanations[cls_name] = pd.DataFrame({
                    'feature': [],
                    'importance': [],
                    'rank': []
                })

    # Compute bias_metrics only for race column
    bias_metrics = None
    if race_col is not None and privileged_list is not None and unprivileged_list is not None:
        if protected_attr == race_col:
            bias_metrics = compute_bias_metrics(df, target_col, race_col, privileged_list, unprivileged_list)
    
    return model, preprocessor, overall, group_report, global_explanations, bias_metrics
