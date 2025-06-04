import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def evaluate_model_bias(df: pd.DataFrame,
                        target_col: str,
                        protected_attr: str,
                        features: list = None,
                        test_size: float = 0.3,
                        max_iter: int = 1000,
                        random_state: int = 42):
    """
    Trains a logistic regression (binary or multinomial) to predict a categorical target 
    and computes precision & recall per protected group for each class.
    Handles missing values by imputing numeric and categorical features and ensures
    consistent class labels for group-level reports.
    """
    # Drop missing in target and protected_attr
    data = df.dropna(subset=[target_col, protected_attr])
    
    # Auto-select features
    if features is None:
        exclude = {target_col, protected_attr}
        features = [c for c in df.columns if c not in exclude]
    
    X = data[features]
    y = data[target_col].astype(str)
    groups = data[protected_attr].astype(str)
    
    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(range(len(le.classes_)))
    class_names = le.classes_
    
    # Pipelines
    num_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_feats = X.select_dtypes(include=['object','category']).columns.tolist()
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_feats),
        ('cat', categorical_pipeline, cat_feats)
    ])
    
    X_proc = preprocessor.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X_proc, y_enc, groups, test_size=test_size,
        stratify=y_enc, random_state=random_state
    )
    
    # Train
    model = LogisticRegression(max_iter=max_iter, multi_class='auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Overall report
    overall_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    overall = {}
    for cls in class_names:
        if cls in overall_dict:
            overall[cls] = {
                'precision': overall_dict[cls]['precision'],
                'recall': overall_dict[cls]['recall'],
                'f1-score': overall_dict[cls]['f1-score']
            }
    
    # Group-wise report
    rows = []
    for grp in sorted(g_test.unique()):
        mask = g_test == grp
        if mask.sum() == 0:
            continue
        grp_dict = classification_report(
            y_test[mask], y_pred[mask],
            labels=classes, target_names=class_names, output_dict=True
        )
        for cls_name in class_names:
            if cls_name in grp_dict:
                cls_metrics = grp_dict[cls_name]
                rows.append({
                    protected_attr: grp,
                    'class': cls_name,
                    'precision': cls_metrics['precision'],
                    'recall': cls_metrics['recall'],
                    'support': cls_metrics['support']
                })

    # ---- SHAP Explanation ----
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    explainer = shap.Explainer(model, X_train_df)
    shap_values = explainer(X_test_df.iloc[:1])

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    shap_html = f"<img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"
    
    return {
        'overall': overall,
        'group_report': rows,
        'class_names': class_names.tolist(),
        'shap_html': shap_html
    }

