import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
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
        
        results.append({
            "Category": category,
            "Privileged Rate": priv_rate,
            "Unprivileged Rate": unpriv_rate,
            "Statistical Parity Difference": metric.statistical_parity_difference(),
            "Disparate Impact": metric.disparate_impact(),
            "Mean Difference": metric.mean_difference(),
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
    unprivileged_list: list = None
):
    """
    1) Trains a RandomForestClassifier (3‐class) on df[features] → df[target_col].
    2) Prints overall + per‐protected‐group precision/recall for each class.
    3) Builds a SHAP "long" table via nested Python loops to ensure lengths match, and
       prints a waterfall plot for sample 0 of each class.
    4) Optionally collapses rare categories so OneHotEncoder doesn't explode into too many columns.

    Returns:
      - model
      - fitted ColumnTransformer
      - overall classification_report (DataFrame)
      - per‐group report (DataFrame)
      - a dict of SHAP‐tables: { class_name → DataFrame(row_id, feature, feature_value, base_value, shap_value) }
    """

    # --- 0) Drop rows missing target or protected_attr
    data = df.dropna(subset=[target_col, protected_attr]).copy()

    # --- 1) Auto‐select features (everything except target + protected_attr + known ID columns)
    exclude = {
        target_col, protected_attr,
        
    }
    features = [c for c in data.columns if c not in exclude]

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

    # --- 4) Build numeric + categorical pipelines
    num_feats = X_trimmed.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_feats = X_trimmed.select_dtypes(include=['object','category']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale',   StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_feats),
        ('cat', categorical_pipeline, cat_feats)
    ])

    # Fit/transform X
    X = preprocessor.fit_transform(X_trimmed)
    feature_names = preprocessor.get_feature_names_out()

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
    overall = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            output_dict=True
        )
    ).transpose()[['precision','recall','f1-score']]

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

    # --- 9) Build a SHAP TreeExplainer on the RF, using an Independent masker on X_train
    masker = shap.maskers.Independent(X_train, max_samples=100)
    explainer = shap.TreeExplainer(model, data=masker)
    sv = explainer(X_test, check_additivity=False)
    shap_tables = {}
    for class_idx in range(len(class_names)):
        cls_name = class_names[class_idx]
        exp_cls = shap.Explanation(
            sv.values[:, class_idx, :],
            sv.base_values[:, class_idx],
            data=X_test,
            feature_names=feature_names
        )
        n_test, n_feats = exp_cls.values.shape
        row_ids = []
        features_list = []
        feat_values = []
        base_values = []
        shap_values = []
        for i in range(n_test):
            base_i = exp_cls.base_values[i]
            for j in range(n_feats):
                row_ids.append(i)
                features_list.append(feature_names[j])
                feat_values.append(exp_cls.data[i, j])
                base_values.append(base_i)
                shap_values.append(exp_cls.values[i, j])
        df_shap = pd.DataFrame({
            'row_id': row_ids,
            'feature': features_list,
            'feature_value': feat_values,
            'base_value': base_values,
            'shap_value': shap_values
        })
        shap_tables[cls_name] = df_shap
    # Compute bias_metrics only for race column
    bias_metrics = None
    if race_col is not None and privileged_list is not None and unprivileged_list is not None:
        if protected_attr == race_col:
            bias_metrics = compute_bias_metrics(df, target_col, race_col, privileged_list, unprivileged_list)
    return model, preprocessor, overall, group_report, shap_tables, bias_metrics
